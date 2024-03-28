from __future__ import unicode_literals, print_function, division
import torch
from model.ner_data_utils import (
    NerDataUtils,
    covert_word_to_token_spans
)


from abc import ABC, abstractmethod
class ECost(ABC):
    def __init__(self, chain_type, w_lambda):
        self.chain_type = chain_type
        self.w_lambda = w_lambda

    @abstractmethod
    def __call__(self, prob, mask, **kwargs):
        raise NotImplementedError("ECost (base class) call not implemented.")

    @staticmethod
    def create(config):
        functors = []
        for functor in config.functors:
            func = functor["functor"]
            if func == "info_gain":
                functors.append(InfoGainCost(functor["chain_type"], functor["w_lambda"]))
            elif func == "joint_entropy":
                functors.append(JointEntropyCost(functor["chain_type"], functor["w_lambda"]))
            else:
                raise ValueError(f"Unknown NER certainty functor {func}.")
        return functors

class JointEntropyCost(ECost):
    def __call__(self, prob, mask, **kwargs):
        epsilon = kwargs.pop("epsilon")
        reduction = kwargs.pop("reduction")
        avg_jentropy = kwargs.pop("avg_jentropy")
        chain_func = torch.sum if self.chain_type == "sum" else torch.cumsum
        need_keep_dim_flag = True if self.chain_type == "sum" else False

        prob = prob.clamp(min=epsilon, max=1.0)
        # Entropy derived from prob.
        entropy = -prob*prob.log()
        # nb, nl, nP = entropy.size()
        entropy = entropy*mask[...,None]
        entropy = entropy.mean(dim=-1)#dim=(-1,-2))
        dim = 0 if len(entropy.shape) == 1 else 1
        if need_keep_dim_flag:
            jentropy = chain_func(entropy, dim=dim, keepdim=True)
        else:
            jentropy = chain_func(entropy, dim=dim)
        nl = jentropy.shape[-1]
        jentropy = jentropy*mask[:,:nl]
        if avg_jentropy:
            if need_keep_dim_flag:
                chained_ns = chain_func(mask, dim=dim, keepdim=True)
            else:
                chained_ns = chain_func(mask, dim=dim)
            jentropy = jentropy / chained_ns[:,:nl]
        if reduction:
            entropy = entropy.mean(dim=-1)
            jentropy = jentropy.mean(dim=-1)
        return {"entropy": entropy, "cost": jentropy}

class InfoGainCost(ECost):
    def __call__(self, prob, mask, **kwargs):
        epsilon = kwargs.pop("epsilon")
        reduction = kwargs.pop("reduction")
        avg_jentropy = kwargs.pop("avg_jentropy")
        chain_func = torch.sum if self.chain_type == "sum" else torch.cumsum
        need_keep_dim_flag = True if self.chain_type == "sum" else False

        # Entropy derived from prob.
        prob = prob.clamp(min=epsilon, max=1.0)
        entropy = -prob*prob.log()
        entropy = entropy*mask[...,None]
        # Info gain.
        # nb, nl, nP = entropy.size()
        entropy_changed = torch.diff(
            entropy,
            dim=-2,
            # prepend=torch.zeros((nb,1,nP), device=entropy.device)
        )
        entropy = entropy.mean(dim=-1)#dim=(-1,-2))
        # To mask valid values.
        rolled_mask = torch.roll(mask, shifts=-1, dims=1)
        ig_mask = rolled_mask[:,:-1]
        entropy_changed = entropy_changed*ig_mask[...,None]
        if entropy_changed.shape[0] > 0 and entropy_changed.shape[1] > 0:
            # Only interested in increasing entropy.
            # entropy_changed = entropy_changed.clamp(min=0.0)
            entropy_changed = entropy_changed.mean(dim=-1)#dim=(-1,-2))
            # Chained changes
            if need_keep_dim_flag:
                entropy_changed = chain_func(entropy_changed, dim=1, keepdim=True)
            else:
                entropy_changed = chain_func(entropy_changed, dim=1)
            nl = entropy_changed.shape[-1]
            entropy_changed = entropy_changed*ig_mask[:,:nl]
            if avg_jentropy:
                applicable = ig_mask > 0
                applicable = applicable[:,:nl]
                if need_keep_dim_flag:
                    chained_ns = chain_func(ig_mask, dim=1, keepdim=True)
                else:
                    chained_ns = chain_func(ig_mask, dim=1)
                entropy_changed = entropy_changed[applicable] / chained_ns[applicable]
        else:
            nb = entropy.shape[0]
            entropy_changed = torch.zeros((nb,1), device=entropy_changed.device)
        if reduction:
            entropy = entropy.mean(dim=-1)
            # if entropy_changed.shape[0] > 0:
            entropy_changed = entropy_changed.mean(dim=-1)
            # else:
            #     entropy_changed = torch.tensor(0.0, device=entropy_changed.device)
        return {"entropy": entropy,
                "cost": entropy_changed}


class NerCertainty():
    def __init__(self, config, ner_vocab=None):
        self.functors = ECost.create(config)
        # self.ner_vocab = ner_vocab

    def forward(self, config, inputs):
        device = inputs["device"]
        cost = torch.tensor(0.0, device=device)
        if config.ner_certainty.source_ner:
            x_logits = inputs["x_logits"]
            attention_mask = inputs["attention_mask"]
            struct_inputs = inputs["struct_inputs"]
            src_cost = self.reduce_uncertainty(config, x_logits, attention_mask, struct_inputs)
            cost = cost + src_cost
        if config.ner_certainty.summary_ner:
            y_logits = inputs["y_logits"]
            decoder_attention_mask = inputs["decoder_attention_mask"]
            struct_labels = inputs["struct_labels"]
            smy_cost = self.reduce_uncertainty(config, y_logits, decoder_attention_mask, struct_labels)
            cost = cost + smy_cost
        return cost*config.ner_certainty.lambda_w

    def reduce_uncertainty(
        self,
        config,
        h_states,
        attention_mask,
        struct,
    ):
        # Prepare batched ner index data
        ner_indices, ner_masks, ner_ctxts = NerDataUtils.prepare_data(
                                                h_states,
                                                attention_mask,
                                                struct,
                                            )
        # Convert word-level ner data to token-level ner span
        ner_token_spans, ner_token_span_masks, max_span_size = \
            covert_word_to_token_spans(token_mask=struct["token_mask"],
                                        token_mask_mask=struct["token_mask_mask"],
                                        indices=ner_indices,
                                        indices_mask_mask=ner_masks,
                                        token_span_count=struct["subword_span"],
                                        token_span_count_mask=struct["subword_span_mask"],
                                        indices_offset_n=1, # offset due to model-prepended BOS token.
            )
        kwargs = {
            "epsilon": config.ner_certainty.epsilon,
            "reduction": True,
            "avg_jentropy": config.ner_certainty.avg_jentropy,
        }
        cost = self.compute_cost(h_states, ner_token_spans, ner_token_span_masks, **kwargs)
        return cost

    def compute_cost(
        self,
        h_states,
        ner_token_spans,
        ner_token_span_masks,
        **kwargs
    ):
        (ner_spans, ner_span_mask, _, ner_batch_count) = \
            NerDataUtils.get_ner_hidden_states(
                h_states=h_states,
                ner_token_spans=ner_token_spans,
                ner_token_span_masks=ner_token_span_masks,
                return_tensor=True
            )
        if len(ner_spans) > 0:
            ner_spans.masked_fill_(ner_span_mask[...,None]==0, -1e14)
            ner_probs = torch.softmax(ner_spans, dim=-1)
            cost = 0.0
            for functor in self.functors:
                estimate = functor(
                                ner_probs,
                                ner_span_mask,
                                **kwargs
                            )
                cost = estimate["cost"].mean()*functor.w_lambda + cost
        else:
            cost = 0.0
        return cost
