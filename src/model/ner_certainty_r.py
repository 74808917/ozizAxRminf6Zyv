from __future__ import unicode_literals, print_function, division
import torch
from model.ner_data_utils import (
    NerDataUtils
)
from model.ner_certainty import (
    ECost,
)

class NerCertaintyR():
    def __init__(self, config):
        self.functors = ECost.create(config)

    def forward(self, config, inputs):
        cost = torch.tensor(0.0, device=inputs["device"])
        if config.source_ner:
            src_cost = self.reduce_uncertainty(config, inputs["source_ner"])
            cost = cost + src_cost
        if config.summary_ner:
            smy_cost = self.reduce_uncertainty(config, inputs["summary_ner"])
            cost = cost + smy_cost
        return cost*config.lambda_w

    def reduce_uncertainty(
        self,
        config,
        data,
    ):
        kwargs = {
            "epsilon": config.epsilon,
            "reduction": True,
            "avg_jentropy": config.avg_jentropy,
        }
        cost = self.compute_cost(
                    data["logits"],
                    data["ner_token_spans"],
                    data["ner_token_span_masks"],
                    **kwargs
                )
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
