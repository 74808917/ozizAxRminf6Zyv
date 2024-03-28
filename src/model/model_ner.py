from __future__ import unicode_literals, print_function, division
import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, CONFIG_MAPPING, BartConfig
from transformers import BartForConditionalGeneration
from model.ner_certainty import NerCertainty
from model.ner_factum import NerFactum

class Model(nn.Module):
    def __init__(self, args, options, vocabs, logger, **kwargs):
        super().__init__()

        # Instantiate baseline module
        if args.base_model_pretrained_name is not None:
            logger.info("Creating seq2seq model from pretrained weights.")
            self.seq2seq = BartForConditionalGeneration.from_pretrained(args.base_model_pretrained_name)
        elif args.base_model_config_name is not None:
            config = AutoConfig.from_pretrained(args.base_model_config_name)
            logger.info("Creating seq2seq model from scratch using pretrained configuration.")
            self.seq2seq = BartForConditionalGeneration(config)
        elif options.base_model is not None:
            logger.info("Creating seq2seq model from configuration.")
            config = options.base_model.to_dict()
            self.seq2seq = BartForConditionalGeneration(BartConfig(**config))
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            logger.info("Creating seq2seq model from scratch.")
            self.seq2seq = BartForConditionalGeneration(config)

        vocab_size = kwargs.pop("vocab_size")
        self.seq2seq.resize_token_embeddings(vocab_size)

        if options.aux_model.topicy.ner_certainty.source_ner:
            self.source_lm_head = copy.deepcopy(self.seq2seq.get_output_embeddings())
            self.source_final_logits_bias = copy.deepcopy(self.seq2seq.final_logits_bias)
            self.seq2seq._init_weights(self.source_lm_head)

        if not options.aux_model.topicy.transport_optim.on:
            self.topical = NerCertainty(options.aux_model.topicy.ner_certainty)
        else:
            self.topical = NerFactum(options.aux_model.topicy)

    def forward(self, batch, options, iepoch):
        inputs = batch[0]
        struct_inputs = batch[1]
        struct_labels = batch[2]

        outputs = self.seq2seq(
                        **inputs,
                        output_attentions=True,
                        output_hidden_states=True,
                    )

        m_output = {}
        m_output["cost"] = outputs.loss.cpu()

        h_x = outputs.encoder_hidden_states[-1]
        h_y = outputs.decoder_hidden_states[-1]

        x_lm_logits = None
        if options.aux_model.topicy.ner_certainty.source_ner:
            self.source_final_logits_bias = self.source_final_logits_bias.to(device=h_x.device)
            x_lm_logits = self.source_lm_head(h_x) + self.source_final_logits_bias

        tp_inputs = {
                        "device": h_y.device,
                        "h_x": h_x,
                        "h_y": h_y,
                        "x_logits": x_lm_logits,
                        "y_logits": outputs.logits,
                        "attention_mask": inputs["attention_mask"],
                        "struct_inputs": struct_inputs,
                        "decoder_attention_mask": inputs["decoder_attention_mask"],
                        "struct_labels": struct_labels
                    }
        tp_cost = self.topical.forward(options.aux_model.topicy, tp_inputs)
        m_output["cost"] += tp_cost.cpu()

        return m_output

    @torch.no_grad()
    def generate(
        self,
        batch,
        options,
        **model_kwargs,
    ):
        inputs = batch[0]

        return self.seq2seq.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **model_kwargs
                )
