from __future__ import unicode_literals, print_function, division
import torch.nn as nn
from model.ner_data_utils import (
    NerDataUtils,
    covert_word_to_token_spans
)
from model.wdistance import WassersteinDistance
from model.ner_certainty_r import NerCertaintyR


class NerFactum(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ner_certainty = NerCertaintyR(config.ner_certainty)
        self.cwd = None
        if config.transport_optim.on:
            self.cwd = WassersteinDistance(config.transport_optim)

    def forward(self, config, inputs):
        x_ner_token_spans, x_ner_token_span_masks = None, None
        if config.ner_certainty.source_ner:
            # X
            x_struct = inputs["struct_inputs"]
            x_ner_indices, x_ner_masks, x_ner_ctxts = NerDataUtils.prepare_data(
                                                h_states=inputs["x_logits"],
                                                attention_mask=inputs["attention_mask"],
                                                struct=x_struct,
                                            )
            # Convert word-level ner data to token-level ner span
            x_ner_token_spans, x_ner_token_span_masks, x_max_span_size = \
                covert_word_to_token_spans(token_mask=x_struct["token_mask"],
                                        token_mask_mask=x_struct["token_mask_mask"],
                                        indices=x_ner_indices,
                                        indices_mask_mask=x_ner_masks,
                                        token_span_count=x_struct["subword_span"],
                                        token_span_count_mask=x_struct["subword_span_mask"],
                                        indices_offset_n=1, # offset due to model-prepended BOS token.
                )

        y_ner_token_spans, y_ner_token_span_masks = None, None
        if config.ner_certainty.summary_ner:
            # Y
            y_struct = inputs["struct_labels"]
            y_ner_indices, y_ner_masks, y_ner_ctxts = NerDataUtils.prepare_data(
                                                h_states=inputs["y_logits"],
                                                attention_mask=inputs["decoder_attention_mask"],
                                                struct=y_struct,
                                            )
            y_ner_token_spans, y_ner_token_span_masks, y_max_span_size = \
                covert_word_to_token_spans(token_mask=y_struct["token_mask"],
                                        token_mask_mask=y_struct["token_mask_mask"],
                                        indices=y_ner_indices,
                                        indices_mask_mask=y_ner_masks,
                                        token_span_count=y_struct["subword_span"],
                                        token_span_count_mask=y_struct["subword_span_mask"],
                                        indices_offset_n=1, # offset due to model-prepended BOS token.
                )

        cost = 0.0
        if config.ner_certainty.source_ner or config.ner_certainty.summary_ner:
            # Uncertainty cost
            device = inputs["x_logits"].device if config.ner_certainty.source_ner else inputs["y_logits"].device
            certainty_inputs = {
                "source_ner": {
                    "logits": inputs["x_logits"],
                    "ner_token_spans": x_ner_token_spans,
                    "ner_token_span_masks": x_ner_token_span_masks,
                },
                "summary_ner": {
                    "logits": inputs["y_logits"],
                    "ner_token_spans": y_ner_token_spans,
                    "ner_token_span_masks": y_ner_token_span_masks
                },
                "device": device
            }
            cost = self.ner_certainty.forward(config.ner_certainty, certainty_inputs) + cost

        if self.cwd is not None:
            # Transport y to x.
            source = inputs["h_x"] if config.transport_optim.x2y else inputs["h_y"]
            source_mask = inputs["attention_mask"] if config.transport_optim.x2y else inputs["decoder_attention_mask"]
            target = inputs["h_y"] if config.transport_optim.x2y else inputs["h_x"]
            target_mask = inputs["decoder_attention_mask"] if config.transport_optim.x2y else inputs["attention_mask"]
            cwd_cost = self.cwd(
                source=source,
                source_mask=source_mask,
                target=target,
                target_mask=target_mask,
            )
            cost = cwd_cost + cost

        return cost
