from __future__ import unicode_literals, print_function, division
import itertools
import torch


def word_to_token_index_map(token_span, token_span_mask):
    '''
        Example:
            model-tokenized doc indexing:    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,...
                                            ---------------------------------------
            doc token (of word) span :      [1, 1, 2, 1, 1, 2, 2, ...]
            doc token span cumsum:          [1, 2, 4, 5, 6, 8, 10,...] <- word token end indexing (exclusive)
                                         -  token span
                                            ---------------------------------------
                                            [0, 1, 2, 4, 5, 6, 8, ...] <- word token start indexing (inclusive)
                                            ---------------------------------------------
            word-token span bounds:         [[0,1),[1,2),[2,4),[4,5),[5,6),[6,8),[8,10),...]
               word-to-token indexing:             ^     ^     ^     ^     ^     ^      ^
            After model-prepend <BOS>:      [      1,    2,    3,    4,    5,    6,     7]
            word index:                     [      0,    1,    2,    3,    4,    5,     6]
    '''
    token_span_ends = torch.cumsum(token_span*token_span_mask, dim=-1)
    token_span_starts = token_span_ends - token_span
    word_token_index_map = torch.cat((token_span_starts[...,None], token_span_ends[...,None]), dim=-1)
    return word_token_index_map

def covert_word_to_token_spans(
    token_mask,
    token_mask_mask,
    indices,
    indices_mask_mask,
    token_span_count, # subword_span
    token_span_count_mask,
    indices_offset_n=0,
):
    '''
        Map the word-level indices to the corresponding token-level first token indices of
        words tokenized by the word-segmentation token encoding methods e.g. byte-pair encoding.

        Parameters:
            token_mask:
                The first token mask of each word in the model tokenized sequence.
            token_mask_mask:
                Token batch mask.
            indices:
                out of order word indices and may be a subset of word indices.
            indices_mask:
                The first word mask of each entity (by indices).
                An entity is formed of an one or more words.
            indices_mask_mask:
                Indices batch mask.
            token_span_count:
                Token (model tokenized tokens) span of each word.
            token_span_count_mask:
                Token sequence batch mask.
            indices_offset_n:
                Mapping to tokens starts after the model takenizer's prepended tokens (e.g. BOS) 
                assuming that the prepended tokens are single-token-per-word.
                So, right-shift word-level indices by indices_offset_n.
            batch_offset_required:
                Offset indices by batch-wide sequence length if True.
                Has an effect only when flat_batch is True.

        Return:
            A tensor of token indices, and a mask to identify indices' batch dimensions.

        Note:
            A sample may not have NER, for example,
                indices = [[tensor([0]), tensor([23])], []]
                indices_mask_mask = [[1, 1], []]
    '''
    indices = [[idx+indices_offset_n for idx in bidx] for bidx in indices]

    # Each entry of the token span cumsums corresponds to the first token index (of a word)
    # while the indexing position to the token_span_count_cumsums 'list' itself has one-to-one 
    # mapping to input word-level indices.
    # Thus, acquiring token indices from token_span_count_cumsums can be done
    # simply by using the input word-level indices to slice it.
    # For example, use 2 in the input word-level indices to get the first token index
    # by token_span_count_cumsums[2].
    word_token_index_map = word_to_token_index_map(token_span_count, token_span_count_mask)

    nb, _ = token_mask.size()
    batch_token_spans = []
    batch_token_span_masks = []
    max_span_size = 0
    for ib in range(nb):
        token_spans = []
        token_span_masks = []
        for idxs in indices[ib]:
            tok_mapping = word_token_index_map[ib,idxs]
            n_mapping, _ = tok_mapping.size()
            first_tok_index = tok_mapping[0,0] # Inclusive
            last_tok_index = tok_mapping[n_mapping-1,-1] # Exclusive
            tok_span = torch.arange(start=first_tok_index,
                                    end=last_tok_index,
                                    device=tok_mapping.device)
            token_spans.append(tok_span)
            span_mask = torch.ones_like(tok_span, device=tok_span.device).long()
            token_span_masks.append(span_mask)
            max_span_size = max(max_span_size, span_mask.shape[-1])
        batch_token_spans.append(token_spans)
        batch_token_span_masks.append(token_span_masks)
    return batch_token_spans, batch_token_span_masks, max_span_size


class NerDataUtils():
    @staticmethod
    def prepare_data(
        h_states,
        attention_mask,
        struct,
        ignore_value=-100
    ):
        '''
            A sample:
                "named_entity": [
                    {"text": "3", "ner": 5, "sentenceindex": 0, "mentionindex": 0, "sentTokenBegin": 12, "docTokenBegin": 12, "numTokens": 1},
                    {"text": "model", "ner": 6, "sentenceindex": 0, "mentionindex": 1, "sentTokenBegin": 19, "docTokenBegin": 19, "numTokens": 1},
                    {"text": "September last year", "ner": 3, "sentenceindex": 1, "mentionindex": 0, "sentTokenBegin": 16, "docTokenBegin": 37, "numTokens": 3},
                    {"text": "model", "ner": 6, "sentenceindex": 4, "mentionindex": 0, "sentTokenBegin": 9, "docTokenBegin": 96, "numTokens": 1}
                ]
        '''
        nb = h_states.shape[0]
        batch_ner_indices = []
        batch_ner_masks = []
        batch_ner_ctxts = [] # as its sentence.
        for ib in range(nb):
            ners = struct["named_entity"][ib]
            ner_indices = []
            ner_masks = []
            ner_ctxts = []
            for ner in ners:
                ner_index=torch.arange(start=ner["docTokenBegin"],
                             end=ner["docTokenBegin"]+ner["numTokens"],
                             device=h_states.device)
                ner_indices.append(ner_index)
                ner_masks.append(ner["numTokens"])
                ner_ctxts.append(ner["sentenceindex"])
            batch_ner_indices.append(ner_indices)
            batch_ner_masks.append(ner_masks)
            batch_ner_ctxts.append(ner_ctxts)
        return batch_ner_indices, batch_ner_masks, batch_ner_ctxts

    @staticmethod
    def get_ner_hidden_states(
        h_states,
        ner_token_spans,
        ner_token_span_masks,
        ner_ctxt=None,
        return_tensor=True
    ):
        '''
            Return may be empty in case that no named entity exists.
        '''
        def get_max_span_size(ner_token_spans):
            c_spans = list(itertools.chain.from_iterable(ner_token_spans))
            if len(c_spans) > 0:
                max_span_size = max([len(span) for span in c_spans])
            else:
                max_span_size = 0
            return max_span_size
        nb, nl, nd = h_states.size()
        h_spans = []
        h_masks = []
        h_ctxts = []
        batch_count = [0]*nb
        max_span_size = get_max_span_size(ner_token_spans)
        if max_span_size > 0:
            for ib in range(nb):
                batch_count[ib] += len(ner_token_spans[ib])
                if batch_count[ib] == 0:
                    continue
                for span, mask in zip(ner_token_spans[ib], ner_token_span_masks[ib]):
                    h_span = h_states[ib,span]
                    if return_tensor:
                        n_pad = max_span_size - sum(mask)
                        h_pad = torch.zeros((n_pad, h_span.shape[-1]),
                                            dtype=h_span.dtype,
                                            device=h_span.device)
                        h_span = torch.cat((h_span, h_pad), dim=0)
                        h_mask = torch.tensor(mask.tolist() + [0]*n_pad, device=mask.device)
                    else:
                        h_mask = mask
                    h_spans.append(h_span)
                    h_masks.append(h_mask)
                if ner_ctxt is not None:
                    h_ctxts += ner_ctxt[ib] # Flatten

            if return_tensor:
                h_spans = torch.stack(h_spans, dim=0)
                h_masks = torch.stack(h_masks, dim=0)
                batch_count = torch.tensor(batch_count, device=h_states.device)
                if len(h_ctxts) > 0:
                    h_ctxts = torch.tensor(h_ctxts, device=h_states.device)

        return (h_spans, h_masks, h_ctxts, batch_count)
