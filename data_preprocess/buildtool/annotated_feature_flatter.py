from __future__ import unicode_literals, print_function, division
import itertools
from operator import add
from buildtool.annotated_utils import cumulative


class AnnotatedFeatureFlatter():
    def __init__(self):
        super().__init__()

    def __call__(self, doc, doc_mask, doc_sentence_sizes, tk_doc, logger):
        doc = self.flatten_doc(doc, logger)
        doc_size = len(doc.split(" "))

        struct_data = {"doc": doc,
                       "sentence_sizes": doc_sentence_sizes}

        if sum(doc_sentence_sizes) != doc_size:
            raise ValueError(f"Annotated sentences mismatch doc: {sum(doc_sentence_sizes)} != {doc_size}")

        doc_mask = self.flatten_feature(doc_mask, logger)
        struct_data["doc_mask"] = doc_mask
        if len(doc_mask) != doc_size:
            raise ValueError(f"Annotated token mismatch doc: {len(doc_mask)} != {doc_size}")

        input_ids, attention_masks, token_masks = None, None, None
        if tk_doc:
            input_ids, attention_masks, token_masks = self.flatten_tk_doc(tk_doc, logger)
            struct_data["input_ids"] = input_ids
            struct_data["attention_mask"] = attention_masks
            struct_data["token_mask"] = token_masks
            if sum(token_masks) != doc_size + 2: # Count BOS and EOS
                raise ValueError(f"Tokens mismatch doc: {sum(token_masks)} != {doc_size + 2}")

        return struct_data

    @staticmethod
    def flatten_feature_by_offset(feature, logger):
        if isinstance(feature, str):
            feature = eval(feature)
        values = [len(feat) for feat in feature]
        cumsum = cumulative(values)
        cumsum = cumsum[:len(feature)]
        # Offset
        for i, (aa, cc) in enumerate(zip(feature, cumsum)):
            feature[i] = list(map(add, aa, [cc]*len(aa)))
        # Flatten
        feature = list(itertools.chain(*feature))
        return feature

    @staticmethod
    def flatten_feature(feature, logger):
        if isinstance(feature, str):
            feature = eval(feature)
        feature = list(itertools.chain(*feature))
        return feature

    @staticmethod
    def flatten_tk_doc(doc, logger):
        if isinstance(doc, str):
            doc = eval(doc)
        size = len(doc)
        input_ids = []
        attention_masks = []
        token_masks = []
        for i, sentence in enumerate(doc):
            if i == 0:
                input_ids += sentence[0]["input_ids"][:-1]
                attention_masks += sentence[0]["attention_mask"][:-1]
                token_masks += sentence[0]["token_mask"][:-1]
            elif i == size - 1:
                input_ids += sentence[0]["input_ids"][1:]
                attention_masks += sentence[0]["attention_mask"][1:]
                token_masks += sentence[0]["token_mask"][1:]
            else:
                input_ids += sentence[0]["input_ids"][1:-1]
                attention_masks += sentence[0]["attention_mask"][1:-1]
                token_masks += sentence[0]["token_mask"][1:-1]
        return input_ids, attention_masks, token_masks

    @staticmethod
    def flatten_doc(doc, logger):
        if isinstance(doc, str):
            doc = eval(doc)
        doc = list(itertools.chain(*doc))
        doc = " ".join(doc)
        return doc
