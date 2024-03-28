from __future__ import unicode_literals, print_function, division
import json
from transformers.file_utils import ModelOutput
from buildtool.corenlp_preprocess_handlers import *
from buildtool.annotated_utils import (
    AnnotateSentenceIteratorByField,
    AnnotateSentenceIteratorByFields
)


class AnnotatedFeatureExtractor():
    def __init__(
        self,
        first_sentence_only,
        missing_token,
        lower_case,
        self_loop_root=True,
        use_ner=False,
        aws_masker=None
    ):
        self.ner = NerParsingProcess() if use_ner else None
        self.first_sentence_only = first_sentence_only
        self.missing_token = missing_token
        self.lower_case = lower_case
        self.self_loop_root = self_loop_root
        self.aws_masker = aws_masker

    def __call__(self, line, doc, logger):
        annotation = json.loads(line, strict=False)
        doc_feat = self.get_token_document(
                        annotation,
                        doc,
                        self.first_sentence_only,
                        self.lower_case,
                        logger
                    )

        ner_feat = {}
        if self.ner:
            ner_feat = self.get_ner_feature(
                            annotation,
                            first_sentence_only=self.first_sentence_only
                        )
        return ModelOutput({**doc_feat,
                            **ner_feat})

    def get_token_document(self, annotation, doc, first_sentence_only, lower_case, logger):
        titer = AnnotateSentenceIteratorByField(annotation, "tokens",
                                                first_sentence_only=first_sentence_only)
        document = []
        sent_sizes = []
        for tokens in titer: # Iterate tokens of each sentence
            sent_sizes.append(len(tokens))
            # Concatenate tokens of a sentence
            sentence = " ".join([token["originalText"].lower() \
                                    if lower_case else token["originalText"] \
                                        for token in tokens])
            document.append([sentence.strip()])
        mask = None
        if self.aws_masker:
            mask = self.aws_masker(document, doc, logger)
        return {"tdoc":document, "tdoc_mask":mask, "sent_sizes":sent_sizes}

    def get_ner_feature(self, annotation, first_sentence_only):
        sentence_fields = ["index", "entitymentions", "tokens"]
        iter = AnnotateSentenceIteratorByFields(
                    annotation,
                    sentence_fields,
                    first_sentence_only=first_sentence_only
                )
        all_mentioned = []
        sentence_stats = {}
        for sentence in iter:
            ner = self.ner(sentence)
            sentence_stats[ner["sentenceindex"]] = ner["sentencelength"]
            if len(ner["entitymentions"]) > 0:
                all_mentioned.append({
                    "sentenceindex": ner["sentenceindex"],
                    "entitymentions": ner["entitymentions"],
                })
        return {"ners": all_mentioned,
                "sentencestat": sentence_stats}
