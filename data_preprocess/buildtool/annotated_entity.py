from __future__ import unicode_literals, print_function, division


class AnnotatedEntity():
    '''
        Link entities between source document and summary.
    '''
    NER_INTEREST = ["NUMBER", "ORDINAL", "MONEY", "PERCENT",
                    "DATE", "TIME", "DURATION",
                    "ORGANIZATION", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY",
                    "DEGREE", "TITLE", "PERSON", "NATIONALITY",
                    "IDEOLOGY", "RELIGION",
                    "CAUSE_OF_DEATH", "CRIMINAL_CHARGE",
                    # "EMAIL", "HANDLE", "URL", "SET", "MISC"
                ]

    def __init__(self, len_threshold=10):
        self.len_threshold = len_threshold

    def __call__(self, ners):
        entities = []
        for a_ner in ners:
            for idx, a_men in enumerate(a_ner["entitymentions"]):
                if a_men["ner"] in self.NER_INTEREST:
                    if self.len_threshold >= len(a_men["text"].split()):
                        entity = {  "text": a_men["text"],
                                    "ner": a_men["ner"],
                                    "sentenceindex": a_ner["sentenceindex"],
                                    "mentionindex": idx,
                                    "sentTokenBegin": a_men["tokenBegin"],
                                    "docTokenBegin": a_men["docTokenBegin"],
                                    "numTokens": a_men["docTokenEnd"]-a_men["docTokenBegin"]
                                }
                        entities.append(entity)
        return entities
