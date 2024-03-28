from __future__ import unicode_literals, print_function, division
import regex

debugging=False

def stringify_ranges(ranges):
    if isinstance(ranges, list):
        return "_".join([str(i) for i in ranges])
    elif isinstance(ranges, str):
        return regex.sub(r"(\d+\)-(\d+)", r"\1_\2", ranges)
    else:
        return ranges


class NerParsingProcess():
    def __init__(self):
        self.menfields = [
            "ner",
            "text",
            "tokenBegin", "tokenEnd",
            "docTokenBegin", "docTokenEnd"
        ]

    def __call__(self, sentence):
        # Check if any annotated word has actually more than one words"
        # Specify spliting by whitespace to avoid spliting on special chars,
        # e.g. 0845\xa06010128 which may be a phone number.
        # But be aware of a side-effect that contiguous whitespaces are counted as words.
        # # whats = [token["word"].split(" ") for token in sentence["tokens"]]
        every_word_counts = [len(token["word"].split(" ")) for token in sentence["tokens"]]
        total_word_counts = sum(every_word_counts)
        sentence_length = len(sentence["tokens"])
        if sentence_length != total_word_counts:
            raise ValueError("Found word containing subwords.")

        mentioned = []
        for mentions in sentence["entitymentions"]:
            men = {k: mentions[k] for k in self.menfields}
            mentioned.append(men)

        return {"sentenceindex": sentence["index"],
                "sentencelength": sentence_length,
                "entitymentions": mentioned}