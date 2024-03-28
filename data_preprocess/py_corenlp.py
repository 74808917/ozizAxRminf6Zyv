from __future__ import unicode_literals, print_function, division
import requests, json

class PyCoreNLP(object):
    PROP_DEFAULT = {
        "annotators":"tokenize,ssplit,pos,lemma,parse",
        "outputFormat":"json"
    }
    PROP_NER = {
        "annotators":"tokenize,ssplit,pos,lemma,ner,parse",
        "outputFormat":"json"
    }
    PROP_NER_COREF = {
        "annotators":"tokenize,ssplit,pos,lemma,ner,parse,coref",
        "outputFormat":"json"
    }
    PROP_TOKENIZE = {
        "annotators":"tokenize,ssplit,pos,lemma",
        "outputFormat":"json"
        }
    PROP_SSPLIT = {
        "annotators":"ssplit",
        "outputFormat":"json"
    }

    URL_DEFAULT = "http://localhost:9000"

    def __init__(self, url = URL_DEFAULT):
        if url[-1] == '/':
            url = url[:-1]
        self.url = url

    def annotate(self, text, mode = None):
        if mode != None:
            prop = eval('self.'+mode)
        else:
            prop = self.PROP_DEFAULT

        r = requests.post(url=self.url,
                          params={"properties":str(prop)},
                          data=text.encode('utf-8'))
        output = json.loads(r.text, strict=False)
        return output
