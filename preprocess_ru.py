import pymorphy2
from pymystem3 import Mystem

mystem = Mystem()
morph = pymorphy2.MorphAnalyzer()

def ru_mystem(text, normalize=True, pos=True):
    joined = ""
    for an in mystem.analyze(text):
        if an.get('analysis',None):
            pos = an['analysis'][0]['gr'].split(",")[0]
            lex = an['analysis'][0]['lex']
            st = "_".join([lex,pos])
        else:
            st = an['text']
        joined += st
    return joined.strip()

def ru_opencorpora_normalize(text):
    text = " ".join([self.morph.parse(w)[0].normal_form for w in text.split(" ")])
    return text



