import os
import kenlm
import numpy as np
import re
import sys
from khmercut import tokenize
from khmernormalizer import normalize
from itertools import combinations, chain

RE_JUNKS = re.compile(r"[^\u1780-\u17ff a-zA-Z0-9\៖\?\!\"\(\)]+")

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def text_junk_combinations(text):
    items = []
    matches = [m for m in RE_JUNKS.finditer(text)]
    for combo in powerset(matches):
        t = text
        offset = 0
        for m in list(combo):
            t = t[0 : m.start() + offset] + t[m.end() + offset :]
            offset -= m.end() - m.start()
        items.append(t)
    return items

if __name__ == "__main__":
    if not os.path.exists("kenlm.bin"):
        sys.stderr.write("kenlm.bin file is missing.\n")
        exit(1)        
    model = kenlm.LanguageModel("kenlm.bin")
    text = sys.stdin.read()
    text = normalize(text)
    possiblities = text_junk_combinations(text)
    scores = list(map(lambda x: model.score(" ".join(tokenize(x))), possiblities))
    sys.stdout.write(possiblities[np.argmax(scores)])