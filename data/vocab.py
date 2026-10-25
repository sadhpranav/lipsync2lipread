"""Character vocabulary for CTC. Index 0 is the CTC blank."""
CHARS = " abcdefghijklmnopqrstuvwxyz"
BLANK = 0
NUM_CLASSES = len(CHARS) + 1  # 28
_C2I = {c: i + 1 for i, c in enumerate(CHARS)}


def encode(text):
    return [_C2I[c] for c in text.lower() if c in _C2I]


def decode(ids):
    return "".join(CHARS[i - 1] for i in ids if i != BLANK)


# GRID grammar: command colour preposition letter digit adverb
GRID_SLOTS = [
    ["bin", "lay", "place", "set"],
    ["blue", "green", "red", "white"],
    ["at", "by", "in", "with"],
    list("abcdefghijklmnopqrstuvxyz"),  # no 'w' in GRID
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    ["again", "now", "please", "soon"],
]
