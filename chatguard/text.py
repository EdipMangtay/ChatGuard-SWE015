"""Shared normalization and an independent rule baseline."""
import html
import re
import unicodedata

from functools import lru_cache

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION = re.compile(r"(?<!\w)@\w+")
LEET = str.maketrans("013457@$", "oieastas")
REPEAT = re.compile(r"([a-z])\1{2,}")
LEXICON = frozenset({
    "asshole", "bastard", "bitch", "bullshit", "dumbass", "fuck",
    "fucking", "motherfucker", "shit", "shitty",
})


def normalize(raw: str) -> str:
    text = html.unescape(raw).casefold().replace("ı", "i").replace("’", "'")
    text = "".join(
        character for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    text = URL.sub(" urltoken ", text)
    text = MENTION.sub(" usertoken ", text)
    text = re.sub(r"(?<!\w)rt(?=\s+usertoken)", " ", text)
    text = re.sub(r"(?<=[a-z])[013457@$](?=[a-z013457@$])|(?<=[a-z])[$@]",
                  lambda match: match[0].translate(LEET), text)
    text = REPEAT.sub(r"\1\1", text)
    return " ".join(TOKEN.findall(text))


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


@lru_cache(maxsize=4096)
def lexicon_match(token: str) -> bool:
    if token in LEXICON:
        return True
    return len(token) >= 4 and any(
        abs(len(candidate) - len(token)) <= 1 and distance(candidate, token) == 1
        for candidate in LEXICON
    )


def rule_label(normalized: str) -> int:
    return int(any(map(lexicon_match, normalized.split())))
