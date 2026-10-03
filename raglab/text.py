"""Text utilities: tokenization and stopwords shared by all retrievers."""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an and are as at be by for from has he in is it its of on that the to was
    were will with you your this these those they them their then than so such
    or not no nor but if into over under between both each few more most other
    some only own same too very can just don should now what which who whom
    when where why how all any there here been being have had having do does
    did doing would could ought i me my we our ours out about after before
    during above below up down
    """.split()
)


def tokenize(text, stem=True):
    """Lowercase alphanumeric tokens with stopwords removed."""
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return [stem_word(t) if stem else t for t in tokens]


def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


_VOWELS = frozenset("aeiou")


def _measure(word):
    """Porter 'measure' m: count of VC sequences."""
    m, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in _VOWELS or (ch == "y" and not prev_vowel)
        if not is_vowel and prev_vowel:
            m += 1
        prev_vowel = is_vowel
    return m


def _has_vowel(word):
    return any(ch in _VOWELS for ch in word)


def stem_word(word):
    """Porter-style stemmer: strips plurals, -ing/-ed, -ly, -tion, etc.

    Collapses morphological variants ('dream'/'dreaming'/'dreams' -> 'dream')
    so retrieval isn't defeated by word form. Not a full Porter implementation,
    but covers the suffixes that matter for search.
    """
    if len(word) <= 3:
        return word
    # plurals
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-3] + "i"
    elif word.endswith("ss"):
        pass
    elif word.endswith("s") and not word.endswith("us"):
        word = word[:-1]
    # -ing / -ed
    if word.endswith("ing") and len(word) > 5 and _has_vowel(word[:-3]):
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4 and _has_vowel(word[:-2]):
        word = word[:-2]
    # agent nouns: computer -> comput (matches computing -> comput)
    if word.endswith("er") and len(word) > 4 and _measure(word[:-2]) > 1:
        word = word[:-2]
    # -ly, -tion/-sion, -ness, -ment, -ity, -ous, -ive (require m > 1 like Porter)
    for suffix, cut in (("ational", 5), ("tional", 2), ("ly", 2), ("ness", 4),
                        ("ment", 4), ("ity", 3), ("ous", 3), ("ive", 3)):
        if word.endswith(suffix) and len(word) > len(suffix) + 2 and _measure(word[:-cut]) > 1:
            word = word[:-cut]
            break
    # trailing e (keep 'ee')
    if word.endswith("e") and not word.endswith("ee") and len(word) > 4 \
            and _measure(word[:-1]) > 1:
        word = word[:-1]
    return word
