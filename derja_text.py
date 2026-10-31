"""Text normalisation and error rates for scoring Tunisian Derja ASR.

The normalisation follows the "normalized WER" used by TheBiggyBang/whisper-small-tunisian-derja,
so bake-off numbers stay comparable with that model card: it removes Derja spelling variants
that are not recognition errors.
"""

import re
import unicodedata
from collections.abc import Callable, Sequence

_HARAKAT_AND_TATWEEL = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06edـ]")
_SPELLING_VARIANTS = str.maketrans(
    "إأآٱىة٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "اااايه" + "0123456789" * 2,
)
_WA = "و"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # zero-width, bidi and other format marks
    text = _HARAKAT_AND_TATWEEL.sub("", text)
    text = text.lower().translate(_SPELLING_VARIANTS)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(_join_standalone_wa(text.split()))


def _join_standalone_wa(words: list[str]) -> list[str]:
    joined, prefix = [], ""
    for word in words:
        if word == _WA:
            prefix += word
            continue
        joined.append(prefix + word)
        prefix = ""
    if prefix:
        joined.append(prefix)
    return joined


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return _corpus_error_rate(refs, hyps, str.split)


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return _corpus_error_rate(refs, hyps, list)


def _corpus_error_rate(refs: Sequence[str], hyps: Sequence[str], units: Callable[[str], list[str]]) -> float:
    edits = total = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        ref_units = units(ref)
        edits += _edit_distance(ref_units, units(hyp))
        total += len(ref_units)
    if total == 0:
        raise ValueError("the references contain nothing to score")
    return edits / total


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def is_degenerate(text: str) -> bool:
    """True for empty or punctuation-only output, such as the '!!!!' that fp16 NaNs decode to."""
    return not any(ch.isalnum() for ch in text)
