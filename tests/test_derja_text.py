import pytest

from derja_text import cer, is_degenerate, normalize, wer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("أنا", "انا"),  # alef with hamza above
        ("إنتي، ماشي؟", "انتي ماشي"),  # hamza below, Arabic comma and question mark
        ("آش", "اش"),  # alef madda
        ("ٱلله", "الله"),  # alef wasla
        ("برشة", "برشه"),  # taa marbuta
        ("على", "علي"),  # alef maqsura
        ("م\u064eر\u0652ح\u064eب\u064bا", "مرحبا"),  # harakat
        ("كـــبير", "كبير"),  # tatweel
        ("ﻻ", "لا"),  # presentation-form ligature
        ("Bonjour, ça va ٣", "bonjour ça va 3"),  # Latin case, punctuation, Arabic-Indic digit
        ("۱۲", "12"),  # Eastern Arabic-Indic digits
        ("<bonjour>", "bonjour"),  # symbols
        ("  اهلا \u200f  بيك ", "اهلا بيك"),  # bidi mark and extra spaces
        ("بر\u061cشة", "برشه"),  # Arabic letter mark
        ("bon\u00adjour\u2060", "bonjour"),  # soft hyphen and word joiner
        ("و مشيت للمدرسة", "ومشيت للمدرسه"),  # standalone wa joins the next word
        ("مشيت و", "مشيت و"),  # trailing wa has nothing to join
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("refs", "hyps", "expected"),
    [
        (["a b c"], ["a x c"], 1 / 3),  # substitution
        (["a b"], ["a b c"], 1 / 2),  # insertion
        (["a b c"], ["a c"], 1 / 3),  # deletion
        (["a b"], [""], 1.0),  # empty hypothesis
        (["a b", "c d e f"], ["a b", "x d e f"], 1 / 6),  # pooled over the corpus, not averaged per clip
    ],
)
def test_wer(refs, hyps, expected):
    assert wer(refs, hyps) == pytest.approx(expected)


def test_cer_counts_a_substituted_character():
    assert cer(["ab"], ["ax"]) == pytest.approx(1 / 2)


def test_cer_counts_a_missing_space():
    assert cer(["ab cd"], ["abcd"]) == pytest.approx(1 / 5)


def test_wer_rejects_corpus_without_reference_words():
    with pytest.raises(ValueError):
        wer([""], ["a"])


def test_wer_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        wer(["a"], ["a", "b"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("!!!!", True),  # what fp16 NaN logits decode to
        (" ! ! ", True),
        ("يا شبينا", False),
        ("ok!", False),
        ("٣", False),
    ],
)
def test_is_degenerate(text, expected):
    assert is_degenerate(text) is expected
