from argmax_shim.argmax.custom_vocabulary import FuzzyKeywordCorrector, sanitize_vocabulary
from argmax_shim.whisperkit.datatypes import WordTiming


def _w(*tokens):
    return [WordTiming(" " + t, [], i, i + 0.5) for i, t in enumerate(tokens)]


def test_multiword_split_is_merged():
    words, n = FuzzyKeywordCorrector(["WhisperKitPro"]).correct(_w("try", "whisper", "kit", "pro", "today"))
    assert [w.word.strip() for w in words] == ["try", "WhisperKitPro", "today"]
    assert n == 1 and words[1].start == 1 and words[1].end == 3.5


def test_close_misspelling_is_fixed_and_unrelated_left_alone():
    words, _ = FuzzyKeywordCorrector(["Argmax"]).correct(_w("arg", "max", "is", "great"))
    assert [w.word.strip() for w in words] == ["Argmax", "is", "great"]
    words, _ = FuzzyKeywordCorrector(["Rongxiang"]).correct(_w("rongshiang", "said", "hi"))
    assert words[0].word.strip() == "Rongxiang"
    words, n = FuzzyKeywordCorrector(["Argmax"]).correct(_w("the", "weather", "is", "nice"))
    assert n == 0


def test_sanitize_drops_invalid():
    assert sanitize_vocabulary(["Argmax", "bad<tag>", "  ", "AT&T"]) == ["Argmax", "AT&T"]
