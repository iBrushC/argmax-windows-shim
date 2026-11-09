from argmax_shim.speakerkit import DiarizationResult, SpeakerInfoStrategy, SpeakerKit, SpeakerSegment
from argmax_shim.speakerkit.alignment import relabel_by_first_appearance, speaker_for_interval
from argmax_shim.whisperkit.datatypes import TranscriptionResult, TranscriptionSegment, WordTiming


def _words(spec):
    return [WordTiming(" " + w, [], s, e) for w, s, e in spec]


def test_relabel_orders_by_first_appearance():
    raw = [("SPEAKER_07", 5.0, 6.0), ("SPEAKER_02", 0.0, 1.0), ("SPEAKER_07", 1.5, 2.0)]
    assert relabel_by_first_appearance(raw) == [(0, 0.0, 1.0), (1, 1.5, 2.0), (1, 5.0, 6.0)]


def test_speaker_for_interval_overlap_and_tolerance():
    segs = [SpeakerSegment(0, 0.0, 2.0), SpeakerSegment(1, 2.0, 4.0)]
    assert speaker_for_interval(1.5, 2.6, segs) == 1
    assert speaker_for_interval(4.3, 4.5, segs, tolerance=0.5) == 1
    assert speaker_for_interval(9.0, 9.5, segs, tolerance=0.5) is None


def test_add_speaker_info_subsegment_splits_on_speaker_change():
    words = _words([("hello", 0.0, 0.4), ("there", 0.5, 0.9), ("hi", 2.1, 2.3), ("back", 2.4, 2.8)])
    tr = TranscriptionResult(text="hello there hi back", segments=[TranscriptionSegment(0, 0, 0.0, 2.8, " hello there hi back", words=words)], language="en")
    dia = DiarizationResult([SpeakerSegment(0, 0.0, 1.0), SpeakerSegment(1, 2.0, 3.0)])
    groups = dia.add_speaker_info(to=[tr])[0]
    assert [(g.speaker, g.text.strip()) for g in groups] == [(0, "hello there"), (1, "hi back")]
    whole = dia.add_speaker_info(to=[tr], strategy=SpeakerInfoStrategy.SEGMENT)[0]
    assert len(whole) == 1


def test_rttm():
    dia = DiarizationResult([SpeakerSegment(0, 0.0, 1.25)])
    lines = SpeakerKit.generate_rttm(dia, "meeting")
    assert str(lines[0]) == "SPEAKER meeting 1 0.000 1.250 <NA> <NA> speaker_0 <NA> <NA>"
    assert dia.speaker_count == 1
