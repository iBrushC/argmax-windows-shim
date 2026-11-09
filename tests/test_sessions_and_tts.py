import threading

import numpy as np

from argmax_shim.argmax import (
    DecodingOptionsPro,
    SpeakerKitPro,
    StreamTranscriptionMode,
    TranscribeDiarizeResultType,
    WhisperKitPro,
    WhisperKitProConfig,
    make_audio_stream,
)
from argmax_shim.core.audio import SAMPLE_RATE
from argmax_shim.speakerkit import DiarizationResult, SpeakerSegment
from argmax_shim.ttskit import GenerationOptions, TTSKit, TTSKitConfig, TTSModelVariant
from argmax_shim.ttskit.backends.base import TTSBackend
from argmax_shim.whisperkit.whisperkit import ModelState

from conftest import FakeASRBackend


class FakeDiarizer:
    """Speaker 0 for the first 2 s, speaker 1 after; first pass wrongly says everyone is speaker 0."""

    def __init__(self):
        self.calls = 0

    def ensure_loaded(self):
        pass

    def unload(self):
        pass

    def diarize(self, audio, options=None):
        self.calls += 1
        dur = len(audio) / SAMPLE_RATE
        if self.calls == 1:
            return DiarizationResult([SpeakerSegment(0, 0.0, dur)])
        return DiarizationResult([SpeakerSegment(0, 0.0, 2.0), SpeakerSegment(1, 2.0, dur)])


def test_combined_session_emits_transcripts_and_revisions(tone):
    wk = WhisperKitPro(WhisperKitProConfig(load=False))
    wk._backend = FakeASRBackend(wk.model_spec, wk.config, offset_of=lambda: 0.0)
    wk.model_state = ModelState.LOADED
    sk = SpeakerKitPro()
    sk._backend = FakeDiarizer()
    sk.config.rediarize_interval = 0.05

    ts = wk.make_stream_session(DecodingOptionsPro(transcribe_interval=0.0, stream_transcription_mode=StreamTranscriptionMode.always_on(max_buffer_length=60)))
    combined = sk.make_stream_session(transcription_session=ts)
    stream, cont = make_audio_stream()
    combined.start(stream)

    def feed():
        audio = tone(4.0)
        for i in range(0, len(audio), SAMPLE_RATE // 10):
            cont.yield_(audio[i : i + SAMPLE_RATE // 10])
            threading.Event().wait(0.005)
        cont.finish()

    threading.Thread(target=feed, daemon=True).start()
    results = list(combined.results)
    assert any(r.type is TranscribeDiarizeResultType.NEW_TRANSCRIPTION for r in results)
    final_text = [r.text for r in results if r.type is TranscribeDiarizeResultType.NEW_TRANSCRIPTION][-1]
    assert final_text.startswith("the quick brown fox")
    # After revisions, words after 2 s must belong to speaker 1.
    latest = {}
    for r in results:
        for w in r.confirmed_words_with_speakers:
            latest[(w.word_timing.start, w.word_timing.word)] = w.speaker
    late = [spk for (start, _), spk in latest.items() if start >= 2.1]
    assert late and all(s == 1 for s in late)


class FakeTTS(TTSBackend):
    sample_rate = 8_000

    def load(self):
        self.loaded = True

    def synthesize(self, text, speaker, language, options):
        return np.full(len(text) * 10, 0.1, np.float32), self.sample_rate


def test_tts_generate_chunks_and_cancel():
    tts = TTSKit(TTSKitConfig(model=TTSModelVariant.SYSTEM))
    tts._backend = FakeTTS(tts.config)
    text = "This is the first sentence. This is the second sentence. And here is the third one."
    seen = []
    result = tts.generate(text, callback=lambda p: seen.append(p) or None)
    assert len(seen) == 3 and seen[0].step_time is not None and seen[1].step_time is None
    assert result.sample_rate == 8_000 and result.audio_duration > 0

    cancelled = tts.generate(text, callback=lambda p: False)
    assert len(cancelled.audio) == len(seen[0].audio)

    whole = []
    tts.generate(text, options=GenerationOptions(chunking_strategy="none"), callback=lambda p: whole.append(p) or None)
    assert len(whole) == 1
