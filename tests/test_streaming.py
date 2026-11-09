import threading

from argmax_shim.argmax import (
    CustomSource,
    DecodingOptionsPro,
    LiveResult,
    LiveTranscriber,
    StreamTranscriptionMode,
    WhisperKitPro,
    WhisperKitProConfig,
    make_audio_stream,
)
from argmax_shim.argmax.streaming import TranscribeStreamSession
from argmax_shim.core.audio import SAMPLE_RATE
from argmax_shim.whisperkit.whisperkit import ModelState

from conftest import SCRIPT, FakeASRBackend


def _kit(session_holder):
    wk = WhisperKitPro(WhisperKitProConfig(load=False))
    wk._backend = FakeASRBackend(wk.model_spec, wk.config)
    wk.model_state = ModelState.LOADED
    return wk


def _feed(continuation, audio, chunk=0.1):
    step = int(chunk * SAMPLE_RATE)
    for i in range(0, len(audio), step):
        continuation.yield_(audio[i : i + step])
    continuation.finish()


def test_local_agreement_confirms_whole_script(tone):
    wk = _kit(None)
    opts = DecodingOptionsPro(transcribe_interval=0.0, stream_transcription_mode=StreamTranscriptionMode.always_on(max_buffer_length=3.0))
    session = TranscribeStreamSession(wk, opts)
    # The fake backend needs to know where the session's buffer starts in stream time.
    original = session._transcribe

    def tracked(buf, buf_start):
        wk._backend.offset_of = lambda: buf_start
        return original(buf, buf_start)

    session._transcribe = tracked
    stream, cont = make_audio_stream()
    session.start(stream)
    threading.Thread(target=_feed, args=(cont, tone(len(SCRIPT) * 0.4 + 0.5)), daemon=True).start()

    confirmed = []
    saw_hypothesis = False
    for r in session.results:
        if r.text:
            confirmed.append(r.text)
        saw_hypothesis |= bool(r.hypothesis_text)
    assert " ".join(confirmed).split() == SCRIPT
    assert saw_hypothesis


def test_live_transcriber_custom_source(tone):
    wk = _kit(None)
    wk._backend.offset_of = lambda: 0.0
    lt = LiveTranscriber(wk)
    stream, cont = make_audio_stream()
    src = CustomSource(id="test", audio_stream=stream, audio_continuation=cont)
    lt.register_stream(src, DecodingOptionsPro(transcribe_interval=0.0, stream_transcription_mode=StreamTranscriptionMode.always_on(max_buffer_length=60)))
    results = lt.start_transcription(src)
    threading.Thread(target=_feed, args=(cont, tone(3.0)), daemon=True).start()

    text = ""
    kinds = set()
    for r in results:
        match r:
            case LiveResult.Confirm(t, seconds, _):
                text += " " + t
                kinds.add("confirm")
                assert seconds >= 0
            case LiveResult.Hypothesis(t, _, _):
                kinds.add("hypothesis")
    assert text.split() == SCRIPT[: len(text.split())] and len(text.split()) >= 6
    assert kinds == {"confirm", "hypothesis"}
    lt.stop_and_remove_all_transcriptions()


def test_voice_triggered_skips_silence(tone):
    import numpy as np

    wk = _kit(None)
    calls = []
    real = wk._backend.transcribe

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    wk._backend.transcribe = counting
    wk._backend.offset_of = lambda: 0.0
    session = TranscribeStreamSession(wk, DecodingOptionsPro(transcribe_interval=0.0, stream_transcription_mode=StreamTranscriptionMode.voice_triggered(min_process_interval=0.0)))
    stream, cont = make_audio_stream()
    session.start(stream)
    _feed(cont, np.zeros(3 * SAMPLE_RATE, np.float32))
    list(session.results)
    assert len(calls) <= 1  # only the final flush, never during silence
