"""
Real-time transcription session.

The SDK uses Argmax's own streaming algorithm (ICML 2025). The shim
approximates its *behavior* with the LocalAgreement-2 policy: re-transcribe a
rolling audio buffer, and confirm the words two consecutive passes agree on.
Confirmed text never changes; hypothesis text may.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

import numpy as np

from .._reference import DOCS_REALTIME, Fidelity, mirrors
from ..core.audio import SAMPLE_RATE, rms
from ..whisperkit.datatypes import TranscriptionSegment, WordTiming
from .whisperkit_pro import DecodingOptionsPro

log = logging.getLogger(__name__)
_END = object()
_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


@mirrors(
    "TranscriptionResultPro (stream session result)",
    module="Argmax",
    docs=DOCS_REALTIME,
    fidelity=Fidelity.APPROXIMATE,
    notes="text holds only newly confirmed text for this update (TODO verify against SDK). "
    "Timestamps are seconds since stream start.",
)
@dataclass
class TranscribeStreamResult:
    text: str
    hypothesis_text: str | None
    confirmed_segments: list[TranscriptionSegment] = field(default_factory=list)
    hypothesis_segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None
    seek_time: float = 0.0

    @property
    @mirrors("(alias of confirmed_segments)", module="-", fidelity=Fidelity.SHIM_ONLY)
    def segments(self) -> list[TranscriptionSegment]:
        return self.confirmed_segments


def _norm(word: str) -> str:
    return _PUNCT.sub("", word.lower())


def _agree(previous: list[WordTiming], current: list[WordTiming]) -> int:
    n = 0
    while n < min(len(previous), len(current)) and _norm(previous[n].word) == _norm(current[n].word):
        n += 1
    return n


def _join(words: list[WordTiming]) -> str:
    return "".join(w.word for w in words).strip()


def _segment(words: list[WordTiming]) -> TranscriptionSegment:
    return TranscriptionSegment(id=0, seek=0, start=words[0].start, end=words[-1].end, text=" " + _join(words), words=list(words))


@mirrors(
    "TranscribeStreamSession",
    module="Argmax",
    docs=DOCS_REALTIME,
    fidelity=Fidelity.APPROXIMATE,
    notes="Runs on a background thread. results is a blocking iterator (Swift: AsyncThrowingStream).",
)
class TranscribeStreamSession:
    @mirrors("(created by WhisperKitPro.makeStreamSession)", module="Argmax", docs=DOCS_REALTIME, fidelity=Fidelity.SHIM_ONLY, notes="Use WhisperKitPro.make_stream_session(options).")
    def __init__(self, whisper_kit, options: DecodingOptionsPro) -> None:
        self._wk = whisper_kit
        self._options = options
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @mirrors("TranscribeStreamSession.start(audioInputStream:)", module="Argmax", docs=DOCS_REALTIME)
    def start(self, audio_input_stream: Iterable[np.ndarray]) -> None:
        if self._thread is not None:
            raise RuntimeError("Session already started")
        self._thread = threading.Thread(target=self._run_safe, args=(audio_input_stream,), daemon=True, name="TranscribeStreamSession")
        self._thread.start()

    @property
    @mirrors("TranscribeStreamSession.results", module="Argmax", docs=DOCS_REALTIME, notes="Consume from a single thread.")
    def results(self) -> Iterator[TranscribeStreamResult]:
        while True:
            item = self._out.get()
            if item is _END:
                if self._error is not None:
                    raise self._error
                return
            yield item

    @mirrors("(finish the input stream)", module="Argmax", fidelity=Fidelity.SHIM_ONLY, notes="Prefer finishing the audio continuation, as in Swift.")
    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _run_safe(self, stream) -> None:
        try:
            self._run(stream)
        except BaseException as exc:  # noqa: BLE001
            log.exception("Stream session failed")
            self._error = exc
        finally:
            self._out.put(_END)

    def _transcribe(self, buf: np.ndarray, buf_start: float):
        opts = replace(self._options.base, word_timestamps=True, chunking_strategy=None)
        result = self._wk._transcribe_array(buf, opts, streaming=True)
        words = []
        for seg in result.segments:
            seg_words = seg.words or _estimate_words(seg)
            words += [replace(w, start=w.start + buf_start, end=w.end + buf_start) for w in seg_words]
        return words, result.language

    def _run(self, stream) -> None:
        opts = self._options
        mode = opts.stream_transcription_mode
        always_on = mode.kind == "alwaysOn"
        interval = opts.transcribe_interval if always_on else max(opts.transcribe_interval, mode.min_process_interval)

        buf = np.zeros(0, np.float32)
        buf_start = 0.0
        total = 0.0
        committed_until = 0.0
        pending: list[WordTiming] = []
        last_hyp: str | None = None
        last_process = float("-inf")
        last_voice = float("-inf")
        language = None

        def emit(newly: list[WordTiming], hyp: list[WordTiming]) -> None:
            nonlocal last_hyp
            hyp_text = _join(hyp) or None
            if not newly and hyp_text == last_hyp:
                return
            last_hyp = hyp_text
            self._out.put(
                TranscribeStreamResult(
                    text=_join(newly),
                    hypothesis_text=hyp_text,
                    confirmed_segments=[_segment(newly)] if newly else [],
                    hypothesis_segments=[_segment(hyp)] if hyp else [],
                    language=language,
                    seek_time=total,
                )
            )

        def trim_to(t: float) -> None:
            nonlocal buf, buf_start
            cut = max(0, int((t - buf_start) * SAMPLE_RATE))
            buf, buf_start = buf[cut:], buf_start + cut / SAMPLE_RATE

        for chunk in stream:
            if self._stop.is_set():
                break
            buf = np.concatenate([buf, chunk])
            total += len(chunk) / SAMPLE_RATE
            if not always_on and rms(buf[-SAMPLE_RATE // 2 :]) >= mode.silence_threshold:
                last_voice = total
            now = time.monotonic()
            if now - last_process < interval:
                continue

            if not always_on and total - last_voice >= 0.5:  # silence: skip inference
                if pending and total - last_voice >= mode.endpoint_silence:
                    newly, pending = pending, []
                    committed_until = newly[-1].end
                    emit(newly, [])
                    trim_to(committed_until)
                elif not pending and buf.size > SAMPLE_RATE:
                    trim_to(total - 1.0)  # keep a little leading context
                continue

            last_process = now
            words, language = self._transcribe(buf, buf_start)
            words = [w for w in words if (w.start + w.end) / 2 > committed_until]
            n = _agree(pending, words)
            newly, pending = words[:n], words[n:]
            if newly:
                committed_until = newly[-1].end
            emit(newly, pending)

            if total - buf_start > mode.max_buffer_length:
                if committed_until <= buf_start and pending:  # nothing confirmed in the window: force it
                    newly, pending = pending, []
                    committed_until = newly[-1].end
                    emit(newly, [])
                trim_to(max(committed_until, total - mode.max_buffer_length))

        # Stream finished: confirm everything that is left.
        if buf.size >= SAMPLE_RATE // 10:
            words, language = self._transcribe(buf, buf_start)
            final = [w for w in words if (w.start + w.end) / 2 > committed_until]
        else:
            final = pending
        if final:
            emit(final, [])
        elif last_hyp:
            emit([], [])


def _estimate_words(seg: TranscriptionSegment) -> list[WordTiming]:
    """Spread segment text evenly over its span when a backend gives no word timings."""
    tokens = seg.text.split()
    if not tokens:
        return []
    step = (seg.end - seg.start) / len(tokens)
    return [WordTiming(" " + t, [], seg.start + i * step, seg.start + (i + 1) * step) for i, t in enumerate(tokens)]
