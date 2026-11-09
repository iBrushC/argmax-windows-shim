"""SpeakerKitPro: SpeakerKit plus Sortformer / Nemotron diarization and real-time transcription with speakers."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import numpy as np

from .._reference import DOCS_DIARIZATION, DOCS_FILE, DOCS_REALTIME_SPEAKERS, Fidelity, mirrors
from ..core.audio import SAMPLE_RATE, make_audio_stream
from ..speakerkit.alignment import speaker_for_interval
from ..speakerkit.backends.sortformer_backend import DEFAULT_MODEL_ID
from ..speakerkit.config import PyannoteConfig
from ..speakerkit.speakerkit import SpeakerKit
from ..whisperkit.datatypes import WordTiming
from .streaming import TranscribeStreamSession

log = logging.getLogger(__name__)
_END = object()


@mirrors("SortformerStreamingConfig", module="Argmax", docs=DOCS_REALTIME_SPEAKERS, fidelity=Fidelity.APPROXIMATE)
class SortformerStreamingConfig(str, Enum):
    REALTIME = "realtime"
    PRERECORDED = "prerecorded"


@mirrors("SortformerMode", module="Argmax", docs=DOCS_DIARIZATION, notes="Swift type name unverified; cases .prerecorded / .realtime.")
class SortformerMode(str, Enum):
    PRERECORDED = "prerecorded"
    REALTIME = "realtime"


@mirrors(
    "SortformerDiarizationOptions",
    module="Argmax",
    docs=DOCS_REALTIME_SPEAKERS,
    fidelity=Fidelity.APPROXIMATE,
    notes="Defaults for max_word_gap_interval / tolerance are taken from the docs' tuning example, not verified SDK defaults.",
)
@dataclass
class SortformerDiarizationOptions:
    sortformer_mode: SortformerMode = SortformerMode.PRERECORDED
    max_word_gap_interval: float = 2.0
    tolerance: float = 0.5


@mirrors(
    "SortformerConfig",
    module="Argmax",
    docs=DOCS_REALTIME_SPEAKERS,
    fidelity=Fidelity.APPROXIMATE,
    notes="model_id, device and rediarize_interval are shim-only.",
)
@dataclass
class SortformerConfig:
    load: bool | None = None
    streaming_config: SortformerStreamingConfig = SortformerStreamingConfig.REALTIME
    model_folder: str | None = None
    # ---- shim-only ----
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    rediarize_interval: float = 2.0
    """Seconds between re-diarization passes in streaming mode (shim-only)."""

    def _create_backend(self):
        from ..speakerkit.backends.sortformer_backend import SortformerBackend

        return SortformerBackend(model_id=self.model_id, model_folder=self.model_folder, device=self.device)


@mirrors(
    "SpeakerKitDiarizer",
    module="Argmax",
    docs=DOCS_FILE,
    fidelity=Fidelity.APPROXIMATE,
)
class SpeakerKitDiarizer:
    @mirrors("(created by SpeakerKitDiarizer.pyannote())", module="Argmax", docs=DOCS_FILE, fidelity=Fidelity.SHIM_ONLY)
    def __init__(self, backend) -> None:
        self._backend = backend

    @classmethod
    @mirrors(
        "SpeakerKitDiarizer.pyannote()",
        module="Argmax",
        docs=DOCS_FILE,
        fidelity=Fidelity.APPROXIMATE,
        notes="variant/hf_token are shim-only; 'precision' falls back to community.",
    )
    def pyannote(cls, variant: str = "community", hf_token: str | None = None) -> "SpeakerKitDiarizer":
        from ..speakerkit.backends.pyannote_backend import PyannoteBackend

        return cls(PyannoteBackend(variant=variant, hf_token=hf_token))

    @mirrors("SpeakerKitDiarizer.downloadModels()", module="Argmax", docs=DOCS_FILE, fidelity=Fidelity.APPROXIMATE, notes="Download happens inside load_models().")
    def download_models(self) -> None:
        pass

    @mirrors("SpeakerKitDiarizer.loadModels()", module="Argmax", docs=DOCS_FILE)
    def load_models(self) -> None:
        self._backend.ensure_loaded()


@mirrors("WordWithSpeaker", module="Argmax", docs=DOCS_REALTIME_SPEAKERS)
@dataclass
class WordWithSpeaker:
    word_timing: WordTiming
    speaker: int | None


@mirrors("TranscribeDiarizeResultType", module="Argmax", docs=DOCS_REALTIME_SPEAKERS)
class TranscribeDiarizeResultType(str, Enum):
    NEW_TRANSCRIPTION = "newTranscription"
    SPEAKER_REVISION = "speakerRevision"


@mirrors(
    "TranscribeDiarizeResult (combinedSession.results element)",
    module="Argmax",
    docs=DOCS_REALTIME_SPEAKERS,
    fidelity=Fidelity.APPROXIMATE,
    notes="text is the cumulative confirmed transcript; confirmed_words_with_speakers is this update's batch. "
    "Use seek_time to find the batch a SPEAKER_REVISION replaces.",
)
@dataclass
class TranscribeDiarizeResult:
    type: TranscribeDiarizeResultType
    text: str
    hypothesis_text: str | None
    confirmed_words_with_speakers: list[WordWithSpeaker] = field(default_factory=list)
    hypothesis_words_with_speakers: list[WordWithSpeaker] = field(default_factory=list)
    seek_time: float = 0.0
    speaker_count: int = 0


@dataclass
class _Batch:
    seek_time: float
    words: list[WordWithSpeaker]


@mirrors(
    "(combined transcription + diarization stream session)",
    module="Argmax",
    docs=DOCS_REALTIME_SPEAKERS,
    fidelity=Fidelity.APPROXIMATE,
    notes="Re-diarizes the whole buffer every rediarize_interval seconds (cost grows with length), "
    "instead of Sortformer's streaming speaker cache.",
)
class TranscribeDiarizeStreamSession:
    @mirrors("(created by SpeakerKitPro.makeStreamSession)", module="Argmax", docs=DOCS_REALTIME_SPEAKERS, fidelity=Fidelity.SHIM_ONLY, notes="Use SpeakerKitPro.make_stream_session(...).")
    def __init__(self, speaker_kit: SpeakerKit, transcription_session: TranscribeStreamSession, diarization_options: SortformerDiarizationOptions) -> None:
        self._sk = speaker_kit
        self._ts = transcription_session
        self._opts = diarization_options
        self._interval = getattr(speaker_kit.config, "rediarize_interval", 2.0)
        self._chunks: list[np.ndarray] = []
        self._audio_lock = threading.Lock()
        self._segments: list = []
        self._speaker_count = 0
        self._batches: list[_Batch] = []
        self._state_lock = threading.Lock()
        self._diar_lock = threading.Lock()
        self._text = ""
        self._out: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._loop: threading.Thread | None = None
        self._error: BaseException | None = None

    @mirrors("combinedSession.start(audioInputStream:)", module="Argmax", docs=DOCS_REALTIME_SPEAKERS)
    def start(self, audio_input_stream) -> None:
        tee_stream, tee_cont = make_audio_stream()

        def pump() -> None:
            try:
                for chunk in audio_input_stream:
                    with self._audio_lock:
                        self._chunks.append(chunk)
                    tee_cont.yield_(chunk)
            finally:
                tee_cont.finish()

        threading.Thread(target=pump, daemon=True, name="diarize-pump").start()
        self._ts.start(tee_stream)
        self._loop = threading.Thread(target=self._diarize_loop, daemon=True, name="diarize-loop")
        self._loop.start()
        threading.Thread(target=self._consume, daemon=True, name="diarize-consume").start()

    @property
    @mirrors("combinedSession.results", module="Argmax", docs=DOCS_REALTIME_SPEAKERS)
    def results(self) -> Iterator[TranscribeDiarizeResult]:
        while True:
            item = self._out.get()
            if item is _END:
                if self._error is not None:
                    raise self._error
                return
            yield item

    # ------------------------------------------------------------------
    def _assign(self, words: list[WordTiming]) -> list[WordWithSpeaker]:
        with self._state_lock:
            segs = list(self._segments)
        return [WordWithSpeaker(w, speaker_for_interval(w.start, w.end, segs, self._opts.tolerance)) for w in words]

    def _diarize_now(self) -> None:
        with self._audio_lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        if audio.size < SAMPLE_RATE:
            return
        with self._diar_lock:
            result = self._sk.diarize(audio, SortformerDiarizationOptions(SortformerMode.PRERECORDED))
            with self._state_lock:
                self._segments = result.segments
                self._speaker_count = result.speaker_count
                segs = list(self._segments)
                for batch in self._batches:
                    revised = [WordWithSpeaker(w.word_timing, speaker_for_interval(w.word_timing.start, w.word_timing.end, segs, self._opts.tolerance)) for w in batch.words]
                    if [r.speaker for r in revised] != [w.speaker for w in batch.words]:
                        batch.words = revised
                        self._out.put(
                            TranscribeDiarizeResult(
                                type=TranscribeDiarizeResultType.SPEAKER_REVISION,
                                text=self._text,
                                hypothesis_text=None,
                                confirmed_words_with_speakers=revised,
                                seek_time=batch.seek_time,
                                speaker_count=self._speaker_count,
                            )
                        )

    def _diarize_loop(self) -> None:
        while not self._done.wait(self._interval):
            try:
                self._diarize_now()
            except Exception:  # noqa: BLE001
                log.exception("Diarization pass failed")

    def _consume(self) -> None:
        try:
            for r in self._ts.results:
                confirmed = [w for s in r.confirmed_segments for w in (s.words or [])]
                hypothesis = [w for s in r.hypothesis_segments for w in (s.words or [])]
                cw, hw = self._assign(confirmed), self._assign(hypothesis)
                with self._state_lock:
                    if r.text:
                        self._text = f"{self._text} {r.text}".strip()
                    if cw:
                        self._batches.append(_Batch(r.seek_time, cw))
                    count = self._speaker_count
                    self._out.put(
                        TranscribeDiarizeResult(
                            type=TranscribeDiarizeResultType.NEW_TRANSCRIPTION,
                            text=self._text,
                            hypothesis_text=r.hypothesis_text,
                            confirmed_words_with_speakers=cw,
                            hypothesis_words_with_speakers=hw,
                            seek_time=r.seek_time,
                            speaker_count=count,
                        )
                    )
            self._done.set()
            if self._loop is not None:
                self._loop.join()
            self._diarize_now()  # final pass may revise earlier batches
        except BaseException as exc:  # noqa: BLE001
            log.exception("Combined session failed")
            self._error = exc
            self._done.set()
        finally:
            self._out.put(_END)


@mirrors(
    "SpeakerKitPro",
    module="Argmax",
    docs=DOCS_DIARIZATION,
    fidelity=Fidelity.APPROXIMATE,
    notes="SpeakerKitPro() with no config uses Sortformer, matching the docs' Sortformer example.",
)
class SpeakerKitPro(SpeakerKit):
    @mirrors("SpeakerKitPro.init(_ config: PyannoteConfig | SortformerConfig)", module="Argmax", docs=DOCS_DIARIZATION)
    def __init__(self, config: PyannoteConfig | SortformerConfig | None = None) -> None:
        super().__init__(config or SortformerConfig())

    @mirrors(
        "SpeakerKitPro.makeStreamSession(transcriptionSession:diarizationConfig:)",
        module="Argmax",
        docs=DOCS_REALTIME_SPEAKERS,
        fidelity=Fidelity.APPROXIMATE,
    )
    def make_stream_session(
        self,
        transcription_session: TranscribeStreamSession,
        diarization_config: SortformerDiarizationOptions | None = None,
    ) -> TranscribeDiarizeStreamSession:
        return TranscribeDiarizeStreamSession(
            self, transcription_session, diarization_config or SortformerDiarizationOptions(SortformerMode.REALTIME)
        )
