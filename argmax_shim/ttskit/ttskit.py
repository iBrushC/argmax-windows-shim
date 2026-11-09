"""The TTSKit text-to-speech pipeline."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from .._reference import OSS_TTSKIT, Fidelity, mirrors
from ..core.errors import MissingDependencyError, warn_once
from .backends import create_tts_backend
from .config import TTSKitConfig
from .datatypes import (
    GenerationOptions,
    GenerationProgress,
    PlaybackStrategy,
    Qwen3Language,
    Qwen3Speaker,
    SpeechResult,
    SpeechTimings,
    TextChunkingStrategy,
)
from .text import split_into_sentences

log = logging.getLogger(__name__)
_END = object()


@mirrors(
    "TTSKit",
    module="TTSKit",
    docs=OSS_TTSKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Synchronous Python methods. Streaming is per sentence chunk rather than per decoder frame.",
)
class TTSKit:
    """Text-to-speech. Defaults to Qwen3-TTS 0.6B like the real SDK; see TTSModelVariant for lighter engines."""

    @mirrors(
        "TTSKit.init(_ config: TTSKitConfig)",
        module="TTSKit",
        docs=OSS_TTSKIT,
        notes="Models load lazily on the first generate() call, as in TTSKit.",
    )
    def __init__(self, config: TTSKitConfig | None = None) -> None:
        self.config = config or TTSKitConfig()
        self._backend = create_tts_backend(self.config)
        self._lock = threading.Lock()
        self._loaded = False
        if self.config.load:
            self.load_models()

    @mirrors("TTSKit.loadModels()", module="TTSKit", docs=OSS_TTSKIT)
    def load_models(self) -> None:
        with self._lock:
            if not self._loaded:
                self._backend.load()
                self._loaded = True

    @mirrors("TTSKit.unloadModels()", module="TTSKit", docs=OSS_TTSKIT)
    def unload_models(self) -> None:
        with self._lock:
            self._backend.unload()
            self._loaded = False

    @mirrors(
        "TTSKit.generate(text:speaker:language:options:callback:)",
        module="TTSKit",
        docs=OSS_TTSKIT,
        fidelity=Fidelity.APPROXIMATE,
        notes="callback mirrors the trailing progress closure; return False to cancel.",
    )
    def generate(
        self,
        text: str,
        speaker: Qwen3Speaker | None = None,
        language: Qwen3Language | None = None,
        options: GenerationOptions | None = None,
        callback: Callable[[GenerationProgress], bool | None] | None = None,
    ) -> SpeechResult:
        self.load_models()
        options = options or GenerationOptions()
        speaker = speaker or Qwen3Speaker.RYAN
        language = language or Qwen3Language.ENGLISH
        if options.instruction and not self._backend.supports_instruction:
            warn_once(log, f"instr-{self.config.model.value}", f"{self.config.model.value} ignores style instructions (1.7B only in TTSKit).")
        chunks = split_into_sentences(text) if options.chunking_strategy is TextChunkingStrategy.SENTENCE else [text]

        started = time.perf_counter()
        pieces: list[np.ndarray] = []
        sample_rate = self._backend.sample_rate
        first_step: float | None = None
        for chunk in chunks:
            t0 = time.perf_counter()
            with self._lock:
                audio, sample_rate = self._backend.synthesize(chunk, speaker, language, options)
            step = time.perf_counter() - t0
            pieces.append(audio)
            if first_step is None:
                first_step = step
            if callback is not None:
                keep_going = callback(GenerationProgress(audio=audio, step_time=step if len(pieces) == 1 else None, sample_rate=sample_rate))
                if keep_going is False:
                    break
        audio = np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)
        return SpeechResult(audio=audio, sample_rate=sample_rate, timings=SpeechTimings(time.perf_counter() - started, first_step))

    @mirrors(
        "TTSKit.play(text:speaker:language:options:playbackStrategy:)",
        module="TTSKit",
        docs=OSS_TTSKIT,
        fidelity=Fidelity.APPROXIMATE,
        notes="Plays through sounddevice. Returns the full SpeechResult when playback ends.",
    )
    def play(
        self,
        text: str,
        speaker: Qwen3Speaker | None = None,
        language: Qwen3Language | None = None,
        options: GenerationOptions | None = None,
        playback_strategy: PlaybackStrategy | None = None,
    ) -> SpeechResult:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise MissingDependencyError("sounddevice", "mic") from exc
        strategy = playback_strategy or PlaybackStrategy.auto()

        if strategy.kind == "generateFirst":
            result = self.generate(text, speaker, language, options)
            sd.play(result.audio, result.sample_rate)
            sd.wait()
            return result

        chunks: queue.Queue = queue.Queue()
        holder: dict = {}

        def produce() -> None:
            try:
                holder["result"] = self.generate(
                    text, speaker, language, options, callback=lambda p: chunks.put((p.audio, p.sample_rate))
                )
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
            finally:
                chunks.put(_END)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()

        buffered: list[np.ndarray] = []
        rate = None
        need = strategy.seconds if strategy.kind == "buffered" else 0.0
        done = False
        while True:  # pre-buffer
            item = chunks.get()
            if item is _END:
                done = True
                break
            buffered.append(item[0])
            rate = item[1]
            if sum(len(b) for b in buffered) / rate >= need:
                break
        if rate is not None:
            with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as out:
                for b in buffered:
                    out.write(b.reshape(-1, 1))
                while not done:
                    item = chunks.get()
                    if item is _END:
                        break
                    out.write(item[0].reshape(-1, 1))
        worker.join()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]
