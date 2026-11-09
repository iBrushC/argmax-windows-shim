"""
Audio I/O: file loading, microphone capture, and the stream type used by real-time APIs.

Everything in the shim is 16 kHz mono float32, matching what the Argmax SDK feeds its models.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .._reference import DOCS_REALTIME, OSS_SPEAKERKIT, OSS_WHISPERKIT, Fidelity, mirrors
from .errors import MissingDependencyError

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_END = object()


@mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY, notes="Shim-internal resampling helper.")
def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling. Shim-internal helper."""
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Stream type (stand-in for Swift's AsyncStream<[Float]>)
# ---------------------------------------------------------------------------


@mirrors(
    "AsyncStream<[Float]>",
    module="Swift Concurrency",
    docs=DOCS_REALTIME,
    fidelity=Fidelity.APPROXIMATE,
    notes="Blocking iterator instead of an async sequence. Consume it from one thread.",
)
class AudioStream:
    """Iterator of 16 kHz mono float32 chunks. Ends when its continuation calls ``finish()``."""

    @mirrors("(created by AsyncStream.makeStream)", module="Swift Concurrency", fidelity=Fidelity.SHIM_ONLY, notes="Use make_audio_stream().")
    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._finished = threading.Event()

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            item = self._queue.get()
            if item is _END:
                return
            yield item

    @property
    @mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY)
    def is_finished(self) -> bool:
        return self._finished.is_set()


@mirrors(
    "AsyncStream<[Float]>.Continuation",
    module="Swift Concurrency",
    docs=DOCS_REALTIME,
    fidelity=Fidelity.APPROXIMATE,
)
class AudioStreamContinuation:
    """Producer side of an :class:`AudioStream`."""

    @mirrors("(created by AsyncStream.makeStream)", module="Swift Concurrency", fidelity=Fidelity.SHIM_ONLY, notes="Use make_audio_stream().")
    def __init__(self, stream: AudioStream, on_finish: Callable[[], None] | None = None) -> None:
        self._stream = stream
        self._on_finish = on_finish
        self._lock = threading.Lock()

    @mirrors(
        "AsyncStream.Continuation.yield(_:)",
        module="Swift Concurrency",
        notes="Named yield_ because `yield` is a Python keyword.",
    )
    def yield_(self, samples: np.ndarray | list[float]) -> None:
        """Push a chunk of 16 kHz mono float32 samples. Ignored after ``finish()``."""
        if self._stream.is_finished:
            return
        self._stream._queue.put(np.asarray(samples, dtype=np.float32).reshape(-1))

    @mirrors("AsyncStream.Continuation.finish()", module="Swift Concurrency")
    def finish(self) -> None:
        """End the stream. Safe to call more than once."""
        with self._lock:
            if self._stream.is_finished:
                return
            self._stream._finished.set()
        if self._on_finish is not None:
            try:
                self._on_finish()
            except Exception:  # noqa: BLE001
                log.exception("Error while closing audio source")
        self._stream._queue.put(_END)


@mirrors(
    "AsyncStream.makeStream(of: [Float].self)",
    module="Swift Concurrency",
    fidelity=Fidelity.APPROXIMATE,
    notes="Use this to build a CustomSource or to feed TranscribeStreamSession.start(audio_input_stream=...).",
)
def make_audio_stream(
    on_finish: Callable[[], None] | None = None,
) -> tuple[AudioStream, AudioStreamContinuation]:
    """Return a connected ``(stream, continuation)`` pair."""
    stream = AudioStream()
    return stream, AudioStreamContinuation(stream, on_finish)


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------


@mirrors("AudioDevice", module="WhisperKit", docs=OSS_WHISPERKIT, fidelity=Fidelity.APPROXIMATE)
@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str


def _import_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415
    except (ImportError, OSError) as exc:
        raise MissingDependencyError("sounddevice", "mic") from exc
    return sd


@mirrors(
    "AudioProcessor",
    module="WhisperKit",
    docs=OSS_WHISPERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Microphone capture uses PortAudio via sounddevice instead of AVAudioEngine.",
)
class AudioProcessor:
    """Audio loading and live capture helpers."""

    @mirrors("AudioProcessor()", module="WhisperKit", docs=OSS_WHISPERKIT, notes="Usually accessed as whisperKit.audio_processor.")
    def __init__(self) -> None:
        self._active: list[AudioStreamContinuation] = []

    @staticmethod
    @mirrors(
        "AudioProcessor.loadAudioAsFloatArray(fromPath:)",
        module="WhisperKit",
        docs=OSS_SPEAKERKIT,
        notes="Reads wav/flac/ogg/mp3 with soundfile; anything else (m4a, mp4, ...) needs ffmpeg on PATH.",
    )
    def load_audio_as_float_array(from_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Load any audio file as 16 kHz mono float32."""
        path = str(from_path)
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return resample(data.mean(axis=1), sr, sample_rate)
        except Exception:  # noqa: BLE001 - fall back to ffmpeg for formats libsndfile can't read
            pass
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"Could not decode {path!r}. Install ffmpeg and make sure it is on PATH.")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()

    @staticmethod
    @mirrors("AudioProcessor.getAudioDevices()", module="WhisperKit", docs=DOCS_REALTIME, fidelity=Fidelity.APPROXIMATE)
    def get_audio_devices() -> list[AudioDevice]:
        """List input devices. Device ids are PortAudio indices, not CoreAudio ids."""
        sd = _import_sounddevice()
        return [
            AudioDevice(id=i, name=d["name"])
            for i, d in enumerate(sd.query_devices())
            if d.get("max_input_channels", 0) > 0
        ]

    @mirrors(
        "AudioProcessor.startStreamingRecordingLive()",
        module="WhisperKit",
        docs=DOCS_REALTIME,
        fidelity=Fidelity.APPROXIMATE,
        notes="input_device_id and chunk_duration are shim parameters (Swift parameter labels unverified).",
    )
    def start_streaming_recording_live(
        self, input_device_id: int | None = None, chunk_duration: float = 0.1
    ) -> tuple[AudioStream, AudioStreamContinuation]:
        """Start the microphone and return ``(stream, continuation)``. Call ``continuation.finish()`` to stop."""
        sd = _import_sounddevice()
        holder: dict = {}

        def close() -> None:
            s = holder.get("stream")
            if s is not None:
                s.stop()
                s.close()

        stream, continuation = make_audio_stream(on_finish=close)

        def open_at(rate: int):
            def callback(indata, frames, time_info, status):  # noqa: ARG001
                if status:
                    log.debug("Input status: %s", status)
                mono = indata[:, 0].astype(np.float32, copy=True)
                continuation.yield_(resample(mono, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else mono)

            s = sd.InputStream(
                samplerate=rate,
                channels=1,
                dtype="float32",
                device=input_device_id,
                blocksize=max(1, int(rate * chunk_duration)),
                callback=callback,
            )
            s.start()
            return s

        try:
            holder["stream"] = open_at(SAMPLE_RATE)
        except Exception:  # noqa: BLE001 - some devices refuse 16 kHz; capture natively and resample
            info = sd.query_devices(input_device_id, "input")
            holder["stream"] = open_at(int(info["default_samplerate"]))
        self._active.append(continuation)
        return stream, continuation

    @mirrors("AudioProcessor.stopRecording()", module="WhisperKit", fidelity=Fidelity.APPROXIMATE)
    def stop_recording(self) -> None:
        """Stop every live recording started from this processor."""
        for c in self._active:
            c.finish()
        self._active.clear()


@mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY, notes="Shim-internal energy helper for voice-triggered mode.")
def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude. Shim-internal helper for voice-triggered mode."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
