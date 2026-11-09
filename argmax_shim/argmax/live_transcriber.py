"""LiveTranscriber: the high-level real-time API that manages sessions and audio sources."""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

from .._reference import DOCS_REALTIME, Fidelity, mirrors
from ..core.audio import AudioStream, AudioStreamContinuation
from .streaming import TranscribeStreamResult, TranscribeStreamSession
from .whisperkit_pro import DecodingOptionsPro

_ids = itertools.count()


@mirrors("StreamType", module="Argmax", docs=DOCS_REALTIME, fidelity=Fidelity.APPROXIMATE)
@dataclass(frozen=True)
class StreamType:
    kind: str
    device_id: int | None = None

    @staticmethod
    @mirrors("StreamType.device(_:)", module="Argmax", docs=DOCS_REALTIME, notes="device_id is a PortAudio index (see AudioProcessor.get_audio_devices).")
    def device(device_id: int | None = None) -> "StreamType":
        return StreamType("device", device_id)


class StreamSource:
    id: str


@mirrors("ArgmaxSource", module="Argmax", docs=DOCS_REALTIME, fidelity=Fidelity.APPROXIMATE)
class ArgmaxSource(StreamSource):
    """Built-in source: a microphone captured by the SDK."""

    @mirrors("ArgmaxSource.init(streamType:)", module="Argmax", docs=DOCS_REALTIME)
    def __init__(self, stream_type: StreamType | None = None) -> None:
        self.stream_type = stream_type or StreamType.device()
        self.id = f"device-{self.stream_type.device_id if self.stream_type.device_id is not None else 'default'}-{next(_ids)}"


@mirrors("CustomSource", module="Argmax", docs=DOCS_REALTIME)
class CustomSource(StreamSource):
    """Your own audio (e.g. system audio, a file, a WebSocket). Build the pair with make_audio_stream()."""

    @mirrors("CustomSource.init(id:audioStream:audioContinuation:)", module="Argmax", docs=DOCS_REALTIME)
    def __init__(self, id: str, audio_stream: AudioStream, audio_continuation: AudioStreamContinuation) -> None:  # noqa: A002
        self.id = id
        self.audio_stream = audio_stream
        self.audio_continuation = audio_continuation


@mirrors("LiveResult.confirm(text, seconds, result)", module="Argmax", docs=DOCS_REALTIME)
@dataclass(frozen=True)
class LiveConfirm:
    text: str
    """Newly confirmed text. Append it to your transcript."""
    seconds: float
    """Elapsed seconds since transcription started (monotonic clock)."""
    result: TranscribeStreamResult


@mirrors("LiveResult.hypothesis(text, seconds, result)", module="Argmax", docs=DOCS_REALTIME)
@dataclass(frozen=True)
class LiveHypothesis:
    text: str
    """Current full hypothesis. Replace (don't append) what you displayed before."""
    seconds: float
    result: TranscribeStreamResult


@mirrors(
    "LiveResult",
    module="Argmax",
    docs=DOCS_REALTIME,
    notes="Swift enum -> two dataclasses. Use `match r: case LiveResult.Confirm(text, seconds, result): ...`.",
)
class LiveResult:
    Confirm = LiveConfirm
    Hypothesis = LiveHypothesis


@dataclass
class _Registration:
    source: StreamSource
    options: DecodingOptionsPro
    session: TranscribeStreamSession | None = None
    continuation: AudioStreamContinuation | None = None
    extra: dict = field(default_factory=dict)


@mirrors(
    "LiveTranscriber",
    module="Argmax",
    docs=DOCS_REALTIME,
    fidelity=Fidelity.APPROXIMATE,
    notes="Streams share one model instance; inference is serialized with a lock.",
)
class LiveTranscriber:
    @mirrors("LiveTranscriber.init(whisperKit:)", module="Argmax", docs=DOCS_REALTIME)
    def __init__(self, whisper_kit) -> None:
        self.whisper_kit = whisper_kit
        self._streams: dict[str, _Registration] = {}
        self._lock = threading.Lock()

    @mirrors("LiveTranscriber.registerStream(streamSource:options:)", module="Argmax", docs=DOCS_REALTIME)
    def register_stream(self, stream_source: StreamSource, options: DecodingOptionsPro | None = None) -> None:
        with self._lock:
            if stream_source.id in self._streams:
                raise ValueError(f"Stream {stream_source.id!r} is already registered")
            self._streams[stream_source.id] = _Registration(stream_source, options or DecodingOptionsPro())

    @mirrors(
        "LiveTranscriber.startTranscription(for:)",
        module="Argmax",
        docs=DOCS_REALTIME,
        notes="`for` is a Python keyword, so the parameter is for_ (or pass positionally).",
    )
    def start_transcription(self, for_: StreamSource) -> Iterator[LiveConfirm | LiveHypothesis]:
        with self._lock:
            reg = self._streams.get(for_.id)
        if reg is None:
            raise KeyError(f"Stream {for_.id!r} is not registered; call register_stream first")
        if isinstance(for_, ArgmaxSource):
            stream, continuation = self.whisper_kit.audio_processor.start_streaming_recording_live(for_.stream_type.device_id)
        elif isinstance(for_, CustomSource):
            stream, continuation = for_.audio_stream, for_.audio_continuation
        else:
            raise TypeError(f"Unsupported source type {type(for_).__name__}")
        reg.continuation = continuation
        reg.session = TranscribeStreamSession(self.whisper_kit, reg.options)
        reg.session.start(stream)
        return self._results(reg.session)

    @staticmethod
    def _results(session: TranscribeStreamSession) -> Iterator[LiveConfirm | LiveHypothesis]:
        started = time.monotonic()
        for r in session.results:
            elapsed = time.monotonic() - started
            if r.text:
                yield LiveResult.Confirm(r.text, elapsed, r)
            if r.hypothesis_text:
                yield LiveResult.Hypothesis(r.hypothesis_text, elapsed, r)

    @mirrors("LiveTranscriber.stopAndRemoveStream(for:)", module="Argmax", docs=DOCS_REALTIME)
    def stop_and_remove_stream(self, for_: StreamSource) -> None:
        with self._lock:
            reg = self._streams.pop(for_.id, None)
        if reg is not None and reg.continuation is not None:
            reg.continuation.finish()

    @mirrors("LiveTranscriber.stopAndRemoveAllTranscriptions()", module="Argmax", docs=DOCS_REALTIME)
    def stop_and_remove_all_transcriptions(self) -> None:
        with self._lock:
            regs = list(self._streams.values())
            self._streams.clear()
        for reg in regs:
            if reg.continuation is not None:
                reg.continuation.finish()
