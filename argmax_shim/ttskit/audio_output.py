"""Saving generated audio."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .._reference import OSS_TTSKIT, Fidelity, mirrors
from .datatypes import AudioFileFormat, SpeechResult


@mirrors("AudioOutput", module="TTSKit", docs=OSS_TTSKIT)
class AudioOutput:
    @staticmethod
    @mirrors(
        "AudioOutput.saveAudio(_:toFolder:filename:format:)",
        module="TTSKit",
        docs=OSS_TTSKIT,
        fidelity=Fidelity.APPROXIMATE,
        notes="Accepts a SpeechResult, or an array plus sample_rate (shim parameter). m4a needs ffmpeg.",
    )
    def save_audio(
        audio: SpeechResult | np.ndarray,
        to_folder: str | Path,
        filename: str,
        format: AudioFileFormat | str = AudioFileFormat.WAV,  # noqa: A002 - mirrors Swift label
        sample_rate: int | None = None,
    ) -> Path:
        import soundfile as sf

        if isinstance(audio, SpeechResult):
            samples, rate = audio.audio, audio.sample_rate
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when passing a raw array.")
            samples, rate = np.asarray(audio, np.float32), sample_rate
        fmt = AudioFileFormat(format)
        folder = Path(to_folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{filename}.{fmt.value}"
        if fmt is AudioFileFormat.WAV:
            sf.write(str(target), samples, rate)
            return target
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Saving .m4a needs ffmpeg on PATH.")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "out.wav"
            sf.write(str(wav), samples, rate)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", str(target)], check=True)
        return target
