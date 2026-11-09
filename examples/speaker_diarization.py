"""
Speaker diarization on a file, with RTTM output.

Mirrors: https://app.argmaxinc.com/docs/examples/speaker-diarization

    python examples/speaker_diarization.py pyannote --audio-path meeting.wav --num-speakers 2
    python examples/speaker_diarization.py sortformer --audio-path meeting.wav --rttm-path out.rttm

Pyannote models are gated: accept the terms for pyannote/speaker-diarization-community-1
on Hugging Face and set HF_TOKEN.
"""

import argparse
from pathlib import Path

from argmax_shim.argmax import (
    ArgmaxConfig,
    ArgmaxSDK,
    AudioProcessor,
    PyannoteConfig,
    PyannoteDiarizationOptions,
    SortformerDiarizationOptions,
    SortformerMode,
    SpeakerKit,
    SpeakerKitDiarizer,
    SpeakerKitPro,
)


def output_rttm(result, audio_path: str, rttm_path: str | None) -> None:
    file_name = Path(audio_path).stem
    rttm_lines = SpeakerKit.generate_rttm(result, file_name=file_name)
    content = "\n".join(line.description for line in rttm_lines)
    if rttm_path:
        Path(rttm_path).write_text(content + "\n")
        print(f"RTTM saved to {rttm_path}")
    else:
        print(content)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("engine", choices=["pyannote", "sortformer"])
    p.add_argument("--api-key", default="ax_dev")
    p.add_argument("--audio-path", required=True)
    p.add_argument("--rttm-path")
    p.add_argument("--num-speakers", type=int)
    args = p.parse_args()

    ArgmaxSDK.with_(ArgmaxConfig(api_key=args.api_key))
    audio_frames = AudioProcessor.load_audio_as_float_array(args.audio_path)

    if args.engine == "pyannote":
        diarizer = SpeakerKitDiarizer.pyannote()
        diarizer.download_models()
        diarizer.load_models()
        speaker_kit = SpeakerKitPro(PyannoteConfig(diarizer=diarizer))
        options = PyannoteDiarizationOptions(number_of_speakers=args.num_speakers) if args.num_speakers else None
    else:
        speaker_kit = SpeakerKitPro()  # Sortformer / Nemotron by default
        options = SortformerDiarizationOptions(sortformer_mode=SortformerMode.PRERECORDED)

    try:
        result = speaker_kit.diarize(audio_frames, options=options)
        output_rttm(result, args.audio_path, args.rttm_path)
        if result.timings:
            print(result.timings.debug_description)
    finally:
        speaker_kit.unload_models()


if __name__ == "__main__":
    main()
