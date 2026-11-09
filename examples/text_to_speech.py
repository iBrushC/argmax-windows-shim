"""
Text-to-speech with TTSKit, plus the shim's lightweight engines.

Mirrors: https://github.com/argmaxinc/argmax-oss-swift#ttskit
(Swift CLI equivalent: `argmax-cli tts --text "..." --play`)

    python examples/text_to_speech.py --text "Hello from TTSKit!" --play
    python examples/text_to_speech.py --engine kokoro --text "Light and fast on a laptop CPU." --output-path out.wav
    python examples/text_to_speech.py --engine qwen-1.7b --speaker serena --instruction "Speak slowly and warmly"
"""

import argparse
from pathlib import Path

from argmax_shim.ttskit import (
    AudioOutput,
    GenerationOptions,
    PlaybackStrategy,
    Qwen3Language,
    Qwen3Speaker,
    TTSKit,
    TTSKitConfig,
    TTSModelVariant,
)

ENGINES = {
    "qwen-0.6b": TTSModelVariant.QWEN3_TTS_0_6B,  # same family as the real TTSKit default
    "qwen-1.7b": TTSModelVariant.QWEN3_TTS_1_7B,
    "kokoro": TTSModelVariant.KOKORO_82M,  # shim-only lightweight engines below
    "piper": TTSModelVariant.PIPER,
    "system": TTSModelVariant.SYSTEM,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello from TTSKit!")
    p.add_argument("--engine", default="qwen-0.6b", choices=list(ENGINES))
    p.add_argument("--speaker", default="ryan", choices=[s.name.lower() for s in Qwen3Speaker])
    p.add_argument("--language", default="english", choices=[l.name.lower() for l in Qwen3Language])
    p.add_argument("--instruction")
    p.add_argument("--play", action="store_true")
    p.add_argument("--output-path")
    args = p.parse_args()

    # Swift: let tts = try await TTSKit(TTSKitConfig(model: .qwen3TTS_0_6b))
    tts = TTSKit(TTSKitConfig(model=ENGINES[args.engine]))
    speaker = Qwen3Speaker[args.speaker.upper()]
    language = Qwen3Language[args.language.upper()]
    options = GenerationOptions(instruction=args.instruction)

    if args.play:
        # Swift: try await tts.play(text:, playbackStrategy: .auto)
        result = tts.play(args.text, speaker=speaker, language=language, options=options, playback_strategy=PlaybackStrategy.auto())
    else:
        # Swift: let result = try await tts.generate(text:speaker:language:options:) { progress in ... }
        def on_progress(progress):
            print(f"Audio chunk: {len(progress.audio)} samples")
            if progress.step_time is not None:
                print(f"First step took {progress.step_time:.2f}s")
            return True  # return False to cancel

        result = tts.generate(args.text, speaker=speaker, language=language, options=options, callback=on_progress)

    print(f"Generated {result.audio_duration:.2f}s of audio at {result.sample_rate}Hz")
    if args.output_path:
        out = Path(args.output_path)
        # Swift: try await AudioOutput.saveAudio(result.audio, toFolder:, filename:, format:)
        saved = AudioOutput.save_audio(result, to_folder=out.parent or ".", filename=out.stem, format=out.suffix.lstrip(".") or "wav")
        print(f"Saved to {saved}")


if __name__ == "__main__":
    main()
