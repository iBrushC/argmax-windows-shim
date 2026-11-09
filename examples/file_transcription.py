"""
File transcription (optionally with speakers and custom vocabulary).

Mirrors: https://app.argmaxinc.com/docs/examples/file-transcription
         https://app.argmaxinc.com/docs/examples/custom-vocabulary

    python examples/file_transcription.py --audio-path meeting.wav
    python examples/file_transcription.py --audio-path meeting.wav --model parakeet-v2_476MB --diarize
    python examples/file_transcription.py --audio-path clip.m4a --detect-language
    python examples/file_transcription.py --audio-path call.wav --model parakeet-v3_494MB --vocab Argmax WhisperKitPro
"""

import argparse

from argmax_shim.argmax import (
    ArgmaxConfig,
    ArgmaxSDK,
    AudioProcessor,
    ChunkingStrategy,
    CustomVocabularyConfig,
    DecodingOptions,
    PyannoteConfig,
    SpeakerKitDiarizer,
    SpeakerKitPro,
    WhisperKitPro,
    WhisperKitProConfig,
    WhisperKitProUtils,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-path", required=True)
    p.add_argument("--api-key", default="ax_dev")
    p.add_argument("--model", default="large-v3-v20240930_626MB")
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--detect-language", action="store_true")
    p.add_argument("--vocab", nargs="*", default=None)
    args = p.parse_args()

    # Swift: await ArgmaxSDK.with(ArgmaxConfig(apiKey: "ax_*****"))
    ArgmaxSDK.with_(ArgmaxConfig(api_key=args.api_key))

    # Swift: let config = WhisperKitProConfig(model: "...", customVocabularyConfig: .init(words: [...]))
    config = WhisperKitProConfig(
        model=args.model,
        custom_vocabulary_config=CustomVocabularyConfig(words=args.vocab) if args.vocab else None,
    )
    whisper_kit_pro = WhisperKitPro(config)

    if not args.diarize:
        # Swift: let results = try? await whisperKitPro.transcribe(audioPath:, decodeOptions:)
        options = DecodingOptions(detect_language=True) if args.detect_language else DecodingOptions()
        results = whisper_kit_pro.transcribe(audio_path=args.audio_path, decode_options=options)
        transcript = WhisperKitProUtils.merge_transcription_results(results)
        if args.detect_language:
            print(f"Detected language: {transcript.language}")
        print(transcript.text)
        return

    # ---- With speakers ----
    audio_array = AudioProcessor.load_audio_as_float_array(args.audio_path)

    # Swift: let diarizer = SpeakerKitDiarizer.pyannote(); downloadModels(); loadModels()
    diarizer = SpeakerKitDiarizer.pyannote()
    diarizer.download_models()
    diarizer.load_models()
    speaker_kit = SpeakerKitPro(PyannoteConfig(diarizer=diarizer))

    # Word timestamps are required for merging transcripts with diarization
    decoding_options = DecodingOptions(word_timestamps=True, chunking_strategy=ChunkingStrategy.VAD)
    transcribe_result = whisper_kit_pro.transcribe(audio_array=audio_array, decode_options=decoding_options)
    merged_result = WhisperKitProUtils.merge_transcription_results(transcribe_result)

    diarization_result = speaker_kit.diarize(audio_array)

    # Swift: diarizationResult.addSpeakerInfo(to: [mergedResult], strategy: .subsegment)
    for segments in diarization_result.add_speaker_info(to=[merged_result]):
        for segment in segments:
            print(f"[{segment.start:7.2f}-{segment.end:7.2f}] Speaker {segment.speaker}: {segment.text.strip()}")


if __name__ == "__main__":
    main()
