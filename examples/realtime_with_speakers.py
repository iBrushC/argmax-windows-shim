"""
Real-time transcription with speakers (Nemotron / Sortformer diarization).

Mirrors the Advanced Example: https://app.argmaxinc.com/docs/examples/real-time-transcription#with-speakers

    python examples/realtime_with_speakers.py --model-name parakeet-v2_476MB

Requires the `sortformer` extra (NeMo). Press Ctrl+C to stop.
"""

import argparse
import signal
from datetime import datetime

from argmax_shim.argmax import (
    ArgmaxConfig,
    ArgmaxSDK,
    ChunkingStrategy,
    DecodingOptions,
    DecodingOptionsPro,
    DecodingTask,
    SortformerConfig,
    SortformerDiarizationOptions,
    SortformerMode,
    SortformerStreamingConfig,
    SpeakerKitPro,
    StreamTranscriptionMode,
    TranscribeDiarizeResultType,
    WhisperKitPro,
    WhisperKitProConfig,
)


def group_words_by_speaker(words) -> list[str]:
    """Groups consecutive words from the same speaker into labeled runs (same helper as the Swift example)."""
    lines, current_speaker, current_text = [], None, ""
    for w in words:
        if w.speaker != current_speaker:
            if current_text:
                label = f"Speaker {current_speaker}" if current_speaker is not None else "Unknown"
                lines.append(f"[{label}]: {current_text.strip()}")
            current_speaker, current_text = w.speaker, w.word_timing.word
        else:
            current_text += w.word_timing.word
    if current_text:
        label = f"Speaker {current_speaker}" if current_speaker is not None else "Unknown"
        lines.append(f"[{label}]: {current_text.strip()}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default="ax_dev")
    p.add_argument("--model-name", default="large-v3-v20240930_626MB")
    args = p.parse_args()

    ArgmaxSDK.with_(ArgmaxConfig(api_key=args.api_key, keychain_persistence=False))
    whisper_kit_pro = WhisperKitPro(WhisperKitProConfig(model=args.model_name))
    speaker_kit = SpeakerKitPro(SortformerConfig(load=True, streaming_config=SortformerStreamingConfig.REALTIME))

    options = DecodingOptionsPro(
        base=DecodingOptions(task=DecodingTask.TRANSCRIBE, word_timestamps=True, chunking_strategy=ChunkingStrategy.VAD),
        transcribe_interval=0.1,
        stream_transcription_mode=StreamTranscriptionMode.voice_triggered(min_process_interval=0.3),
        align_timestamps_to_global=True,
    )
    transcribe_session = whisper_kit_pro.make_stream_session(options=options)
    combined_session = speaker_kit.make_stream_session(
        transcription_session=transcribe_session,
        diarization_config=SortformerDiarizationOptions(sortformer_mode=SortformerMode.REALTIME),
    )

    mic_stream, mic_continuation = whisper_kit_pro.audio_processor.start_streaming_recording_live()
    signal.signal(signal.SIGINT, lambda *_: (print("\nStop recording..."), mic_continuation.finish()))
    combined_session.start(mic_stream)
    print("\nRecording from microphone... (Ctrl+C to stop)\n")

    accumulated_text = ""
    confirmed_batches = []
    for result in combined_session.results:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if result.type is TranscribeDiarizeResultType.NEW_TRANSCRIPTION:
            if result.hypothesis_text:
                formatted = "\n  ".join(group_words_by_speaker(result.hypothesis_words_with_speakers))
                if formatted:
                    print(f"[{ts}] [t={result.seek_time:.2f}s] [Speakers: {result.speaker_count}] Hypothesis:\n  {formatted}")
            if result.confirmed_words_with_speakers:
                formatted = "\n  ".join(group_words_by_speaker(result.confirmed_words_with_speakers))
                print(f"[{ts}] [t={result.seek_time:.2f}s] [Speakers: {result.speaker_count}] Confirmed:\n  {formatted}")
                confirmed_batches.append((result.seek_time, result.text))
                accumulated_text = result.text
        elif result.type is TranscribeDiarizeResultType.SPEAKER_REVISION:
            # In a UI app: find the batch whose seek_time == result.seek_time and replace its speakers.
            formatted = "\n  ".join(group_words_by_speaker(result.confirmed_words_with_speakers))
            print(f"[{ts}] [t={result.seek_time:.2f}s] [Speaker revision]:\n  {formatted}")

    print(f"\n\nFinal transcript:\n{accumulated_text}")


if __name__ == "__main__":
    main()
