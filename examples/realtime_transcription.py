"""
Real-time transcription from the microphone (LiveTranscriber, the recommended high-level API).

Mirrors the Basic Example: https://app.argmaxinc.com/docs/examples/real-time-transcription

    python examples/realtime_transcription.py
    python examples/realtime_transcription.py --model-name parakeet-v2_476MB --mode alwaysOn

Confirmed text is printed normally; hypothesis text is shown in blue. Press Ctrl+C to stop.
"""

import argparse
import signal
import threading
from datetime import datetime

from argmax_shim.argmax import (
    ArgmaxConfig,
    ArgmaxSDK,
    ArgmaxSource,
    DecodingOptions,
    DecodingOptionsPro,
    DecodingTask,
    LiveResult,
    LiveTranscriber,
    ModelStore,
    StreamTranscriptionMode,
    StreamType,
    WhisperKitPro,
    WhisperKitProConfig,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default="ax_dev")
    p.add_argument("--model-name", default="large-v3-v20240930_626MB", help="e.g. parakeet-v2_476MB, tiny.en, large-v3-v20240930_626MB")
    p.add_argument("--mode", default="voiceTriggered", choices=["alwaysOn", "voiceTriggered", "batteryOptimized"])
    p.add_argument("--device-id", type=int, default=None, help="see AudioProcessor.get_audio_devices()")
    args = p.parse_args()

    print("Initializing Argmax Pro SDK...")
    ArgmaxSDK.with_(ArgmaxConfig(api_key=args.api_key, keychain_persistence=False))

    print(f"Downloading {args.model_name} model using ModelStore...")
    model_folder = ModelStore().download_model(name=args.model_name, progress_callback=lambda f: print(f"\rDownload progress: {int(f * 100)}%", end=""))
    print(f"\nDownload completed: {model_folder}")

    whisper_kit_pro = WhisperKitPro(WhisperKitProConfig(model=args.model_name, model_folder=str(model_folder), verbose=True))
    live_transcriber = LiveTranscriber(whisper_kit_pro)

    stream_mode = {
        "alwaysOn": StreamTranscriptionMode.always_on(),
        "batteryOptimized": StreamTranscriptionMode.battery_optimized(),
    }.get(args.mode, StreamTranscriptionMode.voice_triggered(min_process_interval=0.3))
    print(f"Stream mode: {stream_mode.kind}")

    device_source = ArgmaxSource(stream_type=StreamType.device(args.device_id))
    options = DecodingOptionsPro(
        base=DecodingOptions(task=DecodingTask.TRANSCRIBE, word_timestamps=True),
        transcribe_interval=0.1,
        stream_transcription_mode=stream_mode,
        align_timestamps_to_global=True,
    )
    live_transcriber.register_stream(stream_source=device_source, options=options)
    device_results = live_transcriber.start_transcription(device_source)

    # Swift uses a DispatchSource for SIGINT; here a signal handler stops the stream.
    def stop(*_):
        print("\nStop recording...")
        threading.Thread(target=live_transcriber.stop_and_remove_stream, args=(device_source,)).start()

    signal.signal(signal.SIGINT, stop)

    accumulated_confirmed_text = ""
    for result in device_results:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        hypothesis_text = ""
        match result:
            case LiveResult.Confirm(text, seconds, _):
                accumulated_confirmed_text += " " + text
            case LiveResult.Hypothesis(text, seconds, _):
                hypothesis_text = text
        print(f"[{timestamp}] {accumulated_confirmed_text}\033[34m {hypothesis_text}\033[0m")

    print(f"\n\nTranscription:\n\n{accumulated_confirmed_text.strip()}\n")


if __name__ == "__main__":
    main()
