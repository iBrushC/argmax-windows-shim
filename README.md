# argmax-shim

> **For development only.** This is a Python stand-in for the [Argmax SDK](https://app.argmaxinc.com/docs) so you can build on Windows and Linux machines. It is **not** a replacement for the real SDK and should never ship in a product.

The real Argmax SDK runs on Apple Silicon and Android. It uses models that Argmax has compiled and compressed for the Neural Engine and mobile NPUs, a streaming algorithm whose real-time accuracy matches file transcription, audio-based keyword spotting for custom vocabulary, a streaming speaker cache for live diarization, battery and thermal management, background model downloads, and production licensing and support. This shim has none of that. It runs the same open-source model families through generic PyTorch / CTranslate2 / ONNX runtimes and approximates the API's behavior so that the code you write here ports over with minimal changes.

Expect this shim to be slower, heavier, and less accurate in real-time mode than the real thing. If something works here, it will very likely work better on the real SDK. If something is slow or flaky here, don't conclude the same about Argmax.

## Layout

The package mirrors the SDK's module split, one kit per subpackage:

| Swift module | Python package | What it does |
|---|---|---|
| `import WhisperKit` | `argmax_shim.whisperkit` | File transcription, language detection, word timestamps |
| `import SpeakerKit` | `argmax_shim.speakerkit` | Speaker diarization (Pyannote), RTTM, merging speakers into transcripts |
| `import TTSKit` | `argmax_shim.ttskit` | Text-to-speech (Qwen3-TTS, plus lighter dev engines) |
| `import Argmax` (Pro) | `argmax_shim.argmax` | `WhisperKitPro`, `SpeakerKitPro`, real-time transcription, `LiveTranscriber`, custom vocabulary, Sortformer / Nemotron diarization |

Like Swift's `import Argmax`, `argmax_shim.argmax` re-exports the open-source types the Pro examples use (`DecodingOptions`, `AudioProcessor`, `PyannoteConfig`, ...).

## Install

Python 3.10+ (3.11 or 3.12 recommended). Install the core package plus the extras you need:

```bash
pip install -e ".[light]"              # mic + Whisper + Pyannote + Kokoro TTS: a good CPU laptop setup
pip install -e ".[mic,whisper]"        # just transcription
pip install -e ".[parakeet]"           # NVIDIA Parakeet v2/v3 via NeMo (Linux or WSL2 recommended)
pip install -e ".[qwen-asr]"           # Qwen3-ASR (GPU recommended)
pip install -e ".[sortformer]"         # Nemotron / Sortformer real-time diarization (NeMo)
pip install -e ".[tts-qwen]"           # Qwen3-TTS (GPU recommended)
pip install -e ".[tts-piper]"          # or tts-kokoro / tts-system for lighter TTS
```

A few setup notes:

- **ffmpeg** must be on `PATH` to load `.m4a`, `.mp4` and similar files (wav, flac, ogg and mp3 work without it).
- **Pyannote is gated.** Accept the terms for `pyannote/speaker-diarization-community-1` on Hugging Face, then set `HF_TOKEN`.
- **`qwen-asr` and `qwen-tts` conflict** on their `transformers` version. Put them in separate virtual environments.
- **Windows + NeMo** (Parakeet, Sortformer) is painful. Use WSL2, or stick to Whisper and Pyannote on native Windows.

## Quickstart

Every snippet below is a direct port of a Swift example from the Argmax docs.

**File transcription**

```python
from argmax_shim.argmax import ArgmaxSDK, ArgmaxConfig, WhisperKitPro, WhisperKitProConfig, WhisperKitProUtils

ArgmaxSDK.with_(ArgmaxConfig(api_key="ax_*****"))           # no-op in the shim
whisper_kit_pro = WhisperKitPro(WhisperKitProConfig(model="large-v3-v20240930_626MB"))
results = whisper_kit_pro.transcribe(audio_path="path/to/audio.m4a")
transcript = WhisperKitProUtils.merge_transcription_results(results)
print(transcript.text)
```

**Real-time transcription**

```python
from argmax_shim.argmax import ArgmaxSource, DecodingOptionsPro, LiveResult, LiveTranscriber, StreamType

live = LiveTranscriber(whisper_kit_pro)
mic = ArgmaxSource(stream_type=StreamType.device())
live.register_stream(stream_source=mic, options=DecodingOptionsPro())
for result in live.start_transcription(mic):
    match result:
        case LiveResult.Confirm(text, seconds, _):    print("confirmed:", text)
        case LiveResult.Hypothesis(text, seconds, _): print("hypothesis:", text)
```

**Diarization**

```python
from argmax_shim.argmax import AudioProcessor, SpeakerKit, SpeakerKitPro

speaker_kit = SpeakerKitPro()                                # Sortformer / Nemotron by default, as in the docs
audio = AudioProcessor.load_audio_as_float_array("meeting.wav")
result = speaker_kit.diarize(audio)
print("\n".join(str(line) for line in SpeakerKit.generate_rttm(result, file_name="meeting")))
```

**Text-to-speech**

```python
from argmax_shim.ttskit import TTSKit, TTSKitConfig, TTSModelVariant, Qwen3Speaker

tts = TTSKit(TTSKitConfig(model=TTSModelVariant.KOKORO_82M))   # light engine for laptops; default is Qwen3-TTS 0.6B
result = tts.generate("Hello from TTSKit!", speaker=Qwen3Speaker.SERENA)
print(f"Generated {result.audio_duration:.2f}s of audio at {result.sample_rate}Hz")
```

Runnable versions with CLI flags matching the Swift examples are in `examples/`.

## Models

Use the same model names as the real SDK. The shim resolves each one to its upstream open-source checkpoint:

| Argmax name | Upstream checkpoint | Python runtime |
|---|---|---|
| `large-v3-v20240930_626MB` (also `_turbo`) | openai/whisper-large-v3-turbo (CTranslate2 build) | faster-whisper |
| `tiny`, `base`, `small` (+ `.en`) | openai/whisper-* (CTranslate2 builds) | faster-whisper |
| `parakeet-v2_476MB` | nvidia/parakeet-tdt-0.6b-v2 | NeMo |
| `parakeet-v3_494MB` | nvidia/parakeet-tdt-0.6b-v3 | NeMo |
| `WhisperKitProConfig.qwen3_asr()` | Qwen/Qwen3-ASR-1.7B (+ Qwen3-ForcedAligner-0.6B for word timestamps) | qwen-asr |
| `SpeakerKitPro()` / `SortformerConfig` | nvidia/Nemotron-3-Diarization (fallback: diar_streaming_sortformer_4spk-v2.1) | NeMo |
| `PyannoteConfig` / `SpeakerKitDiarizer.pyannote()` | pyannote/speaker-diarization-community-1 | pyannote.audio 4.x |
| `pyannote/precision` | *closed-source (pyannoteAI Precision-2)*, falls back to Community-1 | — |
| `TTSModelVariant.QWEN3_TTS_0_6B` / `_1_7B` | Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice | qwen-tts |

The shim's default ASR model is Whisper turbo, because faster-whisper is the most portable backend on Windows. `qwen3-asr-0.6b` is also accepted as a lighter, shim-only option.

### Lighter TTS engines (shim-only)

On Apple Silicon, TTSKit runs Qwen3-TTS comfortably. On a laptop CPU it can't, so the shim offers lighter stand-ins with the same `TTSKit` API:

| Variant | Size | Quality | Languages | Notes |
|---|---|---|---|---|
| `QWEN3_TTS_0_6B` | ~1 GB | Best (same family as TTSKit) | 10 | Needs a GPU to be pleasant |
| `KOKORO_82M` | ~330 MB | Very good | en, ja, zh, es, fr, it, pt | Best CPU choice. ja/zh need `misaki[ja]` / `misaki[zh]` |
| `PIPER` | ~60 MB/voice | Clear, slightly robotic | en, de, fr, es, it, ru, pt, zh | Fastest neural option |
| `SYSTEM` | 0 | Robotic | Whatever the OS has | pyttsx3: SAPI5 on Windows, eSpeak NG on Linux |

The nine Qwen3 speakers (`RYAN`, `SERENA`, ...) map to similar-sounding voices on Kokoro. `TTSKitConfig(voice=...)` overrides the voice for Kokoro, Piper and System. Style instructions only work on `QWEN3_TTS_1_7B`, as in TTSKit.

## Naming conventions

Swift names translate to Python mechanically, so a coding agent can port in either direction:

| Swift | Python shim |
|---|---|
| `camelCase` methods, properties and labels | `snake_case` keyword arguments |
| `try await kit.transcribe(audioPath: p)` | `kit.transcribe(audio_path=p)` (synchronous) |
| `.voiceTriggered(minProcessInterval: 0.3)` | `StreamTranscriptionMode.voice_triggered(min_process_interval=0.3)` |
| `enum` cases like `.vad`, `.subsegment` | `ChunkingStrategy.VAD`, `SpeakerInfoStrategy.SUBSEGMENT` |
| `AsyncStream<[Float]>` + continuation | `make_audio_stream()` → `(AudioStream, AudioStreamContinuation)`, a blocking iterator |
| `for try await r in session.results` | `for r in session.results` (from one thread) |
| Enum with associated values (`.confirm(text, seconds, result)`) | Dataclasses + `match` (`case LiveResult.Confirm(text, seconds, result)`) |
| Labels that are Python keywords: `with`, `for:`, `from:`, `yield` | `with_`, `for_`, `from_`, `yield_` |

## Swapping to the real SDK

Every public class, function and method carries a decorator naming its Argmax equivalent, right above the declaration:

```python
@mirrors(
    "WhisperKitPro.setCustomVocabulary(_:)",
    module="Argmax",
    docs="https://app.argmaxinc.com/docs/examples/custom-vocabulary",
    fidelity=Fidelity.APPROXIMATE,
    notes="Whisper models: disabled, as in the SDK. ...",
)
def set_custom_vocabulary(self, words): ...
```

The same information is appended to each docstring (`help(WhisperKitPro.set_custom_vocabulary)`), and `MAPPING.md` collects all of it in one table. Regenerate it after changes with `python -m argmax_shim.reference`.

Each reference has a fidelity level:

- **exact**: same inputs, outputs and behavior. Translate the call and you're done.
- **approximate**: same shape, different internals or unverified details. Check the notes and re-test after swapping.
- **no-op**: accepted for compatibility but does nothing here (license init, battery mode, CoreML options). On the real SDK it will do something.
- **shim-only**: no Argmax equivalent. Delete it when swapping. The lightweight TTS engines are the main example; replace them with `.qwen3TTS_0_6b`.

`argmax_shim.IS_SHIM` is `True` here and doesn't exist in the real SDK. Remove any branches on it when you port.

### Prompt for a coding agent

When you're ready to move a project to the real SDK, give your agent the project, this package, and something like:

> This project was built against `argmax_shim`, a Python stand-in for the Argmax SDK. Port it to the Argmax Pro SDK in Swift (or Kotlin). Every shim declaration has an `@mirrors(...)` decorator naming its Swift equivalent and a fidelity level; `MAPPING.md` lists them all. Translate exact and approximate calls using the naming conventions in the shim's README, drop no-op arguments only after checking whether the real SDK needs them, and delete or replace every shim-only declaration (for example, `TTSModelVariant.KOKORO_82M` becomes `.qwen3TTS_0_6b`). Read the notes on approximate entries, and flag anything marked `TODO(verify)` for manual testing.

## What is not mirrored

Device-specific features are intentionally left out or reduced to no-ops:

- `.batteryOptimized` mode (aliased to `voice_triggered`), `ModelOptimizationMode`, CoreML speech decoder modes
- `ModelWarmup`, background downloads, pause/resume, network-type restrictions, crash recovery
- License activation (`ArgmaxSDK.with_` accepts and ignores the key), keychain persistence, obfuscated keys, active license counting
- iOS background audio, Argmax Local Server

## Known gaps

- **Real-time transcription** uses LocalAgreement-2 (re-transcribe a rolling buffer and confirm words two passes agree on), not Argmax's own streaming algorithm. It is noticeably laggier and its confirmed text can differ from file transcription.
- **Custom vocabulary** uses text-level fuzzy matching on Parakeet output instead of audio keyword spotting, and is passed as context to Qwen3-ASR. Like the real SDK, it's disabled for Whisper models. Recall is lower and false positives higher than the real feature.
- **Real-time diarization** re-diarizes the whole buffer every 2 seconds instead of using Sortformer's streaming speaker cache. Cost grows with session length, so keep demo sessions to a few minutes.
- **Parakeet v3** doesn't report a detected language label (`result.language` is `None`).
- **Timestamps** in streaming are always relative to stream start, whether or not `align_timestamps_to_global` is set.
- Code marked `TODO(verify)` relies on details of third-party packages that couldn't be confirmed when this was written: the Qwen3-ASR `time_stamps` structure and context argument, Qwen3-TTS sampling arguments, and loading Nemotron-3-Diarization through NeMo. Several Swift type names (e.g. `Qwen3Speaker`, `SpeakerSegment`) are inferred from the docs and marked "unverified" in their notes.

## Development

```bash
pip install -e ".[dev]"
pytest                                  # logic tests with fake models (no downloads)
python -m argmax_shim.reference         # regenerate MAPPING.md
```

`tests/test_reference.py` fails if a public declaration is missing its `@mirrors` reference, so new code keeps the swap path intact.

## License

The shim's code is MIT-licensed. Each model is governed by its own license and terms (for example, Nemotron-3-Diarization uses the OpenMDW license, and the Pyannote models are gated behind accepting their terms). Check each model card before using it beyond development. "Argmax", "WhisperKit", "SpeakerKit" and "TTSKit" refer to Argmax, Inc.'s products. This shim is an unofficial development aid and isn't affiliated with or endorsed by Argmax unless they say otherwise.
