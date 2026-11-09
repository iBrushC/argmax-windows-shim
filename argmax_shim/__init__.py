"""
argmax_shim: a DEVELOPMENT-ONLY Python stand-in for the Argmax SDK.

This package approximates the public API of Argmax's WhisperKit, SpeakerKit
and TTSKit (open-source and Pro) using open-source models, so developers on
Windows and Linux can build against the same shapes. It is NOT a replacement
for the Argmax SDK, which is faster, more energy-efficient, more accurate in
real-time mode, and built for production on Apple Silicon and Android.

Kits mirror the Swift module layout:

    argmax_shim.whisperkit  ->  import WhisperKit   (open-source SDK)
    argmax_shim.speakerkit  ->  import SpeakerKit   (open-source SDK)
    argmax_shim.ttskit      ->  import TTSKit       (open-source SDK)
    argmax_shim.argmax      ->  import Argmax       (Pro SDK)

Every public declaration carries an ``@mirrors(...)`` decorator naming the
equivalent Argmax declaration. Run ``python -m argmax_shim.reference`` to
regenerate MAPPING.md from those decorators.
"""

__version__ = "0.1.0"

IS_SHIM = True
"""Always True here and absent from the real SDK. Remove any branches on it when swapping."""
