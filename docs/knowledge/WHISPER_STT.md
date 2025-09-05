# Open WebUI's Whisper STT

---
**📚 LiteTTS Documentation Navigation**

**Core Documentation:** [Features](../../FEATURES.md) | [Configuration](../../CONFIGURATION.md) | [Performance](../../PERFORMANCE.md) | [Monitoring](../../MONITORING.md) | [Testing](../../TESTING.md) | [Troubleshooting](../../TROUBLESHOOTING.md)

**Setup & Usage:** [Dependencies](../../DEPENDENCIES.md) | [Quick Start](../../usage/QUICK_START_COMMANDS.md) | [Docker Deployment](../../usage/DOCKER-DEPLOYMENT.md) | [OpenWebUI Integration](../../usage/OPENWEBUI-INTEGRATION.md)

**Advanced:** [API Reference](../../api/API_REFERENCE.md) | [Development](../README.md) | [Voice System](../../voices/README.md) | [Watermarking](../../WATERMARKING.md)

**Project:** [Changelog](../../CHANGELOG.md) | [Roadmap](../../ROADMAP.md) | [Contributing](../../CONTRIBUTIONS.md) | [Beta Features](../../BETA_FEATURES.md)

---
Open WebUI's Whisper STT (Speech-to-Text) integration offers several base options for local processing, primarily distinguished by the model size and its capabilities:

## Model Sizes:
* Open WebUI supports various Whisper models, which vary in size and performance. These include:
  - tiny
  - base
  - small
  - medium
  - large
  - turbo (a more recent, faster model, primarily for English)
  - Specific models like "tiny English" can also be downloaded and used.

## Language Support:
* Some models, like tiny.en, base.en, small.en, and medium.en, are specifically trained for English-only transcription and are generally more accurate for English.
* Multilingual models (e.g., tiny, base, small, medium, large) can handle transcription and translation for a wide range of languages.
* Automatic Detection: Whisper can automatically detect the language of the audio.
* Manual Selection: You can choose a specific language from a supported list.
  * Note: If the chosen language and the audio's actual language mismatch, Whisper might attempt to translate the audio to the selected language.
* OpenAI's Whisper models support numerous languages for both transcription and translation.

## Configuration:
* The specific Whisper model used can be configured within Open WebUI's settings, typically in the Admin Panel under "Settings" and then "Audio" or "STT settings."
* Users can select the desired model from a dropdown list or, in some cases, specify it through environment variables (e.g., WHISPER_MODEL=distil-large-v3 for faster-whisper compatible models).

## Considerations:
* Larger models generally offer higher accuracy but require more computational resources and longer download times.
* The turbo model is optimized for speed in English transcription but may not be suitable for translation tasks.
* For non-English speech or translation, multilingual models like medium or large are recommended for better results.


## Other configuration options
* Output Format: Whisper can output transcriptions in various formats like txt, vtt, srt, tsv, and json.
* Audio file splitting: OpenWebUI supports splitting larger audio files into smaller chunks to accommodate API size limits, according to GitHub discussions.
* Voice Activity Detection (VAD): While OpenWebUI itself doesn't explicitly offer built-in VAD controls in the user interface, it is possible to integrate VAD into the process to filter out silence from audio before transcription, improving accuracy and efficiency.

It's important to consult the Open WebUI documentation and related discussions on GitHub for the most up-to-date information on configuring Whisper STT options within the platform.
