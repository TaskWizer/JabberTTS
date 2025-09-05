# Preferences
- User prefers Docker deployments with clear service separation (OpenWebUI for web+LLM, LiteTTS for TTS, ChromaDB for vectors, Caddy for proxy), specific API routing patterns (/api/* to OpenWebUI, /api/v1/* to LiteTTS), and localhost HTTP for development with easy VPS deployment via domain-only changes.
- User prefers Docker deployments that work out-of-the-box without requiring manual fix scripts or workarounds.
- User strongly prefers minimal changes to existing configurations and wants focus on making things work rather than adding extra features or modifications.
- User prefers organizing startup scripts in LiteTTS/scripts/ directory and test files in LiteTTS/tests/ directory for better project structure.
- User needs production-ready solutions with detailed change explanations.
- User prefers configuration hierarchy with override.json taking precedence over config.json, quantized models (model_q4.onnx) over fp16 models, comprehensive error handling with user-friendly messages, Git version control over backup folders, and structured logging for production systems.
- User prefers phonetic processing systems with pluggable notation support (IPA, Arpabet, Unisyn), RTF performance impact <10%, memory usage <100MB additional, maintaining 83.7%+ documentation coverage, and systematic task management approach with comprehensive testing at each phase.
- User requires comprehensive phonetic dictionaries: CMU dict at dictionaries/cmudict.dict (Arpabet), IPA dict at dictionaries/ipa_dict.json (JSON format), Unisyn dict at dictionaries/unisyn_dict.json (multi-accent), with performance targets of RTF impact <10% and memory <150MB.
- User prefers natural pronunciation for contractions (wasn't not waaasant, hmm not hum), numbers as individual digits with pauses (1...2...3... not one hundred twenty-three), question/exclamation intonation reflected in prosody, emotional expression with tone variation and natural pacing, and robust contraction handling with proper config hot reload functionality.
- User prefers comprehensive rebranding with directory renaming, consistent wiki-style navigation headers in documentation (except README/LICENSE), standard Docker Compose commands over platform-specific scripts, and README cleanup removing recent changes sections.
- User prefers dashboard with proper uptime formatting (HH:MM:SS), real-time voice management with actual data display, performance metrics that reset to 0.0 when idle >5 seconds, and minimal logging during normal operation.

# Deployment
- User prefers systematic Docker deployment audits covering port conflicts, build warnings, deprecation fixes, Python version alignment, security best practices, health checks, resource limits, and comprehensive documentation updates with production-ready configurations.
- User prefers systematic project reorganization with static content in static/ directory, docs in docs/, production-ready multi-stage Docker builds with non-root users and health checks, automated deployment scripts with zero-downtime and rollback capability, RTF targets < 0.25, memory overhead < 150MB, and comprehensive testing through actual user workflows rather than just API calls.
- User prefers consolidated single Docker configuration supporting both dev/prod environments.

# Production Environment
- User runs LiteTTS + OpenWebUI in production at console.cyopsys.com with specific requirements for both TTS and STT functionality.
- User prefers comprehensive audit approaches for audio dependencies and container configurations.
- LiteTTS models directory should be mounted as volume (./data/voices:/app/LiteTTS/models:rw) to persist downloaded ONNX models alongside the voices mount.

# Project Structure and Task Management
- User prefers organized project structure with tests in ./kokoro/tests/ and scripts in ./kokoro/scripts/.
- User prefers systematic task management with meaningful 20-minute work units.
- User prefers systematic task management with preserved functionality validation.

# Performance Optimization
- User prefers systematic performance optimization with incremental testing, RTF benchmarking after each change, threading configuration verification, multi-core ONNX Runtime inference, memory pre-allocation, CPU affinity configuration, and maintaining audio quality through subjective listening tests.
- User prefers aggressive CPU utilization (90-95% vs 60%), CPU affinity pinning across P-cores and E-cores, thermal throttling detection, pipeline parallelism for TTS stages, and RTF targets < 0.2 with special focus on short text optimization (< 0.25 RTF for texts under 20 characters).

# Quality Assurance
- User prefers comprehensive automated audio quality testing systems with objective metrics (WER, MOS prediction, prosody analysis), external ASR API integration with fallback mechanisms, CI/CD integration with configurable thresholds, performance benchmarking maintaining RTF < 0.25 and memory < 150MB targets, and systematic task management approach with incremental testing and backward compatibility.
- User requires comprehensive audio quality audits with OpenWebUI-specific testing, quantifiable metrics (WER, RTF, pronunciation accuracy), actual audio file generation for validation, and end-to-end testing through OpenWebUI interface rather than direct API calls.
- User requires comprehensive system audits with real validation through actual testing, objective metrics (WER, pronunciation accuracy, prosody analysis), evidence-based claims with audio file generation, and end-to-end testing through actual interfaces rather than just API calls.
- User prefers comprehensive end-to-end validation with performance targets maintained.
- User prefers comprehensive code audits with systematic comparison between working/broken versions, detailed task management with 20-minute work units, actual audio file generation for validation, objective quality metrics implementation, and end-to-end testing through actual interfaces rather than just API calls.

# Performance Targets and Validation
- User prefers RTF targets < 0.25.
- User prefers systematic task management with 20-minute work units and incremental validation through actual web interface testing.
