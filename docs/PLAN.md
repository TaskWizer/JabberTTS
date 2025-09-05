## **JabberTTS API Implementation Plan**

### **1.0 Executive Summary**

**Project Goal:** To design, build, and deploy a highly efficient, OpenAI-compatible TTS API endpoint using a heavily optimized `fishaudio/openaudio-s1-mini` model. The system will target CPU-only operation with a Real-Time Factor (RTF) < 0.5 on constrained hardware (4-core CPU, 4-8GB RAM) while supporting voice cloning and integrating with platforms like OpenWebUI.

**Core Strategy:** Achieve performance targets through a multi-faceted optimization pipeline: implementing **Unsloth's Dynamic 2.0**-inspired mixed-bit quantization, strategic model pruning, conversion to **ONNX Runtime**, and a sophisticated streaming API architecture.

**Key Outcome:** A scalable, low-latency `/v1/audio/speech` endpoint that makes high-quality, open-source voice synthesis accessible on consumer-grade hardware.

---

### **2.0 Project Goals & Measurable Success Criteria (OKRs)**

| **Objective** | **Key Results (KR)** | **Measurement Method** |
| :--- | :--- | :--- |
| **KR 1: Maximize Performance** | Achieve RTF < 0.5 on a 4-core Intel i5-8300H CPU (AVX2). | Benchmark with 100+ input samples of varying length. |
| **KR 2: Minimize Resource Use** | Reduce total model RAM footprint to < 2.0 GB. | Monitor `resident set size (RSS)` during inference. |
| **KR 3: Ensure High Quality** | Maintain a Mean Opinion Score (MOS) within 0.5 of the original FP16 model. | Blind A/B testing with 20+ participants. |
| **KR 4: Guarantee Compatibility** | Achieve 100% compatibility with the OpenAI Audio API schema for seamless OpenWebUI integration. | Automated endpoint validation against OpenAI spec. |
| **KR 5: Enable Voice Cloning** | Successfully clone voice from a <30s reference audio sample with the quantized model. | Subjective evaluation of similarity to target voice. |

---

### **3.0 Technical Architecture & Decided Direction**

After deep research, we have decided on a hybrid optimization strategy. **ONNX Runtime is the chosen inference engine**, but we will apply advanced quantization techniques *before* conversion to overcome its limitations.

**The Optimization Pipeline:**
`PyTorch (.pth) -> Pruning & Prep -> (Unsloth-inspired) Mixed-Bit Quantization -> ONNX Conversion -> ONNX Runtime Quantization -> ONNX Runtime CPU Deployment`

**The API Architecture:**
`Client -> FastAPI -> (Auth/Rate Limit) -> Request Queue -> Inference Engine (ONNX Runtime) -> Streaming Response`

---

### **4.0 Phase 1: Model Optimization & Quantization (Weeks 1-4)**

**Goal:** Produce a production-ready, optimized ONNX model file.

| **Step** | **Description** | **Tools / Techniques** | **Success Metric** | **Output** |
| :--- | :--- | :--- | :--- | :--- |
| **1.1 Environment Setup** | Establish a reproducible PyTorch environment for model manipulation. | PyTorch 2.4, CUDA (for initial work), `espeak-ng`, `fish-speech` lib. | Environment replicates on a second machine. | Dockerized environment. |
| **1.2 Baseline Establishment** | Benchmark the original S1-mini model on target hardware for RTF, RAM, and quality. | Custom scripts, `psutil`, `timeit`. | Full baseline metrics recorded. | Baseline report. |
| **1.3 Model Analysis & Pruning** | Analyze model structure. Identify & remove redundant layers/heads. Reduce default context window. | `torch.nn.utils.prune`, model summary tools. | 10-15% reduction in parameter count. | Pruned `.pth` model. |
| **1.4 Mixed-Bit Quantization** | **Core Innovation:** Implement a custom quantization script that applies different bit-widths (2-8 bit) to different layers based on sensitivity analysis. | Modified BitsAndBytes, custom calibration on audio text data. | Model size reduced to **300-400MB**. | Quantized `.pth` model. |
| **1.5 ONNX Conversion** | Convert the final quantized PyTorch model to ONNX format. | `torch.onnx.export`, with dynamic axes for variable-length input. | Successful conversion without errors. | `.onnx` model file. |
| **1.6 ONNX Runtime Optimization** | Apply ONNX Runtime's built-in graph optimizations and final quantization passes. | `onnxruntime`, `onnxoptimizer`. | Further 10-15% performance gain vs. pure PyTorch. | **Final `model_optimized.onnx`.** |

---

### **5.0 Phase 2: API Development & Integration (Weeks 5-7)**

**Goal:** Build a robust FastAPI server that serves the model efficiently.

| **Step** | **Description** | **Technology** | **Success Metric** |
| :--- | :--- | :--- | :--- |
| **2.1 Core API Server** | Implement primary FastAPI app with `/v1/audio/speech` endpoint matching OpenAI's spec (`text`, `voice`, `model`, `response_format`). | FastAPI, Pydantic. | API spec validated against OpenAI documentation. |
| **2.2 Inference Engine** | Develop a dedicated class to manage the ONNX Runtime session, handle input text tokenization, and run inference. | ONNX Runtime (`CPUExecutionProvider`). | Stable inference without memory leaks. |
| **2.3 Streaming Response** | Implement response streaming using MP3 encoding at 96kbps. Chunk audio based on sentence punctuation for low latency. | `ffmpeg-python`, FastAPI StreamingResponse. | First audio chunk delivered in <1s on target hardware. |
| **2.4 Voice Cloning Integration** | Add endpoint logic to accept reference audio, extract features, and condition the model on the target voice. | `librosa`, onnxruntime. | API accepts a `voice` parameter and alters output. |
| **2.5 Auxiliary Services** | Integrate `espeak-ng` for optional text pre-processing/ normalization. Add request queuing for stability. | Python `subprocess`, Redis. | Improved pronunciation for complex words. |

---

### **6.0 Phase 3: Testing & Validation (Weeks 8-9)**

**Goal:** Rigorously validate the system against all success criteria.

| **Test Type** | **Methodology** | **Tools** | **Pass Condition** |
| :--- | :--- | :--- | :--- |
| **6.1 Performance Testing** | Scripted load test with 100+ concurrent requests of varying lengths. Measure RTF, RAM, CPU, and latency. | `locust`, `pytest`, custom scripts. | **RTF < 0.5**, RAM < 2GB, P95 latency < 5s. |
| **6.2 Quality Testing (Objective)** | Use ASR to transcribe generated speech; calculate Word Error Rate (WER) against original text. | `whisper`, `jiwer`. | WER < 5% (within 1% of original model). |
| **6.3 Quality Testing (Subjective)** | Conduct blind A/B tests (Original vs. Quantized model). Participants rate naturalness and similarity on a 5-point scale. | Custom web app for evaluation. | **MOS >= 3.8** (Original is ~4.3). |
| **6.4 Compatibility Testing** | Verify the API response format, headers, and streaming behavior work with OpenWebUI's TTS module. | OpenWebUI dev deployment, manual testing. | Full compatibility; audio plays correctly. |
| **6.5 Endurance Testing** | Run the API under constant load for 24 hours. Monitor for memory leaks or performance degradation. | `docker stats`, `prometheus`, `grafana`. | No memory leaks; stable performance. |

---

### **7.0 Phase 4: Deployment & Documentation (Week 10)**

**Goal:** Prepare the system for production use.

| **Step** | **Description** | **Output** |
| :--- | :--- | :--- |
| **7.1 Containerization** | Create a multi-stage Dockerfile that produces a minimal image containing the API, runtime, and optimized model. | `Dockerfile` & `<5GB` container image. |
| **7.2 Configuration Management** | Externalize all configs (model path, port, log level) via environment variables. | `config.py` / `.env` template. |
| **7.3 Production Readiness** | Add health checks (`/health`), metrics endpoints (`/metrics`), and structured logging (JSON). | Docker `HEALTHCHECK`, Prometheus metrics. |
| **7.4 Comprehensive Documentation** | Write docs covering: API usage, deployment instructions, model optimization process, and benchmarking results. | `README.md`, `API.md`, `DEPLOYMENT.md`. |

---

### **8.0 Immediate Next Steps & Action Plan (Next 7 Days)**

1.  **Day 1-2: Environment Setup**
    *   [ ] Create a Git repository for the project.
    *   [ ] Set up a development environment with Python 3.10, PyTorch 2.4, and ONNX Runtime.
    *   [ ] Download the `fishaudio/openaudio-s1-mini` model and its codec from Hugging Face.
    *   [ ] **Success:** Run the original model and generate a test audio file.

2.  **Day 3-4: Establish Baselines**
    *   [ ] Write a benchmark script to measure RTF and RAM usage of the original model on a target CPU.
    *   [ ] Generate a set of 100 test phrases for quality evaluation.
    *   [ ] **Success:** A CSV file with baseline metrics for all test phrases.

3.  **Day 5-7: Begin Phase 1.3 (Pruning)**
    *   [ ] Analyze the model structure using `print(model)` and PyTorch summary tools.
    *   [ ] Research and implement a simple pruning strategy (e.g., magnitude pruning for linear layers).
    *   [ ] **Success:** A pruned model file that is 10% smaller but can still generate intelligible audio.

This plan provides a clear, actionable roadmap. The key to success lies in the iterative process of **Optimize -> Validate -> Measure** in Phase 1. By focusing on the model first, we ensure the API we build in Phase 2 is performant and stable from day one.
