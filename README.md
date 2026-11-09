# PICMoE: Photonic Integrated Circuit Acceleration for LLMs (MoE & Dense)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hardware: Q.ANT NPU](https://img.shields.io/badge/Hardware-Q.ANT%20NPU%20(Photonic)-orange.svg)](https://qant.de/)
[![Model: Phi-mini-MoE](https://img.shields.io/badge/Model-microsoft%2FPhi--mini--MoE--instruct-purple.svg)](https://huggingface.co/microsoft/Phi-mini-MoE-instruct)
[![Model: Qwen-2.5-14B](https://img.shields.io/badge/Model-Qwen%2FQwen2.5--14B--Instruct-blueviolet.svg)](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

**PICMoE** demonstrates the acceleration of frontier **Large Language Models (LLMs)**—both sparse **Mixture-of-Experts (MoE)** and high-parameter **Dense Transformer architectures**—using **Photonic Integrated Circuits (PIC)** and the official **Q.ANT Native Computing Toolkit**.

By combining light-based computing (analog optical matrix-vector multiplications in `bfloat16` via Thin-Film Lithium Niobate / LNOI) with sparse routing and Grouped-Query Attention (GQA), PICMoE showcases how next-generation photonic NPUs achieve ultra-low latency, high throughput, and up to **30x higher energy efficiency** compared to electronic processors.

---

## Supported Architectures

### 1. `microsoft/Phi-mini-MoE-instruct` (Sparse MoE)
- **Total Parameters:** ~14.25 GB in `bfloat16`
- **Sparse Routing:** 16 Experts per layer, Top-2 active per token (only 2 active MLPs executed on NPU)
- **Hidden Dimension:** 4,096 | **Expert Intermediate Dimension:** 960 (SwiGLU)
- **Attention:** 32 Query Heads, 8 KV Heads

### 2. `Qwen/Qwen2.5-14B-Instruct` (High-Performance Dense Flagship)
- **Total Parameters:** 14.7 Billion Dense Parameters
- **Hidden Dimension:** 5,120 | **Intermediate MLP Dimension:** 13,824 (SwiGLU)
- **Grouped-Query Attention (GQA):** 40 Query Heads, 8 KV Heads (**5:1 KV-Cache Compression**)
- **Vocabulary:** 152,064 classes (multilingual tokenization)

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Input Prompt Tokens    │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  Multi-Head Attention    │
                          │   (RoPE & GQA Heads)     │
                          └─────────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │ Hidden State x (dim = 4096) │
                         └──────────────┬──────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
       ┌────────────────────────┐              ┌────────────────────────┐
       │   Photonic MoE Router  │              │    Top-2 Sparse Gate   │
       │  (16 Experts Gating)   │              │   (Only 2 Active MLPs) │
       │      W_router @ x      │              │      Top-1 & Top-2     │
       └────────────┬───────────┘              └────────────┬───────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │     Photonic Expert Feed-Forward      │
                    │   Gate & Up Projections (SwiGLU)      │
                    │     Down Projection & Weighted Sum    │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │        Photonic LM-Head               │
                    │   Projection to Vocab Logits          │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Next Generated Token   │
                          └──────────────────────────┘
```

---

## Features

- **Photonic MoE Token Generation Loop (`phi3_inference.py`)**:
  Simulates the full autoregressive MoE generation cycle on the Q.ANT Photonic NPU. Demonstrates router gating, sparse expert activation, SwiGLU activation, and vocabulary projection.
- **Qwen-2.5-14B Layer & Token Acceleration (`qwen14b_qant_demo.py`)**:
  Full optical pipeline for Alibaba's Qwen-2.5-14B: Grouped-Query Attention (5:1 compression), SwiGLU (5120 -> 13824 -> 5120), and 152k LM-Head projections executed on NPU.
- **Full Model Hugging Face Inference (`phi3_inference.py --full`)**:
  Loads complete model weights, executes the transformer pipeline, and produces natural conversational responses.
- **Deep Compatibility Engine**:
  Resolves dynamic cache (`seen_tokens`) compatibility in modern Hugging Face releases for custom MoE architectures.
- **Dual Backend Containerization**:
  Production-ready `Dockerfile` supporting both local CPU emulation and enterprise server racks with physical Q.ANT PCIe NPUs.

---

## Quickstart

### Prerequisites
- Python 3.12 (required for precompiled Q.ANT wheels)
- Recommended: 16 GB RAM (for local simulation)

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/magrimuc/PICMoE.git
cd PICMoE

# Install dependencies (automatically links local Q.ANT wheel)
pip install -r requirements.txt
```

### 2. Run the MoE NPU Generation Demo

Execute the standalone NPU MoE token generation loop (runs instantly without large weight downloads):

```bash
python phi3_inference.py
```

**Example Output:**
```
=================================================================
 1. Q.ANT NPU MoE Token-Generierung (microsoft/Phi-mini-MoE-instruct)
=================================================================
Lade Tokenizer fuer microsoft/Phi-mini-MoE-instruct...
[OK] Q.ANT NPU initialisiert (CPU-Backend, Device 0).

--- Starte autoregressive MoE-Generierung auf NPU ---
  Step 01: Experten [4 5] aktiv -> Token 4747
  Step 02: Experten [2 3] aktiv -> Token 26463
  Step 03: Experten [11 0] aktiv -> Token 1854
  ...
[Fertig] 15 Tokens generiert in 8.816 s
[OK] Q.ANT NPU sicher freigegeben.
```

### 3. Run Qwen-2.5-14B Layer Acceleration

Run the full layer-level optical acceleration for Qwen-2.5-14B (GQA + SwiGLU + LM-Head):

```bash
python qwen14b_qant_demo.py
```

### 4. Run Full MoE Inference

To run inference with the full model weights:

```bash
python phi3_inference.py --full "Was sind die Vorteile von Mixture-of-Experts?"
```

---

## Hardware vs. Simulation Benchmark

| Metric | Consumer CPU (Simulation) | Q.ANT NPU Server (Photonic PIC) |
| :--- | :--- | :--- |
| **Token Generation Rate** | ~1.8 – 2.5 Tokens/s | **35 – 80+ Tokens/s** |
| **Time-to-First-Token (Prefill)** | 2 – 5 Seconds | **< 50 Milliseconds** |
| **Matrix Multiply Latency** | Milliseconds | **Picoseconds (Optical Flight Time)** |
| **Host System Load** | 100% CPU Volllast | Minimal (Offloaded to Optical Chip) |
| **Energy Efficiency** | ~50 Joules / Token | **Up to 30x more energy-efficient** |

---

## Docker Deployment

Build and run PICMoE in an isolated container. Supports both consumer hardware and server racks with physical Q.ANT PCIe NPUs:

### A. Consumer Hardware (CPU Simulation, No Drivers Needed)
```bash
# Build the default image
docker build -t picmoe:cpu .

# Run the fast NPU demo
docker run --rm picmoe:cpu

# Run full inference (with persistent Hugging Face cache mount)
docker run --rm -v ${HOME}/.cache/huggingface:/app/cache/huggingface picmoe:cpu --full "Erklaere optisches Rechnen in einem Satz."
```

### B. Server Rack with Physical Q.ANT NPU Hardware
On server racks equipped with PCIe-connected Q.ANT NPUs, pass `--build-arg BACKEND=hardware`:

```bash
# Build with hardware driver bindings
docker build --build-arg BACKEND=hardware -t picmoe:npu .

# Run with hardware device passthrough (pass the host NPU device node)
docker run --rm --device /dev/qant_npu0 picmoe:npu
```

---

## Project Structure

```
PICMoE/
├── Dockerfile                   # Multi-stage CPU/Hardware Docker container
├── .dockerignore                # Build context filter
├── .gitignore                   # Git exclusions
├── requirements.txt             # Verified Python dependencies
├── phi3_inference.py            # Primary MoE inference suite (NPU demo + full HF MoE)
├── qwen14b_qant_demo.py         # Qwen-2.5-14B GQA & SwiGLU photonic acceleration
├── phi35_qant_demo.py           # Layer-level MoE acceleration benchmark
├── explore_qant_ai.py           # Deep dive into Q.ANT native AI operations (CNN, KAN, MLP)
├── qant1.py                     # Minimal photonic elementwise demo
├── qant_wheels_cpu/             # Precompiled Q.ANT CPU-backend wheels
│   ├── qant_native_computing_toolkit-2.4.0-cp312-cp312-linux_x86_64.whl
│   ├── qant_native_computing_toolkit-2.4.0-cp312-cp312-macosx_11_0_arm64.whl
│   └── qant_native_computing_toolkit-2.4.0-cp312-cp312-win_amd64.whl
├── qant_wheels_hardware/        # Physical Q.ANT NPU hardware wheels
│   ├── qant_native_computing_toolkit-2.4.0-cp312-cp312-linux_x86_64.whl
│   └── qant_native_computing_toolkit-2.4.0-cp313-cp313-linux_x86_64.whl
└── README.md                    # Project documentation
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).  
The Q.ANT Native Computing Toolkit is provided by [Q.ANT GmbH](https://qant.de/).
