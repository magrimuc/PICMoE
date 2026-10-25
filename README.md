# PICMoE: Photonic Integrated Circuit Mixture-of-Experts

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hardware: Q.ANT NPU](https://img.shields.io/badge/Hardware-Q.ANT%20NPU%20(Photonic)-orange.svg)](https://qant.de/)
[![Model: Phi-mini-MoE-instruct](https://img.shields.io/badge/Model-microsoft%2FPhi--mini--MoE--instruct-purple.svg)](https://huggingface.co/microsoft/Phi-mini-MoE-instruct)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

**PICMoE** demonstrates the acceleration of **Mixture-of-Experts (MoE)** large language models using **Photonic Integrated Circuits (PIC)** and the **Q.ANT Native Computing Toolkit**.

By combining light-based native computing (analog optical matrix-vector multiplications in `bfloat16`) with sparse expert routing, PICMoE demonstrates how next-generation photonic NPUs can execute sparse autoregressive transformer inference with high energy efficiency and ultra-low latency.

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
                          │   (RoPE & Causal Mask)   │
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
                    │   Projection to 32,064 Vocab Logits   │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Next Generated Token   │
                          └──────────────────────────┘
```

### Model Specifications: `microsoft/Phi-mini-MoE-instruct`
- **Total Parameters:** 7.6 Billion
- **Active Parameters per Token:** 2.4 Billion (Top-2 out of 16 experts)
- **Hidden Dimension:** 4,096
- **Expert Intermediate Dimension:** 960
- **Attention Heads:** 32 Query Heads, 8 Key/Value Heads (Grouped-Query Attention)
- **Precision:** `bfloat16` Native Representation

---

## Features

- **Q.ANT NPU Token Generation Loop (`demo` mode)**:
  Simulates the full autoregressive MoE generation cycle on the Q.ANT Photonic NPU (using the CPU-accelerated native computing backend). Demonstrates router gating, sparse expert activation, SwiGLU activation, and vocabulary projection. Runs instantly without requiring large weight downloads.
- **Full Model Hugging Face Inference (`--full` mode)**:
  Loads the complete 14.25 GB model weights (`microsoft/Phi-mini-MoE-instruct`), executes the full 32-layer transformer pipeline, and produces natural conversational responses using Microsoft's official chat template.
- **Deep Compatibility Engine**:
  Resolves internal differences between modern Hugging Face Transformers releases and the custom MoE dynamic modules (`modeling_slimmoe.py`), ensuring numerically stable KV-caching and correct attention masking.
- **Containerized Deployment**:
  Production-ready `Dockerfile` based on Python 3.12 with OpenMP runtime and CPU-optimized PyTorch.

---

## Quickstart

### Prerequisites
- Python 3.12 (required for the precompiled Q.ANT CPU wheels)
- 16 GB RAM (minimum for full model weights)

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/magrimuc/PICMoE.git
cd PICMoE

# Install dependencies (automatically links local Q.ANT wheel)
pip install -r requirements.txt
```

### 2. Run the Q.ANT NPU MoE Generation Demo

Execute the standalone NPU token generation loop:

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
  Step 01: Experten [11  3] aktiv -> Token  5337 'attack'
  Step 02: Experten [ 5 12] aktiv -> Token 17595 'sona'
  Step 03: Experten [ 2 15] aktiv -> Token  6607 'Oper'
  ...
[Fertig] 15 Tokens generiert in 8.308 s (1.8 Tokens/s)
[OK] Q.ANT NPU sicher freigegeben.
```

### 3. Run Full MoE Inference

To run inference with the full 14.25 GB model weights:

```bash
python phi3_inference.py --full "Was sind die Vorteile von MoE-Architekturen?"
```

---

## Docker Deployment

Build and run PICMoE in an isolated container. PICMoE supports both **consumer hardware (CPU simulation)** and **enterprise server racks with physical Q.ANT NPUs**:

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
On server racks equipped with PCIe-connected Q.ANT NPUs, pass `--build-arg BACKEND=hardware` to build with the native hardware driver bindings:

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
├── Dockerfile                   # Multi-stage CPU-optimized Docker container
├── .dockerignore                # Build context filter
├── .gitignore                   # Git exclusions
├── requirements.txt             # Verified Python dependencies
├── phi3_inference.py            # Primary inference suite (Q.ANT NPU demo + full HF MoE)
├── phi35_qant_demo.py           # Layer-level acceleration benchmark
├── explore_qant_ai.py           # Deep dive into Q.ANT native AI operations
├── qant1.py                     # Minimal photonic elementwise demo
├── qant_wheels_cpu/             # Precompiled Q.ANT Native Computing Toolkit wheels
│   ├── qant_native_computing_toolkit-2.4.0-cp312-cp312-linux_x86_64.whl
│   ├── qant_native_computing_toolkit-2.4.0-cp312-cp312-macosx_11_0_arm64.whl
│   └── qant_native_computing_toolkit-2.4.0-cp312-cp312-win_amd64.whl
└── README.md                    # Project documentation
```

---

## Technical Notes & Version Pinning

- **Hugging Face Transformers Compatibility:**
  `microsoft/Phi-mini-MoE-instruct` relies on custom dynamic modules (`modeling_slimmoe.py`) designed for Transformers 4.43/4.44. The project strictly pins `transformers==4.44.2` and `tokenizers==0.19.1` to ensure correct `DynamicCache` indexing, sliding-window causal masks, and position ID handling.
- **bfloat16 Numerical Emulation:**
  Q.ANT's optical compute engine uses native `bfloat16`. On standard x86 CPU hardware, operations run via AVX2 emulation through `ml_dtypes`.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
The Q.ANT Native Computing Toolkit is provided by [Q.ANT GmbH](https://qant.de/).
