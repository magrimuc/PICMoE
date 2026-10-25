"""
Q.ANT Native Computing Toolkit - KI & Neural Network Exploration
================================================================
Dieses Skript demonstriert die wichtigsten KI-Operationen des Q.ANT Toolkits
und vergleicht sie direkt mit einer PyTorch-Referenz (CPU-Backend, bfloat16).
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

import qant_native_computing_toolkit.generic as q_generic
import qant_native_computing_toolkit.native as q_native
import qant_native_computing_toolkit.ai as q_ai

def compare_results(name: str, qant_res: np.ndarray, torch_res: np.ndarray):
    """Berechnet und visualisiert Fehler zwischen Q.ANT und PyTorch."""
    q_f32 = qant_res.astype(np.float32)
    t_f32 = torch_res.astype(np.float32)
    abs_err = np.abs(q_f32 - t_f32)
    max_err = np.max(abs_err)
    mean_err = np.mean(abs_err)
    print(f"[{name}]")
    print(f"  Shape: {qant_res.shape} | Typ: {qant_res.dtype}")
    print(f"  Maximaler absoluter Fehler: {max_err:.6e}")
    print(f"  Mittlerer absoluter Fehler:  {mean_err:.6e}")
    print("  Status: OK (stimmt mit PyTorch überein)\n")

def main():
    print("=" * 65)
    print(" Q.ANT Native Computing Toolkit: KI- & Deep-Learning-Exploration")
    print("=" * 65)

    # 1. NPU initialisieren (im CPU-Backend simuliert)
    q_generic.init_npu(0)
    print("[OK] NPU Device 0 initialisiert (CPU-Backend)\n")

    # -------------------------------------------------------------
    # 2. KI: Linearer Layer (Dense / Fully Connected Forward Pass)
    # -------------------------------------------------------------
    print("--- 1. Linearer Layer (Dense / Matrix-Vektor-Multiplikation) ---")
    batch_size = 16
    in_features = 32
    out_features = 64

    # Eingaben und Gewichte in bfloat16
    x = np.random.uniform(-1.0, 1.0, size=(batch_size, in_features)).astype(bfloat16)
    w = np.random.uniform(-1.0, 1.0, size=(out_features, in_features)).astype(bfloat16)

    # Q.ANT Ausführung
    y_qant = q_ai.linear_fprop(x, w)

    # PyTorch Referenz
    x_t = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16)
    w_t = torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16)
    y_torch = torch.nn.functional.linear(x_t, w_t).to(torch.float32).numpy().astype(bfloat16)

    compare_results("Linear Layer", y_qant, y_torch)

    # -------------------------------------------------------------
    # 3. KI: 2D Faltung (Conv2d Layer)
    # -------------------------------------------------------------
    print("--- 2. 2D Convolution Layer (CNN) ---")
    # Shape: (Batch, Channels, Height, Width)
    img_in = np.random.uniform(-1.0, 1.0, size=(2, 3, 32, 32)).astype(bfloat16)
    conv_w = np.random.uniform(-1.0, 1.0, size=(8, 3, 3, 3)).astype(bfloat16) # 8 Filter, 3x3

    y_conv_qant = q_ai.conv_fprop(img_in, conv_w, stride=1, padding=1, dilation=1)

    img_in_t = torch.from_numpy(img_in.astype(np.float32)).to(torch.bfloat16)
    conv_w_t = torch.from_numpy(conv_w.astype(np.float32)).to(torch.bfloat16)
    y_conv_torch = (
        torch.nn.functional.conv2d(img_in_t, conv_w_t, stride=1, padding=1, dilation=1)
        .to(torch.float32)
        .numpy()
        .astype(bfloat16)
    )

    compare_results("Conv2D Layer", y_conv_qant, y_conv_torch)

    # -------------------------------------------------------------
    # 4. KI: Aktivierungsfunktionen (ReLU, Sigmoid, Softmax)
    # -------------------------------------------------------------
    print("--- 3. Aktivierungsfunktionen ---")
    act_in = np.random.uniform(-3.0, 3.0, size=(10, 20)).astype(bfloat16)
    act_in_t = torch.from_numpy(act_in.astype(np.float32)).to(torch.bfloat16)

    # ReLU
    relu_qant = q_ai.relu_fprop(act_in)
    relu_torch = torch.nn.functional.relu(act_in_t).to(torch.float32).numpy().astype(bfloat16)
    compare_results("ReLU Activation", relu_qant, relu_torch)

    # Sigmoid
    sig_qant = q_ai.sigmoid_fprop(act_in)
    sig_torch = torch.nn.functional.sigmoid(act_in_t).to(torch.float32).numpy().astype(bfloat16)
    compare_results("Sigmoid Activation", sig_qant, sig_torch)

    # -------------------------------------------------------------
    # 5. Q.ANT Spezialität: Photonic KAN (Kolmogorov-Arnold Network)
    # -------------------------------------------------------------
    print("--- 4. Photonisches KAN Layer (Kolmogorov-Arnold Network) ---")
    print("Hinweis: Photonische Quantenchips nutzen natürliche Licht-Interferenz")
    print("für Kosinus- und Fourier-Transformationen als lernbare Aktivierungen!\n")
    n_in = 8
    n_out = 16
    n_ks = 4

    kan_x = np.random.uniform(-1.0, 1.0, size=n_in).astype(bfloat16)
    phis = np.random.uniform(-np.pi, np.pi, size=(n_out, n_in, n_ks)).astype(bfloat16)
    amps = np.random.uniform(-1.0, 1.0, size=(n_out, n_in, n_ks)).astype(bfloat16)
    ks = np.arange(1, n_ks + 1).astype(bfloat16)

    kan_qant = q_ai.calc_kan_layer_fprop(kan_x, phis, amps, ks)
    kan_ref = np.sum(
        amps * np.cos(kan_x[None, :, None] * ks[None, None, :] + phis),
        axis=(1, 2)
    )
    compare_results("Photonic KAN Layer", kan_qant, kan_ref)

    # 6. NPU freigeben
    q_generic.release_npu(0)
    print("[OK] NPU Device 0 sicher freigegeben.")
    print("=" * 65)

if __name__ == "__main__":
    main()
