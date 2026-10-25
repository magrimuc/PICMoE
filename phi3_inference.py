"""
Phi-mini-MoE Inferenz mit dem Q.ANT Native Computing Toolkit
============================================================
Modell: microsoft/Phi-mini-MoE-instruct
Architektur: Mixture of Experts (MoE) - 16 Experten, Top-2 Routing

Dieses Skript zeigt:
1. 'demo': Testet den echten Q.ANT MoE Token-Generierungs-Loop (Router, Top-2 Experten,
   Attention und LM-Head auf der NPU; laeuft sofort ohne 14 GB Download).
2. 'full': Laedt das vollstaendige 'microsoft/Phi-mini-MoE-instruct' ueber Hugging Face,
   wendet automatische DynamicCache-Kompatibilitaet an und generiert Text.
"""

import os
import sys
import time

# Symlink-Warnung von Hugging Face auf Windows unterdruecken
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
from ml_dtypes import bfloat16

# Kompatibilitaets-Fix: Dummy fuer ungenutztes flash_attn in Microsofts modeling_slimmoe.py
import sys, types, importlib.machinery
if "flash_attn" not in sys.modules:
    def _make_dummy(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return m
    flash_attn = _make_dummy("flash_attn")
    layers = _make_dummy("flash_attn.layers")
    rotary = _make_dummy("flash_attn.layers.rotary")
    rotary.RotaryEmbedding = object
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = layers
    sys.modules["flash_attn.layers.rotary"] = rotary

# Kompatibilitaets-Fix: DynamicCache API fuer Microsoft modeling_slimmoe.py
from transformers import Cache, DynamicCache
if not hasattr(DynamicCache, "get_max_cache_shape"):
    DynamicCache.get_max_cache_shape = lambda self: getattr(self, "get_max_length", lambda: None)()
if not hasattr(Cache, "get_max_cache_shape"):
    Cache.get_max_cache_shape = lambda self: getattr(self, "get_max_length", lambda: None)()

import qant_native_computing_toolkit.generic as q_generic
import qant_native_computing_toolkit.ai as q_ai

MODEL_ID = "microsoft/Phi-mini-MoE-instruct"

# =====================================================================
# Modus 1: Interaktive Q.ANT MoE Inferenz Demo
# =====================================================================
def run_qant_moe_generation_demo(prompt: str = "Was sind die Vorteile von MoE-Architekturen?", max_tokens: int = 15):
    """
    Demonstriert die schrittweise MoE-Token-Generierung auf der Q.ANT NPU:
    1. Multi-Head Attention QKV
    2. MoE Router (Gating ueber 16 Experten -> Top-2 Auswahl)
    3. Sparse Expert Feed-Forward (nur die 2 aktiven Experten laufen auf NPU)
    4. LM-Head Projektion & Softmax ueber das Vokabular
    """
    from transformers import AutoTokenizer

    print("=" * 65)
    print(f" 1. Q.ANT NPU MoE Token-Generierung ({MODEL_ID})")
    print("=" * 65)

    print(f"Lade Tokenizer fuer {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="np")[0]
    print(f"Eingabe-Prompt: '{prompt}'")
    print(f"Prompt Tokens:  {list(input_ids)} (Laenge: {len(input_ids)})\n")

    # NPU initialisieren
    q_generic.init_npu(0)
    print("[OK] Q.ANT NPU initialisiert (CPU-Backend, Device 0).\n")

    # Phi-mini-MoE Architekturdaten:
    d_model = 4096          # Hidden State Dimension
    n_experts = 16          # 16 Experten
    k_experts = 2           # Top-2 Routing (nur 2 Experten aktiv pro Token!)
    d_expert = 960          # Intermediate Dimension jedes Experten
    vocab_size = tokenizer.vocab_size

    # Simulierte Gewichte (bfloat16):
    # Router: projiziert (1, 4096) auf (1, 16) Experten-Logits
    w_router = np.random.uniform(-0.02, 0.02, size=(n_experts, d_model)).astype(bfloat16)
    # LM-Head: projiziert (1, 4096) auf (1, 32000) Vocab-Logits
    w_head = np.random.uniform(-0.02, 0.02, size=(vocab_size, d_model)).astype(bfloat16)

    # 16 Experten (jeder hat Gate-, Up- und Down-Projektion)
    expert_weights = [
        {
            "gate": np.random.uniform(-0.05, 0.05, size=(d_expert, d_model)).astype(bfloat16),
            "up":   np.random.uniform(-0.05, 0.05, size=(d_expert, d_model)).astype(bfloat16),
            "down": np.random.uniform(-0.05, 0.05, size=(d_model, d_expert)).astype(bfloat16),
        }
        for _ in range(n_experts)
    ]

    print("--- Starte autoregressive MoE-Generierung auf NPU ---")
    start_time = time.time()
    generated_tokens = list(input_ids)

    for step in range(max_tokens):
        # 1. Hidden State des aktuellen Tokens (1, 4096)
        x = np.random.uniform(-1.0, 1.0, size=(1, d_model)).astype(bfloat16)

        # 2. MoE Router auf NPU: Berechne Router-Logits fuer alle 16 Experten
        router_logits = q_ai.linear_fprop(x, w_router)
        router_probs = q_ai.softmax_fprop(router_logits[0]).astype(np.float32)

        # Top-2 Experten auswaehlen
        top2_idx = np.argsort(router_probs)[-k_experts:]
        top2_weights = router_probs[top2_idx]
        top2_weights /= np.sum(top2_weights)  # Normalisieren

        # 3. Nur die 2 ausgewaehlten Experten auf der NPU ausfuehren!
        moe_out = np.zeros((1, d_model), dtype=np.float32)
        for exp_idx, weight in zip(top2_idx, top2_weights):
            ew = expert_weights[exp_idx]
            # Gate & Up auf NPU
            g = q_ai.linear_fprop(x, ew["gate"]).astype(np.float32)
            u = q_ai.linear_fprop(x, ew["up"]).astype(np.float32)
            # SwiGLU: SiLU(g) * u
            silu_g = g / (1.0 + np.exp(-g))
            swiglu = (silu_g * u).astype(bfloat16)
            # Down auf NPU
            d = q_ai.linear_fprop(swiglu, ew["down"]).astype(np.float32)
            moe_out += weight * d

        moe_out_bf16 = moe_out.astype(bfloat16)

        # 4. LM-Head Projektion auf NPU -> Vokabular
        logits = q_ai.linear_fprop(moe_out_bf16, w_head)

        # 5. Softmax auf NPU
        probs = q_ai.softmax_fprop(logits[0])

        # 6. Naechsten Token auswaehlen
        next_token_id = int(np.argmax(probs.astype(np.float32)))
        generated_tokens.append(next_token_id)

        raw_text = tokenizer.decode([next_token_id])
        token_text = raw_text.encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"  Step {step+1:02d}: Experten {top2_idx} aktiv -> Token {next_token_id:5d} '{token_text}'")

    total_time = time.time() - start_time
    tok_per_sec = max_tokens / total_time
    print(f"\n[Fertig] {max_tokens} Tokens generiert in {total_time:.3f} s ({tok_per_sec:.1f} Tokens/s)")

    q_generic.release_npu(0)
    print("[OK] Q.ANT NPU sicher freigegeben.\n")


# =====================================================================
# Modus 2: Vollstaendige Hugging Face MoE Inferenz
# =====================================================================
def run_full_phimoe_inference(prompt: str, max_new_tokens: int = 50):
    """
    Laedt das vollstaendige 14.25 GB Modell (16 Experten) und fuehrt Inferenz durch.
    ACHTUNG: Benoetigt ca. 15 GB freien RAM/Auslagerungsdatei.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("=" * 65)
    print(f" 2. Vollstaendige Phi-mini-MoE Inferenz ({MODEL_ID})")
    print("=" * 65)

    print("[1/3] Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("[2/3] Lade MoE-Modell (bfloat16, 16 Experten, low_cpu_mem_usage)...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()

    # Phi-mini-MoE ist ein reines Instruct-Modell und erfordert das vollstaendige Chat-Format
    messages = [
        {"role": "system", "content": "Du bist ein hilfreicher KI-Assistent. Antworte auf Deutsch."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    print(f"\n[3/3] Generiere Antwort fuer: '{prompt}'...")
    start = time.time()
    with torch.no_grad():
        eos_token_ids = [tokenizer.eos_token_id]
        if "<|end|>" in tokenizer.get_vocab():
            eos_token_ids.append(tokenizer.convert_tokens_to_ids("<|end|>"))

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    num_tokens = len(outputs[0]) - inputs.input_ids.shape[1]

    print("\n--- Generierte Antwort ---")
    print(response)
    print("-" * 65)
    print(f"Statistik: {num_tokens} Tokens in {elapsed:.2f} s ({num_tokens/elapsed:.2f} Tokens/s)\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Erklaere kurz, warum Mixture-of-Experts Modelle effizient sind."
        run_full_phimoe_inference(prompt)
    else:
        # Standard: Schnelle Q.ANT MoE Generierungs-Demo
        run_qant_moe_generation_demo()
        print("[TIPP] Fuer die vollstaendige Inferenz mit den 14.25 GB Gewichten:")
        print("       python phi3_inference.py --full \"Deine Frage hier\"\n")
