import numpy as np
from ml_dtypes import bfloat16
import qant_native_computing_toolkit.generic as q_generic
import qant_native_computing_toolkit.native as q_native

# NPU initialisieren (im CPU-Backend simuliert)
q_generic.init_npu(0)

# Daten im bfloat16-Format erstellen
a = np.array([1.0, 2.0, 3.0], dtype=bfloat16)
b = np.array([4.0, 5.0, 6.0], dtype=bfloat16)

# Elementweise Multiplikation durchführen
result = q_native.mul_elementwise(a, b)
print("Ergebnis:", result)

# NPU freigeben
q_generic.release_npu(0)