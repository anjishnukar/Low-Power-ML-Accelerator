"""
Generate test vectors for the SystemVerilog testbench.
Writes hex files: w_int8_0.hex, a_int8_0.hex, e_int8_0_relu.hex, e_int8_0_tanh.hex
"""
import numpy as np
from golden_model import systolic_matmul, generate_random_test
import os

N = 4
VEC_DIR = os.path.join(os.path.dirname(__file__), "..", "tb", "vectors")
os.makedirs(VEC_DIR, exist_ok=True)

w, a = generate_random_test(N)

# Write weights
with open(os.path.join(VEC_DIR, "w_int8_0.hex"), "w") as f:
    for val in w.flatten():
        v = int(val) & 0xFF
        f.write(f"{v:02x}\n")

# Write activations
with open(os.path.join(VEC_DIR, "a_int8_0.hex"), "w") as f:
    for val in a.flatten():
        v = int(val) & 0xFF
        f.write(f"{v:02x}\n")

# Write expected (ReLU)
e_relu = systolic_matmul(w, a, act_mode="relu")
with open(os.path.join(VEC_DIR, "e_int8_0_relu.hex"), "w") as f:
    for val in e_relu.flatten():
        v = int(val) & 0xFFFFFFFF
        f.write(f"{v:08x}\n")

# Write expected (Tanh)
e_tanh = systolic_matmul(w, a, act_mode="tanh")
with open(os.path.join(VEC_DIR, "e_int8_0_tanh.hex"), "w") as f:
    for val in e_tanh.flatten():
        v = int(val) & 0xFFFFFFFF
        f.write(f"{v:08x}\n")

print(f"Vectors written to {VEC_DIR}/")
print(f"Weights:\n{w}")
print(f"Activations:\n{a}")
print(f"Expected (ReLU):\n{e_relu}")
print(f"Expected (Tanh):\n{e_tanh}")
