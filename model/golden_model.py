"""
Golden reference model for systolic array matrix multiplication.
Supports INT8 and FP16 (simulated via NumPy float16).
"""
import numpy as np


def quantized_tanh(x, bits=32):
    """Quantized Tanh approximation matching the RTL."""
    max_val = 2 ** (bits - 1) - 1
    min_val = -(2 ** (bits - 1))
    result = np.where(x >= 2, max_val,
                      np.where(x <= -2, min_val,
                               (x >> 1).astype(np.int32)))
    return np.clip(result, min_val, max_val).astype(np.int32)


def relu(x):
    """ReLU: max(0, x)."""
    return np.maximum(0, x).astype(np.int32)


def systolic_matmul(weights, activations, act_mode="linear", use_fp16=False):
    """
    Perform matrix multiplication using a systolic dataflow.

    weights      : (N, N) matrix  — stationary weights
    activations  : (N, N) matrix  — input activations (one column per cycle)
    act_mode     : "linear" | "relu" | "tanh"
    use_fp16     : If True, cast to float16 for computation

    Returns:
        output : (N, N) matrix of partial sums
    """
    N = weights.shape[0]

    if use_fp16:
        weights = weights.astype(np.float16)
        activations = activations.astype(np.float16)

    # Systolic execution: weights stay, activations flow right,
    # partial sums accumulate downward.
    psum = np.zeros((N, N), dtype=np.int32 if not use_fp16 else np.float16)

    for i in range(N):
        for j in range(N):
            mac_val = weights[j, i] * activations[i, j]
            # Accumulate from top neighbour
            if j > 0:
                mac_val += psum[j - 1, i]
            psum[j, i] = mac_val

    # Apply activation
    if act_mode == "relu":
        psum = relu(psum)
    elif act_mode == "tanh":
        psum = quantized_tanh(psum)
    # linear: no-op

    return psum


def generate_random_test(N=4, use_fp16=False):
    """Generate random weights and activations for verification."""
    rng = np.random.default_rng(42)
    if use_fp16:
        w = rng.uniform(-2.0, 2.0, (N, N)).astype(np.float16)
        a = rng.uniform(-2.0, 2.0, (N, N)).astype(np.float16)
    else:
        w = rng.integers(-128, 127, (N, N), dtype=np.int8).astype(np.int32)
        a = rng.integers(-128, 127, (N, N), dtype=np.int8).astype(np.int32)
    return w, a


def compute_throughput(elapsed_us, ops):
    """Compute throughput in TOPS."""
    return ops / (elapsed_us * 1e-6) / 1e12


if __name__ == "__main__":
    N = 4
    w, a = generate_random_test(N)

    print("=== INT8 Test ===")
    result = systolic_matmul(w, a, act_mode="relu")
    print(result)

    print("\n=== FP16 Test ===")
    w16, a16 = generate_random_test(N, use_fp16=True)
    result16 = systolic_matmul(w16, a16, use_fp16=True)
    print(result16)
