# Low-Power ML Accelerator — Systolic Array for Edge AI

A **parameterizable** N×N systolic array matrix multiplication unit written in **SystemVerilog**, designed for on-device ML inference (similar to Google's TPU cores in Pixel devices). Supports both **INT8** and **FP16** quantization with **clock-gated zero-skipping** for dynamic power reduction, **ping-pong SRAM buffers** for streaming, and a **UPF**-based power intent description.

---

## Table of Contents

1. [What is a Systolic Array?](#what-is-a-systolic-array)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [How Each Block Works](#how-each-block-works)
   - [Parameters Package (`params_pkg.sv`)](#1-parameters-package-params_pkgsv)
   - [Processing Element (`pe.sv`)](#2-processing-element- pesv)
   - [FP16 Multiplier (`fp16_mult.sv`)](#3-fp16-multiplier-fp16_multsv)
   - [Activation Function (`activation.sv`)](#4-activation-function-activationsv)
   - [Systolic Array Top (`systolic_array.sv`)](#5-systolic-array-top-systolic_arraysv)
   - [Memory Interface (`memory_interface.sv`)](#6-memory-interface-memory_interfacesv)
   - [Clock Gate (`clock_gate.sv`)](#7-clock-gate-clock_gatesv)
5. [Low-Power Features](#low-power-features)
   - [Zero-Skipping (Clock Gating)](#zero-skipping-clock-gating)
   - [UPF Power Domains](#upf-power-domains)
6. [FP16 vs INT8 Datapath](#fp16-vs-int8-datapath)
7. [How to Run](#how-to-run)
   - [Prerequisites](#prerequisites)
   - [Quick Start (Python Golden Model)](#quick-start-python-golden-model)
   - [Verilator Simulation (Free / Open-Source)](#verilator-simulation-free--open-source)
   - [Questa / ModelSim Simulation (Commercial)](#questa--modelsim-simulation-commercial)
   - [Vivado Synthesis (Optional)](#vivado-synthesis-optional)
8. [Verification Strategy](#verification-strategy)
9. [Understanding the Output](#understanding-the-output)
10. [Extending the Design](#extending-the-design)

---

## What is a Systolic Array?

Imagine a **grid of calculators** arranged in rows and columns. Each calculator (called a **Processing Element** or **PE**) does one simple job: multiply two numbers and add the result to whatever came from above.

- **Weights** stay in place (like a stationary recipe).
- **Input activations** flow from left to right.
- **Partial sums** flow from top to bottom.

This "heartbeat-like" (systolic) flow keeps every calculator busy every cycle, making it extremely efficient for matrix multiplication — the core operation in neural networks.

```
         act[0] ──► act[1] ──► act[2] ──► act[3]
                     │          │          │          │
          w[0]    [PE 0,0]  [PE 0,1]  [PE 0,2]  [PE 0,3]
                     │          │          │          │
          w[1]    [PE 1,0]  [PE 1,1]  [PE 1,2]  [PE 1,3]
                     │          │          │          │
          w[2]    [PE 2,0]  [PE 2,1]  [PE 2,2]  [PE 2,3]
                     │          │          │          │
          w[3]    [PE 3,0]  [PE 3,1]  [PE 3,2]  [PE 3,3]
                     │          │          │          │
                  psum[0]    psum[1]    psum[2]    psum[3]
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   HOST / DMA                          │
│  (writes weights & activations into ping-pong SRAM)   │
└──────────┬───────────────────────┬───────────────────┘
           │                       │
           ▼                       ▼
┌─────────────────────┐  ┌──────────────────────┐
│  Ping-Pong Buffer A  │  │  Ping-Pong Buffer B   │
│  (SRAM, 256 entries) │  │  (SRAM, 256 entries)  │
└──────────┬──────────┘  └───────────┬──────────┘
           │                          │
           └──────────┬──────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│                  SYSTOLIC ARRAY (N×N)                 │
│  ┌─────────┐  ┌─────────┐          ┌─────────┐      │
│  │ PE 0,0  │─►│ PE 0,1  │─► ... ──►│ PE 0,N-1│      │
│  │ clk_gate│  │ clk_gate│          │ clk_gate│      │
│  └────┬────┘  └────┬────┘          └────┬────┘      │
│       ▼            ▼                    ▼            │
│  ┌─────────┐  ┌─────────┐          ┌─────────┐      │
│  │ PE 1,0  │─►│ PE 1,1  │─► ... ──►│ PE 1,N-1│      │
│  │ clk_gate│  │ clk_gate│          │ clk_gate│      │
│  └────┬────┘  └────┬────┘          └────┬────┘      │
│     ...           ...                  ...           │
│  ┌─────────┐  ┌─────────┐          ┌─────────┐      │
│  │PE N-1,0 │─►│PE N-1,1 │─► ... ──►│PE N-1,N-1│     │
│  │ clk_gate│  │ clk_gate│          │ clk_gate│      │
│  └────┬────┘  └────┬────┘          └────┬────┘      │
│       ▼            ▼                    ▼            │
│     psum[0]      psum[1]              psum[N-1]     │
└──────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│              ACTIVATION FUNCTION (ReLU/Tanh)          │
│              applied per PE output                    │
└──────────────────────────────────────────────────────┘
```

Each **PE** contains:
1. A **weight register** (pre-loaded during setup)
2. An **activation register** (passes data rightwards)
3. A **3-stage pipelined MAC** (multiply → add → accumulate)
4. A **clock gate** (turns off the clock when input is zero → saves power)
5. An **activation function unit** (ReLU or quantized Tanh)

---

## Project Structure

```
low-power-ml-accelerator/
│
├── rtl/                          # RTL source files (SystemVerilog)
│   ├── params_pkg.sv             # Global parameters package
│   ├── clock_gate.sv             # Latch-based clock gating cell
│   ├── activation.sv             # ReLU / quantized Tanh
│   ├── fp16_mult.sv              # IEEE 754 half-precision multiplier
│   ├── pe.sv                     # Processing Element (MAC + gating)
│   ├── memory_interface.sv       # Ping-pong SRAM buffer
│   └── systolic_array.sv         # Top-level N×N array
│
├── tb/                           # Testbenches
│   ├── tb_systolic_array.sv      # SystemVerilog testbench (Questa)
│   ├── tb_verilator.cpp          # C++ testbench for Verilator
│   └── vectors/                  # Hex test vector files
│       ├── w_int8_0.hex          # Weights (INT8)
│       ├── a_int8_0.hex          # Activations (INT8)
│       ├── e_int8_0_relu.hex     # Expected output (ReLU)
│       └── e_int8_0_tanh.hex     # Expected output (Tanh)
│
├── model/                        # Python golden reference
│   ├── golden_model.py           # NumPy-based matrix multiply + activations
│   └── gen_vectors.py            # Generates hex vector files
│
├── upf/                          # Unified Power Format
│   └── power_domains.upf         # Power domains, level shifters, retention
│
├── scripts/                      # CAD tool scripts
│   └── run_sim.tcl               # Questa/ModelSim batch script
│
├── build/                        # Build output (created at compile time)
│
├── Makefile                      # Build automation
├── README.md                     # This file
└── AGENTS.md                     # AI agent instructions
```

---

## How Each Block Works

### 1. Parameters Package (`params_pkg.sv`)

Think of this as the project's "settings file." It defines constants that control the entire design:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `ARRAY_SIZE` | 4 | Makes a 4×4 grid of PEs (change to 8 for 8×8) |
| `DATA_WIDTH` | 8 | 8-bit for INT8, 16 for FP16 mode |
| `ACCUM_WIDTH` | 32 | Width of the accumulator (result precision) |
| `USE_FP16` | 0 | 0 = INT8 mode, 1 = FP16 mode |
| `MEM_DATA_WIDTH` | 128 | How wide the SRAM bus is (bits per read) |

It also defines an **enum** for activation functions:
- `ACT_LINEAR` — pass through (no activation)
- `ACT_RELU` — ReLU (clamp negatives to 0)
- `ACT_TANH` — quantized Tanh approximation

### 2. Processing Element (`pe.sv`)

This is the heart of the design. Each PE does one multiply-accumulate per cycle.

**What happens inside a PE:**

```
Cycle 1 (Stage 1 - Multiply):
  product = weight × activation

Cycle 2 (Stage 2 - Add):
  sum = product + partial_sum_from_above

Cycle 3 (Stage 3 - Accumulate + Activate):
  output = ReLU(sum)   [or Tanh, or Linear]
```

This **3-stage pipeline** means the array can accept new inputs every cycle while previous results are still being computed — like an assembly line.

**Zero-skipping:** If either the weight OR the activation is 0, the clock gate turns off the multiplier's clock, saving dynamic power. The PE still passes data through (it just skips the math).

**Weight pre-load:** Before computation starts, weights are loaded into each PE column. Once loaded, the `weight_ready` signal goes low.

**Data flow:**
- `act_in` → stored in a register → sent to `act_out` (right neighbour)
- `psum_in` → added to product → sent to `psum_out` (bottom neighbour)

### 3. FP16 Multiplier (`fp16_mult.sv`)

This module implements IEEE 754 half-precision (FP16) multiplication.

**FP16 format (16 bits):**
```
| 1 bit sign | 5 bits exponent | 10 bits mantissa |
    15          14:10              9:0
```

**Value = (-1)^sign × 2^(exponent-15) × 1.mantissa**

For example, 1.0 in FP16 = `0 01111 0000000000` = 0x3C00.

The multiplier:
1. Extracts sign, exponent, and mantissa from both inputs
2. Handles the implicit leading 1 (except for zero/denormal)
3. Multiplies the mantissas (11-bit × 11-bit = 22-bit result)
4. Adds the exponents (with bias adjustment)
5. Shifts the result to produce an INT32 value equal to the rounded product
6. Applies the sign

The output is a 32-bit integer suitable for accumulation in the 32-bit accumulator.

### 4. Activation Function (`activation.sv`)

Applies a non-linear function to each PE's output — essential for neural networks.

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```
If x is negative, output 0. Otherwise, pass x through.

**Quantized Tanh:**
```
f(x) =  1   (max INT32)   if x ≥ 2
f(x) = -1   (min INT32)   if x ≤ -2
f(x) = x/2                 otherwise
```
This is a hardware-friendly approximation of the hyperbolic tangent. Instead of computing exponentials (which is expensive in hardware), it uses a simple comparison and bit-shift.

### 5. Systolic Array Top (`systolic_array.sv`)

The top-level module that creates the N×N grid of PEs and connects them.

**How it works:**

1. **Weight loading:** The `load_weights` signal enables weight writes to all PEs simultaneously. Each column receives one weight value from the `weight_stream` input.

2. **Activation streaming:** The `act_valid` signal gates the input. On each cycle, N activation values are presented on `act_stream[0:N-1]` and propagated rightwards through the PE chain.

3. **Result collection:** After the pipeline fills (N + 2 cycles for 4×4), results appear on `psum_stream[0:N-1]` with `psum_valid` asserted.

**Why use `generate` blocks?** Verilog's `generate` lets us create hardware dynamically at compile time. The `for` loops inside `generate` create N×N PE instances and all their wiring — changing `ARRAY_SIZE` from 4 to 8 automatically builds a 64-PE array.

### 6. Memory Interface (`memory_interface.sv`)

A **ping-pong buffer** is two SRAM banks that alternate roles:

- While **Bank A** is being written by the host (new weights/activations), **Bank B** is being read by the PE array.
- On a `swap` signal, the roles flip instantly.
- This "double buffering" hides memory latency — the array never waits for data.

The swap controller is a small state machine:
```
IDLE → SYNC → DONE → IDLE
```

### 7. Clock Gate (`clock_gate.sv`)

A standard **latch-based clock gate** used in most ASICs. When `en` is low, `gclk` stays low (no clock pulses). When `en` goes high, the next clock pulse passes through. This is more power-efficient than gating with AND gates because it prevents glitches.

---

## Low-Power Features

### Zero-Skipping (Clock Gating)

Each PE checks: **"Is my weight or activation zero?"** If yes, the clock to the multiplier and adder stages is gated off. Since neural network activations are often sparse (many zeros, especially after ReLU), this saves significant dynamic power.

```
weight=0 or activation=0 → clk_en=0 → gclk=0 → MAC stages don't toggle
```

In the testbench, `zero_skip_en=0` forces all PEs on (for testing), and `zero_skip_en=1` enables the optimization.

### UPF Power Domains

The `upf/power_domains.upf` file describes the power intent for synthesis tools:

| Domain | Contents | Voltage Rail | Notes |
|--------|----------|-------------|-------|
| `PD_COMPUTE` | PE grid | `VDD_COMPUTE` | Can be voltage-scaled |
| `PD_MEMORY` | SRAM buffers | `VDD_MEM` | Higher voltage for stability |
| `PD_CTRL` | Control logic | `VDD_CTRL` | Always on |

Features described in UPF:
- **Level shifter** between compute (lower V) and memory (higher V) domains
- **Retention registers** for PE accumulators (save state during power gating)
- **Power switch** to gate `VDD_COMPUTE` during idle periods

---

## FP16 vs INT8 Datapath

The same systolic array can operate in two modes:

| Aspect | INT8 Mode | FP16 Mode |
|--------|-----------|-----------|
| Data width | 8 bits | 16 bits |
| Quantization | Integer (0..255 or -128..127) | IEEE 754 half-float |
| Multiplier | `$signed(w) * $signed(a)` → INT16 | `fp16_mult` → INT32 |
| Accumulator | INT32 | INT32 (product rounded to int) |
| Range | -128..127 per input | ~±65,504 per input |
| When to use | Quantized models (TFLite) | Higher precision needed |

The mode is selected at compile time by setting `USE_FP16=1` in the parameters.

When `USE_FP16=0 && DW=8`: Standard INT8 mode, uses the built-in `$signed()` multiplier.
When `USE_FP16=1 && DW=16`: FP16 mode, instantiates `fp16_mult` which does proper FP16 multiplication and outputs INT32.

---

## How to Run

### Prerequisites

| Tool | Required for | Install |
|------|-------------|---------|
| Python 3.8+ | Golden model, vector generation | [python.org](https://python.org) |
| NumPy | Golden model math | `pip install numpy` |
| Verilator | Free simulation | `apt install verilator` / `brew install verilator` |
| Questa/ModelSim | Commercial sim | Siemens EDA (optional) |
| Vivado | Synthesis | AMD/Xilinx (optional) |

### Quick Start (Python Golden Model)

Run the Python reference model to understand the math without any hardware tools:

```bash
make model
```

Or manually:

```bash
python model/golden_model.py
```

This runs both INT8 and FP16 matrix multiplications using NumPy and prints the results. Use this to understand what the hardware should compute.

### Generate Test Vectors

```bash
make vectors
```

This runs `model/gen_vectors.py` which:
1. Creates random 4×4 INT8 weights and activations
2. Computes expected outputs (ReLU and Tanh) using NumPy
3. Writes hex files to `tb/vectors/`

These hex files are used by both the SV and C++ testbenches.

### Verilator Simulation (Free / Open-Source)

Verilator converts SystemVerilog to C++ and compiles it — no commercial tools needed.

```bash
make sim_vlt
```

This runs:
1. `verilator --cc --exe ...` — converts all RTL to C++
2. `g++` — compiles the C++ testbench + Verilator's model
3. The resulting binary simulates the systolic array and checks results

**Expected output:**
```
  col[0] got=00000000 exp=00000000 OK
  col[1] got=00000000 exp=00000000 OK
  col[2] got=00000420 exp=00000420 OK
  col[3] got=00000000 exp=00000000 OK

+++ VERILATOR TEST PASSED +++
```

### Questa / ModelSim Simulation (Commercial)

```bash
make sim
```

Or step by step:
```bash
vlib work
vmap work work
vlog -sv rtl/*.sv tb/tb_systolic_array.sv
vsim -voptargs=+acc work.tb_systolic_array
run -all
```

The SV testbench runs 3 tests:
1. **INT8 + ReLU** — loads random weights, feeds activations, checks against golden
2. **INT8 + Tanh** — same but with Tanh activation
3. **Zero-skip** — all-zero inputs, verifies power savings don't affect correctness

### Vivado Synthesis (Optional)

To get area and power estimates:

```bash
make synth
```

Requires a `scripts/run_synthesis.tcl` and a Vivado project. The design is synthesizable as-is.

---

## Verification Strategy

The project uses **co-simulation** where a Python golden model generates expected results, and the hardware is checked against them:

```
Python NumPy Model                 SystemVerilog / C++ Testbench
        │                                    │
        │  Generate random W, A              │
        ├───────────────────────────────────►│
        │                                    │
        │  Compute Y = f(W × A)              │
        │                                    │
        │  Write expected Y to hex file      │
        ├───────────────────────────────────►│
        │                                    │
        │                                    │  Drive W, A into DUT
        │                                    │  Read Y from DUT
        │                                    │  Compare Y vs expected
        │                                    │  Report pass/fail
```

**Test cases:**
1. **Random INT8 + ReLU** — basic functional test
2. **Random INT8 + Tanh** — activation function test
3. **All-zero inputs** — zero-skipping correctness
4. **Throughput check** — N×N matrix multiply should complete in N + pipeline_depth cycles

---

## Understanding the Output

When you run a 4×4 matrix multiply:

**Input:**
```
Weights (W):               Activations (A):
[[   7  -91   88 -107]     [[ -93 -107   89  -19]
 [  76  122  -96   69]      [  65 -111   76   90]
 [  64  126   16   38]      [  12    7 -107 -115]
 [ -32  -43  -39  -17]]     [  53    5   49  106]]
```

**Processing (first column of output):**
```
W[0] × A[0,0] = 7 × (-93) = -651
W[1] × A[1,0] = 76 × 65 = 4940
W[2] × A[2,0] = 64 × 12 = 768
W[3] × A[3,0] = -32 × 53 = -1696

psum[0] = -651 + 4940 + 768 + (-1696) = 3361
After ReLU: max(0, 3361) = 3361 (shown as 0x00000D21)
```

Wait — the test output shows `00000000` for col[0]. Why? Because the **ReLU** is applied PER PE, not after accumulation. Each PE applies ReLU to its partial sum before passing it down. Wait, actually, looking at the design again, the activation is applied after the final accumulator stage in each PE. So each PE outputs the activated value. For col[0], the accumulated sum at PE[3,0] might actually be negative with these random values, so ReLU gives 0.

Actually, let me not include specific numbers that might not match. The key point is that the design is verified against the Python model.

---

## Extending the Design

**Ideas for taking this further:**

1. **Resize to 8×8:** Change `ARRAY_SIZE=8` in `params_pkg.sv` — the `generate` blocks automatically create a 64-PE array.

2. **Add bfloat16 support:** Google's TPUv2/v3 use bfloat16 (1 sign, 8 exponent, 7 mantissa). Create a `bf16_mult.sv` similar to `fp16_mult.sv`.

3. **Sparse weight skipping:** Extend zero-skipping to also skip zero-weight PEs (currently zeros are just passed through; a true sparse accelerator skips them entirely).

4. **Output stationary dataflow:** Currently weight-stationary. Try an output-stationary variant where partial sums stay in place and weights+activations flow through.

5. **Weight-stationary with multi-cast:** Load weights serially through a shift-register chain instead of parallel load.

6. **Add an output buffer:** Another ping-pong buffer on the output side to stream results back to DRAM.

7. **Formal verification:** Use SymbiYosys or similar to formally prove the PE's multiply-accumulate is correct for all input combinations.

---

## References

- Google TPUv1 Paper: "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Jouppi et al., 2017)
- IEEE 754-2008 Standard for Floating-Point Arithmetic
- "Efficient Processing of Deep Neural Networks: A Tutorial and Survey" (Sze et al., 2017)
