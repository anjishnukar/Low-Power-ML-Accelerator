#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <verilated.h>
#include "Vsystolic_array.h"

// =============================================================
// Configuration (matches params_pkg defaults)
// =============================================================
constexpr int N  = 4;
constexpr int DW = 8;
constexpr int AW = 32;

// =============================================================
// Helper: read hex file into array
// =============================================================
template <typename T>
int read_hex(const std::string &path, T *buf, int max_cnt) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: cannot open " << path << std::endl;
    exit(1);
  }
  std::string line;
  int cnt = 0;
  while (std::getline(f, line) && cnt < max_cnt) {
    if (line.empty()) continue;
    buf[cnt++] = static_cast<T>(std::stoul(line, nullptr, 16));
  }
  return cnt;
}

// =============================================================
// Helper: format output in hex
// =============================================================
std::string hex32(uint32_t v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%08x", v);
  return std::string(buf);
}

// =============================================================
// Main
// =============================================================
int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  // -----------------------------------------------------------------
  // Instantiate DUT
  // -----------------------------------------------------------------
  Vsystolic_array *dut = new Vsystolic_array;

  // -----------------------------------------------------------------
  // Load test vectors
  // -----------------------------------------------------------------
  int8_t  weights[N * N];
  int8_t  acts[N * N];
  int32_t expected[N * N];

  int nw = read_hex("tb/vectors/w_int8_0.hex", weights, N * N);
  int na = read_hex("tb/vectors/a_int8_0.hex", acts, N * N);
  int ne = read_hex("tb/vectors/e_int8_0_relu.hex", expected, N * N);

  if (nw != N * N || na != N * N || ne != N * N) {
    std::cerr << "ERROR: vector files incomplete" << std::endl;
    return 1;
  }

  // -----------------------------------------------------------------
  // Reset
  // -----------------------------------------------------------------
  dut->clk = 0;
  dut->rst_n = 0;
  dut->load_weights = 0;
  dut->act_valid = 0;
  dut->zero_skip_en = 1;
  dut->act_mode = 1;  // ACT_RELU

  for (int i = 0; i < 4; i++) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
  }
  dut->rst_n = 1;

  // -----------------------------------------------------------------
  // Load weights (cycle 0)
  // -----------------------------------------------------------------
  dut->load_weights = 1;
  for (int c = 0; c < N; c++) {
    dut->weight_stream[c] = weights[c];
  }
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
  dut->load_weights = 0;

  // -----------------------------------------------------------------
  // Feed activations (N cycles)
  // -----------------------------------------------------------------
  for (int t = 0; t < N; t++) {
    dut->act_valid = 1;
    for (int c = 0; c < N; c++) {
      dut->act_stream[c] = acts[t * N + c];
    }
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
  }
  dut->act_valid = 0;

  // -----------------------------------------------------------------
  // Drain pipeline (N + 2 cycles)
  // -----------------------------------------------------------------
  for (int t = 0; t < N + 2; t++) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
  }

  // -----------------------------------------------------------------
  // Check results
  // -----------------------------------------------------------------
  int errors = 0;
  for (int c = 0; c < N; c++) {
    uint32_t got = dut->psum_stream[c];
    uint32_t exp = expected[c] & 0xFFFFFFFF;
    std::cout << "  col[" << c << "] got=" << hex32(got)
              << " exp=" << hex32(exp);
    if (got == exp) {
      std::cout << " OK" << std::endl;
    } else {
      std::cout << " FAIL" << std::endl;
      errors++;
    }
  }

  // -----------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------
  if (errors == 0) {
    std::cout << "\n+++ VERILATOR TEST PASSED +++" << std::endl;
  } else {
    std::cout << "\n--- VERILATOR TEST FAILED (" << errors << " errors) ---" << std::endl;
  }

  delete dut;
  return errors;
}
