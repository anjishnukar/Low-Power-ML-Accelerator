`include "params_pkg.sv"
import params_pkg::*;

module tb_systolic_array;

  parameter int N  = ARRAY_SIZE;
  parameter int DW = DATA_WIDTH;
  parameter int AW = ACCUM_WIDTH;

  // -------------------------------------------------------------------
  // Clock & reset
  // -------------------------------------------------------------------
  logic clk, rst_n;
  initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100 MHz
  end

  // -------------------------------------------------------------------
  // DUT signals
  // -------------------------------------------------------------------
  logic            load_weights;
  logic [DW-1:0]   weight_stream [0:N-1];
  logic            weight_load_done;
  logic            act_valid;
  logic [DW-1:0]   act_stream [0:N-1];
  logic [AW-1:0]   psum_stream [0:N-1];
  logic            psum_valid;
  act_mode_t       act_mode;
  logic            zero_skip_en;

  systolic_array #(
    .N (N),
    .DW(DW),
    .AW(AW)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .load_weights   (load_weights),
    .weight_stream  (weight_stream),
    .weight_load_done(weight_load_done),
    .act_valid      (act_valid),
    .act_stream     (act_stream),
    .psum_stream    (psum_stream),
    .psum_valid     (psum_valid),
    .act_mode       (act_mode),
    .zero_skip_en   (zero_skip_en)
  );

  // -------------------------------------------------------------------
  // Test stimulus
  // -------------------------------------------------------------------
  int num_tests;
  int num_errors;
  int cycle_count;

  // Expected results from Python golden model (loaded via $readmemh)
  logic [AW-1:0] expected [0:N-1];

  // Test vectors stored as hex files
  logic [DW-1:0] weights_vec [0:N*N-1];
  logic [DW-1:0] activ_vec   [0:N*N-1];
  logic [AW-1:0] expected_vec[0:N*N-1];

  task automatic load_test_vectors(string file_w, string file_a, string file_e);
    $readmemh(file_w, weights_vec);
    $readmemh(file_a, activ_vec);
    $readmemh(file_e, expected_vec);
  endtask

  task automatic apply_weights();
    for (int c = 0; c < N; c++) begin
      weight_stream[c] = weights_vec[c];
    end
    load_weights = 1'b1;
    @(posedge clk);
    load_weights = 1'b0;
  endtask

  task automatic apply_activations(input logic [N*DW-1:0] flattened);
    for (int c = 0; c < N; c++) begin
      act_stream[c] = flattened[c*DW +: DW];
    end
    act_valid = 1'b1;
    @(posedge clk);
    act_valid = 1'b0;
  endtask

  // -------------------------------------------------------------------
  // Check routine
  // -------------------------------------------------------------------
  task automatic check_result(string label);
    logic [AW-1:0] got;
    logic [AW-1:0] exp;
    int col_err;

    col_err = 0;
    for (int c = 0; c < N; c++) begin
      got = psum_stream[c];
      exp = expected_vec[c];
      if (got !== exp) begin
        $error("[%s] Column %0d: got %0d, expected %0d", label, c, got, exp);
        col_err++;
      end
    end
    if (col_err == 0)
      $display("[%s] PASS", label);
    else
      num_errors++;
  endtask

  // -------------------------------------------------------------------
  // Main test sequence
  // -------------------------------------------------------------------
  initial begin
    num_tests   = 0;
    num_errors  = 0;
    cycle_count = 0;
    act_valid   = 1'b0;
    load_weights = 1'b0;
    zero_skip_en = 1'b1;
    act_mode    = ACT_RELU;

    // Reset
    rst_n = 0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ---------------------------------------------------------------
    // Test 1: INT8 random (ReLU)
    // ---------------------------------------------------------------
    $display("\n=== Test 1: INT8 ReLU ===");
    load_test_vectors("tb/vectors/w_int8_0.hex",
                      "tb/vectors/a_int8_0.hex",
                      "tb/vectors/e_int8_0_relu.hex");
    apply_weights();
    for (int t = 0; t < N; t++) begin
      apply_activations({>>{activ_vec[t*N +: N]}});
      @(posedge clk);
    end
    repeat (N+2) @(posedge clk);  // pipeline delay
    check_result("INT8_RELU");
    num_tests++;

    // ---------------------------------------------------------------
    // Test 2: INT8 random (Tanh)
    // ---------------------------------------------------------------
    $display("\n=== Test 2: INT8 Tanh ===");
    act_mode = ACT_TANH;
    load_test_vectors("tb/vectors/w_int8_0.hex",
                      "tb/vectors/a_int8_0.hex",
                      "tb/vectors/e_int8_0_tanh.hex");
    apply_weights();
    for (int t = 0; t < N; t++) begin
      apply_activations({>>{activ_vec[t*N +: N]}});
      @(posedge clk);
    end
    repeat (N+2) @(posedge clk);
    check_result("INT8_TANH");
    num_tests++;

    // ---------------------------------------------------------------
    // Test 3: Zero-skipping (all-zero input)
    // ---------------------------------------------------------------
    $display("\n=== Test 3: Zero-skip (all zero input) ===");
    act_mode = ACT_LINEAR;
    for (int i = 0; i < N*N; i++) begin
      weights_vec[i] = '0;
      activ_vec[i]   = '0;
      expected_vec[i] = '0;
    end
    apply_weights();
    for (int t = 0; t < N; t++) begin
      apply_activations('0);
      @(posedge clk);
    end
    repeat (N+2) @(posedge clk);
    check_result("ZERO_SKIP");
    num_tests++;

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    $display("\n================================");
    $display("Tests completed: %0d", num_tests);
    if (num_errors == 0)
      $display("ALL TESTS PASSED");
    else
      $display("%0d / %0d TESTS FAILED", num_errors, num_tests);
    $display("================================");
    $finish;
  end

  // Cycle counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cycle_count <= 0;
    else       cycle_count <= cycle_count + 1;
  end

endmodule
