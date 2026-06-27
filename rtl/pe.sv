`include "params_pkg.sv"
import params_pkg::*;

module pe #(
  parameter int DW = DATA_WIDTH,
  parameter int AW = ACCUM_WIDTH
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        clk_en,        // clock enable (zero-skip)
  // Weight load interface
  input  logic        weight_valid,
  input  logic [DW-1:0] weight_in,
  output logic        weight_ready,
  // Activation input (from left neighbour)
  input  logic        act_valid,
  input  logic [DW-1:0] act_in,
  output logic        act_ready,
  // Partial sum input (from top neighbour)
  input  logic [AW-1:0] psum_in,
  // Outputs to right & bottom neighbours
  output logic [DW-1:0] act_out,
  output logic [AW-1:0] psum_out,
  output logic        psum_valid,
  // Mode
  input  act_mode_t   act_mode
);

  // -------------------------------------------------------------------
  // Internal registers
  // -------------------------------------------------------------------
  logic [DW-1:0] weight_reg;
  logic [DW-1:0] act_reg;
  logic [AW-1:0] accum_reg;
  logic [AW-1:0] mult_result;
  logic [AW-1:0] add_result;
  logic          weight_loaded;

  // -------------------------------------------------------------------
  // Clock gating – local instance
  // -------------------------------------------------------------------
  logic gclk;
  clock_gate cg_i (
    .clk (clk),
    .rst_n(rst_n),
    .en  (clk_en),
    .gclk(gclk)
  );

  // -------------------------------------------------------------------
  // Weight loading (pre-load phase)
  // -------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_reg    <= '0;
      weight_loaded <= 1'b0;
    end else if (weight_valid && weight_ready) begin
      weight_reg    <= weight_in;
      weight_loaded <= 1'b1;
    end
  end

  assign weight_ready = !weight_loaded;

  // -------------------------------------------------------------------
  // Activation passthrough (systolic direction: left -> right)
  // -------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_reg <= '0;
    end else if (act_valid) begin
      act_reg <= act_in;
    end
  end

  assign act_out  = act_reg;
  assign act_ready = 1'b1;

  // -------------------------------------------------------------------
  // Pipelined MAC with zero-skipping
  // Stage 1: multiply
  // Stage 2: add to partial sum
  // Stage 3: accumulate + activation
  // -------------------------------------------------------------------

  // Detect zero operands for clock gating
  logic zero_skip;
  assign zero_skip = (weight_reg == '0) || (act_reg == '0);

  // Stage 1: Multiplier
  always_ff @(posedge gclk or negedge rst_n) begin
    if (!rst_n) begin
      mult_result <= '0;
    end else if (!zero_skip) begin
      mult_result <= $signed(weight_reg) * $signed(act_reg);
    end else begin
      mult_result <= '0;
    end
  end

  // Stage 2: Adder
  always_ff @(posedge gclk or negedge rst_n) begin
    if (!rst_n) begin
      add_result <= '0;
    end else begin
      add_result <= mult_result + psum_in;
    end
  end

  // Stage 3: Accumulate & activation
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      accum_reg  <= '0;
      psum_valid <= 1'b0;
    end else begin
      accum_reg  <= add_result;
      psum_valid <= act_valid;  // valid one cycle after input
    end
  end

  // -------------------------------------------------------------------
  // Activation function block
  // -------------------------------------------------------------------
  logic [AW-1:0] activated;
  activation #(
    .DW(AW)
  ) act_i (
    .data_in (accum_reg),
    .mode    (act_mode),
    .data_out(activated)
  );

  assign psum_out = activated;

endmodule
