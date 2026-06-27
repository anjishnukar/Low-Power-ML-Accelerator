`include "params_pkg.sv"
import params_pkg::*;

module systolic_array #(
  parameter int N   = ARRAY_SIZE,
  parameter int DW  = DATA_WIDTH,
  parameter int AW  = ACCUM_WIDTH
) (
  input  logic            clk,
  input  logic            rst_n,
  // Weight loading interface
  input  logic            load_weights,
  input  logic [DW-1:0]   weight_stream [0:N-1],
  output logic            weight_load_done,
  // Activation input
  input  logic            act_valid,
  input  logic [DW-1:0]   act_stream [0:N-1],
  // Partial sum output
  output logic [AW-1:0]   psum_stream [0:N-1],
  output logic            psum_valid,
  // Mode
  input  act_mode_t       act_mode,
  // Clock gating enable (0 = force all PEs on)
  input  logic            zero_skip_en
);

  // -------------------------------------------------------------------
  // Internal wiring
  // -------------------------------------------------------------------
  logic [DW-1:0] act_h [0:N][0:N-1];   // horizontal act propagation
  logic [AW-1:0] psum_v [0:N-1][0:N];  // vertical psum propagation
  logic          clk_en_mat [0:N-1][0:N-1];
  logic [DW-1:0] w_stream [0:N-1];
  logic          w_valid_mat [0:N-1][0:N-1];
  logic          w_ready_mat [0:N-1][0:N-1];
  logic          act_valid_mat [0:N-1][0:N-1];
  logic          act_ready_mat [0:N-1][0:N-1];
  logic          psum_valid_mat [0:N-1][0:N-1];

  // -------------------------------------------------------------------
  // Weight fan-out: load each column
  // -------------------------------------------------------------------
  genvar r, c;
  generate
    for (c = 0; c < N; c++) begin : col_wt
      assign w_stream[c] = weight_stream[c];
    end
  endgenerate

  // -------------------------------------------------------------------
  // Generate the PE grid
  // -------------------------------------------------------------------
  generate
    for (r = 0; r < N; r++) begin : row
      for (c = 0; c < N; c++) begin : col

        // Clock gating: each PE checks its own neighbours' input
        assign clk_en_mat[r][c] = !zero_skip_en ||
          !((weight_stream[c] == '0) && (act_stream[c] == '0));

        // Weight valid: only valid during load phase
        assign w_valid_mat[r][c] = load_weights;

        // Act valid: top row gets external, others from above
        if (r == 0) begin : top_row
          assign act_valid_mat[r][c] = act_valid;
        end else begin : other_rows
          assign act_valid_mat[r][c] = act_valid_mat[r-1][c];
        end

        pe #(
          .DW(DW),
          .AW(AW)
        ) pe_inst (
          .clk          (clk),
          .rst_n        (rst_n),
          .clk_en       (clk_en_mat[r][c]),
          .weight_valid (w_valid_mat[r][c]),
          .weight_in    (w_stream[c]),
          .weight_ready (w_ready_mat[r][c]),
          .act_valid    (act_valid_mat[r][c]),
          .act_in       (act_h[r][c]),
          .act_ready    (act_ready_mat[r][c]),
          .psum_in      (psum_v[r][c]),
          .act_out      (act_h[r+1][c]),
          .psum_out     (psum_v[r][c+1]),
          .psum_valid   (psum_valid_mat[r][c]),
          .act_mode     (act_mode)
        );

      end
    end
  endgenerate

  // -------------------------------------------------------------------
  // Connect boundaries
  // -------------------------------------------------------------------
  generate
    for (c = 0; c < N; c++) begin : in_act
      assign act_h[0][c] = act_stream[c];
    end
    for (r = 0; r < N; r++) begin : init_psum
      assign psum_v[r][0] = '0;
    end
    for (c = 0; c < N; c++) begin : out_psum
      assign psum_stream[c] = psum_v[N-1][c+1];
    end
    for (c = 0; c < N; c++) begin : out_valid
      assign psum_valid_mat[0][c] = act_valid;
    end
  endgenerate

  // Global valid signal
  assign psum_valid = &psum_valid_mat[N-1];

  // Weight loading done
  assign weight_load_done = &{w_ready_mat[0][0]};

endmodule
