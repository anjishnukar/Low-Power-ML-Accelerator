`include "params_pkg.sv"
import params_pkg::*;

module activation #(
  parameter int DW = ACCUM_WIDTH
) (
  input  logic [DW-1:0] data_in,
  input  act_mode_t     mode,
  output logic [DW-1:0] data_out
);

  always_comb begin
    case (mode)
      ACT_LINEAR: data_out = data_in;

      ACT_RELU: begin
        if ($signed(data_in) < 0)
          data_out = '0;
        else
          data_out = data_in;
      end

      ACT_TANH: begin
        // Quantized Tanh approximation:
        //   x >=  2 ->  1
        //   x <= -2 -> -1
        //   otherwise -> x/2  (linear approximation around origin)
        if ($signed(data_in) >= $signed(32'd2))
          data_out = {1'b0, {DW-1{1'b1}}};  // max positive
        else if ($signed(data_in) <= $signed(-32'd2))
          data_out = {1'b1, {DW-1{1'b0}}};  // max negative
        else
          data_out = $signed(data_in) >>> 1; // x/2
      end

      default: data_out = data_in;
    endcase
  end

endmodule
