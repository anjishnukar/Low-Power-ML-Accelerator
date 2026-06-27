module clock_gate (
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  output logic gclk
);
  logic latch;

  always_latch begin
    if (!clk) begin
      latch <= en;
    end
  end

  assign gclk = clk & latch;

endmodule
