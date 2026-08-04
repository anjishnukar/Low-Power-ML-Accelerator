module fp16_mult (
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [31:0] result  // INT32 product (truncated toward zero)
);

  logic          sign;
  logic [4:0]    exp_a, exp_b;
  logic [10:0]   man_a, man_b;   // 11-bit mantissa (1.xxx or 0.xxx for denorm/zero)
  logic [21:0]   man_prod;       // 22-bit mantissa product
  logic [5:0]    exp_sum;
  logic          is_zero_a, is_zero_b;
  logic [31:0]   shifted_val;
  logic [31:0]   abs_result;

  assign is_zero_a = (a[14:0] == 15'd0);
  assign is_zero_b = (b[14:0] == 15'd0);

  assign sign   = a[15] ^ b[15];
  assign exp_a  = a[14:10];
  assign exp_b  = b[14:10];

  // Mantissa: implicit leading 1 for normal numbers, 0 for denormal/zero
  assign man_a = (exp_a == 5'd0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]};
  assign man_b = (exp_b == 5'd0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]};

  always_comb begin
    if (is_zero_a || is_zero_b) begin
      result = 32'd0;
    end else begin
      man_prod = man_a * man_b;               // 22-bit product
      exp_sum  = exp_a + exp_b;               // biased sum (bias 15+15=30)

      // Real value = man_prod * 2^(exp_sum - 50)
      //   man_prod is a 22-bit value with binary point after bit 20
      //   (see derivation in comments above)
      //   Shift = exp_sum - 50:
      //     positive → left shift (integer part)
      //     negative → right shift (fractional part)

      if (exp_sum >= 6'd50) begin
        shifted_val = $unsigned(man_prod) << (exp_sum - 6'd50);
      end else begin
        shifted_val = $unsigned(man_prod) >> (6'd50 - exp_sum);
      end

      // Clamp to INT32 range
      if (shifted_val > 32'd2147483647) begin
        abs_result = 32'd2147483647;
      end else begin
        abs_result = shifted_val;
      end

      result = sign ? (~abs_result + 32'd1) : abs_result;
    end
  end

endmodule
