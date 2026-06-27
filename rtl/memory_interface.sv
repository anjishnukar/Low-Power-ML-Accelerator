`include "params_pkg.sv"
import params_pkg::*;

// Ping-pong SRAM buffer for weight / activation streaming
module memory_interface #(
  parameter int DATA_W = MEM_DATA_WIDTH,
  parameter int ADDR_W = MEM_ADDR_WIDTH,
  parameter int DEPTH  = PE_MEM_DEPTH
) (
  input  logic           clk,
  input  logic           rst_n,
  // Host side (write into buffer)
  input  logic           wr_en,
  input  logic [ADDR_W-1:0] wr_addr,
  input  logic [DATA_W-1:0] wr_data,
  // PE side (read from buffer)
  input  logic           rd_en,
  input  logic [ADDR_W-1:0] rd_addr,
  output logic [DATA_W-1:0] rd_data,
  // Ping-pong swap control
  input  logic           swap,
  output logic           swap_done
);

  // Two banks: A and B (ping-pong)
  logic [DATA_W-1:0] mem_a [0:DEPTH-1];
  logic [DATA_W-1:0] mem_b [0:DEPTH-1];

  // Active port mapping
  logic               wr_port_sel, rd_port_sel;
  logic [DATA_W-1:0]  mem_a_rd, mem_b_rd;

  // Swap state machine
  typedef enum logic [1:0] {
    SWAP_IDLE,
    SWAP_SYNC,
    SWAP_DONE
  } swap_state_t;

  swap_state_t state, next;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= SWAP_IDLE;
      wr_port_sel <= MEM_PORT_A;
      rd_port_sel <= MEM_PORT_B;
    end else begin
      state <= next;
      case (state)
        SWAP_IDLE: if (swap) begin
          wr_port_sel <= !wr_port_sel;
          rd_port_sel <= !rd_port_sel;
        end
        default:;
      endcase
    end
  end

  always_comb begin
    next = state;
    swap_done = 1'b0;
    unique case (state)
      SWAP_IDLE: if (swap)        next = SWAP_SYNC;
      SWAP_SYNC:                  next = SWAP_DONE;
      SWAP_DONE: begin
        swap_done = 1'b1;
        next      = SWAP_IDLE;
      end
    endcase
  end

  // Write ports
  always_ff @(posedge clk) begin
    if (wr_en) begin
      if (wr_port_sel == MEM_PORT_A)
        mem_a[wr_addr] <= wr_data;
      else
        mem_b[wr_addr] <= wr_data;
    end
  end

  // Read ports
  always_ff @(posedge clk) begin
    if (rd_en) begin
      if (rd_port_sel == MEM_PORT_A)
        mem_a_rd <= mem_a[rd_addr];
      else
        mem_b_rd <= mem_b[rd_addr];
    end
  end

  assign rd_data = (rd_port_sel == MEM_PORT_A) ? mem_a_rd : mem_b_rd;

endmodule
