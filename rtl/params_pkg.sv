package params_pkg;
  // ------------------------------------------------------------------
  // Architecture parameters
  // ------------------------------------------------------------------
  parameter int ARRAY_SIZE      = 4;           // N x N systolic array (4 or 8)
  parameter int DATA_WIDTH      = 8;           // 8 for INT8, 16 for FP16
  parameter int ACCUM_WIDTH     = 32;          // Accumulator width
  parameter int MEM_ADDR_WIDTH  = 10;          // Address width for SRAM
  parameter int MEM_DATA_WIDTH  = 128;         // Data bus width for SRAM
  parameter int PE_MEM_DEPTH    = 256;         // Depth per PE local storage

  // Data type selection
  parameter bit USE_FP16        = 0;           // 0 -> INT8, 1 -> FP16

  // Activation function select
  typedef enum logic [1:0] {
    ACT_LINEAR = 2'b00,
    ACT_RELU   = 2'b01,
    ACT_TANH   = 2'b10
  } act_mode_t;

  // Memory port identifier
  typedef enum logic {
    MEM_PORT_A = 1'b0,
    MEM_PORT_B = 1'b1
  } mem_port_t;

endpackage
