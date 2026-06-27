# Makefile for Low-Power ML Accelerator
# Targets: sim, vectors, model, clean

SIM    = vsim
WORK   = work
SCRIPTS = scripts
TB     = tb
RTL    = rtl
MODEL  = model

.PHONY: all vectors sim clean model

all: vectors sim

# Generate test vectors from Python golden model
vectors:
	python $(MODEL)/gen_vectors.py

# Run the golden model standalone
model:
	python $(MODEL)/golden_model.py

# Run Questa/ModelSim simulation
sim: vectors
	vlib $(WORK); \
	vmap work $(WORK); \
	vlog -sv $(RTL)/params_pkg.sv $(RTL)/clock_gate.sv $(RTL)/activation.sv \
	      $(RTL)/pe.sv $(RTL)/memory_interface.sv $(RTL)/systolic_array.sv \
	      $(TB)/tb_systolic_array.sv; \
	vsim -voptargs=+acc -c -do "run -all; quit -f" work.tb_systolic_array

# Clean generated files
clean:
	rm -rf $(WORK) transcript *.wlf modelsim.ini
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

# Vivado synthesis (optional)
synth:
	vivado -mode batch -source $(SCRIPTS)/run_synthesis.tcl
