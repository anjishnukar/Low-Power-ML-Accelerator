# ===========================================================================
# Makefile for Low-Power ML Accelerator — Systolic Array for Edge AI
# ===========================================================================
# Targets:
#   all        — generate vectors + run Verilator sim (default)
#   vectors    — generate test vectors via Python gold model
#   model      — run the Python golden reference standalone
#   sim        — run with Questa/ModelSim
#   sim_vlt    — run with Verilator (recommended, free)
#   clean      — remove generated files
#   synth      — run Vivado synthesis (requires Vivado)
# ===========================================================================

RTL    := rtl
TB     := tb
MODEL  := model
SCRIPTS := scripts
UPF    := upf
BUILD  := build

VSOURCES := $(RTL)/params_pkg.sv \
            $(RTL)/clock_gate.sv \
            $(RTL)/activation.sv \
            $(RTL)/fp16_mult.sv \
            $(RTL)/pe.sv \
            $(RTL)/memory_interface.sv \
            $(RTL)/systolic_array.sv

.PHONY: all vectors model sim sim_vlt clean synth

# ---- Default: vectors + Verilator simulation ----------------------------
all: vectors sim_vlt

# ---- Generate test vectors from Python golden model ----------------------
vectors:
	python $(MODEL)/gen_vectors.py

# ---- Run the golden model standalone ------------------------------------
model:
	python $(MODEL)/golden_model.py

# ---- Questa / ModelSim simulation (commercial) --------------------------
sim: vectors
	vlib work; \
	vmap work work; \
	vlog -sv $(VSOURCES) $(TB)/tb_systolic_array.sv; \
	vsim -voptargs=+acc -c -do "run -all; quit -f" work.tb_systolic_array

# ---- Verilator simulation (free / open-source) --------------------------
sim_vlt: vectors $(BUILD)/Vsim_top
	./$(BUILD)/Vsim_top

$(BUILD)/Vsim_top: $(VSOURCES) $(TB)/tb_verilator.cpp
	@mkdir -p $(BUILD)
	verilator --sv --cc --exe \
		--build \
		--top-module systolic_array \
		--Mdir $(BUILD) \
		--build-jobs 4 \
		-Wno-UNUSEDSIGNAL \
		-Wno-WIDTHTRUNC \
		-Wno-CASEOVERLAP \
		-Wno-CASEX \
		$(VSOURCES) \
		$(TB)/tb_verilator.cpp \
		-o Vsim_top

# ---- Vivado synthesis (optional) ----------------------------------------
synth:
	vivado -mode batch -source $(SCRIPTS)/run_synthesis.tcl

# ---- Clean ---------------------------------------------------------------
clean:
	rm -rf $(BUILD) work transcript *.wlf modelsim.ini
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
