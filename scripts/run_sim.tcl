# Questa/ModelSim run script
# Usage: vsim -do scripts/run_sim.tcl

set WORK work

vlib $WORK
vmap work $WORK

# Compile
vlog -sv -work $WORK rtl/params_pkg.sv
vlog -sv -work $WORK rtl/clock_gate.sv
vlog -sv -work $WORK rtl/activation.sv
vlog -sv -work $WORK rtl/pe.sv
vlog -sv -work $WORK rtl/memory_interface.sv
vlog -sv -work $WORK rtl/systolic_array.sv
vlog -sv -work $WORK tb/tb_systolic_array.sv

# Simulate (run for 1000 ns at 100 MHz -> 100 cycles)
vsim -voptargs=+acc work.tb_systolic_array

log -r *

run 1000 ns

# Check results
if {[examine -radix decimal /tb_systolic_array/num_errors] == 0} {
    puts "\n+++ SIMULATION PASSED +++"
} else {
    puts "\n--- SIMULATION FAILED ---"
}

quit -f
