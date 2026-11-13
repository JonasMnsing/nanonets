import logging
import nanonets
import numpy as np
from nanonets.utils.parallel import batch_launch, run_static_simulation, setup_compute_env, distribute_array_across_processes
from nanonets.utils.voltages import uniform_sample

# ─── Configuration ────────────────────────────────────────────────────────────────
SELF_CAP    = 0.0
N_VOLT      = 50000
U_E         = 0.03
GATE_VOLT   = [0.0]
L           = 3
TOPOLOGY    = {
    "Nx": L, "Ny": L, "electrode_type" : ['constant','constant','constant','constant','constant','constant','constant','constant'],
    "e_pos" : [[L//2,0],[0,0],[L-1,0],[0,L//2],[L-1,L//2],[0,L-1],[L-1,L-1],[L//2,L-1]]
}

# PATH and CPU count
OUTPUT_DIR, CPU_CNT = setup_compute_env(
    cluster_base_path="/scratch/j_mens07/nanonets/data/",
    script_path=__file__)
LOG_LEVEL = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    volt = uniform_sample(U_E, N_VOLT, TOPOLOGY) 

    for i in range(CPU_CNT):
        volt_p = distribute_array_across_processes(i, volt, CPU_CNT)
        args = (volt_p, TOPOLOGY, OUTPUT_DIR)
        kwargs = {
            'net_kwargs': {
                'add_to_path': f"_self_cap{SELF_CAP:.2f}_part{i}",
                'self_cap': SELF_CAP
            },
            'sim_kwargs': {
                'save_th': 100,
            }
        }
        tasks.append((args, kwargs))

    logging.info(f"Queued {len(tasks)} simulation tasks. Starting batch launch on {CPU_CNT} cores.")
    batch_launch(run_static_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()