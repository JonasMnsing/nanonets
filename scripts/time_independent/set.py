import logging
import nanonets
import numpy as np
from nanonets.utils.parallel import batch_launch, run_static_simulation, setup_compute_env

# ─── Configuration ────────────────────────────────────────────────────────────────
SELF_CAP    = 0.0
N_SOURCE    = 100
N_GATE      = 100
SOURCE_VOLT = np.linspace(-0.1,0.1,N_SOURCE)
GATE_VOLT   = [0.0] #np.linspace(-0.1,0.1,N_GATE)
TOPOLOGY    = {"Nx": 1, "Ny": 1, "electrode_type" : ['constant','constant']}

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
    for v_gate in GATE_VOLT:

        volt = np.zeros(shape=(N_SOURCE, 3))
        volt[:,0] = SOURCE_VOLT
        volt[:,-1] = v_gate 
        
        # Define Task
        args = (volt, TOPOLOGY, OUTPUT_DIR)
        kwargs = {
            'net_kwargs': {
                'add_to_path': f"_v_gate{v_gate:.3f}_self_cap{SELF_CAP:.2f}",
                'self_cap': SELF_CAP
            },
            'sim_kwargs': {
                'sim_dic': {
                    "n_trajectories" : 100,
                    "max_jumps"      : 2000,
                    "max_eq_jumps"   : 10000
                },
                'save_th': 10,
                'verbose': True,
            }
        }
        tasks.append((args, kwargs))

    logging.info(f"Queued {len(tasks)} simulation tasks. Starting batch launch on {CPU_CNT} cores.")
    batch_launch(run_static_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()