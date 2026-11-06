import logging
import nanonets
import numpy as np

# ACHTUNG: Importiere hier deine neue Puls-Funktion!
from nanonets.utils import rectangular_pulses  
from nanonets.utils.parallel import batch_launch, run_dynamic_simulation, setup_compute_env

# ─── Configuration ────────────────────────────────────────────────────────────────
N_TRAJECTORIES  = 1000
TAU_0           = 2e-9  # s
TARGET_CURRENT  = 20.0   # pA 
R_MAX           = 25.0   # MΩ
R_MIN           = 5.0    # MΩ
DT              = 1.5e-11

# Single Electron Transistor
TOPO        = {"Nx": 1, "Ny": 1, "electrode_type" : ['constant','constant']}
sim         = nanonets.Simulation(TOPO, 0.0)
AMP_MULT    = 2.2
V_WRITE     = [AMP_MULT * sim.electrostatic.get_charging_energies()[0]*1e-3] * 10
T_WRITE     = [1.6e-9] * 10
T_WAIT      = [1.0e-9] * 7 + [12.0e-9] + [1.0e-9]

# Path Setup
OUTPUT_DIR, CPU_CNT = setup_compute_env(
    cluster_base_path="/scratch/j_mens07/nanonets/data/",
    script_path=__file__
)
LOG_LEVEL = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def calculate_I0(target_current_pA: float, tau_0: float) -> float:
    """Helper to compute memristive threshold from physical current."""
    jumps_per_sec = (target_current_pA * 1e-12) / 1.602e-19
    return jumps_per_sec * tau_0

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []

    # Memristor
    res_info = {"mean_R": R_MAX, "std_R": 0.0, "dynamic": True}
    dyn_res = {
        'tau_0': TAU_0,
        'I0': calculate_I0(TARGET_CURRENT, TAU_0),
        'R_max': R_MAX,
        'R_min': R_MIN
    }
            
    # 1. Two Pulses
    time_steps, V_signal = rectangular_pulses(
        V_write=V_WRITE, 
        t_write=T_WRITE, 
        t_wait=T_WAIT, 
        dt=DT, 
        n_pulses=2
    )
    volt = np.zeros(shape=(len(V_signal), 3))
    volt[:,0] = V_signal
    
    # 2. Tasks
    args = (time_steps, volt, TOPO, OUTPUT_DIR)
    kwargs = {
        'net_kwargs': {
            'add_to_path': f"_pulse_train",
            'res_info': res_info,
            'self_cap': 0.0
        },
        'sim_kwargs': {
            'n_trajectories': N_TRAJECTORIES, 
            'save': True,
            'verbose': True,
            **dyn_res
        }
    }
    tasks.append((args, kwargs))

    logging.info(f"Queued {len(tasks)} simulation tasks. Starting batch launch on {CPU_CNT} cores.")
    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()