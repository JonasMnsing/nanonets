import logging
from pathlib import Path

from nanonets.utils import sinusoidal_voltages, get_time_setup_from_frequency
from nanonets.utils.parallel import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
N_TRAJECTORIES  = 500
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40

# Fixe System-Parameter
FREQ_MHZ        = 60.0
TARGET_CURRENT  = 50.0  # pA threshold for the memristor switching
R_MAX           = 25.0  # MΩ
R_MIN           = 5.0   # MΩ

# Der 2D Parameter-Sweep
AMPLITUDE_LIST  = [0.07, 0.14, 0.21, 0.28, 0.35]
BETA_LIST       = [0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0]

OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/memristors/amp_beta/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 10
# ────────────────────────────────────────────────────────────────────────────────

def calculate_I0(target_current_pA: float, tau_0: float) -> float:
    """Helper to compute memristive threshold from physical current."""
    jumps_per_sec = (target_current_pA * 1e-12) / 1.602e-19
    return jumps_per_sec * tau_0

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    f0_hz = FREQ_MHZ * 1e6
    
    topo = {
        "Nx": N_NP, "Ny": N_NP,
        "e_pos": [[0, 0], [N_NP-1, N_NP-1]],
        "electrode_type": ['constant', 'constant']
    }
    
    # Activate dynamic resistances
    res_info = {"mean_R": R_MAX, "std_R": 0.0, "dynamic": True}

    # Number of samples and time step
    N_steps, dt = get_time_setup_from_frequency(
        f0_hz=f0_hz, 
        n_periods=N_PERIODS, 
        samples_per_period=SAMPLE_P_PERIOD
    )
    
    for amp in AMPLITUDE_LIST:
        time_steps, volt = sinusoidal_voltages(
            N_samples=N_steps, 
            topology_parameter=topo,
            amplitudes=[amp, 0.0], 
            frequencies=[f0_hz, 0.0], 
            time_step=dt
        )
        for beta in BETA_LIST:
            
            # Memristor parameter
            tau_0 = beta / f0_hz
            dyn_res = {
                'tau_0': tau_0,
                'I0': calculate_I0(TARGET_CURRENT, tau_0),
                'R_max': R_MAX,
                'R_min': R_MIN
            }
            
            # Define Task
            args = (time_steps, volt, topo, OUTPUT_DIR)
            kwargs = {
                'net_kwargs': {
                    'add_to_path': f"_amp{amp:.3f}_beta{beta:.2f}",
                    'res_info': res_info,
                    'self_cap': 0.0
                },
                'sim_kwargs': {
                    'n_trajectories': N_TRAJECTORIES, 
                    'save': True,
                    'verbose': True
                    **dyn_res
                }
            }
            tasks.append((args, kwargs))

    logging.info(f"Queued {len(tasks)} simulation tasks. Starting batch launch on {CPU_CNT} cores.")
    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()