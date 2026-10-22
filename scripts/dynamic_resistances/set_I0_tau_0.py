import logging
import nanonets
from nanonets.utils import sinusoidal_voltages, get_time_setup_from_frequency
from nanonets.utils.parallel import batch_launch, run_dynamic_simulation, setup_compute_env

# ─── Configuration ────────────────────────────────────────────────────────────────
N_TRAJECTORIES  = 100
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40
SELF_CAP        = 0.0

# Fix Parameter
FREQ_MHZ    = 500.0
AMP_MULT    = 3
TOPO        = {"Nx": 1, "Ny": 1, "electrode_type" : ['constant','constant']}
sim         = nanonets.Simulation(TOPO, SELF_CAP)
AMPLITUDE   = AMP_MULT * sim.electrostatic.get_charging_energies()[0]*1e-3
R_MAX       = 25.0
R_MIN       = 5.0 

# 2D Parameter-Sweep
TAU_LIST            = [0.1e-9, 0.2e-9, 1.0e-9, 2.0e-9, 6.0e-9, 20.0e-9, 100.0e-9]
TARGET_CURRENT_LIST = [1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0]

# PATH and CPU count
OUTPUT_DIR, CPU_CNT = setup_compute_env(
    cluster_base_path="/scratch/j_mens07/nanonets/data/",
    script_path=__file__)
LOG_LEVEL = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def calculate_I0(target_current_pA: float, tau_0: float) -> float:
    jumps_per_sec = (target_current_pA * 1e-12) / 1.602e-19
    return jumps_per_sec * tau_0

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    f0_hz = FREQ_MHZ * 1e6
    
    # Activate dynamic resistances
    res_info = {"mean_R": R_MAX, "std_R": 0.0, "dynamic": True}

    # Number of samples and time step
    N_steps, dt = get_time_setup_from_frequency(f0_hz=f0_hz, n_periods=N_PERIODS, samples_per_period=SAMPLE_P_PERIOD)
    time_steps, volt = sinusoidal_voltages(N_samples=N_steps, topology_parameter=TOPO,amplitudes=[AMPLITUDE, 0.0], frequencies=[f0_hz, 0.0], time_step=dt)

    for i_target in TARGET_CURRENT_LIST:
        for tau_0 in TAU_LIST:
            i_0 = calculate_I0(i_target, tau_0)

            # Memristor parameter
            dyn_res = {
                'tau_0': tau_0,
                'I0': i_0,
                'R_max': R_MAX,
                'R_min': R_MIN
            }

            tau_ns = tau_0 * 1e9

            # Define Task
            args = (time_steps, volt, TOPO, OUTPUT_DIR)
            kwargs = {
                'net_kwargs': {
                    'add_to_path': f"_tau{tau_ns:.3f}_i_target{i_target:.2f}",
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