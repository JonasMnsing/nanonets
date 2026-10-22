import logging
import nanonets
import numpy as np
from nanonets.utils import sinusoidal_voltages, get_time_setup_from_frequency
from nanonets.utils.parallel import batch_launch, run_dynamic_simulation, setup_compute_env

# ─── Configuration ────────────────────────────────────────────────────────────────
N_TRAJECTORIES  = 100
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40
SELF_CAP        = 0.0

# 2D Parameter-Sweep
AMPLITUDE_MULT  = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
FREQ_LIST       = np.round(np.logspace(0,4,30),2)

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
    topo = {"Nx": 1, "Ny": 1, "electrode_type" : ['constant','constant']}
    sim = nanonets.Simulation(topo, SELF_CAP)
    v_base = sim.electrostatic.get_charging_energies()[0]*1e-3
    
    for amp_mult in AMPLITUDE_MULT:
        amp = v_base * amp_mult
        for f0_mhz in FREQ_LIST:
            f0_hz = f0_mhz * 1e6
            # Number of samples and time step
            N_steps, dt = get_time_setup_from_frequency(
                f0_hz=f0_hz, n_periods=N_PERIODS, samples_per_period=SAMPLE_P_PERIOD
            )
            time_steps, volt = sinusoidal_voltages(
                N_samples=N_steps, topology_parameter=topo, amplitudes=[amp, 0.0], frequencies=[f0_hz, 0.0], time_step=dt
            )

            # Define Task
            args = (time_steps, volt, topo, OUTPUT_DIR)
            kwargs = {
                'net_kwargs': {
                    'add_to_path': f"_amp{amp:.3f}_f0{f0_mhz:.2f}_self_cap{SELF_CAP:.2f}",
                    'self_cap': SELF_CAP
                },
                'sim_kwargs': {
                    'n_trajectories': N_TRAJECTORIES, 
                    'save': True,
                    'verbose': True,
                }
            }
            tasks.append((args, kwargs))

    logging.info(f"Queued {len(tasks)} simulation tasks. Starting batch launch on {CPU_CNT} cores.")
    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()