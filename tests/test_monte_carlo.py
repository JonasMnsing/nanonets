import numpy as np
import time
from nanonets.monte_carlo import MonteCarlo

def run_test():

    print("=== Setup a Dummy-SET-System (Single-Electron Transistor) ===")
    
    # 1. Constants
    N_electrodes = 2
    N_particles = 1
    e_charge = 0.1602176634

    # 2. Topology (adv_indices)
    # Nodes: 0=Source, 1=Drain, 2=NP
    # Event 0: Source -> NP (0 -> 2)
    # Event 1: NP -> Source (2 -> 0)
    # Event 2: NP -> Drain  (2 -> 1)
    # Event 3: Drain -> NP  (1 -> 2)
    adv_index_rows = np.array([0, 2, 2, 1], dtype=np.int64)
    adv_index_cols = np.array([2, 0, 1, 2], dtype=np.int64)

    # 3. Electrostatic
    # V_Source = 0.5 V, V_Drain = -0.5 V, V_NP = 0.0 V
    potential_vector = np.array([0.5, -0.5, 0.0], dtype=np.float64)
    charge_vector = np.array([0.0], dtype=np.float64)
    
    # Capacitance matrix (1x1 for a single particle, C = 2 aF -> inv_C = 0.5)
    inv_capacitance_matrix = np.array([[0.5]], dtype=np.float64)
    
    # Coulomb charging energy for each event (constant offsets)
    const_capacitance_values = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    
    # 4. Tunneling physics (T_val = k_B * T, res = R_T * e^2)
    temperatures = np.array([0.025, 0.025, 0.025, 0.025], dtype=np.float64)
    base_resistance_MOhm = 25.0
    scaled_res = base_resistance_MOhm * (e_charge**2) * 1e-12
    resistances = np.array([scaled_res, scaled_res, scaled_res, scaled_res], dtype=np.float64)

    # Empty array for floating electrodes
    floating_electrodes = np.empty(0, dtype=np.int64)

    print("Initialize MonteCarlo Engine...")
    t0 = time.time()
    
    model = MonteCarlo(
        charge_vector=charge_vector,
        potential_vector=potential_vector,
        inv_capacitance_matrix=inv_capacitance_matrix,
        const_capacitance_values=const_capacitance_values,
        temperatures=temperatures,
        resistances=resistances,
        adv_index_rows=adv_index_rows,
        adv_index_cols=adv_index_cols,
        N_electrodes=N_electrodes,
        N_particles=N_particles,
        floating_electrodes=floating_electrodes
    )
    print(f"Compiling '__init__' finished in {time.time()-t0:.2f} seconds.\n")

    # =========================================================================
    # TEST 1: Standard Equilibration
    # =========================================================================
    print("=== Test 1: run_equilibration_steps ===")
    t0 = time.time()
    model.run_equilibration_steps(5000)
    print(f"Finished in {time.time()-t0:.4f} seconds.")
    print(f"Letztes Event: {model.jump}, Vergangene KMC-Zeit: {model.time:.2e} s\n")

    # =========================================================================
    # TEST 2: Standard Production Run
    # =========================================================================
    print("=== Test 2: kmc_simulation_trajectory ===")
    target_electrode = 1
    t0 = time.time()
    model.kmc_simulation_trajectory(target_electrode, 10000)
    print(f"Finished in {time.time()-t0:.4f} seconds.")
    print(f"Averaged NP-Charge: {model.charge_mean[0]:.4f} aC")
    print(f"Measured Current (Drain): {model.target_observable_mean*1e-9:.4e} nA\n")

    # =========================================================================
    # TEST 3: Time dependent KMC
    # =========================================================================
    print("=== Test 3: kmc_time_simulation_trajectory ===")
    target_time = model.time + 1e-6
    t0 = time.time()
    model.kmc_time_simulation_trajectory(target_electrode, target_time)
    print(f"Finished in {time.time()-t0:.4f} seconds.")
    print(f"System time snapped to target_time? {model.time == target_time}")
    print(f"Measured Current (Drain) in Slice: {model.target_observable_mean*1e-9:.4e} nA\n")

    # =========================================================================
    # TEST 4: Memristive Function
    # =========================================================================
    print("=== Test 4: Memristive KMC (Eq + Prod) ===")
    t0 = time.time()
    # Equilibration
    model.run_equilibration_steps_var_resistance(2000, 7.5, 1e-8, 25.0, 10.0)
    # Production
    model.kmc_simulation_trajectory_var_resistance(target_electrode, 5000, 7.5, 1e-8, 25.0, 10.0)
    print(f"Finished in {time.time()-t0:.4f} seconds.")
    print(f"Memristive track (I_tilde) of each event: {model.I_tilde}")
    print(f"Measured memristive Current (Drain): {model.target_observable_mean*1e-9:.4e} nA\n")

    print("+++ ALL TESTS PASSED +++")

if __name__ == "__main__":
    run_test()