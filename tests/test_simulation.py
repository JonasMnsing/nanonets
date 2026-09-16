import pytest
import numpy as np
from nanonets.simulation import Simulation 

@pytest.fixture
def lattice_params():
    return {
        'Nx': 3,
        'Ny': 3,
        'e_pos': [[0, 0], [2, 2]],
        'electrode_type': ['constant', 'constant']
    }

@pytest.fixture
def random_params():
    return {
        'Np': 9,
        'e_pos': [[-1, -1], [1, 1]],
        'electrode_type': ['constant', 'constant']
    }

# =============================================================================
# INITIALIZATION
# =============================================================================

def test_simulation_init_lattice(lattice_params):
    """Test composition of classes for lattice network."""
    sim = Simulation(topology_parameter=lattice_params)
    
    assert sim.network_topology_type == 'lattice', "Topology not recognized."
    assert sim.topology.N_particles == 9, "Number of particles not correct."
    assert sim.topology.N_electrodes == 2, "Number of electrodes not correct."
    assert sim.electrostatic.capacitance_matrix is not None, "Capacitance not calculated."
    assert hasattr(sim, 'tunneling'), "Tunneling class not initialized."

def test_simulation_init_random(random_params):
    """Test composition of classes for random network and its kwargs."""
    sim = Simulation(topology_parameter=random_params, delta=0.2, max_attempts=10)
    
    assert sim.network_topology_type == 'random', "Topology not recognized."
    assert sim.topology.N_particles == 9, "Number of particles not correct."
    assert sim.topology.N_electrodes == 2, "Number of electrodes not correct."

# =============================================================================
# TESTS: STATIC SIMULATION (KMC)
# =============================================================================

def test_run_static_voltages_standard(lattice_params):
    """Test a static simulation."""
    sim = Simulation(topology_parameter=lattice_params)
    
    voltages = np.array([
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0]
    ])
    
    test_sim_dic = {
        "n_trajectories": 2,
        "max_jumps": 50,
        "max_eq_jumps": 20
    }
    
    sim.run_static_voltages(voltages=voltages, target_electrode=1, sim_dic=test_sim_dic, verbose=True)
    
    # Assertions
    assert len(sim.observable_storage) == 2, "There should be two calculated averages."
    assert len(sim.observable_error_storage) == 2, "There should be two calculated errors."
    assert len(sim.potential_storage) == 2, "There should be two calculated potential arrays."

def test_run_static_voltages_memristive(lattice_params):
    """Test a static simulation for dynamic resistances."""

    res_info_dyn = {"mean_R": 25.0, "std_R": 0.0, "dynamic": True}
    sim = Simulation(topology_parameter=lattice_params, res_info=res_info_dyn, seed=42)
    
    voltages = np.array([
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0]
        ])
    
    test_sim_dic = {
        "n_trajectories": 2,
        "max_jumps": 50,
        "max_eq_jumps": 20
    }
    
    sim.run_static_voltages(voltages=voltages, target_electrode=1, sim_dic=test_sim_dic, 
                            verbose=False, I0=7.5, R_max=25.0, R_min=10.0)
    
    # Assertions
    assert len(sim.observable_storage) == 2, "There should be two calculated averages."
    assert len(sim.observable_error_storage) == 2, "There should be two calculated errors."
    assert len(sim.potential_storage) == 2, "There should be two calculated potential arrays."

# =============================================================================
# TESTS: DYNAMIC SIMULATION
# =============================================================================

def test_run_dynamic_voltages_sine_wave(lattice_params):
    """Test a time dependent simulation."""

    sim = Simulation(topology_parameter=lattice_params)
    
    # 1. Time Steps
    n_steps = 10
    time_steps = np.linspace(0, 1e-6, n_steps)
    
    # 2. Voltages
    voltages = np.zeros((n_steps, 3))
    voltages[:, 0] = 0.2 * np.sin(2 * np.pi * 1e6 * time_steps)
    
    # 3. Run simuation
    sim.run_dynamic_voltages(voltages=voltages, time_steps=time_steps,
                             target_electrode=1, n_trajectories=2, verbose=True)
    
    # 4. Assertions
    assert len(sim.observable_storage) == n_steps, "Observable Storage has wrong length."
    assert len(sim.observable_error_storage) == n_steps, "Observable Error Storage has wrong length."
    assert len(sim.jump_storage) == n_steps, "Jump Storage has wrong length."
    
    # 5. Verbose-Array shapes
    assert sim.state_storage.shape == (n_steps, sim.topology.N_particles), "State Storage Shape is wrong."
    assert sim.potential_storage.shape == (n_steps, sim.topology.N_particles + sim.topology.N_electrodes), "Potential Storage Shape  is wrong."
    
    # Check if there are NaNs
    assert not np.isnan(sim.observable_storage).any(), "There are NaN currents!"