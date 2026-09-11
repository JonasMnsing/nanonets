import numpy as np
import pytest
from nanonets.topology import NanoparticleTopology
from nanonets.electrostatic import NanoparticleElectrostatic

def test_electrostatic_pipeline_lattice():
    """Test electrostatic Initilization for a 3x3 lattice."""

    topo = NanoparticleTopology(seed=42)
    topo.lattice_network(N_x=3, N_y=3, mean_radius=5.0)
    topo.add_electrodes_to_lattice_net([(0, 0), (2, 2)])
    
    # Electrostatic-instance via composition
    electro = NanoparticleElectrostatic(topology=topo, electrode_type=['constant', 'constant'])

    # Calc capacitance
    electro.constant_self_capacitance(self_cap=0.28)
    electro.calc_capacitance_matrix(eps_r=2.6, eps_s=3.9, short_range=True)
    electro.calc_electrode_capacitance_matrix(short_range=True)
    
    N = topo.N_particles
    N_elec = topo.N_electrodes
    
    # Capacitance Matrix
    C_mat = electro.get_capacitance_matrix()
    assert C_mat.shape == (N, N)
    assert np.all(np.diag(C_mat) > 0)
    
    # Inverse Capacitance Matrix
    C_inv = electro.get_inv_capacitance_matrix()
    assert C_inv.shape == (N, N)
    
    # Electrode Capacitance Matrix
    C_elec = electro.get_electrode_capacitance_matrix()
    assert C_elec.shape == (N_elec, N)
    
    # Charge vector
    voltages = np.array([1.0, 0.0, 0.0])
    electro.init_charge_vector(voltages)
    charge_vec = electro.get_charge_vector()
    assert charge_vec.shape == (N,)

def test_electrostatic_exceptions():
    """Check for wrong build order"""

    topo = NanoparticleTopology()
    topo.lattice_network(N_x=2, N_y=2, mean_radius=5.0)
    topo.add_electrodes_to_lattice_net([(0, 0), (1, 1)])
    
    electro = NanoparticleElectrostatic(topology=topo, electrode_type=['constant', 'constant'])
    
    with pytest.raises(RuntimeError):
        electro.get_capacitance_matrix()
        
    with pytest.raises(RuntimeError):
        electro.init_charge_vector(np.array([1.0, 0.0, 0.0]))

def test_floating_electrode_validation():
    """Check for wrong voltages for floating electrodes."""

    topo = NanoparticleTopology()
    topo.lattice_network(N_x=2, N_y=2, mean_radius=5.0)
    topo.add_electrodes_to_lattice_net([(0, 0), (1, 1)])
    
    electro = NanoparticleElectrostatic(topology=topo, electrode_type=['constant', 'floating'])
    electro.constant_self_capacitance(0.28)
    electro.calc_capacitance_matrix()
    electro.calc_electrode_capacitance_matrix()
    
    invalid_voltages = np.array([1.0, 5.0, 0.0])
    
    with pytest.raises(ValueError, match="Floating electrode voltages must be initialized to zero"):
        electro.init_charge_vector(invalid_voltages)

def test_lattice_capacitance_diagonal_values():
    """
    Check if diagonal elements of the capacitance matrix for center, edges or corner particles match the expected number of neighbors.
    """

    topo = NanoparticleTopology()
    topo.lattice_network(N_x=3, N_y=3, mean_radius=10.0)
    topo.add_electrodes_to_lattice_net([(0, 0), (2, 2)])
    
    electro = NanoparticleElectrostatic(topology=topo, electrode_type=['constant', 'constant'])
    electro.constant_self_capacitance(self_cap=0.28)
    
    eps_r = 2.6
    electro.calc_capacitance_matrix(eps_r=eps_r, eps_s=3.9, short_range=True)
    electro.calc_electrode_capacitance_matrix(short_range=True)
    
    C_mat = electro.get_capacitance_matrix()
    
    # Estimated mutual capacitance
    r = 10.0
    dist = 11.0
    C_m = electro.mutual_capacitance_adjacent_spheres(eps_r, r, r, dist)
    C_s = 0.28 # Self-Capacitance

    centers = [4]
    corners = [2,6]
    corner_and_electrode = [0,8]
    edges = [1,3,5,7]
    
    expected_center_diag = 4 * C_m + C_s
    expected_edge_diag = 3 * C_m + C_s
    expected_corner_diag = 2 * C_m + C_s

    for i in centers:
        assert np.isclose(C_mat[i, i], expected_center_diag, rtol=1e-5), \
            f"Center Particle {i} wrong, got: {C_mat[i, i]}"

    for i in corners:
        assert np.isclose(C_mat[i, i], expected_corner_diag, rtol=1e-5), \
            f"Corner Particle {i} wrong, got: {C_mat[i, i]}"

    for i in corner_and_electrode:
        assert np.isclose(C_mat[i, i], expected_edge_diag, rtol=1e-5), \
            f"Corner Particle {i} wrong, got: {C_mat[i, i]}"

    for i in edges:
        assert np.isclose(C_mat[i, i], expected_edge_diag, rtol=1e-5), \
            f"Edge Particle {i} wrong, got: {C_mat[i, i]}"