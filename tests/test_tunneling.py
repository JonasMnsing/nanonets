import numpy as np
import pytest
from nanonets.topology import NanoparticleTopology
from nanonets.electrostatic import NanoparticleElectrostatic
from nanonets.tunneling import NanoparticleTunneling

@pytest.fixture
def base_tunneling_setup():
    """Fixture providing a tunneling instance for a 3x3 lattice."""
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=3, N_y=3, mean_radius=10.0)
    topo.add_electrodes_to_lattice_net([(0, 0), (2, 2)])
    
    electro = NanoparticleElectrostatic(topology=topo, electrode_type=['constant', 'constant'])
    electro.constant_self_capacitance(0.28)
    electro.calc_capacitance_matrix(short_range=True)
    electro.calc_electrode_capacitance_matrix(short_range=True)
    
    tunnel = NanoparticleTunneling(electrostatics=electro)
    return tunnel

def test_tunneling_adv_indices(base_tunneling_setup):
    """Check if tunneling indices are properly initialized containing bi-directional properties."""
    tunnel = base_tunneling_setup
    tunnel.init_adv_indices()
    
    rows, cols = tunnel.get_advanced_indices()
    
    assert len(rows) > 0
    assert len(rows) == len(cols)
    
    # If there is i->j there should also be j->i
    pairs = set(zip(rows, cols))
    for r, c in pairs:
        assert (c, r) in pairs, f"For ({r}, {c}), ({c}, {r}) is missing!"

def test_tunneling_adv_indices_exact_mapping(base_tunneling_setup):
    """Check if the network topology is correctly translated into adv indices."""
    tunnel = base_tunneling_setup
    tunnel.init_adv_indices()
    
    rows, cols = tunnel.get_advanced_indices()
    
    # get adjacency
    actual_connections = {}
    for r, c in zip(rows, cols):
        if r not in actual_connections:
            actual_connections[r] = set()
        actual_connections[r].add(c)

    # Truth
    expected_connections = {
        0: {2},          # E1 (0) is connected with 0 (2)
        1: {10},         # E2 (1) is connected with NP 8 (10)
        2: {0, 3, 5},    # NP 0 (2) connected with E1 (0), NP 1 (3), NP 3 (5)
        3: {2, 4, 6},    # NP 1 (3)
        4: {3, 7},       # NP 2 (4)
        5: {2, 6, 8},    # NP 3 (5)
        6: {3, 5, 7, 9}, # NP 4 (6)
        7: {4, 6, 10},   # NP 5 (7)
        8: {5, 9},       # NP 6 (8)
        9: {6, 8, 10},   # NP 7 (9)
        10: {1, 7, 9}    # NP 8 (10) connected with E2 (1), NP 5 (7), NP 7 (9)
    }

    for node, expected_neighbors in expected_connections.items():
        assert actual_connections[node] == expected_neighbors, \
            f"Node {node} has wrong neighbors! Expected: {expected_neighbors}, Got: {actual_connections.get(node, set())}"

    total_expected_events = sum(len(neighbors) for neighbors in expected_connections.values())
    assert len(rows) == total_expected_events, \
        f"Wrong Number of Events! Expected {total_expected_events}, but matrix got {len(rows)}."

def test_junction_resistances_symmetry(base_tunneling_setup):
    """Check if all resistances are undirected i.e. R_ij == R_ji and above limit."""
    tunnel = base_tunneling_setup
    tunnel.init_adv_indices()
    
    # 1. Check Init
    tunnel.init_junction_resistances(R=25.0, Rstd=5.0)
    R_array = tunnel.get_resistance()
    
    assert np.all(R_array >= tunnel.MIN_RESISTANCE_MOHM)
    
    rows, cols = tunnel.get_advanced_indices()

    # Check Symmetry
    for idx, (i, j) in enumerate(zip(rows, cols)):
        reverse_idx = np.where((rows == j) & (cols == i))[0][0]
        assert np.isclose(R_array[idx], R_array[reverse_idx]), "Resistances are non-symmetric!"

    # 2. Check random update
    tunnel.update_junction_resistances_at_random(N=3, R=100.0, Rstd=0.0)
    R_array_updated = tunnel.get_resistance()
    
    # Symmetry must be preserved
    for idx, (i, j) in enumerate(zip(rows, cols)):
        reverse_idx = np.where((rows == j) & (cols == i))[0][0]
        assert np.isclose(R_array_updated[idx], R_array_updated[reverse_idx]), "Symmetry lost after update!"

def test_conductance_matrix_laplacian_properties(base_tunneling_setup):
    """Test if conductance matrix has properties of a laplace matrix."""
    tunnel = base_tunneling_setup
    tunnel.init_adv_indices()
    tunnel.init_junction_resistances(R=25.0, Rstd=5.0)
    tunnel.build_conductance_matrix()
    
    G = tunnel.get_conductance_matrix()
    
    # Shape has to be (N_particles + N_electrodes)
    total_nodes = tunnel.electro.topo.N_particles + tunnel.electro.topo.N_electrodes
    assert G.shape == (total_nodes, total_nodes)
    
    # Diagonal elements must be positive
    assert np.all(np.diag(G) > 0)
    
    # Off Diagonal elements must be <= 0
    G_off_diag = G.copy()
    np.fill_diagonal(G_off_diag, 0)
    assert np.all(G_off_diag <= 0)
    
    # Any currents which enters a node must also exit the node 
    row_sums = np.sum(G, axis=1)
    assert np.allclose(row_sums, 0, atol=1e-10)

def test_slowest_time_constant(base_tunneling_setup):
    """Eigenvalue must be a non-infty positive float."""
    tunnel = base_tunneling_setup
    tunnel.init_adv_indices()
    tunnel.init_junction_resistances(R=25.0)
    
    tau = tunnel.get_slowest_linear_time_constant()
    
    assert isinstance(tau, float)
    assert tau > 0
    assert not np.isinf(tau)

def test_transfer_coeffs(base_tunneling_setup):
    """Checks the transfer coefficients."""
    tunnel = base_tunneling_setup
    tunnel.init_adv_indices()
    tunnel.init_junction_resistances(R=25.0)
    tunnel.build_conductance_matrix()
    
    tunnel.init_transfer_coeffs(output_electrode=1)
    coeffs = tunnel.get_transfer_coeffs()
    
    assert len(coeffs) == tunnel.electro.topo.N_electrodes
    assert coeffs[1] == 0.0