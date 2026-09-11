import pytest
import numpy as np
import networkx as nx
from nanonets.topology import NanoparticleTopology

### LATTICE NETWORK
###################

@pytest.mark.parametrize("Nx, Ny, expected_particles, expected_junctions", [
    (3, 3, 9, 4),
    (5, 3, 15, 4),
    (10, 2, 20, 4),
    (5, 1, 5, 2),
])
def test_lattice_network_dimensions(Nx, Ny, expected_particles, expected_junctions):
    """Basic Properties i.e. Number of Particles, Number of Junctions and Dimension."""
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=Nx, N_y=Ny, mean_radius=5.0)
    
    assert topo.N_particles == expected_particles
    assert topo.N_junctions == expected_junctions
    assert len(topo.G.nodes) == expected_particles
    assert topo.net_topology.shape == (expected_particles, expected_junctions + 1)

def test_net_topology_connections_lattice():
    """Network Topology Array. Check lattice connectivity."""
    neigbor_list = [{1,3},{0,2,4},{1,5},{0,4,6},{1,3,5,7},{2,4,8},{3,7},{4,6,8},{5,7}]
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=3, N_y=3, mean_radius=5.0)
    
    matrix = topo.net_topology
    NO_CONN = topo.NO_CONNECTION

    for i, neighbors_true in enumerate(neigbor_list):
        neighbors = set(matrix[i,1:]) - {NO_CONN}
        assert neighbors == neighbors_true, f"Particle {i} has wrong neihbors!"

def test_net_topology_connections_string():
    """Network Topology Array. Check string connectivity."""
    neigbor_list = [{1},{0,2},{1,3},{2,4},{3}]
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=5, N_y=1, mean_radius=5.0)
    
    matrix = topo.net_topology
    NO_CONN = topo.NO_CONNECTION

    for i, neighbors_true in enumerate(neigbor_list):
        neighbors = set(matrix[i,1:]) - {NO_CONN}
        assert neighbors == neighbors_true, f"Particle {i} has wrong neihbors!"

def test_add_electrodes_topology_matrix():
    """Check if electrodes are properly connected."""
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=3, N_y=3, mean_radius=5.0)
    
    electrode_coords = [(0, 0), (2, 2), (1, 0)]
    expected_indices = [0, 8, 1] 
    
    topo.add_electrodes_to_lattice_net(electrode_coords)
    
    matrix = topo.net_topology
    NO_CONN = topo.NO_CONNECTION
    
    for n, particle_idx in enumerate(expected_indices):
        electrode_number = n + 1
        assert matrix[particle_idx, 0] == electrode_number, \
            f"Particle index {particle_idx} misses electrode {electrode_number}!"
            
    for i in range(topo.N_particles):
        if i not in expected_indices:
            assert matrix[i, 0] == NO_CONN, \
                f"Particle index {i} has an electrode even though it should not!"

def test_net_topology_symmetry():
    """Test connection symmetry i.e. connection of 3-5 also yields 5-3."""
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=4, N_y=4, mean_radius=5.0)
        
    matrix = topo.net_topology
    NO_CONN = topo.NO_CONNECTION
    
    for node_i in range(topo.N_particles):
        neighbors_of_i = [n for n in matrix[node_i, 1:] if n != NO_CONN]
        
        for node_j in neighbors_of_i:
            neighbors_of_j = [n for n in matrix[node_j, 1:] if n != NO_CONN]
            assert node_i in neighbors_of_j, f"Asymmetrie: {node_i} kennt {node_j}, aber nicht umgekehrt!"

def test_lattice_geometry_and_distances():
    """Check if distances are calculated properly."""
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=3, N_y=1, mean_radius=5.0)
    # spacing = 2 * 5.0 + 1.0 = 11.0 nm

    assert topo.pos[0] == (0.0, 0.0)
    assert topo.pos[1] == (11.0, 0.0)
    
    assert np.isclose(topo.dist_matrix[0, 1], 11.0)
    assert np.isclose(topo.dist_matrix[0, 2], 22.0)
    assert np.isclose(topo.dist_matrix[1, 1], 0.0)

def test_graph_consistency():
    """Check Graph to net_topology consistency and electrode connection."""
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=3, N_y=3, mean_radius=5.0)
    topo.add_electrodes_to_lattice_net([(0, 0), (2, 2)])

    # Check consistency between net_topology and graph
    for node in range(topo.N_particles):
        graph_neighbors = set(n for n in topo.G.neighbors(node) if n >= 0)
        topo_neighbors = set(n for n in topo.net_topology[node, 1:] if n != topo.NO_CONNECTION)
        
        # Pytest-Style: assert statt raise ValueError
        assert graph_neighbors == topo_neighbors, \
            f"Inconsistency in connections for node {node}. Graph: {graph_neighbors}, Matrix: {topo_neighbors}"
               
    # Check electrode connections
    for node in range(topo.N_particles):
        electrode = topo.net_topology[node, 0]
        if electrode != topo.NO_CONNECTION:
            # Pytest-Style: assert für beide Richtungen der Kante
            assert topo.G.has_edge(node, -electrode), \
                f"Missing edge from node {node} to electrode {-electrode}"
            assert topo.G.has_edge(-electrode, node), \
                f"Missing edge from electrode {-electrode} to node {node}"

### RANDOM NETWORK
##################

def test_random_network_packing_and_connectivity():
    """Check random network connectivity and no particle overlap."""
    topo = NanoparticleTopology(seed=42)
    topo.random_network(N_particles=20, mean_radius=5.0)

    assert nx.is_strongly_connected(topo.G), "Network is not strongly connected!"
    
    np.fill_diagonal(topo.dist_matrix, np.inf)
    min_dist = np.min(topo.dist_matrix)
    erwarteter_min_abstand = 2 * 5.0 + 1.0 - 1e-5
    
    assert min_dist >= erwarteter_min_abstand, f"Partikel überlappen! Min Distanz: {min_dist}"

def test_random_network_self_healing():
    """Check re-packing when radii change."""
    topo = NanoparticleTopology(seed=42)
    topo.random_network(N_particles=15, mean_radius=5.0)
    
    distance_sum_old = np.sum(topo.dist_matrix)
    
    topo.update_nanoparticle_radius(nanoparticles=[0], mean_radius=30.0)
    
    assert topo.radius_vals[0] == 30.0, "Radius did not change!"
    
    distance_sum_new = np.sum(topo.dist_matrix)
    assert distance_sum_old != distance_sum_new, "Network did not re-pack!"

def test_random_network_symmetry():
    """Test connection symmetry of random networks."""
    topo = NanoparticleTopology(seed=42)
    topo.random_network(N_particles=20, mean_radius=5.0)
    
    matrix = topo.net_topology
    NO_CONN = topo.NO_CONNECTION
    
    for node_i in range(topo.N_particles):
        neighbors_of_i = [n for n in matrix[node_i, 1:] if n != NO_CONN]
        
        for node_j in neighbors_of_i:
            neighbors_of_j = [n for n in matrix[node_j, 1:] if n != NO_CONN]
            
            assert node_i in neighbors_of_j, \
                f"Asymmetry in random network: Connection {node_i} is present {node_j}, but not the other way around!"