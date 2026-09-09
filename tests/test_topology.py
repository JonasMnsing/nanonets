import pytest
from nanonets.topology import NanoparticleTopology

@pytest.mark.parametrize("Nx, Ny, expected_particles, expected_junctions", [
    (3, 3, 9, 4),
    (5, 3, 15, 4),
    (10, 2, 20, 4),
    (5, 1, 5, 2),
])
def test_lattice_network_dimensions(Nx, Ny, expected_particles, expected_junctions):
    topo = NanoparticleTopology()
    topo.lattice_network(N_x=Nx, N_y=Ny, mean_radius=5.0)
    
    # 1. Stimmt die Anzahl der Partikel?
    assert topo.N_particles == expected_particles
    
    # 2. Stimmt die berechnete maximale Anzahl an Nachbarn?
    assert topo.N_junctions == expected_junctions
    
    # 3. Hat der Graph exakt so viele Knoten, wie er haben sollte?
    assert len(topo.G.nodes) == expected_particles
    
    # 4. Ist die Matrix richtig geformt? (Spalte 0 = Elektrode + Junctions)
    assert topo.net_topology.shape == (expected_particles, expected_junctions + 1)