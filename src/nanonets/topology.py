import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Tuple

class NanoparticleTopology:
    """
    Class to set up, modify, and analyze the topology of nanoparticle networks,
    including connections to external electrodes.

    Supports both lattice (grid) and random topologies. Manages network structure via a
    NetworkX directed graph and a compact topology matrix for export/import.

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles (nodes) in the network.
    N_electrodes : int
        Number of electrodes attached to the network.
    N_junctions : int
        Target number of junctions (neighbors) per nanoparticle (for random networks, average).
    N_x, N_y : int
        Grid dimensions for cubic networks. Only set for cubic grids.
    rng : numpy.random.Generator
        Random number generator instance for reproducibility.
    G : nx.DiGraph
        NetworkX directed graph representing the nanoparticle network.
    pos : dict
        Dictionary mapping node indices to their 2D positions for visualization.
    net_topology : np.ndarray
        Matrix describing the network connectivity:
            - First column: electrode connection (if any), or NO_CONNECTION.
            - Remaining columns: indices of connected nanoparticles, or NO_CONNECTION.

    Constants
    ---------
    NO_CONNECTION : int
        Placeholder value for unconnected junctions in topology matrix.

    Methods
    -------
    lattice_network(N_x, N_y)
        Set up a lattice of nanoparticles.
    add_electrodes_to_lattice_net(particle_pos)
        Attach electrodes to specific nanoparticles by their lattice positions.
    random_network(N_particles, N_junctions=0)
        Set up a random network, using Delaunay triangulation.
    add_electrodes_to_random_net(electrode_positions)
        Attach electrodes to closest available nanoparticles in a random network.
    add_np_to_output()
        Add a nanoparticle at the output electrode.
    graph_to_net_topology()
        Synchronize net_topology from the NetworkX graph object.
    get_net_topology(), get_graph(), get_positions()
        Accessors for topology, graph, and node positions.
    validate_network()
        Run consistency and connectivity checks.
    export_network(filepath), import_network(filepath)
        Save/load network state for reproducibility and sharing.
    """

    NO_CONNECTION = -100

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the NanoparticleTopology class.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator (for reproducibility).
        """
        self.rng            = np.random.default_rng(seed)
        self.N_particles    = 0
        self.N_electrodes   = 0
        self.N_junctions    = 0
        self.G              = nx.DiGraph()
        self.pos            = {}
        self.net_topology   = np.array([])

    def lattice_network(self, N_x: int, N_y: int) -> None:
        """
        Define a 2D square lattice of nanoparticles.

        Parameters
        ----------
        N_x : int
            Number of nanoparticles along the x-direction.
        N_y : int
            Number of nanoparticles along the y-direction.

        Notes
        -----
        Each node is connected to its immediate neighbors.
        """
        self.lattice        = True
        self.N_x, self.N_y  = N_x, N_y
        self.N_particles    = N_x * N_y

        # Generate 2D grid positions
        nano_particles_pos = [[x, y] for y in range(N_y) for x in range(N_x)]
        self.pos = {i : [pos[0],pos[1]] for i, pos in enumerate(nano_particles_pos)}
        
        # Initialize graph and add nodes
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.N_particles))

        # Determine number of neighbors (junctions) based on dimensions
        if ((N_x > 1) and (N_y > 1)):
            self.N_junctions = 4+1
        else:
            self.N_junctions = 2+1

        # Allocate topology matrix: first col for electrode, rest for neighbors
        self.net_topology = np.full((self.N_particles, self.N_junctions + 1), fill_value=self.NO_CONNECTION)

        # Connect each nanoparticle to its 2D nearest neighbors
        for idx, pos1 in enumerate(nano_particles_pos):
            n_NN = 0  # Neighbor count
            for jdx, pos2 in enumerate(nano_particles_pos):
                # Distance of 1: Immediate neighbor
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance == 1:
                    self.net_topology[idx, n_NN + 1] = jdx
                    self.G.add_edge(idx,jdx)
                    self.G.add_edge(jdx,idx)
                    n_NN += 1
                if (n_NN == self.N_junctions):
                    break
         
    def add_electrodes_to_lattice_net(self, particle_pos: List[List[int]]) -> None:
        """
        Attach electrodes to nanoparticles at specified positions (for cubic grids).

        Parameters
        ----------
        particle_pos : List of [x, y]
            Each [x, y] specifies the grid coordinate of a nanoparticle to be connected to an electrode.

        Raises
        ------
        RuntimeError
            If called before a cubic network is initialized.
        ValueError
            If any position is out of bounds or assigned multiple times.
        """
        if not getattr(self, "lattice", False) or not hasattr(self, "N_x") or not hasattr(self, "N_y"):
            raise RuntimeError("cubic_network() must be called before add_electrodes_to_lattice_net().")
        
        # Check for duplicate or out-of-bounds positions
        seen_indices = set()
        for pos in particle_pos:
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                raise ValueError(f"Invalid position format: {pos}. Must be [x, y].")
            x, y = pos
            if not (0 <= x < self.N_x and 0 <= y < self.N_y):
                raise ValueError(f"Electrode position {pos} out of bounds (grid size {self.N_x}x{self.N_y}).")
            idx = y * self.N_x + x
            if idx in seen_indices:
                raise ValueError(f"Duplicate electrode assignment at position {pos}.")
            seen_indices.add(idx)
    
        self.N_electrodes = len(particle_pos)

        # Add the electrode nodes to the graph, using negative indices for electrodes
        electrode_nodes = -np.arange(1,self.N_electrodes+1)
        self.G.add_nodes_from(electrode_nodes)

        # Attach each electrode to its corresponding particle based on particle positions
        for n, pos in enumerate(particle_pos):
            # Convert 2D position (x, y) into a 1D index in the nanoparticle network
            p = pos[1]*self.N_x + pos[0]
            self.net_topology[p,0] = 1 + n  # Store the electrode number (starting from 1)
            
            # Connect the electrode node to the nanoparticle
            electrode_node  = electrode_nodes[n]
            self.G.add_edge(electrode_node,p)
            self.G.add_edge(p,electrode_node)

            # Set electrode positions, adjusting based on boundary conditions
            if (pos[0] == 0):           # If the particle is at the left boundary (x == 0)
                self.pos[electrode_node] = (pos[0]-1,pos[1])
            elif (pos[0] == (self.N_x-1)):   # If the particle is at the right boundary (x == N_x-1)
                self.pos[electrode_node] = (pos[0]+1,pos[1])
            elif (pos[1] == 0):         # If the particle is at the bottom boundary (y == 0)
                self.pos[electrode_node] = (pos[0],pos[1]-1)
            else:                       # Default case (otherwise move the electrode in the positive direction)
                self.pos[electrode_node] = (pos[0],pos[1]+1)

    def random_network(self, N_particles: int) -> None:
        """Just predefines attributes for future processing in electrostatic.py

        Parameters
        ----------
        N_particles : int
            _description_
        """
        self.lattice        = False
        self.N_particles    = N_particles

    def add_electrodes_to_random_net(self, electrode_positions: List[List[float]]) -> None:
        """Just predefines attributes for future processing in electrostatic.py

        Parameters
        ----------
        electrode_positions : List[List[float]]
            List of electrode positions [[x1, y1], [x2, y2], ...]. Coordinates should be within the box [-1, 1].

        Raises
        ------
        RuntimeError
            If called before a random network has been created.
        ValueError
            If there are more electrodes than nanoparticles, or invalid position types.
        """
        if getattr(self, "lattice", True):
            raise RuntimeError("add_electrodes_to_random_net() can only be called after random_network().")
        if len(electrode_positions) > self.N_particles:
            raise ValueError("Cannot attach more electrodes than there are nanoparticles.")
        
        self.electrode_positions = electrode_positions
        self.N_electrodes = len(electrode_positions)
            
    # def random_network(self, N_particles: int) -> None:
    #     """
    #     Set up a random 2D planar nanoparticle network using Delaunay triangulation
    #     and Poisson disk sampling for physical plausibility.

    #     The domain size is automatically chosen so that all N_particles fit with the required
    #     minimum separation (based on smallest allowed radius).

    #     Parameters
    #     ----------
    #     N_particles : int
    #         Number of nanoparticles (nodes) in the network.

    #     Raises
    #     ------
    #     ValueError
    #         If N_particles < 3 (Delaunay triangulation requires at least 3 points).
    #     RuntimeError
    #         If Poisson disk sampling cannot generate the requested points.
    #     """
    #     if N_particles < 3:
    #         raise ValueError("At least 3 particles are required for Delaunay triangulation.")

    #     self.lattice        = False
    #     self.N_particles    = N_particles

    #     # Use the minimum possible NP radius for separation BEFORE radii are initialized
    #     min_dist = 2 * 10.0 + 1.0  # [nm]

    #     # Compute required domain radius for random close packing
    #     packing_density     = 0.45  # empirical value for random disk packings
    #     needed_area         = N_particles * (min_dist / 2) ** 2 / packing_density
    #     domain_radius       = np.sqrt(needed_area)
    #     self.domain_radius  = domain_radius

    #     def poisson_disk_sampling(n_points, min_dist, domain_radius=1.0, max_attempts=10000, rng=None):
    #         rng = rng or np.random.default_rng()
    #         points = []
    #         attempts = 0
    #         while len(points) < n_points and attempts < max_attempts:
    #             r = domain_radius * np.sqrt(rng.uniform())
    #             theta = rng.uniform(0, 2 * np.pi)
    #             x, y = r * np.cos(theta), r * np.sin(theta)
    #             if all(np.hypot(x - px, y - py) >= min_dist for px, py in points):
    #                 points.append((x, y))
    #             attempts += 1
    #         if len(points) < n_points:
    #             raise RuntimeError(
    #                 f"Could not place {n_points} points with min_dist={min_dist} "
    #                 f"in {max_attempts} attempts. Try reducing n_points or min_dist."
    #             )
    #         return points

    #     # Generate random positions with conservative minimum separation
    #     pos = poisson_disk_sampling(self.N_particles, min_dist, domain_radius=domain_radius, rng=self.rng)

    #     # Build undirected graph via Delaunay triangulation
    #     temp_G = nx.Graph()
    #     for i, p in enumerate(pos):
    #         temp_G.add_node(i, pos=p)

    #     tri = Delaunay(pos)
    #     edges = set()
    #     for simplex in tri.simplices:
    #         for i in range(3):
    #             edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
    #             edges.add(edge)
    #     temp_G.add_edges_from(edges)

    #     # Store as a directed graph (all edges are bidirectional)
    #     self.G = nx.DiGraph(temp_G)
    #     self.pos = {i: p for i, p in enumerate(pos)}
    #     self.N_junctions = np.max([val for (node, val) in temp_G.degree()]) + 1

    #     self._graph_to_net_topology()
        
    # def add_electrodes_to_random_net(self, electrode_positions: List[List[float]]) -> None:
    #     """
    #     Attach electrodes to a random nanoparticle network.

    #     This method is intended to be used **only after** calling `random_network()`.
    #     For each provided electrode position, the nearest unassigned nanoparticle is located,
    #     and a bidirectional connection is made. Each electrode is represented as a new negative-index node,
    #     and its spatial position is recorded.

    #     Parameters
    #     ----------
    #     electrode_positions : List[List[float]]
    #         List of electrode positions [[x1, y1], [x2, y2], ...]. Coordinates should be within the box [-1, 1].

    #     Raises
    #     ------
    #     RuntimeError
    #         If called before a random network has been created (i.e., if not self.lattice and pos is not set).
    #     ValueError
    #         If there are more electrodes than nanoparticles, or invalid position types.
    #     """
    #     if getattr(self, "lattice", True):
    #         raise RuntimeError("add_electrodes_to_random_net() can only be called after random_network().")
    #     if not hasattr(self, "pos") or not self.pos or not isinstance(self.pos, dict):
    #         raise RuntimeError("No nanoparticle positions found. Call random_network() first.")
    #     if len(electrode_positions) > self.N_particles:
    #         raise ValueError("Cannot attach more electrodes than there are nanoparticles.")
    #     if not hasattr(self, "domain_radius"):
    #         raise RuntimeError("Domain radius not found. Make sure random_network has been called.")
        
    #     # Scale electrode positions from [-1, 1] to device disk
    #     scaled_electrode_positions = [[x * self.domain_radius, y * self.domain_radius] for x, y in electrode_positions]
        
    #     self.N_electrodes = len(scaled_electrode_positions)
    #     used_nodes = set()  # Track which nanoparticles are already connected to an electrode

    #     # Convert positions to DataFrame for easy computation
    #     node_positions = pd.DataFrame(self.pos).T.sort_index()

    #     for n, e_pos in enumerate(scaled_electrode_positions):
    #         if not (isinstance(e_pos, (list, tuple)) and len(e_pos) == 2):
    #             raise ValueError(f"Electrode position {e_pos} is invalid. Must be [x, y].")
            
    #         # Compute Euclidean distance to each nanoparticle
    #         node_positions['d'] = np.sqrt((e_pos[0] - node_positions[0]) ** 2 +
    #                                     (e_pos[1] - node_positions[1]) ** 2)
            
    #         # Find closest unused nanoparticle
    #         found = False
    #         for i in node_positions.sort_values(by='d').index:
    #             if i not in used_nodes:
    #                 used_nodes.add(i)
    #                 closest_nanoparticle = i
    #                 found = True
    #                 break
    #         if not found:
    #             raise RuntimeError("Ran out of unassigned nanoparticles before placing all electrodes.")
            
    #         electrode_node = -n - 1
    #         # Create bidirectional edges
    #         self.G.add_edge(electrode_node, closest_nanoparticle)
    #         self.G.add_edge(closest_nanoparticle, electrode_node)
    #         # Store the electrode's spatial position for plotting
    #         self.pos[electrode_node] = tuple(e_pos)
        
    #     self._graph_to_net_topology()
                
    def add_np_to_output(self):
        """
        Insert a single new nanoparticle in series between the currently connected
        nanoparticle and the output electrode (the last electrode in the network).

        Steps performed:
        1. Identifies the nanoparticle currently connected to the output electrode.
        2. Disconnects this nanoparticle from the output electrode.
        3. Creates a new nanoparticle.
        4. Connects this new nanoparticle between the previously connected nanoparticle and the output electrode.
        5. Updates the net_topology matrix, NetworkX graph, and position dictionary.

        Raises
        ------
        RuntimeError
            If there are no electrodes defined, or if output electrode is not connected.
        """
        if not hasattr(self, 'N_electrodes') or self.N_electrodes < 1:
            raise RuntimeError("No electrodes defined in the network.")
        output_electrode_idx = self.N_electrodes-1

        # Find the nanoparticle connected to the output electrode (search net_topology)
        matches = np.where(self.net_topology[:, 0] == (output_electrode_idx + 1))[0]
        if len(matches) == 0:
            raise RuntimeError(f"No nanoparticle is connected to output electrode (index {output_electrode_idx}).")
        adj_np = matches[0]
        
        prev_particle_count =   self.N_particles
        self.N_particles    +=  1

        # Prepare new topology row for the new nanoparticle
        new_row     = np.full(self.net_topology.shape[1], self.NO_CONNECTION, dtype=int)
        new_row[0]  = output_electrode_idx + 1  # Connect to electrode
        new_row[1]  = adj_np                    # Connect to the previous adjacent nanoparticle

        self.net_topology = np.vstack((self.net_topology, new_row))

        # Update the adjacent nanoparticle's connection:
        #  - Find the first free neighbor slot and connect it to the new node
        #  - Remove its previous electrode connection, if present
        free_spots = np.where(self.net_topology[adj_np, :] == self.NO_CONNECTION)[0]
        if len(free_spots) == 0:
            raise RuntimeError(f"No free neighbor slots in net_topology for nanoparticle {adj_np}.")
        self.net_topology[adj_np, free_spots[0]]    = prev_particle_count
        self.net_topology[adj_np, 0]                = self.NO_CONNECTION  # Remove old electrode connection

        # Update the NetworkX graph
        electrode_node = -output_electrode_idx - 1
        if self.G.has_edge(adj_np, electrode_node):
            self.G.remove_edge(adj_np, electrode_node)
        if self.G.has_edge(electrode_node, adj_np):
            self.G.remove_edge(electrode_node, adj_np)
        self.G.add_node(prev_particle_count)
        self.G.add_edge(prev_particle_count, adj_np)
        self.G.add_edge(adj_np, prev_particle_count)
        self.G.add_edge(prev_particle_count, electrode_node)
        self.G.add_edge(electrode_node, prev_particle_count)

        # Assign a spatial position to the new nanoparticle (place at electrode for now)
        if electrode_node in self.pos:
            self.pos[prev_particle_count] = self.pos[electrode_node]
        else:
            self.pos[prev_particle_count] = (0, 0)  # fallback if no electrode position

        # Move the electrode for clarity (especially for lattice layout)
        x, y = self.pos[prev_particle_count]
        if self.lattice:
            # Move electrode to be just outside the grid boundary in a sensible way
            if x == self.N_x:
                self.pos[electrode_node] = (x + 1, y)
            elif x == -1:
                self.pos[electrode_node] = (x - 1, y)
            elif y == self.N_y:
                self.pos[electrode_node] = (x, y + 1)
            elif y == -1:
                self.pos[electrode_node] = (x, y - 1)
        else:
            # For random networks, shift electrode right or up for clarity
            self.pos[electrode_node] = (x + 0.2, y + 0.2)

    def _graph_to_net_topology(self)->None:
        """
        Rebuild the net_topology matrix from the current directed graph (self.G).

        For each nanoparticle node (nodes 0 to N_particles-1):
            - The first column indicates a connected electrode (index > 0),
            or NO_CONNECTION if none is present. Only the *first* electrode found
            will be listed (multiple electrode connections are not expected).
            - Subsequent columns list the indices of connected nanoparticle neighbors,
            or NO_CONNECTION as a placeholder if not all neighbor slots are filled.

        This method should be called after building or modifying the graph,
        especially after creating a new random network or adding electrodes.

        Updates
        -------
        self.net_topology : np.ndarray

        Raises
        ------
        RuntimeError
            If self.G or basic network attributes are not initialized.
        """
        if not hasattr(self, "G") or not isinstance(self.G, nx.DiGraph):
            raise RuntimeError("Graph object self.G is not initialized.")
        if not hasattr(self, "N_particles") or not hasattr(self, "N_junctions"):
            raise RuntimeError("Basic network attributes (N_particles, N_junctions) not set.")
    
        # Create an empty net_topology array
        net_topology = np.full(shape=(self.N_particles, self.N_junctions + 1), fill_value=self.NO_CONNECTION, dtype=int)

        # For each nanoparticle (node indices 0 ... N_particles-1)
        for node in range(self.N_particles):
            neighbor_idx = 1  # Start filling from column 1 (0 is for electrode)
            for neighbor in self.G.neighbors(node):
                if neighbor >= 0:
                    # Connected to another nanoparticle
                    if neighbor_idx <= self.N_junctions:
                        net_topology[node, neighbor_idx] = neighbor
                        neighbor_idx += 1
                else:
                    # Connected to an electrode (negative indices)
                    # Only one electrode per particle is stored in the topology
                    net_topology[node, 0] = -neighbor  # Store as positive integer (1-based)

        # Store the generated net topology in the class attribute
        self.net_topology = net_topology

    def get_net_topology(self) -> np.ndarray:
        """
        Return a copy of the network topology matrix.

        Returns
        -------
        np.ndarray
            Network topology matrix (see class docstring for structure).
        """
        return self.net_topology.copy()

    def get_graph(self) -> nx.DiGraph:
        """
        Return the NetworkX DiGraph object for the current network.

        Returns
        -------
        nx.DiGraph
        """
        return self.G
    
    def get_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Return a dictionary mapping node indices to their 2D positions.

        Returns
        -------
        dict
            Dictionary of node positions.
        """
        return dict(self.pos)

    def validate_network(self) -> bool:
        """
        Validate the network topology.
        
        Checks:
        1. Network connectivity (weak connectivity for directed graphs)
        2. Consistency between net_topology matrix and NetworkX graph
        3. Electrode connections
        
        Returns
        -------
        bool
            True if the network is valid, False otherwise.
            
        Raises
        ------
        ValueError
            If inconsistencies are found in the network topology
        """
        # Check network connectivity (using weak connectivity for directed graphs)
        if not nx.is_weakly_connected(self.G):
            raise ValueError("Network is not fully connected")
            
        # Check consistency between net_topology and graph
        for node in range(self.N_particles):
            graph_neighbors = set(n for n in self.G.neighbors(node) if n >= 0)
            topo_neighbors = set(n for n in self.net_topology[node, 1:] if n != self.NO_CONNECTION)
            if graph_neighbors != topo_neighbors:
                raise ValueError(f"Inconsistency found in connections for node {node}")
                
        # Check electrode connections
        for node in range(self.N_particles):
            electrode = self.net_topology[node, 0]
            if electrode != self.NO_CONNECTION:
                if not self.G.has_edge(node, -(electrode)) or not self.G.has_edge(-(electrode), node):
                    raise ValueError(f"Missing electrode connection for node {node}")
                    
        return True
    
    def export_network(self, filepath: str) -> None:
        """
        Export the network configuration to a file.
        
        This method saves:
        1. Network topology matrix
        2. Node positions
        3. Electrode configurations
        4. Network parameters (N_particles, N_junctions, etc.)
        
        Parameters
        ----------
        filepath : str
            Path to save the network configuration file
        """
        network_data = {
            'net_topology': self.net_topology.tolist(),
            'positions': {str(k): list(v) for k, v in self.pos.items()},
            'N_particles': self.N_particles,
            'N_electrodes': self.N_electrodes,
            'N_junctions': self.N_junctions
        }
        
        if hasattr(self, 'N_x'):
            network_data['N_x'] = self.N_x
            network_data['N_y'] = self.N_y
            
        np.save(filepath, network_data, allow_pickle=True)

    def import_network(self, filepath: str) -> None:
        """
        Import a network configuration from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the network configuration file
            
        Raises
        ------
        ValueError
            If the file format is invalid or missing required data
        """
        try:
            network_data = np.load(filepath, allow_pickle=True).item()
            
            # Restore basic attributes
            self.net_topology = np.array(network_data['net_topology'])
            self.pos = {int(k) if k.isdigit() else int(k[1:]) if k.startswith('-') else k: 
                       tuple(v) for k, v in network_data['positions'].items()}
            self.N_particles = network_data['N_particles']
            self.N_electrodes = network_data['N_electrodes']
            self.N_junctions = network_data['N_junctions']
            
            if 'N_x' in network_data:
                self.N_x = network_data['N_x']
                self.N_y = network_data['N_y']
                
            # Reconstruct the graph
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.N_particles))
            self.G.add_nodes_from(range(-self.N_electrodes, 0))
            
            # Add edges from topology matrix
            for node in range(self.N_particles):
                # Add electrode connections
                if self.net_topology[node, 0] != self.NO_CONNECTION:
                    electrode = -self.net_topology[node, 0]
                    self.G.add_edge(node, electrode)
                    self.G.add_edge(electrode, node)
                    
                # Add nanoparticle connections
                for neighbor in self.net_topology[node, 1:]:
                    if neighbor != self.NO_CONNECTION:
                        self.G.add_edge(node, neighbor)
                        self.G.add_edge(neighbor, node)
                        
            # Validate the imported network
            self.validate_network()
            
        except Exception as e:
            raise ValueError(f"Failed to import network configuration: {str(e)}")
    
    def __str__(self):
        return f"Topology Class with {self.N_particles} particles, {self.N_junctions} junctions.\nNetwork Topology:\n{self.net_topology}"
    
###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Lattice
    #########
    # N_x, N_y        = 5,3
    # electrode_pos   = [[0,0],[2,0],[0,2],[2,2]]
    N_x, N_y        = 1,1
    electrode_pos   = [[0,0],[0,0]]
    lattice_net     = NanoparticleTopology()

    # Build Network and attach Electrodes
    lattice_net.lattice_network(N_x, N_y)
    lattice_net.add_electrodes_to_lattice_net(electrode_pos)
    # lattice_net.add_np_to_output()
    is_valid = lattice_net.validate_network()
    print(lattice_net)
    print("This Network is valid!\n") if is_valid else print("This network is not valid!\n")
    
    # Disordered Network Topology
    #############################
    N_particles     = 20
    electrode_pos   = [[-1,-1],[-1,1],[1,-1],[1,1]]
    rng_net         = NanoparticleTopology()

    # Build Network and attach Electrodes
    rng_net.random_network(N_particles)
    rng_net.add_electrodes_to_random_net(electrode_pos)
    rng_net.add_np_to_output()
    is_valid = rng_net.validate_network()
    print(lattice_net)
    print("This Network is valid!") if is_valid else print("This network is not valid!")

