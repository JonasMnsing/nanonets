import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Tuple

class NanoparticleTopology:
    """
    Class to set up, modify, and analyze the topology of nanoparticle networks,
    including connections to external electrodes. Supports both lattice (grid) and random topologies.

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
    radius_vals : np.ndarray of float
        Radii of nanoparticles [nm].
    dist_matrix : np.ndarray
        Pairwise NP-NP distance matrix [nm].
    electrode_dist_matrix : np.ndarray
        Pairwise NP-electrode distance matrix [nm].

    Constants
    ---------
    NO_CONNECTION : int
        Placeholder value for unconnected junctions in topology matrix.
    MIN_NP_RADIUS : float
        Minimum allowed nanoparticle radius
    MIN_NP_NP_DISTANCE : float
        Minimum distance between two adjacent nanoparticles
    
    Notes
    -----
    All distances/radii are in nanometers (nm)
    """

    NO_CONNECTION = -100
    MIN_NP_RADIUS = 5.0
    MIN_NP_NP_DISTANCE = 1.0

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the NanoparticleTopology class.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator (for reproducibility).
        """
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.N_particles: int = 0
        self.N_electrodes: int = 0
        self.N_junctions: int = 0
        self.N_x: Optional[int] = None
        self.N_y: Optional[int] = None
        self.lattice: bool = True
        self.G: nx.DiGraph = nx.DiGraph()
        self.pos: Dict[int, Tuple[float,float]] = {}
        self.net_topology: np.ndarray = np.array([])
        self.radius_vals: np.ndarray = None
        self.dist_matrix: Optional[np.ndarray] = None
        self.electrode_dist_matrix: Optional[np.ndarray] = None

    ### NANOPARTICLE RADIUS
    #######################

    def _sample_radii(self, size: int, mean: float, std: float) -> np.ndarray:
        """Sample radii from truncated normal distribution >= MEAN_NP_RADIUS

        Parameters
        ----------
        size : int
            Number of samples
        mean : float
            Mean radius [nm]
        std : float
            Standard Deviation for radii [nm]

        Returns
        -------
        np.ndarray
            Array of radii
        """

        if std == 0.0:
            return np.full(size, mean)
        
        radii = self.rng.normal(loc=mean, scale=std, size=size)
        while np.any(radii < self.MIN_NP_RADIUS):
            bad_mask = radii < self.MIN_NP_RADIUS
            radii[bad_mask] = self.rng.normal(loc=mean, scale=std, size=np.sum(bad_mask))
        return radii

    def init_nanoparticle_radius(self, mean_radius: float = 10.0, std_radius: float = 0.0) -> None:
        """
        Initialize radii for all nanoparticles, ensuring all radii are >= MIN_NP_RADIUS.

        Parameters
        ----------
        mean_radius : float, optional
            Mean nanoparticle radius [nm]. Must be >= MIN_NP_RADIUS. (Default: 10.0)
        std_radius : float, optional
            Standard deviation for radii [nm]. Must be >= 0. (Default: 0.0)

        Raises
        ------
        ValueError
            If mean_radius <= MIN_NP_RADIUS or std_radius < 0.
            If no nanoparticles are present in the network.
        """
        if mean_radius < self.MIN_NP_RADIUS:
            raise ValueError(
                f"Mean radius ({mean_radius}) cannot be smaller than MIN_NP_RADIUS ({self.MIN_NP_RADIUS})."
            )
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}.")
        if self.N_particles <= 0:
            raise ValueError("No nanoparticles defined. Initialize the network before setting radii.")

        self.radius_vals = self._sample_radii(self.N_particles, mean_radius, std_radius)

    def update_nanoparticle_radius(self, nanoparticles: List[int], mean_radius: float = 10.0, std_radius: float = 0.0, delta: float = 0.5, max_attempts: int = 5, **packing_kwargs) -> None:
        """
        Update the radii of specific nanoparticles in the network.

        Parameters
        ----------
        nanoparticles : List[int]
            Indices of nanoparticles to update.
        mean_radius : float, optional
            Mean radius for the new values [nm]. Must be >= MIN_NP_RADIUS. (default: 10.0).
        std_radius : float, optional
            Standard deviation of radius [nm]. Must be >= 0. (default: 0.0).
        delta : float, optional
            Connection tolerance buffer [nm] (default: 0.5).
        max_attempts : int, optional
            Number of repacking attempts if graph is disconnected (default: 5).
            
        Raises
        ------
        ValueError
            If mean_radius <= MIN_NP_RADIUS, std_radius < 0, or if indices are non existent.
        RuntimeError
            If radius_vals has not been initialized.
        """
        if self.radius_vals is None:
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")
        if self.network_type == "lattice":
            raise RuntimeError("Cannot alter individual radii in a perfect lattice network.")

        if mean_radius < self.MIN_NP_RADIUS:
            raise ValueError(
                f"Mean radius ({mean_radius}) cannot be smaller than MIN_NP_RADIUS ({self.MIN_NP_RADIUS})."
            )
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}.")
            
        invalid_indices = [i for i in nanoparticles if i < 0 or i >= self.N_particles]
        if invalid_indices:
            raise ValueError(f"Invalid nanoparticle indices: {invalid_indices}")

        new_radii = self._sample_radii(len(nanoparticles), mean_radius, std_radius)
        self.radius_vals[nanoparticles] = new_radii

        if self.network_type == "random":
            self._rebuild_random_geometry(delta = delta, max_attempts = max_attempts, **packing_kwargs)

    def update_nanoparticle_radius_at_random(self, N: int, mean_radius: float = 10.0, std_radius: float = 0.0, delta: float = 0.5, max_attempts: int = 5, **packing_kwargs) -> None:
        """
        Randomly select N unique nanoparticles and update their radii.

        Parameters
        ----------
        N : int
            Number of distinct nanoparticles to modify.
        mean_radius : float, optional
            Mean radius value [nm]. Must be >= MIN_NP_RADIUS. (default: 10.0).
        std_radius : float, optional
            Standard deviation for radius values [nm]. Must be >= 0. (default: 0.0).
        delta : float, optional
            Connection tolerance buffer [nm] (default: 0.5).
        max_attempts : int, optional
            Number of repacking attempts if graph is disconnected (default: 5).

        Raises
        ------
        ValueError
            If mean_radius <= MIN_NP_RADIUS, std_radius < 0, or N is greater than the total number of nanoparticles.
        RuntimeError
            If radius_vals has not been initialized.
        """
        if self.radius_vals is None:
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")
        if self.network_type == "lattice":
            raise RuntimeError("Cannot alter individual radii in a perfect lattice network.")

        if N > self.N_particles or N <= 0:
            raise ValueError(f"Requested {N} nanoparticles, but total available is {self.N_particles}.")

        if mean_radius < self.MIN_NP_RADIUS:
            raise ValueError(
                f"Mean radius ({mean_radius}) cannot be smaller than MIN_NP_RADIUS ({self.MIN_NP_RADIUS})."
            )
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}.")

        chosen_indices = self.rng.choice(a=self.N_particles, size=N, replace=False)
        new_radii = self._sample_radii(N, mean_radius, std_radius)
        self.radius_vals[chosen_indices] = new_radii

        if self.network_type == "random":
            self._rebuild_random_geometry(delta = delta, max_attempts = max_attempts, **packing_kwargs)

    def _update_distance_matrix(self) -> None:
        """
        Compute the NxN pairwise Euclidean distance matrix between nanoparticles.
        """
        coords = np.array([self.pos[i] for i in range(self.N_particles)])
        # Broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        self.dist_matrix = np.linalg.norm(delta, axis=-1)
    
    def _update_electrode_distance_matrix(self) -> None:
        """
        Compute the (N_electrodes x N_particles) Euclidean distance matrix.
        """
        if self.N_electrodes == 0:
            self.electrode_dist_matrix = np.array([])
            return
            
        el_coords = np.array([self.pos[-i] for i in range(1, self.N_electrodes + 1)])
        np_coords = np.array([self.pos[i] for i in range(self.N_particles)])
        
        # Broadcasting: (N_el, 1, 2) - (1, N_np, 2) -> (N_el, N_np, 2)
        delta = el_coords[:, np.newaxis, :] - np_coords[np.newaxis, :, :]
        self.electrode_dist_matrix = np.linalg.norm(delta, axis=-1)

    ### LATTICE NETWORKS
    ####################

    def lattice_network(self, N_x: int, N_y: int, mean_radius: float = 10.0) -> None:
        """
        Define a 2D square lattice of nanoparticles.

        Parameters
        ----------
        N_x : int
            Number of nanoparticles along the x-direction.
        N_y : int
            Number of nanoparticles along the y-direction.
        mean_radius : float, optional
            Radius of the nanoparticles [nm] (default: 10.0). 
            Note: Lattice networks only support identical radii (std_radius = 0).

        Raises
        ------
        RuntimeError
            If radius_vals has not been initialized.
        """
        self.lattice = True
        self.N_x, self.N_y = N_x, N_y
        self.N_particles = N_x * N_y

        # Define nanoparticle radii
        self.init_nanoparticle_radius(mean_radius=mean_radius, std_radius=0.0)
        r_val = self.radius_vals[0]
        spacing = 2 * r_val + self.MIN_NP_NP_DISTANCE

        # Generate 2D grid positions
        self.pos = {
            y * N_x + x: (float(x * spacing), float(y * spacing))
            for y in range(N_y) for x in range(N_x)
        }
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.N_particles))

        if N_x > 1 and N_y > 1:
            self.N_junctions = 4
        else:
            self.N_junctions = 2

        # Allocate topology matrix: first col for electrode, rest for neighbors
        # Note: N_junctions + 1 columns total (col 0 = electrode, rest = neighbors)
        self.net_topology = np.full((self.N_particles, self.N_junctions + 1), fill_value=self.NO_CONNECTION)

        # Connect each nanoparticle to its 2D nearest neighbors
        for y in range(N_y):
            for x in range(N_x):
                idx = y * N_x + x
                neighbors = []
                
                # Check 4 possible directions (Right, Left, Up, Down)
                potential_neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for nx_coord, ny_coord in potential_neighbors:
                    if 0 <= nx_coord < N_x and 0 <= ny_coord < N_y:
                        jdx = ny_coord * N_x + nx_coord
                        neighbors.append(jdx)
                        self.G.add_edge(idx, jdx)
                        self.G.add_edge(jdx, idx) # Undirected behavior in directed graph

                # Store in topology matrix
                for n_idx, neighbor_jdx in enumerate(neighbors):
                    self.net_topology[idx, n_idx + 1] = neighbor_jdx

        self._update_distance_matrix()
   
    def add_electrodes_to_lattice_net(self, particle_pos: List[List[int]]) -> None:
        """
        Attach electrodes to nanoparticles at specified positions (for lattice networks).

        Parameters
        ----------
        particle_pos : List of [x, y]
            Each [x, y] specifies the grid coordinate of a nanoparticle to be connected to an electrode.

        Raises
        ------
        RuntimeError
            If called before a lattice network is initialized.
        ValueError
            If any position is out of bounds or assigned multiple times.
        """
        if not self.lattice or self.N_x is None or self.N_y is None:
            raise RuntimeError("lattice_network() must be called before add_electrodes_to_lattice_net().")
        
        # Check for duplicate or out-of-bounds positions
        seen_indices = set()
        for pos in particle_pos:
            x, y = pos[0], pos[1]
            if not (0 <= x < self.N_x and 0 <= y < self.N_y):
                raise ValueError(f"Electrode position {pos} out of bounds.")
            idx = y * self.N_x + x
            if idx in seen_indices:
                raise ValueError(f"Duplicate electrode assignment at position {pos}.")
            seen_indices.add(idx)
    
        # Numer of electrodes and add electrode nodes to the graph, using negative indices for electrodes
        self.N_electrodes = len(particle_pos)
        electrode_nodes = -np.arange(1, self.N_electrodes + 1)
        self.G.add_nodes_from(electrode_nodes)
        spacing = 2 * self.radius_vals[0] + self.MIN_NP_NP_DISTANCE

        # Attach each electrode to its corresponding particle based on particle positions
        for n, pos in enumerate(particle_pos):
            x, y = pos
            # Convert 2D position (x, y) into a 1D index in the nanoparticle network
            p = y * self.N_x + x
            self.net_topology[p,0] = 1 + n  # Store the electrode number (starting from 1)
            
            # Connect the electrode node to the nanoparticle
            electrode_node = electrode_nodes[n]
            self.G.add_edge(electrode_node, p)
            self.G.add_edge(p, electrode_node)

            # Set electrode positions, adjusting based on boundary conditions
            if x == 0:           # If the particle is at the left boundary (x == 0)
                self.pos[electrode_node] = ((x - 1) * spacing, y * spacing)
            elif x == (self.N_x - 1):   # If the particle is at the right boundary (x == N_x-1)
                self.pos[electrode_node] = ((x + 1) * spacing, y * spacing)
            elif y == 0:         # If the particle is at the bottom boundary (y == 0)
                self.pos[electrode_node] = (x * spacing, (y - 1) * spacing)
            else:                       # Default case (otherwise move the electrode in the positive direction)
                self.pos[electrode_node] = (x * spacing, (y + 1) * spacing)

        self._update_electrode_distance_matrix()

    ### RANDOM NETWORKS
    ###################

    def pack_circles(self, iterations: int = 20000, dt: float = 0.005, k_repel: float = 25.0, k_attract: float = 0.5,
                     initial_temp:float = 2.0, start_cutoff: float = 40.0, end_cutoff: float = 2.2, safety_passes: int = 5000):
        """
        Packs polydisperse circles into a dense, cohesive cluster using a physics-based 
        simulated annealing ("glass transition") algorithm.

        The algorithm transitions from a high-temperature "liquid" phase (global gathering, 
        high thermal noise) to a low-temperature "glassy" phase (local lattice locking, 
        zero noise) to find an optimal local minimum.

        Parameters
        ----------
        iterations : int, optional
            The total number of physics simulation steps. High values (e.g. 20000) 
            are recommended for high-precision packing. Default is 20000.
        dt : float, optional
            The time step for the physics integration. Lower values (e.g. 0.005) 
            increase stability and precision but slow down effective movement. 
            Default is 0.005.
        k_repel : float, optional
            Stiffness of the particles. Controls how strongly overlapping particles 
            push apart. Higher values resolve overlaps faster but can cause instability 
            if `dt` is too large. Default is 25.0.
        k_attract : float, optional
            Cohesion strength. Controls how strongly particles pull together to fill 
            voids. Default is 0.5.
        initial_temp : float, optional
            The starting magnitude of the thermal noise (Brownian motion). This noise 
            randomly shakes particles to prevent them from getting trapped in loose 
            arches or jams. Decays to zero over the course of the simulation. 
            Default is 2.0.
        start_cutoff : float, optional
            The initial interaction range (as a multiple of target distance). 
            A large value (e.g. 40.0) ensures that initially distant particles 
            can "see" and attract each other to form a single cluster. 
            Default is 40.0.
        end_cutoff : float, optional
            The final interaction range. A small value (e.g. 2.2) ensures that 
            particles eventually only bond with their immediate neighbors, forming 
            a dense local lattice without global crushing. Default is 2.2.
        safety_passes : int, optional
            The number of purely geometric (non-physics) iterations run after the 
            simulation to strictly resolve any remaining floating-point overlaps. 
            Default is 5000.

        Returns
        -------
        dict
            A dictionary where keys are the indices (0 to N-1) corresponding to the 
            input `radii`, and values are lists `[x, y]` of center coordinates.

        Notes
        -----
        The simulation minimizes a potential energy landscape where the ideal distance 
        between two particles $i$ and $j$ is $r_i + r_j + d_{min}$. 
        """

        # Copy NP Radii and Number
        radii = self.radius_vals.copy()
        N = self.N_particles
        
        # --- 1. Smart Initialization ---
        # Shuffle indices to prevent input-order bias (size segregation)
        indices = np.arange(N)
        self.rng.shuffle(indices)
        
        # Estimate total area to determine a safe initial spread
        # We start loose to allow the "liquid" phase to reorganize easily
        total_area = np.sum(np.pi * (radii + self.MIN_NP_NP_DISTANCE)**2)
        start_radius = np.sqrt(total_area) * 2.5
        
        # Random placement in a circular cloud
        theta = self.rng.uniform(0, 2*np.pi, N)
        r_pos = np.sqrt(self.rng.uniform(0, 1, N)) * start_radius
        x = r_pos * np.cos(theta)
        y = r_pos * np.sin(theta)
        
        # Store as (N, 2) array for vectorized operations
        positions = np.column_stack((x, y))

        # --- 2. Main Physics Annealing Loop ---       
        for step in range(iterations):
            progress = step / iterations
            
            # --- A. Annealing Schedule ---
            
            # 1. Interaction Range (Vision): Global -> Local
            # Keep global (start_cutoff) for first 30% to gather isolated clusters
            if progress < 0.3:
                current_cutoff = start_cutoff
            else:
                # Exponential decay to end_cutoff
                # This gently tightens the "vision" to nearest neighbors only
                p_decay = (progress - 0.3) / 0.7
                current_cutoff = start_cutoff * (end_cutoff / start_cutoff)**p_decay

            # 2. Temperature (Thermal Noise): Hot -> Frozen
            # Linearly decay noise to zero at 85% completion to allow final settling
            if progress < 0.85:
                current_temp = initial_temp * (1.0 - progress / 0.85)
            else:
                current_temp = 0.0

            # --- B. Vectorized Distance Calculations ---
            # Calculate N x N distance matrix using broadcasting
            # delta[i, j] is the vector pointing from j to i
            delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] 
            dist_sq = np.sum(delta**2, axis=2)
            dist = np.sqrt(dist_sq)
            
            # Prevent division by zero for self-interaction or exact overlap
            safe_dist = dist.copy()
            safe_dist[safe_dist < 1e-7] = 1e-7
            norm_delta = delta / safe_dist[..., np.newaxis]
            
            # Determine Target Distances
            radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            target_dist = radii_sum + self.MIN_NP_NP_DISTANCE
            
            # Diff: Negative = Overlap, Positive = Gap
            diff = dist - target_dist
            np.fill_diagonal(diff, np.inf) # Ignore self-interaction
            
            # --- C. Force Calculation ---
            
            # 1. Repulsion (Overlap)
            # Active if diff < 0. Force is proportional to overlap depth.
            mask_repel = diff < 0
            repel_mag = diff * k_repel
            # Force clamping avoids numeric explosion from deep initial overlaps
            repel_mag = np.clip(repel_mag, -5.0, 5.0) 
            
            f_repel = np.sum(mask_repel[..., np.newaxis] * norm_delta * repel_mag[..., np.newaxis], axis=1)
            
            # 2. Attraction (Cohesion)
            # Active if gap exists (diff > 0) AND within current interaction range.
            vision_limit = target_dist * current_cutoff
            mask_attract = (diff > 0) & (dist < vision_limit)
            
            attract_mag = diff * k_attract
            attract_mag = np.clip(attract_mag, -5.0, 5.0)
            
            f_attract = np.sum(mask_attract[..., np.newaxis] * norm_delta * attract_mag[..., np.newaxis], axis=1)
            
            # Combine Forces
            # Note: repel_mag is negative (diff<0). norm_delta points j->i.
            # We want to push i away from j.
            # The math: Total Force = - (Repel + Attract)
            total_force = -(f_repel + f_attract)
            
            # 3. Thermal Noise (Brownian Motion)
            # Random kicks to break "jammed" arches and explore configurations
            if current_temp > 0:
                noise = self.rng.normal(0, 1, size=(N, 2)) * current_temp
                total_force += noise
                
            # 4. Global Damping (Viscosity)
            # Simulates a thick fluid, preventing perpetual oscillation
            total_force *= 0.5 
            
            # 5. Drift Correction
            # Gently recenter the cloud to (0,0) so it doesn't float away
            positions -= np.mean(positions, axis=0) * 0.05
            
            # Integration (Euler)
            positions += total_force * dt
            
        # --- 3. Final Safety Polish ---
        # Strictly resolves any remaining microscopic overlaps           
        for i in range(safety_passes):
            delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(delta**2, axis=2))
            dist[dist < 1e-7] = 1e-7
            
            req = radii[:, np.newaxis] + radii[np.newaxis, :] + self.MIN_NP_NP_DISTANCE
            overlap = req - dist
            np.fill_diagonal(overlap, -1)
            
            if not np.any(overlap > 0):
                break
                
            mask = overlap > 0
            norm_delta = delta / dist[..., np.newaxis]
            # Very gentle correction to fix overlaps without breaking the lattice
            correction = np.sum(mask[..., np.newaxis] * norm_delta * overlap[..., np.newaxis], axis=1) * 0.2
            positions += correction
            positions -= np.mean(positions, axis=0)

        # Format output
        self.pos = {i: positions[i].tolist() for i in range(N)}

    def create_packing_graph(self, delta: float = 0.5, max_attempts: int = 5, **packing_kwargs) -> None:
        """
        Create a symmetric directed graph based on physical circle contact.

        Two particles i, j connect if distance <= (r_i + r_j + MIN_NP_NP_DISTANCE + delta).

        Parameters
        ----------
        delta : float, optional
            Connection tolerance buffer [nm] (default: 0.5).
        max_attempts : int, optional
            Number of repacking attempts if graph is disconnected (default: 5).
        """
        if self.radius_vals is None or len(self.pos) < self.N_particles:
            raise RuntimeError("Packing positions and radii must exist before creating the graph.")

        attempt = 0
        while attempt < max_attempts:
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.N_particles))

            coords = np.array([self.pos[i] for i in range(self.N_particles)])

            for i in range(self.N_particles):
                pos_i = coords[i]
                r_i = self.radius_vals[i]

                for j in range(i + 1, self.N_particles):
                    dist = np.linalg.norm(pos_i - coords[j])
                    threshold = r_i + self.radius_vals[j] + self.MIN_NP_NP_DISTANCE + delta

                    if dist <= threshold:
                        self.G.add_edge(i, j)
                        self.G.add_edge(j, i)

            # Check strong connectivity across all particles
            if nx.is_strongly_connected(self.G):
                break

            # Repack if disconnected
            self.pack_circles(**packing_kwargs)
            attempt += 1

        # Max NP-NP neighbor count
        max_deg = max((deg for _, deg in self.G.out_degree()), default=0)
        self.N_junctions = max_deg
    
    def _rebuild_random_geometry(self, delta: float = 0.5, max_attempts: int = 5, **packing_kwargs) -> None:
        """
        Internal pipeline to heal the network after radius changes.
        Repacks circles, rebuilds the graph, and updates matrices.

        Parameters
        ----------
        delta : float, optional
            Connection tolerance buffer [nm] (default: 0.5).
        max_attempts : int, optional
            Number of repacking attempts if graph is disconnected (default: 5).
        """
        self.pack_circles(**packing_kwargs)
        self.create_packing_graph(delta=delta, max_attempts=max_attempts, **packing_kwargs)
        self._graph_to_net_topology()
        self._update_distance_matrix()
        # Falls Elektroden existieren, müssen auch deren Distanzen neu berechnet werden
        if self.N_electrodes > 0:
            self._update_electrode_distance_matrix()

    def random_network(self, N_particles: int, mean_radius: float = 10.0, std_radius: float = 0.0, delta: float = 0.5, max_attempts: int = 5, **packing_kwargs) -> None:
        """
        Initializes a random network from scratch.

        Parameters
        ----------
        mean_radius : float, optional
            Mean nanoparticle radius [nm]. Must be >= MIN_NP_RADIUS. (Default: 10.0)
        std_radius : float, optional
            Standard deviation for radii [nm]. Must be >= 0. (Default: 0.0)
        delta : float, optional
            Connection tolerance buffer [nm] (default: 0.5).
        max_attempts : int, optional
            Number of repacking attempts if graph is disconnected (default: 5).
        """
        if N_particles <= 1:
            raise ValueError("N_particles must be greater than 1.")

        self.lattice = False
        self.network_type = "random"
        self.N_particles = N_particles

        # 1. Radien initialisieren
        self.init_nanoparticle_radius(mean_radius=mean_radius, std_radius=std_radius)

        # 2. Geometrie aufbauen (nutzt unsere neue Pipeline)
        self._rebuild_random_geometry(delta=delta, max_attempts=max_attempts, **packing_kwargs)

    def add_electrodes_to_random_net(self, electrode_positions: List[Tuple[float, float]], electrode_radius: float = 10.0) -> None:
        """
        Attach electrodes to the closest available nanoparticles in a random network.

        The input positions are normalized [-1, 1] and will be scaled to the physical
        bounding box of the packed nanoparticle cluster.

        Parameters
        ----------
        electrode_positions : List of (x, y)
            Normalized coordinates. E.g., (-1, 0) is middle-left, (1, 0) is middle-right.
        electrode_radius : float, optional
            Physical radius of the electrodes to place them just outside the NP cloud.
        """
        if getattr(self, "network_type", None) != "random":
            raise RuntimeError("add_electrodes_to_random_net() requires a random network.")
        if len(electrode_positions) > self.N_particles:
            raise ValueError("Cannot attach more electrodes than nanoparticles.")

        # 1. Physikalische Ausdehnung der Partikelwolke bestimmen
        coords = np.array([self.pos[i] for i in range(self.N_particles)])
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        
        center_x = (max_x + min_x) / 2.0
        center_y = (max_y + min_y) / 2.0
        
        # Halbe Breite/Höhe plus Puffer für die Elektrode
        half_width = (max_x - min_x) / 2.0 + electrode_radius
        half_height = (max_y - min_y) / 2.0 + electrode_radius

        self.N_electrodes = len(electrode_positions)
        electrode_nodes = -np.arange(1, self.N_electrodes + 1)
        self.G.add_nodes_from(electrode_nodes)

        # 2. Für jede Elektrode die physische Position berechnen und ankoppeln
        for n, norm_pos in enumerate(electrode_positions):
            norm_x, norm_y = float(norm_pos[0]), float(norm_pos[1])
            if not (-1.0 <= norm_x <= 1.0 and -1.0 <= norm_y <= 1.0):
                raise ValueError(f"Electrode coords must be within [-1, 1], got {norm_pos}")

            # Skalierung auf echte Nanometer
            phys_x = center_x + norm_x * half_width
            phys_y = center_y + norm_y * half_height
            
            el_node = electrode_nodes[n]
            self.pos[el_node] = (phys_x, phys_y)

            # Finde das nächstgelegene Nanopartikel (euklidischer Abstand)
            dist_to_particles = np.linalg.norm(coords - np.array([phys_x, phys_y]), axis=1)
            closest_np = int(np.argmin(dist_to_particles))

            # Im Graphen verknüpfen
            self.G.add_edge(el_node, closest_np)
            self.G.add_edge(closest_np, el_node)

        # 3. Topologie und Distanzmatrix für die Elektroden updaten
        self._graph_to_net_topology()
        self._update_electrode_distance_matrix()

    def _graph_to_net_topology(self)->None:
        """
        Build the net_topology matrix from the current directed graph (self.G).

        Raises
        ------
        RuntimeError
            If self.G or basic network attributes are not initialized.
        """
        if self.N_particles <= 0 or self.N_junctions <= 0:
            raise RuntimeError("Network attributes (N_particles, N_junctions) not initialized.")

        # Allocate matrix
        net_topology = np.full(
            shape=(self.N_particles, self.N_junctions + 1),
            fill_value=self.NO_CONNECTION,
            dtype=int,
        )

        
        # For each nanoparticle (node indices 0 ... N_particles-1)
        for node in range(self.N_particles):
            neighbor_idx = 1
            for neighbor in self.G.neighbors(node):
                if neighbor >= 0:
                    # Nanoparticle neighbor
                    if neighbor_idx <= self.N_junctions:
                        net_topology[node, neighbor_idx] = int(neighbor)
                        neighbor_idx += 1
                else:
                    # Electrode node (negative index -> 1-based positive electrode number)
                    net_topology[node, 0] = int(-neighbor)

        self.net_topology = net_topology

    def delete_n_junctions(self, n: int) -> None:
        """
        Randomly delete n nanoparticle-nanoparticle junctions (edges), preserving:
        - network connectivity (no bridges/cut-edges are removed)
        - all electrode connectivity (edges connected to electrode-bound NPs are preserved)

        Parameters
        ----------
        n : int
            Number of nanoparticle-nanoparticle junctions to delete.

        Raises
        ------
        ValueError
            If n is negative or more than the number of deletable (non-bridge) edges.
        """
        if n < 0:
            raise ValueError(f"Number of junctions to delete must be non-negative, got {n}")

        if n == 0:
            return

        # Extract subgraph of only nanoparticles (no electrodes)
        G_np = self.G.subgraph([i for i in range(self.N_particles)]).to_undirected()

        # Build list of all possible removable edges
        # Bedingung: Keiner der beiden Knoten darf eine direkte Elektrode in Spalte 0 haben!
        removable = []
        for i in range(self.N_particles):
            if self.net_topology[i, 0] == self.NO_CONNECTION:
                for j in self.net_topology[i, 1:]:
                    if j != self.NO_CONNECTION:
                        j_int = int(j)
                        # Nur hinzufügen, wenn auch j keine Elektrode besitzt und wir die Kante nur 1x betrachten (i < j_int)
                        if i < j_int and self.net_topology[j_int, 0] == self.NO_CONNECTION:
                            removable.append((i, j_int))

        # Find bridges (edges whose removal would disconnect the network)
        bridges = set(nx.bridges(G_np))

        # Only allow deletion of non-bridge edges
        candidates = [edge for edge in removable if edge not in bridges and edge[::-1] not in bridges]
        
        if n > len(candidates):
            raise ValueError(f"Only {len(candidates)} removable (non-bridge) junctions available, cannot delete {n}.")

        # Randomly pick n edges to delete using the class's internal numpy random generator
        idxs = self.rng.choice(len(candidates), size=n, replace=False)
        to_delete = [candidates[i] for i in idxs]

        for i, j in to_delete:
            # Remove from topology matrix (both directions)
            i_idx = np.where(self.net_topology[i, 1:] == j)[0]
            j_idx = np.where(self.net_topology[j, 1:] == i)[0]
            
            if len(i_idx) > 0:
                self.net_topology[i, 1 + i_idx[0]] = self.NO_CONNECTION
            if len(j_idx) > 0:
                self.net_topology[j, 1 + j_idx[0]] = self.NO_CONNECTION
                
            # Remove from NetworkX graph (both directions for DiGraph)
            if self.G.has_edge(i, j):
                self.G.remove_edge(i, j)
            if self.G.has_edge(j, i):
                self.G.remove_edge(j, i)
                            
    # def add_np_to_output(self):
    #     """
    #     Insert a single new nanoparticle in series between the currently connected
    #     nanoparticle and the output electrode (the last electrode in the network).

    #     Steps performed:
    #     1. Identifies the nanoparticle currently connected to the output electrode.
    #     2. Disconnects this nanoparticle from the output electrode.
    #     3. Creates a new nanoparticle.
    #     4. Connects this new nanoparticle between the previously connected nanoparticle and the output electrode.
    #     5. Updates the net_topology matrix, NetworkX graph, and position dictionary.

    #     Raises
    #     ------
    #     RuntimeError
    #         If there are no electrodes defined, or if output electrode is not connected.
    #     """
    #     if not hasattr(self, 'N_electrodes') or self.N_electrodes < 1:
    #         raise RuntimeError("No electrodes defined in the network.")
    #     output_electrode_idx = self.N_electrodes-1

    #     # Find the nanoparticle connected to the output electrode (search net_topology)
    #     matches = np.where(self.net_topology[:, 0] == (output_electrode_idx + 1))[0]
    #     if len(matches) == 0:
    #         raise RuntimeError(f"No nanoparticle is connected to output electrode (index {output_electrode_idx}).")
    #     adj_np = matches[0]
        
    #     prev_particle_count =   self.N_particles
    #     self.N_particles    +=  1

    #     # Prepare new topology row for the new nanoparticle
    #     new_row     = np.full(self.net_topology.shape[1], self.NO_CONNECTION, dtype=int)
    #     new_row[0]  = output_electrode_idx + 1  # Connect to electrode
    #     new_row[1]  = adj_np                    # Connect to the previous adjacent nanoparticle

    #     self.net_topology = np.vstack((self.net_topology, new_row))

    #     # Update the adjacent nanoparticle's connection:
    #     #  - Find the first free neighbor slot and connect it to the new node
    #     #  - Remove its previous electrode connection, if present
    #     free_spots = np.where(self.net_topology[adj_np, :] == self.NO_CONNECTION)[0]
    #     if len(free_spots) == 0:
    #         raise RuntimeError(f"No free neighbor slots in net_topology for nanoparticle {adj_np}.")
    #     self.net_topology[adj_np, free_spots[0]]    = prev_particle_count
    #     self.net_topology[adj_np, 0]                = self.NO_CONNECTION  # Remove old electrode connection

    #     # Update the NetworkX graph
    #     electrode_node = -output_electrode_idx - 1
    #     if self.G.has_edge(adj_np, electrode_node):
    #         self.G.remove_edge(adj_np, electrode_node)
    #     if self.G.has_edge(electrode_node, adj_np):
    #         self.G.remove_edge(electrode_node, adj_np)
    #     self.G.add_node(prev_particle_count)
    #     self.G.add_edge(prev_particle_count, adj_np)
    #     self.G.add_edge(adj_np, prev_particle_count)
    #     self.G.add_edge(prev_particle_count, electrode_node)
    #     self.G.add_edge(electrode_node, prev_particle_count)

    #     # Assign a spatial position to the new nanoparticle (place at electrode for now)
    #     if electrode_node in self.pos:
    #         self.pos[prev_particle_count] = self.pos[electrode_node]
    #     else:
    #         self.pos[prev_particle_count] = (0, 0)  # fallback if no electrode position

    #     # Move the electrode for clarity (especially for lattice layout)
    #     x, y = self.pos[prev_particle_count]
    #     if self.lattice:
    #         # Move electrode to be just outside the grid boundary in a sensible way
    #         if x == self.N_x:
    #             self.pos[electrode_node] = (x + 1, y)
    #         elif x == -1:
    #             self.pos[electrode_node] = (x - 1, y)
    #         elif y == self.N_y:
    #             self.pos[electrode_node] = (x, y + 1)
    #         elif y == -1:
    #             self.pos[electrode_node] = (x, y - 1)
    #     else:
    #         # For random networks, shift electrode right or up for clarity
    #         self.pos[electrode_node] = (x + 0.2, y + 0.2)

    def get_net_topology(self) -> np.ndarray:
        """
        Return a copy of the network topology matrix.

        Returns
        -------
        np.ndarray
            Network topology matrix.
        """
        if self.net_topology.size == 0:
            raise RuntimeError("Topology matrix not defined. Initialize a network first.")
        return self.net_topology.copy()

    def get_graph(self) -> nx.DiGraph:
        """
        Return the NetworkX DiGraph object for the current network.

        Returns
        -------
        nx.DiGraph
        """
        return self.G.copy()
    
    def get_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Return a dictionary mapping node indices to their 2D positions.

        Returns
        -------
        dict
            Dictionary of node positions.
        """
        return dict(self.pos)

    def get_dist_matrix(self) -> np.ndarray:
        """Get the network distance matrix

        Returns
        -------
        np.ndarray
            Array containing network distances [nm]
            Shape: (N_particles, N_particles)
        
        Raises
        ------
        RuntimeError
            If distance matrix hasn't been calculated
        """
        if getattr(self, "dist_matrix", None) is None:
            raise RuntimeError("Distance matrix not calculated. Initialize a network first.")
        return self.dist_matrix.copy()
    
    def get_electrode_dist_matrix(self) -> np.ndarray:
        """Get the electrode distance matrix

        Returns
        -------
        np.ndarray
            Array containing network distances [nm]
            Shape: (N_electrodes, N_particles)
        
        Raises
        ------
        RuntimeError
            If electrode distance matrix hasn't been calculated
        """
        if getattr(self, "electrode_dist_matrix", None) is None:
            raise RuntimeError("Electrode distance matrix not calculated. Attach electrodes first.")
        return self.electrode_dist_matrix.copy()
    
    def get_radius(self) -> np.ndarray:
        """Get the radius of each NP

        Returns
        -------
        np.ndarray
            Array containing nanoparticle radius [nm]
            Shape: (N_particles,)
        
        Raises
        ------
        RuntimeError
            If radius hasn't been calculated
        """
        if self.radius_vals is None:
            raise RuntimeError("Nanoparticle radius not defined. Initialize a network first.")
        return self.radius_vals.copy()
    
    def export_network(self, filepath: str) -> None:
        """
        Export the network configuration to a file.
        
        This method saves the complete geometric and topological state:
        1. Network parameters and type (N_particles, N_electrodes, lattice, etc.)
        2. Network topology matrix
        3. Node positions
        4. Nanoparticle radii
        5. Distance matrices (NP-NP and Electrode-NP)
        
        Parameters
        ----------
        filepath : str
            Path to save the network configuration file (e.g., 'network.npy').
        """
        # Basis-Attribute
        network_data = {
            'network_type': getattr(self, 'network_type', None),
            'lattice': self.lattice,
            'N_particles': self.N_particles,
            'N_electrodes': self.N_electrodes,
            'N_junctions': self.N_junctions,
            'net_topology': self.net_topology, # numpy arrays werden durch pickle nativ unterstützt
            'positions': {str(k): list(v) for k, v in self.pos.items()},
        }
        
        # Gitter-Dimensionen (nur bei Lattice)
        if self.network_type == 'lattice':
            network_data['N_x'] = self.N_x
            network_data['N_y'] = self.N_y
            
        # Geometrische Parameter (falls bereits initialisiert)
        if self.radius_vals is not None:
            network_data['radius_vals'] = self.radius_vals
        if getattr(self, 'dist_matrix', None) is not None:
            network_data['dist_matrix'] = self.dist_matrix
        if getattr(self, 'electrode_dist_matrix', None) is not None:
            network_data['electrode_dist_matrix'] = self.electrode_dist_matrix
            
        np.save(filepath, network_data, allow_pickle=True)

    def import_network(self, filepath: str) -> None:
        """
        Import a network configuration from a file and reconstruct the graph.
        
        Parameters
        ----------
        filepath : str
            Path to the network configuration file.
            
        Raises
        ------
        ValueError
            If the file format is invalid or missing required data.
        """
        try:
            network_data = np.load(filepath, allow_pickle=True).item()
            
            # Basis-Attribute wiederherstellen
            self.network_type = network_data.get('network_type')
            self.lattice = network_data.get('lattice', False)
            self.N_particles = network_data['N_particles']
            self.N_electrodes = network_data['N_electrodes']
            self.N_junctions = network_data['N_junctions']
            
            if self.network_type == 'lattice':
                self.N_x = network_data.get('N_x')
                self.N_y = network_data.get('N_y')
                
            # Arrays wiederherstellen
            self.net_topology = np.array(network_data['net_topology'])
            
            if 'radius_vals' in network_data:
                self.radius_vals = np.array(network_data['radius_vals'])
            if 'dist_matrix' in network_data:
                self.dist_matrix = np.array(network_data['dist_matrix'])
            if 'electrode_dist_matrix' in network_data:
                self.electrode_dist_matrix = np.array(network_data['electrode_dist_matrix'])
                
            # Positionen wiederherstellen (int() versteht auch negative Strings wie "-1")
            self.pos = {int(k): tuple(v) for k, v in network_data['positions'].items()}
                
            # Den Graphen (NetworkX) neu aufbauen
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.N_particles))
            if self.N_electrodes > 0:
                self.G.add_nodes_from(range(-self.N_electrodes, 0))
            
            # Kanten aus der Matrix rekonstruieren
            for node in range(self.N_particles):
                # Elektroden-Kanten (Spalte 0)
                if self.net_topology[node, 0] != self.NO_CONNECTION:
                    electrode = -int(self.net_topology[node, 0])
                    self.G.add_edge(node, electrode)
                    self.G.add_edge(electrode, node)
                    
                # Nanopartikel-Kanten (Spalten 1 bis Ende)
                for neighbor in self.net_topology[node, 1:]:
                    if neighbor != self.NO_CONNECTION:
                        neighbor_idx = int(neighbor)
                        self.G.add_edge(node, neighbor_idx)
                        self.G.add_edge(neighbor_idx, node)
                        
        except Exception as e:
            raise ValueError(f"Failed to import network configuration: {str(e)}")
    
###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    N_x, N_y = 3,3
    electrode_pos = [[0,0],[2,0],[0,2],[2,2]]
    radius = 10.0
    lattice_net = NanoparticleTopology()

    lattice_net.lattice_network(N_x, N_y, radius)
    lattice_net.add_electrodes_to_lattice_net(electrode_pos)

    lattice_pos = lattice_net.get_positions()
    lattice_dist = lattice_net.get_dist_matrix()
    lattice_e_dist = lattice_net.get_electrode_dist_matrix()
    lattice_radius = lattice_net.get_radius()

    print(lattice_pos)
    print(lattice_e_dist)
    print(lattice_radius)