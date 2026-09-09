import numpy as np
import networkx as nx
from . import topology
from typing import List, Optional
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
from itertools import combinations

class NanoparticleElectrostatic(topology.NanoparticleTopology):
    """
    Extends NanoparticleTopology with electrostatic modeling and physical parameters
    for 2D nanoparticle networks.

    This class enables computation of capacitance matrices, induced charges, 
    polydisperse packing, and physically accurate (planar) network geometries.
    Supports both regular lattice and random (Delaunay) planar networks,
    with advanced handling of boundary conditions for 'constant' (voltage-biased)
    and 'floating' electrodes.

    Attributes
    ----------
    Inherited from NanoparticleTopology:
        N_particles : int
            Number of nanoparticles.
        N_electrodes : int
            Number of electrodes.
        N_junctions : int
            Number of junctions per nanoparticle (target/average).
        G : nx.DiGraph
            NetworkX directed graph object.
        pos : dict
            Mapping of node indices to 2D positions.
        net_topology : np.ndarray
            Matrix encoding connections and electrode links.

    Added by this class:
        electrode_type : np.ndarray of str
            Type for each electrode ('constant' or 'floating').
        floating_indices : np.ndarray of int
            Indices of floating electrodes.
        radius_vals : np.ndarray of float
            Radii of nanoparticles [nm].
        capacitance_matrix : np.ndarray
            Full network capacitance matrix [aF].
        inv_capacitance_matrix : np.ndarray
            Inverse capacitance matrix [1/aF].
        charge_vector : np.ndarray
            Vector of induced NP charges [aC].
        self_capacitance : np.ndarray
            Self-capacitance per NP [aF].
        electrode_capacitance_matrix : np.ndarray
            Capacitance matrix between electrodes and NPs [aF].
        dist_matrix : np.ndarray
            Pairwise NP-NP distance matrix [nm].
        electrode_dist_matrix : np.ndarray
            Pairwise NP-electrode distance matrix [nm].

    Physical Constants
    ------------------
        EPSILON_0 : float
            Vacuum permittivity [aF/nm]
        PI : float
            Pi
        ELECTRODE_RADIUS : float
            Electrode radius [nm]
        MIN_NP_NP_DISTANCE : float
            Minimum center-to-center distance between NPs [nm]
        MIN_NP_RADIUS : float
            Minimum allowed NP radius [nm]

    Methods
    -------
        init_nanoparticle_radius(mean, std)
            Initialize all NP radii (with optional polydispersity).
        update_nanoparticle_radius(indices, mean, std)
            Update radii for selected NPs.
        pack_planar_circles(...)
            Adjust NP positions to remove overlaps, keeping network planar.
        calc_capacitance_matrix(eps_r, eps_s)
            Calculate the full NP network capacitance matrix.
        calc_electrode_capacitance_matrix()
            Compute NP-electrode and self-capacitance terms.
        init_charge_vector(voltage_values)
            Set induced NP charges based on electrode voltages.
        get_charge_vector()
            Return induced NP charge vector [aC].
        get_capacitance_matrix()
            Return full NP capacitance matrix [aF].
        get_self_capacitance()
            Return self-capacitance per NP [aF].
        get_dist_matrix()
            Return NP-NP distance matrix [nm].
        delete_n_junctions(n)
            Randomly remove n junctions (ensuring network remains connected).

    Notes
    -----
    - All distances/radii in nanometers (nm); all capacitances in attofarads (aF).
    - The class enforces physical plausibility (no overlaps, realistic capacitances).
    """
    # Physical constants
    EPSILON_0           = 8.85418781762039e-3  # aF/nm, vacuum permittivity
    PI                  = 3.14159265359
    ELECTRODE_RADIUS    = 10.0  # nm
    MIN_NP_NP_DISTANCE  = 1.0   # nm
    MIN_NP_RADIUS       = 5.0   # nm, minimum allowed nanoparticle radius
    
    def __init__(self, electrode_type: Optional[List[str]] = None, seed: Optional[int] = None) -> None:
        """
        Initialize the NanoparticleElectrostatic network.

        Parameters
        ----------
        electrode_type : List[str], optional
            List of electrode types, each element should be 'constant' or 'floating'.
            Length must match number of electrodes to be attached to the network.
            If None, no electrode types are set initially.
        seed : int, optional
            Seed for the random number generator (for reproducibility).

        Raises
        ------
        ValueError
             If electrode_type contains values other than 'constant' or 'floating'.
        """
        super().__init__(seed)
        self.electrode_type     = None
        self.floating_indices   = np.array([], dtype=int)
        if electrode_type is not None:
            arr = np.array(electrode_type)
            if not np.all(np.isin(arr, ['constant', 'floating'])):
                raise ValueError("electrode_type must contain only 'constant' or 'floating'")
            self.electrode_type     = arr
            self.floating_indices   = np.where(arr == 'floating')[0]

    def mutual_capacitance_adjacent_spheres(self, eps_r: float, np_radius1: float, np_radius2: float, distance: float, N_sum: int = 50) -> float:
        """
        Compute the mutual capacitance between two adjacent spherical nanoparticles
        separated by an insulator, using an exact series solution.

        The capacitance is computed in attofarads [aF]. All distances are in nanometers [nm].

        Parameters
        ----------
        eps_r : float
            Relative permittivity of the insulating material between spheres (dimensionless)
        np_radius1 : float
            Radius of first sphere [nm]
        np_radius2 : float
            Radius of second sphere [nm]
        distance : float
            Center-to-center distance between spheres [nm]
        N_sum : int, optional
            Number of terms in the series expansion (default: 50)

        Returns
        -------
        float
            Capacitance value [aF]
        
        Raises
        ------
        ValueError
            If any parameter is non-positive.

        Notes
        -----
        The function enforces a minimum center-to-center distance to avoid unphysical overlaps.
        """

        if eps_r <= 0:
            raise ValueError(f"Relative permittivity eps_r must be positive, got {eps_r}.")
        if np_radius1 <= 0 or np_radius2 <= 0:
            raise ValueError(f"Radii must be positive, got {np_radius1} and {np_radius2}.")
        if distance <= 0:
            raise ValueError(f"Distance must be positive, got {distance}.")
        if N_sum < 1:
            raise ValueError("N_sum must be at least 1.")
        
        # Minimum separation to avoid overlap
        min_sep = np_radius1 + np_radius2 + self.MIN_NP_NP_DISTANCE
        if distance < min_sep:
            distance = min_sep

        factor  = 4 * self.PI * self.EPSILON_0 * eps_r * (np_radius1 * np_radius2) / distance
        arg     = (distance**2 - np_radius1**2 - np_radius2**2) / (2 * np_radius1 * np_radius2)
        U_val   = np.arccosh(arg)
        if U_val * (N_sum + 1) > 700:
            # Use the exponential approximation to prevent overflow
            s = np.sum([2.0 * np.exp(-n * U_val) for n in range(1, N_sum + 1)])
        else:
            # Use the original exact series solution
            s = np.sum([1.0 / np.sinh(n * U_val) for n in range(1, N_sum + 1)])
        # s       = np.sum([1.0 / np.sinh(n * U_val) for n in range(1 ,N_sum + 1)])
        cap     = factor * np.sinh(U_val) * s

        return cap
        
    def self_capacitance_sphere(self, eps_s: float, np_radius: float, h_oxide: float = np.inf, N_sum: int = 50) -> float:
        """
        Compute the self-capacitance of a single spherical nanoparticle in an infinite homogeneous dielectric.

        Parameters
        ----------
        eps_s : float
            Relative permittivity of the surrounding medium [dimensionless]
        np_radius : float
            Radius of the nanoparticle [nm]
        h_oxide : float
            Oxide height [nm]
        N_sum : int, optional
            Number of terms in the series expansion (default: 50)
        
        Returns
        -------
        float
            Capacitance [aF]

        Raises
        ------
        ValueError
            If eps_s <= 0 or np_radius <= 0
        """
        if eps_s <= 0:
            raise ValueError(f"Relative permittivity eps_s must be positive, got {eps_s}.")
        if np_radius <= 0:
            raise ValueError(f"Radius must be positive, got {np_radius}.")
        if h_oxide <= 0:
            raise ValueError(f"Oxide thickness (h_oxide) must be strictly positive to avoid short-circuiting to the substrate, got {h_oxide}.")
        
        factor  = 4 * self.PI * self.EPSILON_0 * eps_s
        if h_oxide == np.inf:
            cap     = factor * np_radius
        else:
            h_oxide = np_radius + h_oxide
            alpha   = np.arccosh(h_oxide/np_radius)
            cap     = factor * np_radius * np.sinh(alpha)*np.sum([1/np.sinh(n*alpha) for n  in range(1, N_sum + 1)])

        # return cap
        return 0.28
        # return 0.07
        # return 0.0
                       
    

    def create_packing_graph(self, delta: float = 0.5):
        """
        Creates a directed graph from circle packing positions.
        Nodes are connected if their physical distance is within a tolerance range.
        
        Parameters
        ----------
        delta : float, optional
            Connection tolerance. Two particles connect if dist < (r_i + r_j + d_min + delta).
        """
        
        is_connected = False
        n_packs = 0
        while ((not is_connected) and (n_packs < 10)):
            self.G = nx.DiGraph()
            
            # 1. Add all nodes explicitly (in case some have no connections)
            # We can store attributes like position and radius in the node for later use
            for i in range(self.N_particles):
                self.G.add_node(i)
            
            # 2. Check all pairs for connections
            # We loop j > i to avoid duplicate checks, then add edges in both directions
            for i in range(self.N_particles):
                pos_i = np.array(self.pos[i])
                r_i = self.radius_vals[i]
                
                for j in range(i + 1, self.N_particles):
                    pos_j = np.array(self.pos[j])
                    r_j = self.radius_vals[j]
                    
                    # Calculate Euclidean distance
                    dist = np.linalg.norm(pos_i - pos_j)
                    
                    # The Connection Condition
                    # "Touching" logic: Sum of radii + Buffer + Tolerance
                    contact_threshold = r_i + r_j + self.MIN_NP_NP_DISTANCE + delta
                    
                    if dist <= contact_threshold:
                        # Add directed edges both ways (symmetric connection)
                        self.G.add_edge(i, j)
                        self.G.add_edge(j, i)
        
            # Check connectivity
            is_connected = nx.is_strongly_connected(self.G)
            if not is_connected:
                self.pack_circles()
                n_packs += 1
        
        self.N_junctions = max([val for (node, val) in self.G.out_degree()]) + 1

    def pack_lattice(self):
        """
        Scales the initial position of each NP in a lattice so that they don't overlap and are spaced by a minimum distance.
        The algorithm considers the fixed radius of each NP. Does not work for disorderd topologies and disordered NP sizes.
        """
        
        r_val = self.radius_vals[0]
        
        for n in range(self.N_particles-1):
            self.pos[n] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[n][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[n][1])
        self.pos[self.N_particles-1] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[self.N_particles-1][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*(self.pos[self.N_particles-1][1]-1) + 2*r_val + self.MIN_NP_NP_DISTANCE)
        for e in range(1,self.N_electrodes):
            self.pos[-e] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[-e][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[-e][1])
        self.pos[-self.N_electrodes] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[-self.N_electrodes][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*(self.pos[-self.N_electrodes][1]-1) + 2*r_val + self.MIN_NP_NP_DISTANCE)

        dist_matrix = np.zeros(shape=(self.N_particles,self.N_particles))
        for i in range(self.N_particles):
            for j in range(self.N_particles):
                dist_matrix[i,j] = np.sqrt((self.pos[i][0]-self.pos[j][0])**2 + (self.pos[i][1]-self.pos[j][1])**2)
        self.dist_matrix = dist_matrix

        el_dist = np.zeros((self.N_electrodes, self.N_particles))
        for i in range(1, self.N_electrodes + 1):
            for j in range(self.N_particles):
                el_dist[i - 1, j] = np.sqrt((self.pos[-i][0]-self.pos[j][0])**2 + (self.pos[-i][1]-self.pos[j][1])**2)
        self.electrode_dist_matrix = el_dist

    def calc_capacitance_matrix(self, eps_r: float = 2.6, eps_s: float = 3.9, short_range: bool = True)->None:
        """
        Calculate the capacitance matrix (NxN) for the nanoparticle network.

        For each nanoparticle:
        - The off-diagonal elements are minus the mutual capacitance to all other nanoparticles.
        - The diagonal element is the sum of all mutual capacitances to other NPs, 
        capacitances to (constant-potential) electrodes, and the nanoparticle's self-capacitance.

        Parameters
        ----------
        eps_r : float, optional
            Relative permittivity of insulating material between nanoparticles (default: 2.6)
        eps_s : float, optional
            Relative permittivity of the environment (e.g., oxide layer) for self-capacitance (default: 3.9)
        
        Raises
        ------
        ValueError
            If permittivity parameters are non-positive.
        RuntimeError
            If radii have not been initialized, or if matrix inversion fails.
        """
        if eps_r <= 0 or eps_s <= 0:
            raise ValueError(f"Permittivities must be positive, got eps_r={eps_r}, eps_s={eps_s}")
        if not hasattr(self, 'radius_vals'):
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")
        if not hasattr(self, 'dist_matrix') or not hasattr(self, 'electrode_dist_matrix'):
            raise RuntimeError("Distance matrices not initialized. Call pack_planar_circles first.")

        # Initialize capacitance matrix
        self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
        self.eps_r              = eps_r
        self.eps_s              = eps_s

        # Loop over nanoparticles to fill capacitance matrix
        for i in range(self.N_particles):
            C_sum = 0.0
            # Get adjacent neighbors
            electrode = self.net_topology[i,0]
            neighbors = self.net_topology[i,1:].copy()
            # Mutual NP-NP capacitances (off-diagonal)
            for j in range(self.N_particles):
                if i!=j:
                    if short_range:
                        if j in neighbors: # Check if j is neighbor
                            val     =   self.mutual_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[j], self.dist_matrix[i,j])
                            C_sum   +=  val
                            self.capacitance_matrix[i,j] = -val
                    else:
                        val     =   self.mutual_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[j], self.dist_matrix[i,j])
                        C_sum   +=  val
                        self.capacitance_matrix[i,j] = -val

            # NP-electrode capacitance (only for constant electrodes)
            for j in range(self.N_electrodes):
                if hasattr(self, 'floating_indices') and j in self.floating_indices:
                    continue  # skip floating electrodes
                
                if short_range:
                    if j+1 == electrode: # Check if electrode is adjacent
                        val     =   self.mutual_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS, self.electrode_dist_matrix[j,i])
                        C_sum   +=  val
                else:
                    val     =   self.mutual_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS, self.electrode_dist_matrix[j,i])
                    C_sum   +=  val
                    
            # Self Capacitance
            C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i])
            self.capacitance_matrix[i,i] = C_sum
        
        # Invert capacitance matrix
        try:
            self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e
    
    def calc_electrode_capacitance_matrix(self, short_range: bool = True):
        """
        Calculate the capacitance matrix between electrodes and nanoparticles.

        For each electrode (constant-potential only), calculates its mutual capacitance
        to every nanoparticle using the specified geometry and physical parameters.

        - Skips floating electrodes (as they are not held at constant potential).
        - Stores result as self.electrode_capacitance_matrix (shape: N_electrodes x N_particles)
        - Stores self-capacitances as self.self_capacitance (length: N_particles)

        Raises
        ------
        RuntimeError
            If required attributes are missing.
        """

        if not hasattr(self, 'N_electrodes') or not hasattr(self, 'N_particles'):
            raise RuntimeError("Network/electrodes not initialized.")
        if not hasattr(self, 'floating_indices'):
            raise RuntimeError("Floating electrode indices not set.")
        if not hasattr(self, 'eps_r') or not hasattr(self, 'eps_s'):
            raise RuntimeError("Physical parameters eps_r, eps_s not set. Call calc_capacitance_matrix first.")
        if not hasattr(self, 'radius_vals') or not hasattr(self, 'electrode_dist_matrix'):
            raise RuntimeError("Required geometry (radius_vals, electrode_dist_matrix) not set. Run packing and init radii.")

        # Boolean mask: True for constant electrodes, False for floating
        constant_mask = np.ones(self.N_electrodes, dtype=bool)
        constant_mask[self.floating_indices] = 0

        # C_lead[j, i] = C between electrode j and particle i (zero if floating)
        C_lead = np.zeros((self.N_electrodes, self.N_particles))
        for j in np.nonzero(constant_mask)[0]:
            for i in range(self.N_particles):
                if short_range:
                    if j+1 == self.net_topology[i,0]: # Check if electrode is adjacent
                        C_lead[j, i] = self.mutual_capacitance_adjacent_spheres(self.eps_r,self.radius_vals[i],
                                                                                self.ELECTRODE_RADIUS,self.electrode_dist_matrix[j, i])
                else:
                    C_lead[j, i] = self.mutual_capacitance_adjacent_spheres(self.eps_r,self.radius_vals[i],
                                                                                self.ELECTRODE_RADIUS,self.electrode_dist_matrix[j, i])

        # Self-capacitances (for reference)
        C_self = np.array([self.self_capacitance_sphere(self.eps_s, r) for r in self.radius_vals])
        self.self_capacitance               = C_self        # shape (N_particles,)
        self.electrode_capacitance_matrix   = C_lead        # shape (N_electrodes, N_particles)

    def load_capacitance_matrix(self, path: str)->None:
        """
        Load the capacitance matrix (NxN) for the nanoparticle network from path.

        Parameters
        ----------
        path : str
            File Path
        """

        # Load capacitance matrix
        self.capacitance_matrix = np.loadtxt(path)

        # Invert capacitance matrix
        try:
            self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e
        
    def load_electrode_capacitance_matrix(self, path: str)->None:
        """
        Load the capacitance matrix between electrodes and nanoparticles from path.

        Parameters
        ----------
        path : str
            File Path
        """

        # Load capacitance matrix
        self.electrode_capacitance_matrix = np.loadtxt(path)

        # Self-capacitances (for reference)
        diagonal_ele = np.diag(self.capacitance_matrix)
        off_diagonal = np.sum(self.capacitance_matrix,axis=1) - diagonal_ele
        self.self_capacitance = diagonal_ele + off_diagonal - np.sum(self.electrode_capacitance_matrix,axis=0)
        
    def init_charge_vector(self, voltage_values: np.ndarray) -> None:
        """
        Initialize the nanoparticle charge vector given electrode and gate voltages.

        The resulting vector (self.charge_vector) has shape (N_particles,).
        Convention: `voltage_values` is
            [V_e1, V_e2, ..., V_eN, V_G]
        where N = self.N_electrodes and V_G is the gate voltage.
        - Floating electrode voltages must be zero.

        Parameters
        ----------
        voltage_values : np.ndarray
            1D array of shape (N_electrodes + 1,) with electrode and gate voltages.

        Raises
        ------
        RuntimeError
            If capacitance matrix has not been calculated.
        ValueError
            If input length is wrong or floating electrode voltages are nonzero.
        """
        if not hasattr(self, 'capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")

        voltage_values = np.asarray(voltage_values)
        if voltage_values.shape[0] != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {voltage_values.shape[0]}")

        # Check floating electrode voltages (allow tiny floating point errors)
        floating_voltages = voltage_values[self.floating_indices]
        if np.any(np.abs(floating_voltages) > 1e-12):
            raise ValueError("Floating electrode voltages must be initialized to zero")
        
        V_e = voltage_values[:self.N_electrodes]
        V_g = voltage_values[-1]

        # C_lead.T @ V_e is shape (N_particles,)
        self.charge_vector = self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g

    def get_charge_vector_offset(self, voltage_values : np.ndarray) -> np.ndarray:
        """
        Compute the charge vector offset induced by the provided electrode and gate voltages.

        This method calculates the induced charge on each nanoparticle as a result of changes
        in electrode and gate voltages, without reinitializing the internal charge vector.
        Useful for rapid recalculation when only external voltages change.

        Parameters
        ----------
        voltage_values : np.ndarray
            1D array, shape (N_electrodes + 1,): [V_e1, V_e2, ..., V_eN, V_G] where V_G is the gate.

        Returns
        -------
        np.ndarray
            Charge offset vector (length N_particles) [aC], representing the change in nanoparticle charges.

        Raises
        ------
        ValueError
            If the input array is the wrong shape.
        """
        voltage_values = np.asarray(voltage_values)
        if voltage_values.shape[0] != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {voltage_values.shape[0]}.")

        V_e = voltage_values[:self.N_electrodes]
        V_g = voltage_values[-1]

        return self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g
    
    # TODO Testing
    def delete_n_junctions(self, n: int) -> None:
        """
        Randomly delete n nanoparticle-nanoparticle junctions (edges), preserving:
        - network connectivity (no bridges/cut-edges are removed)
        - all electrode connectivity

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

        # Extract subgraph of only nanoparticles (no electrodes)
        G_np = self.G.subgraph([i for i in range(self.N_particles)]).to_undirected()

        # Build list of all possible removable edges (no self-loops)
        removable = []
        for i in range(self.N_particles):
            if self.net_topology[i, 0] == self.NO_CONNECTION:
                for j in self.net_topology[i, 1:]:
                    if j != self.NO_CONNECTION and i < int(j):
                        removable.append((i, int(j)))

        # Find bridges (edges whose removal would disconnect the network)
        bridges = set(nx.bridges(G_np))

        # Only allow deletion of non-bridge, non-self-loop edges
        candidates = [edge for edge in removable if edge not in bridges and edge[::-1] not in bridges]
        if n > len(candidates):
            raise ValueError(f"Only {len(candidates)} removable (non-bridge) junctions available, cannot delete {n}.")

        # Randomly pick n edges to delete
        idxs = self.rng.choice(len(candidates), size=n, replace=False)
        to_delete = [candidates[i] for i in idxs]

        for i, j in to_delete:
            # Remove from topology matrix
            i_idx = np.where(self.net_topology[i, 1:] == j)[0]
            j_idx = np.where(self.net_topology[j, 1:] == i)[0]
            if len(i_idx) > 0:
                self.net_topology[i, 1 + i_idx[0]] = self.NO_CONNECTION
            if len(j_idx) > 0:
                self.net_topology[j, 1 + j_idx[0]] = self.NO_CONNECTION
            # Remove from graph (both directions)
            if self.G.has_edge(i, j):
                self.G.remove_edge(i, j)
            if self.G.has_edge(j, i):
                self.G.remove_edge(j, i)

    def get_charge_vector(self) -> np.ndarray:
        """
        Get the current charge vector of the network.

        Returns
        -------
        np.ndarray
            Array of induced charge values for each nanoparticle [aC], shape: (N_particles,)

        Raises
        ------
        RuntimeError
            If charge vector has not been initialized (call init_charge_vector() first).
        """
        if not hasattr(self, 'charge_vector'):
            raise RuntimeError("Charge vector not initialized. Call init_charge_vector first.")
        return self.charge_vector
    
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Get the network capacitance matrix.

        Returns
        -------
        np.ndarray
            Array containing network capacitance values [aF]
            Shape: (N_particles, N_particles)
            
        Raises
        ------
        RuntimeError
            If capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")
        return self.capacitance_matrix
    
    def get_electrode_capacitance_matrix(self) -> np.ndarray:
        """
        Get the electrode capacitance matrix.

        Returns
        -------
        np.ndarray
            Array containing electrode(lead) capacitance values [aF]
            Shape: (N_electrodes, N_particles)
            
        Raises
        ------
        RuntimeError
            If electrode capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'electrode_capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_electrode_capacitance_matrix first.")
        return self.electrode_capacitance_matrix
    
    def get_self_capacitance(self) -> np.ndarray:
        """Get the self capacitance of each NP.

        Returns
        -------
        np.ndarray
            Array containing self capacitance values [aF]
            Shape: (N_particles,)
            
        Raises
        ------
        RuntimeError
            If self capacitance values hasn't been calculated
        """
        if not hasattr(self, 'self_capacitance'):
            raise RuntimeError("Self Capacitance not calculated. Call calc_electrode_capacitance_matrix first.")
        return self.self_capacitance
        
    def get_inv_capacitance_matrix(self) -> np.ndarray:
        """Get the inverse capacitance matrix.

        Returns
        -------
        np.ndarray
            Inverse of capacitance matrix [1/aF]
            Shape: (N_particles, N_particles)
            
        Raises
        ------
        RuntimeError
            If inverse capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'inv_capacitance_matrix'):
            raise RuntimeError("Inverse capacitance matrix not calculated. Call calc_capacitance_matrix first.")
        return self.inv_capacitance_matrix

    
    
    # def calc_capacitance_matrix(self, eps_r: float = 2.6, eps_s: float = 3.9)->None:
    #     """
    #     Calculate the capacitance matrix of the nanoparticle network.

    #     Parameters
    #     ----------
    #     eps_r : float
    #         Relative permittivity of insulating material between nanoparticles
    #     eps_s : float
    #         Relative permittivity of insulating environment (oxide layer)
            
    #     Raises
    #     ------
    #     ValueError
    #         If physical parameters are invalid
    #     RuntimeError
    #         If matrix inversion fails
    #     """
    #     if not hasattr(self, 'radius_vals'):
    #         raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")

    #     # Initialize capacitance matrix
    #     self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
    #     self.eps_r              = eps_r
    #     self.eps_s              = eps_s

    #     # Loop over each particle to calculate capacitance contributions
    #     for i in range(self.N_particles):
    #         C_sum = 0.0
    #         # Iterate over the junctions of the current nanoparticle
    #         for j in range(self.N_junctions+1):
    #             neighbor = self.net_topology[i,j]

    #             # Skip if the neighbor is invalid
    #             if neighbor == self.NO_CONNECTION:
    #                 continue

    #             # Capacitance with the electrode (j == 0)
    #             if (j == 0):
    #                 if neighbor-1 not in self.floating_indices:
    #                     # Add electrode capacitance (using mutual capacitance formula)
    #                     C_sum += self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)

    #             else:
    #                 # Calc mutual capacitance
    #                 val = self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[int(neighbor)])
    #                 self.capacitance_matrix[i,int(neighbor)] = -val
    #                 C_sum += val

    #         # Self Capacitance
    #         C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i])
            
    #         # Set diagonal element (total capacitance)
    #         self.capacitance_matrix[i,i] = C_sum
        
    #     # Calculate the inverse capacitance matrix
    #     try:
    #         self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
    #     except np.linalg.LinAlgError as e:
    #         raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e

    # def init_charge_vector(self, voltage_values: np.array)->None:
    #     """
    #     Initialize the charge vector based on electrode voltages.

    #     Parameters
    #     ----------
    #     voltage_values : array
    #        Electrode voltages as np.array([V_e1, V_e2, V_e3, ..., V_G])
    #        where V_G is the gate voltage

    #     Raises
    #     ------
    #     ValueError
    #         If voltage array length doesn't match number of electrodes + 1
    #         If floating electrode voltages are non-zero
    #     RuntimeError
    #         If capacitance matrix hasn't been calculated
    #     """
    #     if not hasattr(self, 'capacitance_matrix'):
    #         raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")

    #     if len(voltage_values) != self.N_electrodes + 1:
    #         raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {len(voltage_values)}")

    #     floating_voltages = [voltage_values[i] for i in self.floating_indices]
    #     if any(v != 0 for v in floating_voltages):
    #         raise ValueError("Floating electrode voltages must be initialized to zero")

    #     # Initialize charge vector
    #     self.charge_vector = np.zeros(self.N_particles)

    #     # Iterate over all nanoparticles
    #     for i in range(self.N_particles):
    #         electrode_index = int(self.net_topology[i,0] - 1)

    #         if self.net_topology[i,0] != self.NO_CONNECTION:
    #             # If connected to an electrode, calculate charge from electrode voltage
    #             if electrode_index not in self.floating_indices:
    #                 C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)
    #             else:
    #                 C_lead  = 0.0
                
    #             C_self = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             self.charge_vector[i] = voltage_values[electrode_index] * C_lead + voltage_values[-1] * C_self
            
    #         else:
    #             # If not connected to an electrode
    #             C_self = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             self.charge_vector[i] = voltage_values[-1] * C_self

    # def get_charge_vector_offset(self, voltage_values : np.array)->np.array:
    #     """
    #     Calculate the charge vector offset induced by electrode voltages.
        
    #     This method computes the charges induced on nanoparticles by the electrodes
    #     without reinitializing the entire charge vector. This is useful when you want
    #     to update only the portion of charges that changes due to external voltage changes.

    #     Parameters
    #     ----------
    #     voltage_values : np.array
    #         Electrode voltages as np.array([V_e1, V_e2, V_e3, ..., V_G])
    #         where V_G is the gate voltage

    #     Returns
    #     -------
    #     np.array
    #         Charge offset vector [aC] representing charges induced by electrodes
    #     """

    #     offset = np.zeros(self.N_particles)

    #     # For each charge
    #     for i in range(self.N_particles):
    #         electrode_index = int(self.net_topology[i,0] - 1)

    #         if self.net_topology[i,0] != self.NO_CONNECTION:
    #             # If connected to an electrode, calculate charge from electrode voltage
    #             if electrode_index not in self.floating_indices:
    #                 C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)
    #             else:
    #                 C_lead  = 0.0
                
    #             C_self      = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             offset[i]   = voltage_values[electrode_index] * C_lead + voltage_values[-1] * C_self
            
    #         else:
    #             # If not connected to an electrode
    #             C_self      = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             offset[i]   = voltage_values[-1] * C_self

    #     return offset