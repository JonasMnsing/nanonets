import numpy as np
from nanonets.topology import NanoparticleTopology
from typing import List, Optional

class NanoparticleElectrostatic:
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
    electrode_type : np.ndarray of str
        Type for each electrode ('constant' or 'floating').
    floating_indices : np.ndarray of int
        Indices of floating electrodes.
    capacitance_matrix : np.ndarray
        Full network capacitance matrix [aF].
    inv_capacitance_matrix : np.ndarray
        Inverse capacitance matrix [1/aF].
    charge_vector : np.ndarray
        Vector of induced NP charges [aC].
    potential_vector : np.ndarray
        Network potential for all nodes [V]
    self_capacitance : np.ndarray
        Self-capacitance per NP [aF].
    electrode_capacitance_matrix : np.ndarray
        Capacitance matrix between electrodes and NPs [aF].

    Physical Constants
    ------------------
    EPSILON_0 : float
        Vacuum permittivity [aF/nm]
    PI : float
        Pi
    ELECTRODE_RADIUS : float
        Electrode radius [nm]

    Notes
    -----
    All capacitances are in attofarads (aF) and charges are in (aC).
    """
    EPSILON_0           = 8.85418781762039e-3  # aF/nm, vacuum permittivity
    PI                  = 3.14159265359
    ELECTRODE_RADIUS    = 10.0  # nm
    
    def __init__(self, topology: NanoparticleTopology, electrode_type: Optional[List[str]] = None) -> None:
        """
        Initialize the NanoparticleElectrostatic network.

        Parameters
        ----------
        topology : NanoparticleTopology
            A fully initialized topology model of the network.
        electrode_type : List[str], optional
            List of electrode types, each element should be 'constant' or 'floating'.
            Length must match number of electrodes to be attached to the network.
            If None, no electrode types are set initially.

        Raises
        ------
        ValueError
             If electrode_type contains values other than 'constant' or 'floating'.
        """
        self.topo = topology

        if self.topo.N_electrodes == 0 and electrode_type is not None:
            raise ValueError("Provided topology has no electrodes, but electrode_types were specified.")

        self.electrode_type = None
        self.floating_indices = np.array([], dtype=int)

        if electrode_type is not None:
            arr = np.array(electrode_type)
            if not np.all(np.isin(arr, ['constant', 'floating'])):
                raise ValueError("electrode_type must contain only 'constant' or 'floating'")
            
            if len(arr) != self.topo.N_electrodes:
                raise ValueError(
                    f"Length of electrode_type ({len(arr)}) must match number of electrodes ({self.topo.N_electrodes})."
                )

        if electrode_type is not None:
            arr = np.array(electrode_type)
            if not np.all(np.isin(arr, ['constant', 'floating'])):
                raise ValueError("electrode_type must contain only 'constant' or 'floating'")
            
            self.electrode_type = arr
            self.floating_indices = np.where(arr == 'floating')[0]

        self.eps_r = None
        self.eps_s = None

        self.capacitance_matrix = None
        self.inv_capacitance_matrix = None
        self.electrode_capacitance_matrix = None

        self.self_capacitance = None
        self.charge_vector = None
        self.potential_vector = None

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
        min_sep = np_radius1 + np_radius2 + self.topo.MIN_NP_NP_DISTANCE
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
        
    def constant_self_capacitance(self, self_cap: float = 0.28) -> None:
        """
        Define the particles self-capacitance array.

        Parameters
        ----------
        self_cap : float
            Capacitance value [aF]

        Raises
        ------
        ValueError
            If eps_s <= 0 or np_radius <= 0
        """
        if self_cap < 0:
            raise ValueError("Self Capacitance must be positive!")

        self.self_capacitance = np.repeat(self_cap, self.topo.N_particles)

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
        short_range : bool, optional
            If true, there are only mutual capacitance values between neighbors
        
        Raises
        ------
        ValueError
            If permittivity parameters are non-positive.
        RuntimeError
            If radii have not been initialized, or if matrix inversion fails.
        """
        if eps_r <= 0 or eps_s <= 0:
            raise ValueError(f"Permittivities must be positive, got eps_r={eps_r}, eps_s={eps_s}")
        
        if self.topo.radius_vals is None:
            raise RuntimeError("Nanoparticle radii not initialized. Initialize a network first.")

        if self.self_capacitance is None:
            raise ValueError(f"Self capacitance must be initialized first.")

        dist_mat = self.topo.get_dist_matrix()
        elec_dist_mat = self.topo.get_electrode_dist_matrix()
        net_topo = self.topo.get_net_topology()
        no_conn = self.topo.NO_CONNECTION

        # Initialize capacitance matrix
        N = self.topo.N_particles
        self.capacitance_matrix = np.zeros((N, N))
        self.eps_r = eps_r
        self.eps_s = eps_s

        # Loop over nanoparticles to fill capacitance matrix
        for i in range(N):
            C_sum = 0.0
            electrode = net_topo[i, 0]
            neighbors = [n for n in net_topo[i, 1:] if n != no_conn]
            
            # Mutual NP-NP capacitances (off-diagonal)
            for j in range(N):
                if i!=j:
                    if short_range:
                        if j in neighbors: # Check if j is neighbor
                            val = self.mutual_capacitance_adjacent_spheres(eps_r, self.topo.radius_vals[i], self.topo.radius_vals[j], dist_mat[i,j])
                            C_sum +=  val
                            self.capacitance_matrix[i, j] = -val
                    else:
                        val = self.mutual_capacitance_adjacent_spheres(eps_r, self.topo.radius_vals[i], self.topo.radius_vals[j], dist_mat[i,j])
                        C_sum += val
                        self.capacitance_matrix[i, j] = -val

            # NP-electrode capacitance (only for constant electrodes)
            for j in range(self.topo.N_electrodes):
                if self.floating_indices.size > 0 and j in self.floating_indices:
                    continue  # skip floating electrodes
                
                if short_range:
                    if (j + 1) == electrode: # Check if electrode is adjacent
                        val = self.mutual_capacitance_adjacent_spheres(eps_r, self.topo.radius_vals[i], self.ELECTRODE_RADIUS, elec_dist_mat[j,i])
                        C_sum += val
                else:
                    val = self.mutual_capacitance_adjacent_spheres(eps_r, self.topo.radius_vals[i], self.ELECTRODE_RADIUS, elec_dist_mat[j,i])
                    C_sum += val
                    
            # Self Capacitance
            C_sum += self.self_capacitance[i]
            self.capacitance_matrix[i, i] = C_sum
        
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

        Parameters
        ----------
        short_range : bool, optional
            If true, there are only mutual capacitance values between neighbors

        Raises
        ------
        RuntimeError
            If required attributes are missing.
        """
        if self.topo.N_particles == 0:
            raise RuntimeError("Network not initialized.")
        if not hasattr(self, 'eps_r') or not hasattr(self, 'eps_s'):
            raise RuntimeError("Physical parameters eps_r, eps_s not set. Call calc_capacitance_matrix first.")
        if self.topo.radius_vals is None:
            raise RuntimeError("Nanoparticle radii not initialized. Initialize a network first.")

        elec_dist_mat = self.topo.get_electrode_dist_matrix()
        net_topo = self.topo.get_net_topology()
        n_elec = self.topo.N_electrodes
        n_parts = self.topo.N_particles

        # Boolean mask: True for constant electrodes, False for floating
        constant_mask = np.ones(n_elec, dtype=bool)
        if self.floating_indices.size > 0:
            constant_mask[self.floating_indices] = 0

        # C_lead[j, i] = C between electrode j and particle i (zero if floating)
        C_lead = np.zeros((n_elec, n_parts))

        constant_indices = np.nonzero(constant_mask)[0]
        for j in constant_indices:
            for i in range(n_parts):
                electrode_conn = net_topo[i, 0]
                if short_range:
                    if (j + 1) == electrode_conn: # Check if electrode is adjacent
                        C_lead[j, i] = self.mutual_capacitance_adjacent_spheres(self.eps_r, self.topo.radius_vals[i], self.ELECTRODE_RADIUS, elec_dist_mat[j, i])
                else:
                    C_lead[j, i] = self.mutual_capacitance_adjacent_spheres(self.eps_r,self.radius_vals[i], self.ELECTRODE_RADIUS,elec_dist_mat[j, i])

        self.electrode_capacitance_matrix = C_lead # shape (N_electrodes, N_particles)

    def load_capacitance_matrix(self, path: str) -> None:
        """
        Load the capacitance matrix (NxN) for the nanoparticle network from a file.

        Parameters
        ----------
        path : str
            File path to the capacitance matrix data.
            
        Raises
        ------
        RuntimeError
            If the matrix cannot be loaded or is singular during inversion.
        """
        try:
            # Load capacitance matrix (supports text formats like .txt or .csv via np.loadtxt)
            self.capacitance_matrix = np.loadtxt(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load capacitance matrix from {path}: {str(e)}")

        # Invert capacitance matrix
        try:
            self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to invert loaded capacitance matrix. Matrix may be singular.") from e
        
    def load_electrode_capacitance_matrix(self, path: str) -> None:
        """
        Load the capacitance matrix between electrodes and nanoparticles from a file,
        and reconstruct the nanoparticle self-capacitances as a reference.

        Parameters
        ----------
        path : str
            File path to the electrode capacitance matrix data.
            
        Raises
        ------
        RuntimeError
            If the main capacitance matrix hasn't been loaded/calculated yet, 
            or if loading fails.
        """
        if self.capacitance_matrix is None:
            raise RuntimeError(
                "Main capacitance matrix must be loaded (via load_capacitance_matrix) "
                "before loading the electrode capacitance matrix."
            )

        try:
            # Load electrode capacitance matrix
            self.electrode_capacitance_matrix = np.loadtxt(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load electrode capacitance matrix from {path}: {str(e)}")

        # Self-capacitances (reconstructed for reference)
        diagonal_ele = np.diag(self.capacitance_matrix)
        off_diagonal = np.sum(self.capacitance_matrix, axis=1) - diagonal_ele
        self.self_capacitance = diagonal_ele + off_diagonal - np.sum(self.electrode_capacitance_matrix, axis=0)
        
    def init_charge_vector(self, voltage_values: np.ndarray) -> None:
        """
        Initialize the nanoparticle charge vector given electrode and gate voltages.

        The resulting vector (self.charge_vector) has shape (N_particles,).
        Convention: `voltage_values` is
            [V_e1, V_e2, ..., V_eN, V_G]
        where N = self.topo.N_electrodes and V_G is the gate voltage.
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
        if getattr(self, 'capacitance_matrix', None) is None:
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")

        voltage_values = np.asarray(voltage_values)
        n_elec = self.topo.N_electrodes

        if voltage_values.shape[0] != n_elec + 1:
            raise ValueError(f"Expected {n_elec + 1} voltage values, got {voltage_values.shape[0]}")

        # Check floating electrode voltages (allow tiny floating point errors)
        if self.floating_indices.size > 0:
            floating_voltages = voltage_values[self.floating_indices]
            if np.any(np.abs(floating_voltages) > 1e-12):
                raise ValueError("Floating electrode voltages must be initialized to zero")
        
        V_e = voltage_values[:n_elec]
        V_g = voltage_values[-1]

        # C_lead.T @ V_e is shape (N_particles,)
        self.charge_vector = self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g

    def init_potential_vector(self, voltage_values: np.ndarray) -> None:
        """
        Initialize the full potential vector for the network, setting electrode and gate voltages.

        The potential vector is organized for fast KMC/tunneling indexing:
        - First N_electrodes entries: electrode voltages (from voltage_values)
        - Remaining N_particles entries: initialized to zero (updated dynamically during simulation)

        Parameters
        ----------
        voltage_values : np.ndarray or list
            Array of shape (N_electrodes + 1,) = [V_e1, V_e2, ..., V_eN, V_G]
            where V_G is the gate voltage.

        Raises
        ------
        ValueError
            If voltage_values length does not match number of electrodes + 1.
        """
        n_elec = self.topo.N_electrodes
        n_parts = self.topo.N_particles
        
        if n_parts == 0:
            raise RuntimeError("Network not initialized. Build topology first.")

        voltage_values = np.asarray(voltage_values)
        if voltage_values.shape[0] != n_elec + 1:
            raise ValueError(f"Expected {n_elec + 1} voltage values, got {voltage_values.shape[0]}.")
        
        self.potential_vector = np.zeros(n_elec + n_parts)
        if n_elec > 0:
            self.potential_vector[0:n_elec] = voltage_values[:-1]

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
        n_elec = self.topo.N_electrodes
        
        if voltage_values.shape[0] != n_elec + 1:
            raise ValueError(f"Expected {n_elec + 1} voltage values, got {voltage_values.shape[0]}.")

        V_e = voltage_values[:n_elec]
        V_g = voltage_values[-1]

        return self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g
    
    def get_charge_vector(self) -> np.ndarray:
        """
        Get a copy of the current charge vector of the network.

        Returns
        -------
        np.ndarray
            Array of induced charge values for each nanoparticle [aC], shape: (N_particles,)

        Raises
        ------
        RuntimeError
            If charge vector has not been initialized.
        """
        if getattr(self, 'charge_vector', None) is None:
            raise RuntimeError("Charge vector not initialized. Call init_charge_vector first.")
        return self.charge_vector.copy()

    def get_potential_vector(self) -> np.ndarray:
        """
        Returns
        -------
        potential_vector : ndarray
            Potential values for electrodes and nanoparticles [V]
            
        Raises
        ------
        RuntimeError
            If potential vector has not been initialized.
        """
        if getattr(self, 'potential_vector', None) is None:
            raise RuntimeError("Potential vector not initialized. Call init_potential_vector first.")
        return self.potential_vector.copy()
    
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Get a copy of the network capacitance matrix.

        Returns
        -------
        np.ndarray
            Array containing network capacitance values [aF], shape: (N_particles, N_particles)
            
        Raises
        ------
        RuntimeError
            If capacitance matrix hasn't been calculated.
        """
        if getattr(self, 'capacitance_matrix', None) is None:
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")
        return self.capacitance_matrix.copy()
    
    def get_electrode_capacitance_matrix(self) -> np.ndarray:
        """
        Get a copy of the electrode capacitance matrix.

        Returns
        -------
        np.ndarray
            Array containing electrode capacitance values [aF], shape: (N_electrodes, N_particles)
            
        Raises
        ------
        RuntimeError
            If electrode capacitance matrix hasn't been calculated.
        """
        if getattr(self, 'electrode_capacitance_matrix', None) is None:
            raise RuntimeError("Electrode capacitance matrix not calculated. Call calc_electrode_capacitance_matrix first.")
        return self.electrode_capacitance_matrix.copy()
    
    def get_self_capacitance(self) -> np.ndarray:
        """
        Get a copy of the self-capacitance array for each nanoparticle.

        Returns
        -------
        np.ndarray
            Array containing self-capacitance values [aF], shape: (N_particles,)
            
        Raises
        ------
        RuntimeError
            If self-capacitance values haven't been calculated.
        """
        if getattr(self, 'self_capacitance', None) is None:
            raise RuntimeError("Self-capacitance not calculated. Call constant_self_capacitance first.")
        return self.self_capacitance.copy()
        
    def get_inv_capacitance_matrix(self) -> np.ndarray:
        """
        Get a copy of the inverse capacitance matrix.

        Returns
        -------
        np.ndarray
            Inverse of capacitance matrix [1/aF], shape: (N_particles, N_particles)
            
        Raises
        ------
        RuntimeError
            If inverse capacitance matrix hasn't been calculated.
        """
        if getattr(self, 'inv_capacitance_matrix', None) is None:
            raise RuntimeError("Inverse capacitance matrix not calculated. Call calc_capacitance_matrix first.")
        return self.inv_capacitance_matrix.copy()