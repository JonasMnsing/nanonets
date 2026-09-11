import numpy as np
import networkx as nx
from nanonets.electrostatic import NanoparticleElectrostatic
from typing import Tuple, List, Optional
from scipy.linalg import eig

class NanoparticleTunneling:
    """
    Simulates single-electron tunneling dynamics in nanoparticle networks,
    operating on a provided NanoparticleElectrostatic instance via composition.

    This class extends NanoparticleElectrostatic to add functionality for:
    - Tunneling events (NP-NP, NP-electrode)
    - Resistive junctions (static or dynamic)
    - Temperature effects on tunneling
    - Efficient bookkeeping of event indices for KMC and rate equation methods

    Indexing Scheme
    ---------------
    - Nanoparticles are indexed from `self.N_electrodes` to `self.N_electrodes + self.N_particles - 1`
    - Electrodes are indexed from `0` to `self.N_electrodes - 1`
    - Tunneling events are indexed as (i → j) using arrays `adv_index_rows`, `adv_index_cols`
    - All nodes (NPs and electrodes) can be mapped to a dense index for Laplacian/conductance calculations

    Physical Constants (class attributes)
    -------------------------------------
    ELE_CHARGE_A_C : float
        Elementary charge [attoCoulombs, aC = 1e-18 C]
    ELE_CHARGE_C : float
        Elementary charge [C]
    KB_AJ_PER_K : float
        Boltzmann constant [aJ/K = 1e-18 J/K]
    KB_EV_PER_K : float
        Boltzmann constant [eV/K]
    MIN_RESISTANCE_MOHM : float
        Minimum allowed tunnel resistance [MΩ]

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles in the network
    N_electrodes : int
        Number of electrodes
    N_junctions : int
        Number of nearest-neighbor junctions per nanoparticle (max degree)
    net_topology : np.ndarray
        Network topology matrix (from parent class)
    adv_index_rows : np.ndarray
        Origin node indices for all possible tunnel events (i→j)
    adv_index_cols : np.ndarray
        Target node indices for all possible tunnel events (i→j)
    const_capacitance_values : np.ndarray
        Precomputed free energy terms for tunneling events [(aC)^2/aF]
    resistances : np.ndarray
        1D array of tunnel resistances [MΩ], one per tunneling event (always undirected)
    conductance_matrix : np.ndarray
        Network conductance (Laplacian) matrix [1/MΩ], including all NPs and electrodes

    Notes
    -----
    - All units are SI (capacitance in aF, charge in aC, resistance in MΩ, temperature in aJ/K).
    - All tunnel resistances are always treated as undirected (R_ij = R_ji).
    - Index conventions are consistent for fast, vectorized simulation.
    - Intended for use in kinetic Monte Carlo, stochastic, or master equation modeling of nanoparticle networks.
    """
    
    # Physical constants
    ELE_CHARGE_A_C      = 0.160217662       # [aC] (attoCoulombs, 1e-18 C)
    ELE_CHARGE_C        = 1.60217662e-19    # [C]   (Coulombs)
    KB_AJ_PER_K         = 1.380649e-5       # [aJ/K] (attoJoules per Kelvin, 1e-18 J/K)
    KB_EV_PER_K         = 8.617333262e-5    # [eV/K] (electronvolts per Kelvin) 
    MIN_RESISTANCE_MOHM = 1.0               # [MOhm]

    def __init__(self, electrostatics: NanoparticleElectrostatic) -> None:
        """Initialize tunneling class.

        Parameters
        ----------
        electrostatics : NanoparticleElectrostatic
            A fully initialized electrostatic model of the network (which itself contains the topology).
        """
        self.electro = electrostatics

        # Initialisiere die Arrays (werden von den init_-Methoden befüllt)
        self.adv_index_rows = np.array([], dtype=int)
        self.adv_index_cols = np.array([], dtype=int)
        self.const_capacitance_values = None
        self.resistances = None
        self.conductance_matrix = None
        
    def init_adv_indices(self) -> None:
        """
        Initialize advanced indices for all possible tunneling events (i→j).

        This sets up two arrays:
        - self.adv_index_rows: origin indices (i) for each possible tunneling event (i→j)
        - self.adv_index_cols: destination indices (j) for each event (i→j)

        Notes
        -----
        - Only valid, connected pairs are included.
        - Nanoparticle nodes are indexed from self.N_electrodes to self.N_electrodes+self.N_particles-1.
        - Electrodes are indexed from 0 to self.N_electrodes-1 (shifted for array construction).
        - Updates self.adv_index_rows and self.adv_index_cols as 1D arrays.
        
        Raises
        ------
        RuntimeError
            If the network topology has not been built.
        """
        n_parts = self.electro.topo.N_particles
        n_elec = self.electro.topo.N_electrodes
        n_junc = self.electro.topo.N_junctions
        no_conn = self.electro.topo.NO_CONNECTION

        if n_parts == 0:
            raise RuntimeError("Network not initialized. Build topology first.")

        net_topo = self.electro.topo.get_net_topology()

        # Prepare a unified connections array: [electrodes | nanoparticles]
        connections = np.full((n_parts + n_elec, n_junc + 1), no_conn, dtype=int)
        connections[n_elec:, :] = net_topo
        
        # Map which nanoparticle is attached to each electrode (first column of net_topology)
        nth_e, nth_np = 1, 0
        while (nth_np < self.N_particles) and (nth_e <= self.N_electrodes):
            if int(self.net_topology[nth_np, 0]) == nth_e:
                # The nth_e electrode is connected to nth_np nanoparticle
                connections[nth_e - 1, 1] = nth_np
                nth_e += 1
                nth_np = 0
                continue
            nth_np += 1

        # Offset indices for compact event indexing:
        # - Electrodes: [0 ... N_electrodes-1]
        # - Nanoparticles: [N_electrodes ... N_electrodes+N_particles-1]
        connections[:, 0]  -= 1  # if electrode number is 1-based, shift to 0-based
        connections[:, 1:] += n_elec

        # Build tunneling event lists: for each node, find all valid partners
        adv_index_cols = [list(row[row >= 0].astype(int)) for row in connections]
        adv_index_rows = [len(col_list) * [i] for i, col_list in enumerate(adv_index_cols)]

        # Flatten lists for fast vectorized lookup (event i→j: self.adv_index_rows[k], self.adv_index_cols[k])
        self.adv_index_cols = np.array([item for sublist in adv_index_cols for item in sublist], dtype=int)
        self.adv_index_rows = np.array([item for sublist in adv_index_rows for item in sublist], dtype=int)

    def init_SET(self, radius: float, eps_r: float = 2.6, eps_s: float = 3.9, R: float = 25.0, R_std: float = 0.0) -> None:

        # Params
        self.N_particles    = 1
        self.N_electrodes   = 2
        self.net_topology   = []

        # Capacitance Matrix
        d_np_e  = self.ELECTRODE_RADIUS + radius + self.MIN_NP_NP_DISTANCE
        c_m     = self.mutual_capacitance_adjacent_spheres(eps_r, radius, self.ELECTRODE_RADIUS, d_np_e)
        c_s     = self.self_capacitance_sphere(eps_s, radius)
        c_total = 2*c_m + c_s
        self.capacitance_matrix             = np.array([[c_total]])
        self.inv_capacitance_matrix         = np.linalg.inv(self.capacitance_matrix)
        self.electrode_capacitance_matrix   = np.array([[c_m],
                                                        [c_m]])
        self.self_capacitance               = c_s
        self.const_capacitance_values       = np.full(4,(self.inv_capacitance_matrix[0,0]*self.ELE_CHARGE_A_C**2) / 2)

        # Network
        self.G = nx.DiGraph()
        self.G.add_nodes_from([0,-1,-2])
        self.G.add_edges_from([[-1,0],[0,-1],[0,-2],[-2,0]])
        self.pos = {0   : [0,0],
                    -1  : [-d_np_e, 0],
                    -2  : [d_np_e, 0]}
        
        # Adv Indices
        self.adv_index_rows = np.array([0,1,2,2])
        self.adv_index_cols = np.array([2,2,0,1])
        self.init_junction_resistances(R, R_std)

    def init_const_capacitance_values(self) -> None:
        """
        Precompute constant capacitance terms for tunneling free energy calculations.

        This computes the (C_ii + C_jj - 2*C_ij) * (e^2 / 2) for all possible tunneling
        events, using advanced index arrays. Electrode rows/columns are masked out
        because their potentials are fixed macroscopically and they have infinite capacitance.

        Requires:
        ---------
        - self.adv_index_rows, self.adv_index_cols : from init_adv_indices()
        - electro.get_inv_capacitance_matrix() : inverse capacitance matrix (NxN)

        Stores:
        -------
        - self.const_capacitance_values : ndarray, precomputed constant terms for all tunnel events
          [(aC)^2/aF] (which resolves to aJ).

        Raises
        ------
        RuntimeError
            If advanced indices or the inverse capacitance matrix are missing.
        """
        if self.adv_index_rows.size == 0 or self.adv_index_cols.size == 0:
            raise RuntimeError("Advanced indices not initialized. Call init_adv_indices first.")
        
        # Sicherer Zugriff über den Getter der Elektrostatik-Instanz
        inv_cap_mat = self.electro.get_inv_capacitance_matrix()
        n_elec = self.electro.topo.N_electrodes
        
        # Indices relative to nanoparticle-only part (exclude electrodes)
        row_i = self.adv_index_rows - n_elec
        col_i = self.adv_index_cols - n_elec
        
        # Only use valid nanoparticle indices (exclude electrodes)
        row_mask = (row_i >= 0).astype(int)
        col_mask = (col_i >= 0).astype(int)

        e2_half = (self.ELE_CHARGE_A_C ** 2) / 2.0

        # Precompute capacitance terms
        cap_ii = inv_cap_mat[row_i, row_i] * row_mask * e2_half
        cap_jj = inv_cap_mat[col_i, col_i] * col_mask * e2_half
        cap_ij = inv_cap_mat[row_i, col_i] * row_mask * col_mask * e2_half

        self.const_capacitance_values = cap_ii + cap_jj - 2 * cap_ij

    def init_junction_resistances(self, R: float = 25, Rstd: float = 0) -> None:
        """
        Initialize a 1D array of tunnel resistances (in MΩ) for each unique NP-NP or NP-electrode junction.
        Ensures all resistances are undirected (R_ij = R_ji) and above a minimum value.
        
        Parameters
        ----------
        R : float
            Mean tunnel resistance [MΩ]
        Rstd : float
            Standard deviation of resistances [MΩ]
            
        Raises
        ------
        RuntimeError
            If advanced indices have not been initialized.
        ValueError
            If R is not positive or Rstd is negative.
        """
        if self.adv_index_rows.size == 0:
            raise RuntimeError("Advanced indices not initialized. Call init_adv_indices first.")
            
        if R <= 0:
            raise ValueError(f"Mean resistance must be positive, got {R}")
        if Rstd < 0:
            raise ValueError(f"Std deviation must be non-negative, got {Rstd}")

        n_junctions = len(self.adv_index_rows)
        rng = self.electro.topo.rng

        # Sample resistances from normal distribution, resample if below minimum
        resistances = rng.normal(loc=R, scale=Rstd, size=n_junctions)
        
        # Resampling
        while np.any(resistances < self.MIN_RESISTANCE_MOHM):
            bad = resistances < self.MIN_RESISTANCE_MOHM
            resistances[bad] = rng.normal(loc=R, scale=Rstd, size=np.sum(bad))

        # Enforce undirected (symmetry) property
        self.resistances = self._ensure_undirected_resistances(resistances, average=True)

    def _ensure_undirected_resistances(self, resistances: np.ndarray, average: bool = False) -> np.ndarray:
        """
        Ensures that resistances are undirected by making R(i, j) = R(j, i).
        
        Parameters
        ----------
        resistances : np.ndarray
            1D array of resistance values corresponding to the junction pairs.
        average : bool, optional
            If True, the resistance values for both directions (i, j) and (j, i) are averaged.
            If False, the resistance value of (i, j) overwrites (j, i). Default is False.

        Returns
        -------
        np.ndarray
            Updated resistance values with undirected property enforced.
        """
        # Dictionary Comprehension: Das läuft intern in C ab und ist spürbar schneller!
        pair_to_index = {
            (i, j): idx 
            for idx, (i, j) in enumerate(zip(self.adv_index_rows, self.adv_index_cols))
        }
        
        # Iterate through each resistance and enforce symmetry
        for idx, (i, j) in enumerate(zip(self.adv_index_rows, self.adv_index_cols)):
            reverse_pair = (j, i)
            
            if reverse_pair in pair_to_index:
                reverse_idx = pair_to_index[reverse_pair]
                
                # Ensure symmetry
                if average:
                    avg_res = (resistances[idx] + resistances[reverse_idx]) / 2.0
                    resistances[idx] = avg_res
                    resistances[reverse_idx] = avg_res
                else:
                    resistances[reverse_idx] = resistances[idx]

        return resistances
    
    def update_junction_resistances(self, junctions: List[Tuple[int, int]], R: float = 25) -> None:
        """
        Update tunnel resistances for specific junctions.

        Parameters
        ----------
        junctions : List[Tuple[int, int]]
            List of (origin, target) NP index pairs to update (indices relative to NP array).
        R : float, optional
            New resistance value [MΩ], by default 25.0

        Raises
        ------
        ValueError
            If a specified junction does not exist, or if R is below the minimum allowed limit.
        RuntimeError
            If the resistance array has not been initialized.
        """
        if getattr(self, "resistances", None) is None:
            raise RuntimeError("Resistances not initialized. Call init_junction_resistances first.")
            
        if R < self.MIN_RESISTANCE_MOHM:
            raise ValueError(f"Resistance {R} is below the allowed minimum of {self.MIN_RESISTANCE_MOHM} MΩ.")

        n_elec = self.electro.topo.N_electrodes


        for junc in junctions:
            # Update forward direction
            a = np.where(self.adv_index_rows == junc[0] + n_elec)[0]
            b = np.where(self.adv_index_cols == junc[1] + n_elec)[0]
            idx = np.intersect1d(a,b)[0]
            self.resistances[idx] = R

            # Update reverse direction
            a = np.where(self.adv_index_cols == junc[0] + n_elec)[0]
            b = np.where(self.adv_index_rows == junc[1] + n_elec)[0]
            idx = np.intersect1d(a,b)[0]
            self.resistances[idx] = R

    def update_junction_resistances_at_random(self, N: int, R: float = 25.0, Rstd: float = 0.0) -> None:
        """
        Randomly select N unique undirected NP-NP junctions and update their tunnel resistances.

        Parameters
        ----------
        N : int
            Number of distinct (undirected) junctions to modify.
        R : float, optional
            Mean resistance value [MΩ] (default: 25.0).
        Rstd : float, optional
            Standard deviation for resistance values [MΩ] (default: 0.0).

        Raises
        ------
        ValueError
            If N is greater than the total number of available undirected junctions.
        RuntimeError
            If advanced indices have not been initialized.
        """

        if getattr(self, "adv_index_rows", None) is None or len(self.adv_index_rows) == 0:
            raise RuntimeError("Advanced indices not initialized. Call init_adv_indices first.")
            
        n_elec = self.electro.topo.N_electrodes
        rng = self.electro.topo.rng

        all_pairs = []
        seen = set()
        for row, col in zip(self.adv_index_rows, self.adv_index_cols):
            i = row - n_elec
            j = col - n_elec
            if ((i < j) and (i >= 0) and (j >= 0)):
                pair = (i, j)
                if pair not in seen:
                    all_pairs.append((i,j))
                    seen.add(pair)

        if N > len(all_pairs):
            raise ValueError(f"Requested {N} junctions, but only {len(all_pairs)} available.")
        
        # Randomly choose N distinct junctions
        chosen_pairs = rng.choice(a=all_pairs, size=N, replace=False)
        chosen_pairs = [tuple(pair) for pair in chosen_pairs]

        # Generate resistance values
        if Rstd > 0:
            new_resistances = rng.normal(R, Rstd, size=N)
            # Resample if any below minimum
            while np.any(new_resistances < self.MIN_RESISTANCE_MOHM):
                bad = new_resistances < self.MIN_RESISTANCE_MOHM
                new_resistances[bad] = rng.normal(R, Rstd, size=np.sum(bad))
        else:
            new_resistances = np.full(N, R)
        
        # Set each chosen junction (and its reverse) to the sampled value
        for (pair, new_R) in zip(chosen_pairs, new_resistances):
            self.update_junction_resistances([pair], new_R)

    # # TODO: Allow Gauss distributed
    # def update_nanoparticle_resistances(self, nanoparticles: List[int], R: float = 25) -> None:
    #     """
    #     Set all resistances in self.resistances corresponding to jumps
    #     *to or from* the specified nanoparticles to a new value.

    #     Parameters
    #     ----------
    #     nanoparticles : List[int]
    #         Indices of nanoparticles for which all associated resistances should be updated.
    #     R : float, optional
    #         New resistance value [MΩ] to assign, by default 25.

    #     Raises
    #     ------
    #     RuntimeError
    #         If resistances have not been initialized.
    #     ValueError
    #         If any nanoparticle index is out of bounds.
    #     """
    #     if not hasattr(self, "resistances"):
    #         raise RuntimeError("Resistances have not been initialized. Call init_resistances first.")

    #     for idx in nanoparticles:
    #         adv_idx = idx + self.N_electrodes
    #         # Set for all jumps to and from this NP
    #         self.resistances[np.where(self.adv_index_cols == adv_idx)[0]] = R
    #         self.resistances[np.where(self.adv_index_rows == adv_idx)[0]] = R
    
    def build_conductance_matrix(self) -> None:
        """
        Build and store the conductance (Laplacian) matrix for the network [S].

        Each off-diagonal entry -g_{ij} represents a conductance between nodes i and j.
        Diagonal entries sum all conductances connected to node i.
        The resulting matrix can be used for network current/voltage calculations.
        Floating electrodes are completely disconnected (g = 0) in this matrix.

        Raises
        ------
        RuntimeError
            If resistances are not initialized.

        Notes
        -----
        - The node ordering is: [nanoparticles..., electrodes...]
        - Node indices correspond to their new positions in the matrix via raw2dense.
        """
        if getattr(self, "resistances", None) is None:
            raise RuntimeError("Resistances have not been initialized. Call init_junction_resistances first.")

        n_elec = self.electro.topo.N_electrodes
        floating_indices = self.electro.floating_indices

        src = self.adv_index_rows.copy() - n_elec
        tgt = self.adv_index_cols.copy() - n_elec

        # Get sorted list of all node indices (NPs >= 0, electrodes < 0)
        np_nodes = sorted({idx for idx in np.concatenate([src, tgt]) if idx >= 0})
        el_nodes = sorted({idx for idx in np.concatenate([src, tgt]) if idx <  0})
        total_n = len(np_nodes) + len(el_nodes)

        # Map raw node indices to their position in the matrix
        raw2dense = {raw: i for i, raw in enumerate(np_nodes + el_nodes)}
        cond_matrix = np.zeros((total_n, total_n))

        for s_raw, t_raw, R_l in zip(src, tgt, self.resistances):
            g = 1.0 / (2*R_l)
            i = raw2dense[s_raw]
            j = raw2dense[t_raw]

            # Catch Floating Electrodes
            if hasattr(self, 'floating_indices') and i in floating_indices or j in floating_indices:
                continue
            
            # symmetric update
            cond_matrix[i, i] += g
            cond_matrix[j, j] += g
            cond_matrix[i, j] -= g
            cond_matrix[j, i] -= g

        self.conductance_matrix = cond_matrix*1e-6

    def init_transfer_coeffs(self, output_electrode: int = None) -> None:
        """
        Calculate and store the DC transconductance vector for the network.

        This computes the current flowing into the selected output electrode
        when 1V is applied to any other electrode (while others are grounded).
        It uses the Schur complement to trace out the internal nanoparticle degrees of freedom.

        Parameters
        ----------
        output_electrode : int, optional
            Index of the output electrode (0-based relative to electrode list). 
            If None, uses the last electrode.

        Raises
        ------
        RuntimeError
            If conductance matrix is not initialized, or if the NP matrix is singular.
        ValueError
            If output electrode index is invalid.

        Stores
        ------
        self.transfer_coeffs : np.ndarray
            1D array, shape (N_electrodes,). Units: Siemens [S].
            Element [i] is the current at the output when Electrode i is set to 1.0V.
        """
        if getattr(self, "conductance_matrix", None) is None:
            raise RuntimeError("Conductance matrix not calculated. Call build_conductance_matrix first.")

        N_e = self.electro.topo.N_electrodes
        N_p = self.electro.topo.N_particles

        # Indices for unknowns (NPs) and knowns (electrodes)
        u_idx = np.arange(N_p)
        k_idx = np.arange(N_p, N_e + N_p)

        # Output electrode: by default, the last one
        if output_electrode is None:
            rel_out = N_e - 1
        else:
            rel_out = int(output_electrode)
            if not (0 <= rel_out < N_e):
                raise ValueError(f"Invalid output_electrode index: {rel_out}")

        # The actual matrix index for the output electrode
        matrix_out_idx = k_idx[rel_out]

        # Partition Matrix
        Y = self.conductance_matrix
        
        # Internal-Internal Coupling
        Y_uu = Y[np.ix_(u_idx, u_idx)]

        # Internal-Electrode Coupling
        Y_uk = Y[np.ix_(u_idx, k_idx)]

        # Solve Y_uu * A = Y_uk for A
        A = np.linalg.solve(Y_uu, Y_uk)

        # Get the row connecting Internal nodes to the Output Electrode
        Y_out_u = Y[matrix_out_idx, u_idx]

        # Calculate the indirect current path (through the network)
        indirect_coupling = Y_out_u @ A

        # Get direct coupling (Electrode to Output Electrode, usually 0)
        Y_out_k = Y[matrix_out_idx, k_idx]

        # Formula: G_eff = Y_out_k - Y_out_u * A
        self.transfer_coeffs            = -(Y_out_k - indirect_coupling)
        self.transfer_coeffs[rel_out]   = 0.0

    def get_const_capacitance_values(self) -> np.ndarray:
        """
        Returns
        -------
        const_capacitance_values : ndarray
            Capacitance terms for tunneling free energy calculation [(aC)^2/aF]
            
        Raises
        ------
        RuntimeError
            If constant capacitance values have not been computed.
        """
        if getattr(self, 'const_capacitance_values', None) is None:
            raise RuntimeError("Constant capacitance values not calculated. Call init_const_capacitance_values first.")
        return self.const_capacitance_values.copy()
    
    def get_advanced_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        adv_index_rows : ndarray of shape (n_tunnel_events,)
            Origin nanoparticles (i) in tunneling events i→j
        adv_index_cols : ndarray of shape (n_tunnel_events,)
            Target nanoparticles (j) in tunneling events i→j
            
        Raises
        ------
        RuntimeError
            If advanced indices have not been initialized.
        """
        if getattr(self, 'adv_index_rows', None) is None or len(self.adv_index_rows) == 0:
            raise RuntimeError("Advanced indices not initialized. Call init_adv_indices first.")
        return self.adv_index_rows.copy(), self.adv_index_cols.copy()
    
    def get_const_temperatures(self, T: float = 0.28) -> np.ndarray:
        """
        Parameters
        ----------
        T : float
            Network temperature [K]

        Returns
        -------
        T_arr : ndarray
            Array of thermal energies for each tunneling event [aJ]
            (T * kB, one per tunneling event)
        """
        if getattr(self, 'adv_index_rows', None) is None:
            raise RuntimeError("Advanced indices not initialized. Call init_adv_indices first.")
        return np.repeat(T * self.KB_AJ_PER_K, len(self.adv_index_rows))
    
    def get_tunneling_rate_prefactor(self) -> np.ndarray:
        """
        Returns
        -------
        ndarray
            1D array of rate prefactors (R_{ij} * e^2) [(MΩ·(aC)^2)]
            for each tunneling event (i→j).
        """
        if getattr(self, 'resistances', None) is None:
            raise RuntimeError("Junction resistances not initialized. Call init_junction_resistances first.")

        return self.resistances * (self.ELE_CHARGE_A_C ** 2) * 1e-12
    
    def get_resistance(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            1D array of resistance values for each tunneling event (i->j)
        """
        if getattr(self, 'resistances', None) is None:
            raise RuntimeError("Junction resistances not initialized. Call init_junction_resistances first.")
        return self.resistances.copy()
    
    def get_conductance_matrix(self) -> np.ndarray:
        """
        Get the network conductance (Laplacian) matrix.

        Returns
        -------
        np.ndarray
            Symmetric conductance matrix [S], shape: (N_total, N_total),
            where N_total = N_particles + N_electrodes

        Raises
        ------
        RuntimeError
            If conductance matrix hasn't been calculated (call build_conductance_matrix first).
        """
        if getattr(self, 'conductance_matrix', None) is None:
            raise RuntimeError("Conductance matrix not calculated. Call build_conductance_matrix first.")
        return self.conductance_matrix.copy()
    
    def get_transfer_coeffs(self) -> np.ndarray:
        """
        Get the vector of transfer coefficients for the current output electrode.

        Returns
        -------
        np.ndarray
            Transfer coefficients (transconductance), shape (N_electrodes,), units [S]
            
        Raises
        ------
        RuntimeError
            If transfer_coeffs are not initialized (call init_transfer_coeffs first).
        """
        if getattr(self, "transfer_coeffs", None) is None:
            raise RuntimeError("Transfer coefficients not initialized. Call init_transfer_coeffs first.")
        return self.transfer_coeffs.copy()
    
    def get_slowest_linear_time_constant(self, lam_th: float = 1e-9) -> float:
        """
        Get the slowest time constant in the linear regime based on matrix eigenvalues.
        Solves the generalized eigenvalue problem G*x = lambda*C*x.

        Parameters
        ----------
        lam_th : float, optional
            Threshold for strictly positive eigenvalues to ignore zero-modes 
            or numerical noise, by default 1e-9.

        Returns
        -------
        float
            Slowest linear time constant [s]. Returns np.inf if no valid time constant exists.
            
        Raises
        ------
        RuntimeError
            If electrostatics or resistances are not initialized.
        """
        if getattr(self, "resistances", None) is None:
            raise RuntimeError("Resistances not initialized. Call init_junction_resistances first.")
            
        self.build_conductance_matrix()

        g_m = self.get_conductance_matrix()
        cap_m = self.electro.get_capacitance_matrix()*1e-18
        n_elec = self.electro.topo.N_electrodes

        # Smallest Eigenvalue
        eig_v, _ = eig(g_m[:-n_elec,:-n_elec], cap_m)
        eig_real = np.real(eig_v)
        eig_valid = eig_real[eig_real > lam_th]
        lambda_min = np.min(eig_valid)

        # Slowest Time Constant
        tau_0 = 1.0 / lambda_min

        return tau_0