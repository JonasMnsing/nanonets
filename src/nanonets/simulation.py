from . import topology
from . import electrostatic
from . import tunneling
from .monte_carlo import MonteCarlo

import numpy as np
import pandas as pd
import os.path

class Simulation:
    """
    Simulation class for single-electron effects in nanoparticle networks.

    This class builds on NanoparticleTunneling to provide a full device model, supporting:
        - Arbitrary network topology (lattice or random graph)
        - Electrostatic modeling (capacitance, packing, NP radii)
        - Tunnel junction properties (static or dynamic resistances)
        - Flexible electrode placement and type assignment ('constant' or 'floating')
        - Device heterogeneity via multiple NP or resistance types
        - Batch and time-dependent kinetic Monte Carlo simulation

    Attributes
    ----------
    network_topology : str
        Type of network, either 'lattice' or 'random'.
    res_info : dict
        Resistance parameters for primary nanoparticle type.
    res_info2 : dict or None
        Resistance parameters for a second nanoparticle type, if present.
    folder : str
        Folder where simulation outputs are stored.
    path1, path2, path3 : str
        File paths for output data.
    [All attributes from NanoparticleTunneling are inherited.]

    Parameters
    ----------
    topology_parameter : dict
        Defines topology, number of particles, electrode positions, and electrode types.
        Required keys depend on topology:
            For lattice: 'Nx', 'Ny', 'e_pos', 'electrode_type'
            For random:  'Np', 'Nj', 'e_pos', 'electrode_type'
    folder : str, optional
        Output folder for result files.
    add_to_path : str, optional
        Extra string appended to output file names.
    res_info : dict, optional
        Resistance parameters for main NP type. Example:
            {"mean_R": 25.0, "std_R": 0.0, "dynamic": False}
    res_info2 : dict, optional
        Resistance parameters for a second NP type (optional).
    np_info : dict, optional
        Physical parameters for first NP type (dielectric, radius, spacing, etc.).
            Example: {"eps_r": 2.6, "eps_s": 3.9, "mean_radius": 10.0, "std_radius": 0.0, "np_distance": 1.0}
    np_info2 : dict, optional
        Physical parameters for second NP type; may specify 'np_index' for assignment.
    seed : int, optional
        Random seed for reproducibility.
    kwargs : dict, optional
        Additional options (e.g., del_n_junctions: int for disordered/sparse networks).

    Raises
    ------
    ValueError
        If an unsupported network topology is requested.

    Notes
    -----
    - Lattice topology uses a square 2D grid; random uses Delaunay triangulation.
    - Multiple nanoparticle types allow simulation of heterogeneous materials.
    - All matrices and simulation data for electrostatics and transport are constructed at initialization.
    - For further physics, data, and usage details, see parent class docstrings.
    """

    def __init__(self, topology_parameter : dict, self_cap: float = 0.28, folder: str = '', add_to_path: str = "", res_info: dict = None, res_info2: dict = None,
                 np_info: dict = None, np_info2: dict = None, seed: int = None, **kwargs):
        """
        Defines network topology, electrostatic properties, and tunneling junctions for a given topology.

        Parameters
        ----------
        topology_parameter : dict
            Dictionary including number of nanoparticles, electrode positions, and electrode types.
            For lattice: must include 'Nx', 'Ny', 'e_pos', 'electrode_type'.
            For random: must include 'Np', 'e_pos', 'electrode_type'.
        folder : str, optional
            Directory where simulation results are saved (default: '').
        add_to_path : str, optional
            String appended to output file names (default: "").
        res_info : dict, optional
            Resistance parameters for primary NP type. (default: mean_R=25.0, std_R=0.0, dynamic=False)
        res_info2 : dict, optional
            Resistance parameters for secondary NP type (if any).
        np_info : dict, optional
            Parameters for first nanoparticle type (default: see code).
        np_info2 : dict, optional
            Parameters for second nanoparticle type (may include 'np_index' for which NPs are affected).
        seed : int, optional
            Random seed for reproducibility.
        kwargs : dict
            Additional keyword arguments.
        

        Raises
        ------
        ValueError
            If required keys are missing, or unsupported topology is requested.

        Notes
        -----
        - All attributes for electrostatics, topology, and tunneling are set up automatically.
        - Network can have two nanoparticle types for heterogeneity.
        - All physical quantities are initialized, and key file paths are pre-set.
        """

        # --- 1. Parameter Defaults Setup ---
        electrode_type = kwargs.get('electrode_type', topology_parameter.get('electrode_type', 'constant'))

        if np_info is None:
            np_info = {"eps_r": 2.6, "eps_s": 3.9, "mean_radius": 10.0, "std_radius": 0.0}
            
        if res_info is None:
            res_info = {"mean_R": 25.0, "std_R": 0.0, "dynamic": False}

        self.dynamic_resistances = res_info.get('dynamic', False)
        self.dynamic_resistances_info = res_info

        # --- 2. TOPOLOGY COMPOSITION ---
        # Init Topology
        self.topology = topology.NanoparticleTopology(seed=seed)

        # Define network topology
        if 'Nx' in topology_parameter:
            if topology_parameter['Nx'] == 1 and topology_parameter['Ny'] == 1:
                self.network_topology_type = 'set'
            else:
                self.network_topology_type = 'lattice'
        elif 'Np' in topology_parameter:
            self.network_topology_type = 'random'
        else:
            self.network_topology_type = 'custom'

        # Build Topology
        if self.network_topology_type == "lattice":
            self.topology.lattice_network(N_x=topology_parameter["Nx"], N_y=topology_parameter["Ny"], mean_radius=np_info['mean_radius'])
            self.topology.add_electrodes_to_lattice_net(particle_pos=topology_parameter["e_pos"])
            path_var = f'Nx={topology_parameter["Nx"]}_Ny={topology_parameter["Ny"]}_Ne={len(topology_parameter["e_pos"])}{add_to_path}'
            
        elif self.network_topology_type == "random":
            delta = kwargs.get('delta', 0.5)
            max_attempts = kwargs.get('max_attempts', 5)
            packing_kwargs = kwargs.get('packing_kwargs', {})

            self.topology.random_network(N_particles=topology_parameter["Np"], mean_radius=np_info['mean_radius'], std_radius=np_info['std_radius'],
                                         delta=delta, max_attempts=max_attempts, **packing_kwargs)
            if np_info2 is not None:
                if 'np_index' in np_info2:
                    self.topology.update_nanoparticle_radius(np_info2['np_index'], np_info2['mean_radius'], np_info2['std_radius'],
                                                            delta=delta, max_attempts=max_attempts, **packing_kwargs)
                elif 'N' in np_info2:
                    self.topology.update_nanoparticle_radius_at_random(np_info2['N'], np_info2['mean_radius'], np_info2['std_radius'],
                                                                                delta=delta, max_attempts=max_attempts, **packing_kwargs)
                    
            self.topology.add_electrodes_to_random_net(electrode_positions=topology_parameter["e_pos"])
            path_var = f'Np={topology_parameter["Np"]}_Ne={len(topology_parameter["e_pos"])}{add_to_path}'
            
        elif self.network_topology_type == "set":
            pass
            
        else:
            pass

        # Delete Junctions after Topology was build
        del_n_junc = kwargs.get("delete_n_junctions", None)
        if del_n_junc is not None:
            self.topology.delete_n_junctions(del_n_junc)

        # --- 3. ELECTROSTATIC COMPOSITION ---
        # Init Electrostatic
        self.electrostatic = electrostatic.NanoparticleElectrostatic(self.topology, electrode_type)
        
        if self.network_topology_type != "set":
            self.electrostatic.constant_self_capacitance(self_cap=self_cap)
            self.electrostatic.calc_capacitance_matrix(np_info['eps_r'], np_info['eps_s'])
            self.electrostatic.calc_electrode_capacitance_matrix()

        # --- 4. TUNNELING COMPOSITION ---
        # Init Tunneling
        self.tunneling = tunneling.NanoparticleTunneling(self.electrostatic)
        
        if self.network_topology_type != "set":
            self.tunneling.init_adv_indices()
            self.tunneling.init_junction_resistances(res_info['mean_R'], res_info['std_R'])
            if res_info2 is not None:
                self.tunneling.update_junction_resistances_at_random(res_info2['N'], res_info2['mean_R'], res_info2['std_R'])
            self.tunneling.init_const_capacitance_values()

        # --- 5. PATHS ---
        self.folder = folder
        self.path1 = os.path.join(folder, f'{path_var}.csv')
        self.path2 = os.path.join(folder, f'mean_state_{path_var}.parquet')
        self.path3 = os.path.join(folder, f'net_currents_{path_var}.parquet')
        self.path4 = os.path.join(folder, f'resistances_{path_var}.parquet')

    def run_static_voltages(self, voltages: np.ndarray, target_electrode: int, T_val: float = 0.01, sim_dic: dict = None,
                            save_th: int = None, verbose: bool = False, **dyn_res_kwargs):
        """
        Run an ensemble of Kinetic Monte Carlo trajectories at fixed electrode voltages 
        to extract the steady-state macroscopic current.
        """
        
        # --- Default Simulation Parameters ---
        if sim_dic is None:
            sim_dic = {
                "n_trajectories" : 400,     # Number of independent KMC runs per voltage
                "max_jumps"      : 20000,   # Number of productions KMC steps
                "max_eq_jumps"   : 100000   # Number of equilibration KMC steps
            }

        n_trajectories = sim_dic.get('n_trajectories', 400)
        max_jumps = sim_dic.get('max_jumps', 20000)
        max_eq_jumps = sim_dic.get('max_eq_jumps', 100000)

        if self.dynamic_resistances:
            I0_val = dyn_res_kwargs.get('I0', 7.5)
            tau_0 = dyn_res_kwargs.get('tau_0', 1e-8)
            R_max = dyn_res_kwargs.get('R_max', 25.0)
            R_min = dyn_res_kwargs.get('R_min', 10.0)


        n_voltages = len(voltages)
        n_junctions = len(self.tunneling.adv_index_rows)
        N_particles, N_electrodes = self.topology.get_particle_electrode_count()

        # Pre-allocate Arrays
        self.observable_storage = np.zeros(n_voltages)
        self.observable_error_storage = np.zeros(n_voltages)
        self.jump_storage = np.zeros(n_voltages)
        self.eq_jump_storage = np.zeros(n_voltages)
        self.time_storage = np.zeros(n_voltages)
        self.potential_storage = np.zeros((n_voltages, N_particles + N_electrodes))

        if verbose:
            self.state_storage = np.zeros((n_voltages, N_particles))
            self.network_current_storage = np.zeros((n_voltages, n_junctions))
            if self.dynamic_resistances:
                self.resistance_storage = np.zeros((n_voltages, n_junctions))

        j = 0
        for i, voltage_values in enumerate(voltages):
            
            # --- Get all CONSTANT KMC model input arrays ---
            inv_capacitance_matrix = np.ascontiguousarray(self.electrostatic.get_inv_capacitance_matrix(), dtype=np.float64)
            const_capacitance_values = np.ascontiguousarray(self.tunneling.get_const_capacitance_values(), dtype=np.float64)
            adv_index_rows, adv_index_cols = self.tunneling.get_advanced_indices()

            adv_index_rows = np.ascontiguousarray(adv_index_rows, dtype=np.int64)
            adv_index_cols = np.ascontiguousarray(adv_index_cols, dtype=np.int64)
            floating_electrodes = np.ascontiguousarray(np.where(self.electrostatic.electrode_type == 'floating')[0], dtype=np.int64)

            temperatures = np.ascontiguousarray(self.tunneling.get_const_temperatures(T=T_val), dtype=np.float64)
            resistances = np.ascontiguousarray(self.tunneling.get_tunneling_rate_prefactor(), dtype=np.float64)

            # --- Preallocate arrays for this voltage's ensemble ---
            ens_observables = np.zeros(n_trajectories)
            ens_potentials = np.zeros((n_trajectories, N_particles + N_electrodes))
            ens_jumps = np.zeros(n_trajectories)
            ens_times = np.zeros(n_trajectories)

            if verbose:
                ens_charges = np.zeros((n_trajectories, N_particles))
                ens_currents = np.zeros((n_trajectories, len(adv_index_rows)))
                ens_resistances = np.zeros((n_trajectories, len(adv_index_rows)))

            # =======================================================
            # ENSEMBLE TRAJECTORY LOOP
            # =======================================================
            for traj in range(n_trajectories):

                # Update electrostatics strictly for this fresh trajectory
                self.electrostatic.init_charge_vector(voltage_values=voltage_values)
                self.electrostatic.init_potential_vector(voltage_values=voltage_values)
                charge_vector = np.ascontiguousarray(self.electrostatic.get_charge_vector(), dtype=np.float64)
                potential_vector = np.ascontiguousarray(self.electrostatic.get_potential_vector(), dtype=np.float64)

                # Instantiate (Numba-optimized) model
                self.model = MonteCarlo(
                    charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
                    temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles,
                    floating_electrodes
                )
                
                # Equilibrate and Production Run
                if self.dynamic_resistances:
                    eq_jumps = self.model.run_equilibration_steps_var_resistance(max_eq_jumps, I0_val, tau_0, R_max, R_min)
                    self.model.kmc_simulation_trajectory_var_resistance(target_electrode, max_jumps, I0_val, tau_0, R_max, R_min)
                else:
                    eq_jumps = self.model.run_equilibration_steps(max_eq_jumps)
                    self.model.kmc_simulation_trajectory(target_electrode, max_jumps)

                # Store this specific trajectory's results
                ens_observables[traj] = self.model.target_observable_mean
                ens_potentials[traj, :] = self.model.potential_mean
                ens_jumps[traj] = self.model.total_jumps
                ens_times[traj] = self.model.time

                if verbose:
                    ens_charges[traj, :] = self.model.charge_mean
                    ens_currents[traj, :] = self.model.network_currents
                    ens_resistances[traj, :] = self.model.resistances

            # =======================================================
            # ENSEMBLE AVERAGING
            # =======================================================
            self.observable_storage[i] = np.mean(ens_observables)
            self.observable_error_storage[i] = np.std(ens_observables, ddof=1) / np.sqrt(n_trajectories)
            
            self.potential_storage[i, :] = np.mean(ens_potentials, axis=0)
            self.eq_jump_storage[i] = eq_jumps
            self.jump_storage[i] = np.mean(ens_jumps)
            self.time_storage[i] = np.mean(ens_times)

            if verbose:
                self.state_storage[i, :] = np.mean(ens_charges, axis=0)
                self.network_current_storage[i, :] = np.mean(ens_currents, axis=0)
                if self.dynamic_resistances:
                    self.resistance_storage[i, :] = np.mean(ens_resistances, axis=0)

            # --- Periodically save to disk ---
            if (save_th is not None and ((i + 1) % save_th == 0)):
                self.data_to_path(voltages[:(i + 1), :], self.path1)
                self.potential_to_path(self.path2, end_idx=i+1)
                if verbose:
                    self.network_current_to_path(self.path3, end_idx=i+1)
                    if self.dynamic_resistances:
                        self.resistance_to_path(self.path4, end_idx=i+1)
            
    def run_dynamic_voltages(self, voltages: np.ndarray, time_steps: np.ndarray, target_electrode: int,
                             T_val: float = 0.01, save: bool = False,
                             n_trajectories: int = 100, init_charges: bool = None, verbose: bool = False, **dyn_res_kwargs):
        """
        Run kinetic Monte Carlo simulation for time-dependent electrode voltages.

        This method simulates a voltage sequence (e.g., waveform or pulse train) and tracks
        the current or potential at a target electrode. For each time segment, an observable
        is calculated by averaging over multiple statistical runs.

        Parameters
        ----------
        voltages : np.ndarray
            2D array (n_timesteps, n_electrodes+1) of electrode voltages [V]. Each row is a time step.
        time_steps : np.ndarray
            1D array (n_timesteps,) of KMC simulation time (seconds) at each voltage segment (monotonically increasing).
        target_electrode : int
            Index of the electrode for which the observable (current/potential) is recorded.
        T_val : float, optional
            Network temperature [K]. Default: 5.0.
        eq_steps : int, optional
            Number of KMC steps for initial equilibration. Default: 0.
        save : bool, optional
            Whether to save results to file after run. Default: True.
        stat_size : int, optional
            Number of independent stochastic runs for averaging. Default: 10.
        init_charges : np.ndarray, optional
            2D array (stat_size, n_particles) of pre-initialized charge states for each run.
            If None, charges are equilibrated from scratch.
        verbose : bool, optional
            If True, tracks additional simulation data (not used in this method).

        Returns
        -------
        None
            Results are saved to disk and/or stored in class attributes.

        Notes
        -----
        - Each run is independent and averaged for error estimation.
        - The observable is either output current or (for floating electrodes) electrode potential.
        - The last time step in voltages is ignored for observable reporting.
        """
        # --- Dimension Check & Auto-Padding ---
        if len(time_steps) == len(voltages):
            # Extrapolation
            if len(time_steps) > 1:
                dt = time_steps[-1] - time_steps[-2]
                time_steps = np.append(time_steps, time_steps[-1] + dt)
            else:
                raise ValueError(f"For 'time_steps' and 'voltages' having the same length we need at least 2 steps.")
            
        elif len(time_steps) != len(voltages) + 1:
            raise ValueError(
                f"Dimension mismatch: 'time_steps' (len={len(time_steps)}) must have exactly"
                f"one more element than 'voltages' (len={len(voltages)}) to define the intervals."
            )

        if self.dynamic_resistances:
            I0_val = dyn_res_kwargs.get('I0', 7.5)
            tau_0 = dyn_res_kwargs.get('tau_0', 1e-8)
            R_max = dyn_res_kwargs.get('R_max', 25.0)
            R_min = dyn_res_kwargs.get('R_min', 10.0)

        # Round voltages to 0.01 mV
        voltages = np.round(voltages, 5)
        
        # --- Get strictly contiguous arrays for Numba ---
        inv_capacitance_matrix = np.ascontiguousarray(self.electrostatic.get_inv_capacitance_matrix(), dtype=np.float64)
        const_capacitance_values = np.ascontiguousarray(self.tunneling.get_const_capacitance_values(), dtype=np.float64)
        N_particles, N_electrodes = self.topology.get_particle_electrode_count()
        adv_index_rows, adv_index_cols = self.tunneling.get_advanced_indices()

        adv_index_rows = np.ascontiguousarray(adv_index_rows, dtype=np.int64)
        adv_index_cols = np.ascontiguousarray(adv_index_cols, dtype=np.int64)
        floating_electrodes = np.ascontiguousarray(np.where(self.electrostatic.electrode_type == 'floating')[0], dtype=np.int64)
        const_electrodes = np.ascontiguousarray(np.where(self.electrostatic.electrode_type == 'constant')[0], dtype=np.int64)

        temperatures = np.ascontiguousarray(self.tunneling.get_const_temperatures(T=T_val), dtype=np.float64)
        resistances = np.ascontiguousarray(self.tunneling.get_tunneling_rate_prefactor(), dtype=np.float64)

        # --- Initialize network for first time step ---
        self.electrostatic.init_charge_vector(voltage_values=voltages[0])
        self.electrostatic.init_potential_vector(voltage_values=voltages[0])

        initial_charge_vector = np.ascontiguousarray(self.electrostatic.get_charge_vector(), dtype=np.float64)
        potential_vector = np.ascontiguousarray(self.electrostatic.get_potential_vector(), dtype=np.float64)
        
        # --- Instantiate (Numba-optimized) model ---
        self.model = MonteCarlo(
            initial_charge_vector.copy(), potential_vector, inv_capacitance_matrix, const_capacitance_values,
            temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles, 
            floating_electrodes
        )

        # Initial time for the reference state
        self.model.time = time_steps[0]
        
        # Subtract charges induced by initial electrode voltages for charge-neutrality
        offset = self.electrostatic.get_charge_vector_offset(voltage_values=voltages[0])
        self.model.charge_vector = self.model.charge_vector - offset
        
        # --- Ensemble initial states ---
        if init_charges is None:
            self.q_eq = np.tile(self.model.charge_vector.copy(), (n_trajectories, 1))
        else:
            self.q_eq = init_charges.copy()

        # --- Allocate result arrays ---
        n_time = voltages.shape[0]
        n_junctions = len(self.model.adv_index_rows)
        
        self.observable_storage = np.zeros(n_time)
        self.jump_storage = np.zeros(n_time)
        self.potential_storage = np.zeros(shape=(n_time, self.model.N_particles + self.model.N_electrodes))

        if verbose:
            self.state_storage = np.zeros(shape=(n_time, self.model.N_particles))
            self.network_current_storage = np.zeros(shape=(n_time, n_junctions))
            if self.dynamic_resistances:
                self.resistance_storage = np.zeros(shape=(n_time, n_junctions))

        # Store observable for statistical errors
        observable = np.zeros(shape=(n_trajectories, n_time))

        # --- Main simulation loop: ensemble average ---
        for s in range(n_trajectories):
            self.model.charge_vector = self.q_eq[s,:].copy()

            if self.dynamic_resistances:
                self.model.I_tilde = np.zeros(len(adv_index_rows))
                               
            for i, voltage_values in enumerate(voltages):
                # Apply charging state from electrode voltage
                offset = self.electrostatic.get_charge_vector_offset(voltage_values=voltage_values)
                self.model.charge_vector += offset
                
                # Define given time and time target
                self.model.time = time_steps[i]
                time_target = time_steps[i+1]

                # Update constant electrode potentials in the Numba model
                self.model.potential_vector[const_electrodes] = voltage_values[const_electrodes]
                                
                if self.dynamic_resistances:
                    self.model.kmc_time_simulation_trajectory_var_resistance(
                        target_electrode, time_target, I0_val, tau_0, R_max, R_min)
                else:
                    self.model.kmc_time_simulation_trajectory(target_electrode, time_target)

                # Fetch results directly from public attributes (no getters)
                target_observable_mean = self.model.target_observable_mean
                total_jumps = self.model.total_jumps
                
                # Add observables to outputs
                observable[s, i] = target_observable_mean
                self.jump_storage[i] += total_jumps / n_trajectories
                self.potential_storage[i, :] += self.model.potential_mean / n_trajectories

                if verbose:
                    self.state_storage[i, :] += self.model.charge_mean / n_trajectories
                    self.network_current_storage[i, :] += self.model.network_currents / n_trajectories
                    if self.dynamic_resistances:
                        self.resistance_storage[i, :] += self.model.resistances / n_trajectories
                
                # Remove voltage offset for next step (to avoid double-counting)
                self.model.charge_vector -= offset

            # Store last charge vector for each run
            self.q_eq[s, :] = self.model.charge_vector.copy()

        # --- Statistics and final result arrays ---
        self.observable_storage = np.mean(observable, axis=0)
        self.observable_error_storage = 1.96 * np.std(observable, axis=0, ddof=1) / np.sqrt(n_trajectories)

        # Prepare output voltage arrays for saving
        if save:
            V_safe_vals = np.zeros(shape=(n_time, self.topology.N_electrodes + 1))
            if verbose and floating_electrodes.size > 0:
                V_safe_vals[:, floating_electrodes] = self.potential_storage[:, floating_electrodes]
            elif floating_electrodes.size > 0:
                print("Warning: Cannot save floating potentials if verbose=False.")
            V_safe_vals[:, const_electrodes] = voltages[:, const_electrodes]
            V_safe_vals[:, -1] = voltages[:, -1]

            self.data_to_path(V_safe_vals, self.path1)
            self.potential_to_path(self.path2)
            
            if verbose:
                self.network_current_to_path(self.path3)

    def get_observable_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of main observable values for each simulation condition or time point.
        """
        return self.observable_storage.copy()
    
    def get_observable_error_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of standard errors (or 95% CI) for each observable.
        """
        return self.observable_error_storage.copy()
    
    def get_state_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of averaged nanoparticle charge states per condition or time point.
        """
        return self.state_storage.copy()
    
    def get_potential_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of averaged potentials (for all particles and electrodes) per condition or time point.
        """
        return self.potential_storage.copy()
    
    def get_network_current_storage(self) -> dict:
        """
        Returns
        -------
        dict
            Dictionary mapping each (row, col) pair (junction) to its average network current.
            Keys are (origin, target) tuples, values are 1D arrays (current over time/voltages).
        """
        # Werte sicher über Komposition aus dem Tunneling-Modul abrufen
        adv_index_rows, adv_index_cols = self.tunneling.get_advanced_indices()
        
        return {
            (adv_index_rows[i], adv_index_cols[i]) : np.array(self.network_current_storage)[:, i].copy()
            for i in range(len(self.network_current_storage[0]))
        }
    
    def get_resistance_storage(self) -> dict:
        """
        Returns
        -------
        dict
            Dictionary mapping each (row, col) pair (junction) to its average dynamic resistance.
            Keys are (origin, target) tuples, values are 1D arrays (resistance over time/voltages).
            
        Raises
        ------
        RuntimeError
            If dynamic resistances were not tracked (e.g., dynamic=False or verbose=False).
        """
        if not hasattr(self, 'resistance_storage') or self.resistance_storage is None or len(self.resistance_storage) == 0:
            raise RuntimeError("Resistance storage is empty. Ensure dynamic_resistances=True and verbose=True.")
            
        adv_index_rows, adv_index_cols = self.tunneling.get_advanced_indices()
        
        return {
            (adv_index_rows[i], adv_index_cols[i]) : np.array(self.resistance_storage)[:, i].copy()
            for i in range(len(self.resistance_storage[0]))
        }
    
    def get_eq_jump_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of number of jumps performed during equilibration for each simulation point.
        """
        return self.eq_jump_storage.copy()
    
    def get_jump_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of total number of KMC jumps per simulation point.
        """
        return self.jump_storage.copy()
    
    def get_time_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of simulation times for each voltage point or condition.
        """
        return self.time_storage.copy()
    
    def data_to_path(self, voltages : np.ndarray, path : str) -> None:
        """
        Save simulation results for each voltage set to a CSV file.

        Parameters
        ----------
        voltages : np.ndarray
            Array of applied electrode voltages (shape: [n_points, n_electrodes]).
        path : str
            File path for output CSV.

        Notes
        -----
        The output file will have columns for all electrodes, followed by:
        Eq_Jumps (equilibration steps), Jumps (total KMC steps), Observable (main value), and Error (statistical error).
        Appends to file if already exists, otherwise creates a new file with headers.
        """
        end_idx = voltages.shape[0]

        # Only get calculated slices
        val_a = self.get_eq_jump_storage()[:end_idx].reshape(-1, 1)
        val_b = self.get_jump_storage()[:end_idx].reshape(-1, 1)
        val_c = self.get_observable_storage()[:end_idx].reshape(-1, 1)
        val_d = self.get_observable_error_storage()[:end_idx].reshape(-1, 1)
        data = np.hstack([voltages, val_a, val_b, val_c, val_d])

        # Use all electrode columns
        columns = [f'E{i}' for i in range(voltages.shape[1] - 1)]
        columns = np.array(columns + ['G', 'Eq_Jumps', 'Jumps', 'Observable', 'Error'])

        df = pd.DataFrame(data)
        df.columns = columns

        # Overwrite file
        df.to_csv(path, index=False)

    def potential_to_path(self, path: str, end_idx: int = None) -> None:
        """
        Save the potential landscape (microstates) for each voltage point to a CSV file.

        Parameters
        ----------
        path : str
            File path for output CSV.

        Notes
        -----
        Each row corresponds to the full potential vector (all electrodes + particles) for a voltage set.
        Appends to file if it already exists; otherwise, creates a new file with headers.
        """
        data_to_save = self.potential_storage[:end_idx] if end_idx is not None else self.potential_storage
        
        # Parquet verlangt String-Spaltennamen (Pandas Default-Integer 0, 1... sind nicht erlaubt)
        col_names = [str(i) for i in range(data_to_save.shape[1])]
        
        microstates_df = pd.DataFrame(data_to_save, columns=col_names)
        
        # index=False verhindert das Speichern der unbenannten Pandas-Index-Spalte
        microstates_df.to_parquet(path, engine='pyarrow', index=False)

    def network_current_to_path(self, path : str, end_idx: int = None) -> None:
        """
        Save the average tunneling rate (network currents) for each voltage point to a CSV file.

        Parameters
        ----------
        path : str
            File path for output CSV.

        Notes
        -----
        - Each column is labeled by the (row, col) tuple corresponding to a specific junction.
        - Each row corresponds to a voltage set in the simulation.
        - Appends to file if it already exists; otherwise, creates a new file with headers.
        """
        adv_index_rows, adv_index_cols = self.tunneling.get_advanced_indices()
        
        # Tupel (row, col) in lesbare Strings "row_col" konvertieren
        avg_j_cols = [f"{adv_index_rows[i]}_{adv_index_cols[i]}" for i in range(len(adv_index_rows))]
        
        data_to_save = self.network_current_storage[:end_idx] if end_idx is not None else self.network_current_storage
        average_jumps_df = pd.DataFrame(data_to_save, columns=avg_j_cols)

        average_jumps_df.to_parquet(path, engine='pyarrow', index=False)

    def resistance_to_path(self, path: str, end_idx: int = None) -> None:
        """
        Save the average dynamic resistances of the memristive junctions to a Parquet file.
        """
        if not hasattr(self, 'resistance_storage') or self.resistance_storage is None:
            return 
                        
        adv_index_rows, adv_index_cols = self.tunneling.get_advanced_indices()
        avg_r_cols = [f"{adv_index_rows[i]}_{adv_index_cols[i]}" for i in range(len(adv_index_rows))]
        
        data_to_save = self.resistance_storage[:end_idx] if end_idx is not None else self.resistance_storage
        resistance_df = pd.DataFrame(data_to_save, columns=avg_r_cols)

        resistance_df.to_parquet(path, engine='pyarrow', index=False)