from . import tunneling
from . import monte_carlo
import numpy as np
import pandas as pd
import os.path

class Simulation(tunneling.NanoparticleTunneling):
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
    high_C_output : bool, optional
        If True, adds a high-capacitance output electrode.
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

    def __init__(self, topology_parameter : dict, folder: str = '', add_to_path: str = "", res_info: dict = None, res_info2: dict = None,
                 np_info: dict = None, np_info2: dict = None, seed: int = None, high_C_output: bool = False, pack_optimizer: bool = False, **kwargs):
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
        high_C_output : bool, optional
            Whether to add a high-capacitance output electrode (default: False).
        kwargs : dict
            Additional keyword arguments (e.g., 'del_n_junctions').

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

        # --- Electrode type and inheritance ---
        if "electrode_type" in kwargs:
            electrode_type  = kwargs['electrode_type']
        else:
            electrode_type  = topology_parameter['electrode_type']
        super().__init__(electrode_type, seed)

        # --- Topology type ---
        if 'Nx' in topology_parameter:
            if ((topology_parameter['Nx'] == 1) and (topology_parameter['Ny'] == 1)):
                self.network_topology = 'set'
            else:
                self.network_topology = 'lattice'
        elif 'Np' in topology_parameter:
            self.network_topology = 'random'
        else:
            self.network_topology = None

        # --- Default NP info ---
        if np_info is None:
            np_info = {
                "eps_r"         : 2.6,  # Permittivity of molecular junction 
                "eps_s"         : 3.9,  # Permittivity of oxide layer
                "mean_radius"   : 10.0, # average nanoparticle radius
                "std_radius"    : 0.0   # standard deviation of nanoparticle radius
            }

        # --- Default Resistance info ---
        if res_info is None:
            res_info = {
                "mean_R"    : 25.0, # Average resistance
                "std_R"     : 0.0,  # Standard deviation of resistances
                "dynamic"   : False # Dynamic or constant resistances
            }
        
        # --- Dynamic Junction Resistances ---
        self.dynamic_resistances = res_info['dynamic']
        if self.dynamic_resistances:
            self.dynamic_resistances_info = res_info

        # --- Lattice topology ---
        if self.network_topology == "lattice":
            # Path variable
            path_var = f'Nx={topology_parameter["Nx"]}_Ny={topology_parameter["Ny"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # --- Topology ---
            self.lattice_network(N_x=topology_parameter["Nx"], N_y=topology_parameter["Ny"])
            self.add_electrodes_to_lattice_net(topology_parameter["e_pos"])

            if high_C_output:
                self.add_np_to_output()
                                    
        # --- Random Topology ---
        elif self.network_topology == "random":
            # Path variable
            path_var = f'Np={topology_parameter["Np"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # --- Topology ---
            self.random_network(N_particles=topology_parameter["Np"])
            self.add_electrodes_to_random_net(electrode_positions=topology_parameter["e_pos"])

            if high_C_output:
                self.add_np_to_output()

        # --- Single Electron Transistor ---      
        elif self.network_topology == "set":
            self.init_SET(np_info['mean_radius'], np_info['eps_r'], np_info['eps_s'], res_info['mean_R'], res_info['std_R'])
            path_var = 'set'+add_to_path+'.csv'
        
        else:
            # Path variable
            path_var = 'custom_network'+add_to_path+'.csv'
        
        if self.network_topology != "set":
            # --- Electrostatics ---
            if self.network_topology is not None:
                self.init_nanoparticle_radius(np_info['mean_radius'], np_info['std_radius'])
                if np_info2 is not None:
                    self.update_nanoparticle_radius(np_info2['np_index'], np_info2['mean_radius'], np_info2['std_radius'])
                if pack_optimizer:
                    self.pack_planar_circles()
                else:
                    self.pack_lattice()
            else:
                self.net_topology           = kwargs["net_topology"]
                self.dist_matrix            = kwargs["dist_matrix"]
                self.electrode_dist_matrix  = kwargs["electrode_dist_matrix"]
                self.radius_vals            = kwargs["radius_vals"]
                self.N_particles            = self.electrode_dist_matrix.shape[1]
                self.N_electrodes           = self.electrode_dist_matrix.shape[0]
                self.N_junctions            = self.net_topology.shape[1]-1

            self.calc_capacitance_matrix(np_info['eps_r'], np_info['eps_s'])
            self.calc_electrode_capacitance_matrix()

            # --- Tunneling ---
            self.init_adv_indices()
            self.init_junction_resistances(res_info['mean_R'], res_info['std_R'])
            if res_info2 is not None:
                self.update_junction_resistances_at_random(res_info2['N'], res_info2['mean_R'], res_info2['std_R'])
            self.init_const_capacitance_values()

        # --- Path ---
        self.folder = folder
        self.path1  = folder + path_var
        self.path2  = folder + 'mean_state_'    + path_var
        self.path3  = folder + 'net_currents_'  + path_var

    def run_static_voltages(self, voltages: np.ndarray, target_electrode: int, T_val: float = 0.1, sim_dic: dict = None,
                            save_th: int = None, verbose: bool = False):
        """
        Run an ensemble of Kinetic Monte Carlo trajectories at fixed electrode voltages 
        to extract the steady-state macroscopic current.
        """
        
        # --- Default Ensemble Simulation Parameters ---
        if sim_dic is None:
            sim_dic = {
                "n_trajectories" : 400,         # Number of independent KMC runs per voltage
                "sim_time"       : 3.0e-6,      # Production duration per trajectory [s]
                "eq_time"        : 1.5e-6,      # Equilibration duration per trajectory [s]
                "ac_time"        : 150e-9,      # Base parameter for Numba initialization
                "max_jumps"      : 20000,       # Safety break
                "max_eq_jumps"   : 100000
            }

        n_trajectories  = sim_dic.get('n_trajectories', 400)
        sim_time        = sim_dic.get('sim_time', 3.0e-6)
        eq_time         = sim_dic.get('eq_time', 1.5e-6)
        ac_time         = sim_dic.get('ac_time', 150e-9)
        max_jumps       = sim_dic.get('max_jumps', 20000)
        max_eq_jumps    = sim_dic.get('max_eq_jumps', 100000)

        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        
        # Init global storage lists
        self.clear_simulation_outputs()
        
        j = 0
        for i, voltage_values in enumerate(voltages):
            
            # --- Get all CONSTANT KMC model input arrays ---
            inv_capacitance_matrix          = self.get_inv_capacitance_matrix()
            const_capacitance_values        = self.get_const_capacitance_values()
            N_particles, N_electrodes       = self.get_particle_electrode_count()
            adv_index_rows, adv_index_cols  = self.get_advanced_indices()
            temperatures                    = self.get_const_temperatures(T=T_val)
            resistances                     = self.get_tunneling_rate_prefactor()

            # --- Preallocate arrays for this voltage's ensemble ---
            ens_observables = np.zeros(n_trajectories)
            ens_potentials  = np.zeros((n_trajectories, N_particles + N_electrodes))
            ens_jumps       = np.zeros(n_trajectories)
            if verbose:
                ens_charges     = np.zeros((n_trajectories, N_particles))
                ens_currents    = np.zeros((n_trajectories, len(adv_index_rows)))

            # =======================================================
            # ENSEMBLE TRAJECTORY LOOP
            # =======================================================
            for traj in range(n_trajectories):
                
                # Update electrostatics strictly for this fresh trajectory
                self.init_charge_vector(voltage_values=voltage_values)
                self.init_potential_vector(voltage_values=voltage_values)
                charge_vector    = self.get_charge_vector()
                potential_vector = self.get_potential_vector()

                # Instantiate (Numba-optimized) model
                self.model = monte_carlo.MonteCarlo(
                    charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
                    temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles,
                    floating_electrodes, ac_time
                )

                # 1. Equilibrate
                eq_jumps = self.model.run_equilibration_duration(eq_time, max_eq_jumps)
                
                # 2. Production Run
                self.model.kmc_simulation_trajectory(target_electrode, sim_time, max_jumps)

                # 3. Store this specific trajectory's results
                ens_observables[traj]   = self.model.target_observable_mean
                ens_potentials[traj, :] = self.model.potential_mean
                ens_jumps[traj]         = self.model.total_jumps

                if verbose:
                    ens_charges[traj, :]    = self.model.charge_mean
                    ens_currents[traj, :]   = self.model.network_currents

            # =======================================================
            # ENSEMBLE AVERAGING
            # =======================================================
            self.observable_storage.append(np.mean(ens_observables))
            # Standard Error of the Mean = std / sqrt(N)
            self.observable_error_storage.append(np.std(ens_observables, ddof=1) / np.sqrt(n_trajectories))
            
            self.potential_storage.append(np.mean(ens_potentials, axis=0))
            self.eq_jump_storage.append(eq_jumps) 
            self.jump_storage.append(np.mean(ens_jumps))
            self.time_storage.append(sim_time)

            if verbose:
                self.state_storage.append(np.mean(ens_charges, axis=0))
                self.network_current_storage.append(np.mean(ens_currents, axis=0))


            # --- Periodically save to disk ---
            if (save_th is not None and ((i + 1) % save_th == 0)):
                self.data_to_path(voltages[j:(i + 1),:], self.path1)
                self.potential_to_path(self.path2)
                if verbose:
                    self.network_current_to_path(self.path3)
                self.clear_simulation_outputs()
                j = i + 1
    
    
    # def run_static_voltages(self, voltages : np.ndarray, target_electrode : int, T_val: float = 0.1, sim_dic: dict = None, save_th: int = None, verbose: bool = False):
    #     """
    #     Run a kinetic Monte Carlo simulation at fixed electrode voltages, estimating either the steady-state
    #     current or, for floating electrodes, the equilibrium potential at a selected electrode.

    #     Parameters
    #     ----------
    #     voltages : np.ndarray
    #         2D array of electrode voltages. Each row is a distinct set of electrode voltages [V].
    #     target_electrode : int
    #         Index of the electrode for which to record current (constant) or potential (floating).
    #     T_val : float, optional
    #         Network temperature [K]. Default: 1.0.
    #     sim_dic : dict, optional
    #         Dictionary of simulation parameters:
    #             error_th        : Target relative error of the main observable.
    #             max_jumps       : Maximum KMC steps per voltage point.
    #             eq_steps        : Number of equilibration KMC steps before measurement.
    #             jumps_per_batch : KMC steps per batch.
    #             kmc_counting    : If True, use event-counting for current.
    #             min_batches     : Minimum measurement batches.
    #     save_th : int, optional
    #         Frequency (in voltage steps) with which simulation data is saved to file.
    #     verbose : bool, optional
    #         If True, records and stores extra simulation data (longer runtime, larger outputs).

    #     Returns
    #     -------
    #     None
    #         Results are saved to file and/or stored in class attributes.
    #     """
    #     # Round voltages to 0.01 mV
    #     voltages = np.round(voltages,5)
        
    #     # --- Default Simulation Parameter ---
    #     if sim_dic is None:
    #         sim_dic =   {
    #             "duration"        : False,
    #             "ac_time"         : 40e-9,
    #             "error_th"        : 0.05,
    #             "max_jumps"       : 10000000,
    #             "n_eq"            : 100000,
    #             "n_per_batch"     : 2000,
    #             "kmc_counting"    : False,
    #             "min_batches"     : 5
    #         }

    #     # --- Get Simulation Parameter ---            
    #     error_th        = sim_dic['error_th']
    #     max_jumps       = sim_dic['max_jumps']
    #     n_eq            = sim_dic['n_eq']
    #     n_per_batch     = sim_dic['n_per_batch']
    #     kmc_counting    = sim_dic['kmc_counting']
    #     min_batches     = sim_dic['min_batches']
    #     duration        = sim_dic['duration']
    #     ac_time         = sim_dic['ac_time']

    #     # Identify floating electrodes and detect if target electrode is floating
    #     floating_electrodes = np.where(self.electrode_type == 'floating')[0]
    #     output_potential    = (self.electrode_type[target_electrode] == 'floating')
        
    #     # Init storage lists
    #     self.clear_simulation_outputs()
        
    #     j = 0
    #     for i, voltage_values in enumerate(voltages):
            
    #         # --- Update network electrostatics ---
    #         self.init_charge_vector(voltage_values=voltage_values)
    #         self.init_potential_vector(voltage_values=voltage_values)

    #         # --- Get all KMC model input arrays ---
    #         inv_capacitance_matrix          = self.get_inv_capacitance_matrix()
    #         charge_vector                   = self.get_charge_vector()
    #         potential_vector                = self.get_potential_vector()
    #         const_capacitance_values        = self.get_const_capacitance_values()
    #         N_particles, N_electrodes       = self.get_particle_electrode_count()
    #         adv_index_rows, adv_index_cols  = self.get_advanced_indices()
    #         temperatures                    = self.get_const_temperatures(T=T_val)
    #         resistances                     = self.get_tunneling_rate_prefactor()

    #         # --- Instantiate (Numba-optimized) model ---
    #         self.model = monte_carlo.MonteCarlo(
    #             charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
    #             temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles,
    #             floating_electrodes, ac_time)

    #         # --- Run KMC simulation (with or without dynamic resistances) ---
    #         if self.dynamic_resistances:
    #             eq_jumps = self.model.run_equilibration_steps_var_resistance(
    #                 n_eq, 
    #                 self.dynamic_resistances_info['slope'], 
    #                 self.dynamic_resistances_info['shift'],
    #                 self.dynamic_resistances_info['tau_0'],
    #                 self.dynamic_resistances_info['R_max'],
    #                 self.dynamic_resistances_info['R_min']
    #             )
                
    #             # Production Run until Current at target electrode is less than error_th or max_jumps was passed
    #             self.model.kmc_simulation_var_resistance(
    #                 target_electrode, error_th, max_jumps, n_per_batch, 
    #                 self.dynamic_resistances_info['slope'],
    #                 self.dynamic_resistances_info['shift'],
    #                 self.dynamic_resistances_info['tau_0'],
    #                 self.dynamic_resistances_info['R_max'],
    #                 self.dynamic_resistances_info['R_min'],
    #                 kmc_counting, verbose
    #             )
    #         else:
    #             if duration:
    #                 # Check if total rate constant is less than 1e-10[1/s]: pass
    #                 eq_jumps = self.model.run_equilibration_duration(n_eq)
    #                 self.model.kmc_simulation_duration(
    #                     target_electrode, error_th, max_jumps, n_per_batch,
    #                     output_potential, kmc_counting, min_batches
    #                 )
    #             else:
    #                 eq_jumps = self.model.run_equilibration_steps(n_eq)
    #                 self.model.kmc_simulation(
    #                     target_electrode, error_th, max_jumps, n_per_batch,
    #                     output_potential, kmc_counting, min_batches, verbose
    #                 )
            
    #         # --- Collect simulation results ---
    #         self.observable_storage.append(self.model.get_observable())
    #         self.observable_error_storage.append(self.model.get_observable_error())
    #         self.state_storage.append(self.model.get_state())
    #         self.potential_storage.append(self.model.get_potential())
    #         self.network_current_storage.append(self.model.get_network_current())
    #         self.eq_jump_storage.append(eq_jumps)
    #         self.jump_storage.append(self.model.get_jump())
    #         self.time_storage.append(self.model.get_time())

    #         # --- Store extra data if verbose ---
    #         if verbose:
    #             self.observable_per_batch.append(self.model.get_target_observable_per_it())
    #             self.time_per_batch.append(self.model.get_time_per_it())
    #             self.potential_per_batch.append(self.model.get_potential_per_it())
    #             # self.jump_per_batch.append(model.get_jump_per_batch())
    #             if self.dynamic_resistances:
    #                 self.resistance_per_batch.append(self.model.get_resistances_per_it())
                
    #         # --- Periodically save to disk ---
    #         if (save_th is not None and ((i + 1) % save_th == 0)):
    #             self.data_to_path(voltages[j:(i + 1),:], self.path1)
    #             self.potential_to_path(self.path2)
    #             self.network_current_to_path(self.path3)
    #             self.clear_simulation_outputs()

    #             j = i+1

    def run_dynamic_voltages(self, voltages: np.ndarray, time_steps: np.ndarray, target_electrode: int, T_val: float = 0.1, eq_steps: int = 0, save: bool = False,
                         stat_size: int = 10, init_charges: bool = None, verbose: bool = False):
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

        # Round voltages to 0.01 mV
        voltages = np.round(voltages,5)
        
        # Identify floating electrodes and detect if target electrode is floating
        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        const_electrodes    = np.where(self.electrode_type == 'constant')[0]
        output_potential    = (self.electrode_type[target_electrode] == 'floating')

        # --- Initialize network for first time step ---
        self.init_charge_vector(voltage_values=voltages[0])
        self.init_potential_vector(voltage_values=voltages[0])

        # --- Gather system parameters for KMC model ---
        inv_capacitance_matrix          = self.get_inv_capacitance_matrix()
        charge_vector                   = self.get_charge_vector()
        potential_vector                = self.get_potential_vector()
        const_capacitance_values        = self.get_const_capacitance_values()
        N_particles, N_electrodes       = self.get_particle_electrode_count()
        adv_index_rows, adv_index_cols  = self.get_advanced_indices()
        temperatures                    = self.get_const_temperatures(T=T_val)
        resistances                     = self.get_tunneling_rate_prefactor()

        # Slowest Time Constant
        tau_0 = self.get_slowest_linear_time_constant()
        
        # --- Instantiate (Numba-optimized) model ---
        self.model = monte_carlo.MonteCarlo(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
                                temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles, floating_electrodes, tau_0=tau_0)

        # --- Equilibration (unless using pre-initialized states) ---
        if init_charges is None:
            if self.dynamic_resistances:
                eq_jumps = self.model.run_equilibration_steps_var_resistance(
                        eq_steps, 
                        self.dynamic_resistances_info['slope'], 
                        self.dynamic_resistances_info['shift'],
                        self.dynamic_resistances_info['tau_0'],
                        self.dynamic_resistances_info['R_max'],
                        self.dynamic_resistances_info['R_min']
                    )
            else:
                eq_jumps = self.model.run_equilibration_steps(eq_steps)

        # Initial time
        self.model.time = time_steps[0]
        
        # Subtract charges induced by initial electrode voltages for charge-neutrality
        offset                      = self.get_charge_vector_offset(voltage_values=voltages[0])
        self.model.charge_vector    = self.model.charge_vector - offset
        
        # --- Ensemble initial states ---
        if init_charges is None:
            self.q_eq   = np.tile(self.model.charge_vector.copy(), (stat_size,1))
        else:
            eq_jumps    = 0
            self.q_eq   = init_charges

        # --- Allocate result arrays ---
        n_time      = voltages.shape[0]
        n_junctions = len(self.model.adv_index_rows)
        self.observable_storage         = np.zeros(n_time)
        self.state_storage              = np.zeros(shape=(n_time, self.model.N_particles))
        self.potential_storage          = np.zeros(shape=(n_time, self.model.N_particles+self.model.N_electrodes))
        self.network_current_storage    = np.zeros(shape=(n_time, n_junctions))
        self.jump_storage               = np.zeros(n_time)

        # Store equilibrated charge distribution
        observable = np.zeros(shape=(stat_size, n_time))
        
        # --- Main simulation loop: ensemble average ---
        for s in range(stat_size):
            self.model.charge_vector = self.q_eq[s,:].copy()
            for i, voltage_values in enumerate(voltages):
                # Apply charging state from electrode voltage
                offset                      =  self.get_charge_vector_offset(voltage_values=voltage_values)
                self.model.charge_vector    += offset
                
                # Define given time and time target
                self.model.time = time_steps[i]
                time_target     = time_steps[i+1]

                # Update constant electrode potentials
                self.model.potential_vector[const_electrodes] = voltage_values[const_electrodes]
                                
                if self.dynamic_resistances:
                    self.model.kmc_time_simulation_var_resistance(
                        target_electrode, time_target,
                        self.dynamic_resistances_info['slope'], 
                        self.dynamic_resistances_info['shift'],
                        self.dynamic_resistances_info['tau_0'],
                        self.dynamic_resistances_info['R_max'],
                        self.dynamic_resistances_info['R_min']
                    )
                else:
                    self.model.kmc_time_simulation(target_electrode, time_target, output_potential)
                    target_observable_mean  = self.model.get_observable()
                    total_jumps             = self.model.get_jump()
                
                # Add observables to outputs
                observable[s,i]                     =  target_observable_mean
                self.state_storage[i,:]             += self.model.get_state() / stat_size
                self.potential_storage[i,:]         += self.model.get_potential() / stat_size
                self.network_current_storage[i,:]   += self.model.get_network_current() / stat_size
                self.jump_storage[i]                += total_jumps / stat_size

                # Remove voltage offset for next step
                self.model.charge_vector -= offset

            # Store last charge vector for each run
            self.q_eq[s,:] = self.model.charge_vector.copy()

        # --- Statistics and final result arrays ---
        self.observable_storage         = np.mean(observable, axis=0)
        self.observable_error_storage   = 1.96*np.std(observable, axis=0, ddof=1) / np.sqrt(stat_size)
        self.eq_jump_storage            = np.repeat(eq_jumps, len(self.observable_storage))

        # Prepare output voltage arrays for saving
        V_safe_vals                         = np.zeros(shape=(self.potential_storage.shape[0],self.N_electrodes+1))
        V_safe_vals[:,floating_electrodes]  = self.potential_storage[:,floating_electrodes]
        V_safe_vals[:,const_electrodes]     = voltages[:, const_electrodes]
        V_safe_vals[:,-1]                   = voltages[:, -1]

        if save:
            self.data_to_path(V_safe_vals, self.path1)
            self.potential_to_path(self.path2)
            self.network_current_to_path(self.path3)
        
    def clear_simulation_outputs(self) -> None:
        """Clears simulation outputs before running a new set of voltages."""
        
        # Default Data
        self.observable_storage         = []
        self.observable_error_storage   = []
        self.state_storage              = []
        self.potential_storage          = []
        self.network_current_storage    = []
        self.eq_jump_storage            = []
        self.jump_storage               = []
        self.time_storage               = []
        
        # Verbose Data
        self.observable_per_batch   = []
        # self.jump_per_batch         = []
        self.time_per_batch         = []
        self.potential_per_batch    = []
        self.resistance_per_batch   = []

    def get_observable_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of main observable values for each simulation condition or time point.
        """
        return np.array(self.observable_storage)
    
    def get_observable_error_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of standard errors (or 95% CI) for each observable.
        """
        return np.array(self.observable_error_storage)
    
    def get_state_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of averaged nanoparticle charge states per condition or time point.
        """
        return np.array(self.state_storage)
    
    def get_potential_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of averaged potentials (for all particles and electrodes) per condition or time point.
        """
        return np.array(self.potential_storage)
    
    def get_network_current_storage(self) -> dict:
        """
        Returns
        -------
        dict
            Dictionary mapping each (row, col) pair (junction) to its average network current.
            Keys are (origin, target) tuples, values are floats (current).
        """
        return {
            (self.adv_index_rows[i], self.adv_index_cols[i]) : np.array(self.network_current_storage)[:,i].copy()
            for i in range(len(self.network_current_storage[0]))
        }
    
    def get_eq_jump_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of number of jumps performed during equilibration for each simulation point.
        """
        return np.array(self.eq_jump_storage)
    
    def get_jump_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of total number of KMC jumps per simulation point.
        """
        return np.array(self.jump_storage)
    
    def get_time_storage(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of simulation times for each voltage point or condition.
        """
        return np.array(self.time_storage)
    
    def get_observable_per_batch(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of observable values for each batch (if verbose simulation).
        """
        return np.array(self.observable_per_batch)
    
    # def get_jump_per_batch(self) -> np.ndarray:
    #     """
    #     Returns
    #     -------
    #     np.ndarray
    #         Array of jump counts for each batch and each possible event (if verbose).
    #     """
    #     return np.array(self.jump_per_batch)

    def get_time_per_batch(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of simulation time per batch (if verbose simulation).
        """
        return np.array(self.time_per_batch)
    
    def get_potential_per_batch(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Array of averaged potential landscapes per batch (if verbose simulation).
        """
        return np.array(self.potential_per_batch)
    
    def get_resistance_per_batch(self) -> dict:
        """
        Returns
        -------
        dict
            Dictionary mapping (origin, target) pairs to average resistance values per batch.
        """
        return {(self.adv_index_rows[i], self.adv_index_cols[i]) : val for i, val in enumerate(self.resistance_per_batch)}
    
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
        # Get storage arrays and ensure shape is correct for stacking
        val_a   = self.get_eq_jump_storage().reshape(-1, 1)
        val_b   = self.get_jump_storage().reshape(-1, 1)
        val_c   = self.get_observable_storage().reshape(-1, 1)
        val_d   = self.get_observable_error_storage().reshape(-1, 1)

        data    = np.hstack([voltages, val_a, val_b, val_c, val_d])

        # Use all electrode columns
        columns = [f'E{i}' for i in range(voltages.shape[1]-1)]
        columns = np.array(columns + ['G', 'Eq_Jumps', 'Jumps', 'Observable', 'Error'])

        df          = pd.DataFrame(data)
        df.columns  = columns

        # Save or append
        if (os.path.isfile(path)):
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, header=True, index=False)

    def potential_to_path(self, path: str) -> None:
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
        microstates_df = pd.DataFrame(self.potential_storage)

        if (os.path.isfile(path)):
            microstates_df.to_csv(path, mode='a', header=False, index=False)
        else:
            microstates_df.to_csv(path, header=True, index=False)

    def network_current_to_path(self, path : str) -> None:
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
        avg_j_cols                  = [(self.adv_index_rows[i],self.adv_index_cols[i]) for i in range(len(self.adv_index_rows))]
        average_jumps_df            = pd.DataFrame(self.network_current_storage)
        average_jumps_df.columns    = avg_j_cols

        if (os.path.isfile(path)):
            average_jumps_df.to_csv(path, mode='a', header=False, index=False)
        else:
            average_jumps_df.to_csv(path, header=True, index=False)