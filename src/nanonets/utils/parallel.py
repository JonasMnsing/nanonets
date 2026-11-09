
import numpy as np
import os
import socket
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, Callable

from nanonets import Simulation


def setup_compute_env(cluster_base_path: str, script_path: str, experiment_name: str = "") -> Tuple[Path, int]:
    """
    Automatically detects if the code runs locally or on the HPC cluster,
    sets up the output directories, and determines the optimal CPU count.
    
    Parameters
    ----------
    cluster_base_path : str
        The root directory for data on the cluster (e.g., /scratch/...).
    script_path : str
        Always pass __file__ from the calling script to resolve the local project root.
    experiment_name : str, optional
        A subfolder name for this specific sweep (e.g., "amp_beta").
        
    Returns
    -------
    Tuple[Path, int]
        The resolved output directory and the number of CPUs to use.
    """
    script_file = Path(script_path).resolve()
    cluster_base = Path(cluster_base_path)
    
    # --- 1. Path Mirroring ---
    parts = script_file.parts
    if "scripts" in parts:
        # Project-Root
        idx = parts.index("scripts")
        project_root = Path(*parts[:idx])
        
        # Get everything below "scripts" and delte ".py"
        rel_subpath = Path(*parts[idx+1:-1]) / script_file.stem
    else:
        # Fallback, if the script is not inside "scripts"
        project_root = script_file.parents[1]
        rel_subpath = Path(script_file.stem)

    # For an expicit name, overwrite mirroring
    exp_dir_name = experiment_name if experiment_name else rel_subpath

    # --- 2. Environment Routing ---
    is_cluster = cluster_base.parents[1].exists() or "SLURM_JOB_ID" in os.environ

    if is_cluster:
        out_dir = cluster_base / exp_dir_name
        cpu_cnt = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        print(f"🚀 Cluster Mode ({socket.gethostname()}) | CPUs: {cpu_cnt} | Path: {out_dir}")
    else:
        out_dir = project_root / "data" / "raw" / exp_dir_name
        cpu_cnt = 10 # max(1, multiprocessing.cpu_count() - 2) # optimal_workers = max(1, psutil.cpu_count(logical=False) - 2)
        print(f"💻 Local Mode ({socket.gethostname()}) | CPUs: {cpu_cnt} | Path: {out_dir}")

    # Ordnerstruktur anlegen
    out_dir.mkdir(parents=True, exist_ok=True)
    
    return out_dir, cpu_cnt

def run_static_simulation(voltages: np.ndarray, topology: Dict[str, Any], out_folder: Union[Path, str],
                          net_kwargs: Optional[Dict[str,Any]] = None, sim_kwargs: Optional[Dict[str,Any]] = None)->None:
    """Instantiate and run one nanonets static simulation.

    Parameters
    ----------
    voltages : np.ndarray
        2D array of shape (n_volt, n_electrodes) of applied voltages.
    topology : dict
        Network topology
    out_folder : pathlib.Path or str
        Directory where simulation outputs are saved.
    net_kwargs : dict, optional
        Additional keyword arguments for nanonets.simulation(). Defaults to None
    sim_kwargs : dict, optional
        Additional keyword arguments for nanonets.run_static_voltages(). Defaults to None.

    Returns
    -------
    None
    """
    net_kwargs = net_kwargs or {}
    sim_kwargs = sim_kwargs or {}
    run_name = net_kwargs.get('add_to_path', 'Simulation')

    try:
        target = voltages.shape[1] - 2
        sim = Simulation(
            topology_parameter=topology,
            folder=str(out_folder)+"/",
            **net_kwargs,
        )
        sim.run_static_voltages(
            voltages=voltages,
            target_electrode=target,
            **sim_kwargs,
        )
        logging.info(f"Done: {run_name}")
    except Exception:
        logging.exception(f"Error for run: {run_name}")

def run_dynamic_simulation(time_steps: np.ndarray, voltages: np.ndarray, topology: Dict[str, Any], out_folder: Union[Path, str],
                           net_kwargs: Optional[Dict[str,Any]] = None, sim_kwargs: Optional[Dict[str,Any]] = None)->None:
    """Instantiate and run one nanonets simulation.

    Parameters
    ----------
    time_steps : np.ndarray
        1D array of time points (in seconds) for simulation steps.
    voltages : np.ndarray
        2D array of shape (n_steps, n_electrodes) of applied voltages.
    topology : dict
        Network topology with keys 'Nx', 'Ny', 'Nz', 'e_pos', and 'electrode_type'.
    out_folder : pathlib.Path or str
        Directory where simulation outputs are saved.
    net_kwargs : dict, optional
        Additional keyword arguments for nanonets.simulation(). Defaults to None
    sim_kwargs : dict, optional
        Additional keyword arguments for nanonets.run_static_voltages(). Defaults to None.

    Returns
    -------
    None
    """
    sim_kwargs = sim_kwargs or {}
    net_kwargs = net_kwargs or {}
    run_name = net_kwargs.get('add_to_path', 'Simulation')

    try:
        target = voltages.shape[1] - 2
        sim = Simulation(
            topology_parameter=topology,
            folder=str(out_folder)+"/",
            **net_kwargs,
        )
        sim.run_dynamic_voltages(
            voltages=voltages,
            time_steps=time_steps,
            target_electrode=target,
            **sim_kwargs,
        )
        logging.info(f"Done: {run_name}")
    except Exception:
        logging.exception(f"Error for run: {run_name}")

def batch_launch(func: Callable[..., Any], tasks: List[Tuple[Tuple[Any,...]]], max_procs: int)->None:
    """
    Launch tasks in parallel, limiting concurrency to max_procs.

    Parameters
    ----------
    func : callable
        Function to run in each process. Must accept args and kwargs as given.
    tasks : list of (args, kwargs)
        Each task is a tuple: (args, kwargs). `args` is a tuple of positional
        arguments for func; `kwargs` is a dict of keyword arguments.
    max_procs : int
        Maximum number of concurrent processes.

    Returns
    -------
    None
    """
    running = []
    for args, kwargs in tasks:
        while len(running) >= max_procs:
            running.pop(0).join()
        p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        p.start()
        running.append(p)
    for p in running:
        p.join()