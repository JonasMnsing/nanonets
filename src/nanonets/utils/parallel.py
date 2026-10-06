
import numpy as np
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, Callable

from nanonets import Simulation

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
        logging.info(f"Done {topology}")
    except Exception:
        logging.exception(f"Error for topology {topology}")

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
        logging.info(f"Done {topology}")
    except Exception:
        logging.exception(f"Error for topology {topology}")

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