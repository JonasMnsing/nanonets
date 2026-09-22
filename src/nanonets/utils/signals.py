import numpy as np

from scipy import signal
from typing import Union, Tuple, List


def uniform_sample(U_e : Union[float, List[float]], N_samples : int, topology_parameter : dict,
                   U_g : Union[float, List[float]] = 0.0, target_electrode : int = -1) -> np.ndarray:
    """Returns a uniform sample of electrode voltages, with floating and target electrodes set Zero and Gate electrode defined individually.

    Parameters
    ----------
    U_e : Union[float, List[float]]
        Electrode Voltage range 
    N_samples : int
        Number of Samples
    topology_parameter : dict
        Network topology dictonary
    U_g : Union[float, List[float]], optional
        Gate voltage range, by default 0.0
    target_electrode : int
        Electrode set to be grounded at start

    Returns
    -------
    np.array
        Uniform sample
    """

    # Parameter based on electrode specification
    N_electrodes = len(topology_parameter["e_pos"])
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx = np.where(electrode_types=="floating")[0]
    
    # Sample Control Voltages
    if isinstance(U_e, list):
        sample = np.random.uniform(low=U_e[0], high=U_e[1], size=(N_samples,N_electrodes+1))
    else:
        sample = np.random.uniform(low=-U_e, high=U_e, size=(N_samples,N_electrodes+1))  
    
    # Sample Gate Voltages
    if isinstance(U_g, list):
        sample[:,-1] = np.random.uniform(low=U_g[0], high=U_g[1], size=N_samples)
    else:
        sample[:,-1] = np.random.uniform(low=-U_g, high=U_g, size=N_samples)

    # Set target and floating electrodes to 0V
    sample[:,floating_idx] = 0.0
    sample[:,target_electrode-1] = 0.0

    return sample

def lhs_sample(U_e : Union[float, List[float]], N_samples : int, topology_parameter : dict,
               U_g : Union[float, List[float]] = 0.0, target_electrode : int = -1) -> np.ndarray:
    """Returns a Latin Hypercube sample of electrode voltages, with floating and target electrodes set Zero and Gate electrode defined individually.

    Parameters
    ----------
    U_e : Union[float, List[float]]
        Electrode Voltage range
    N_samples : int
        Number of Samples
    topology_parameter : dict
        Network topology dictonary
    U_g : Union[float, List[float]], optional
        Gate voltage range, by default 0.0
    target_electrode : int
        Electrode set to be grounded at start

    Returns
    -------
    np.array
        LHS sample
    """

    # Parameter based on electrode specification
    N_electrodes = len(topology_parameter["e_pos"])
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx = np.where(electrode_types=="floating")[0]

    # lhs sample between [0,1]
    lhs_samples = np.zeros((N_samples, N_electrodes+1))
    for i in range(N_electrodes+1):
        intervals = np.linspace(0, 1, N_samples + 1)
        points = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(points)
        lhs_samples[:, i] = points
    scaled_samples = np.zeros_like(lhs_samples)

    # Sample Control Voltages
    if isinstance(U_e, list):
        scaled_samples[:,:N_electrodes] = U_e[0] + (U_e[1] - U_e[0]) * lhs_samples[:,:N_electrodes]
    else:
        scaled_samples[:,:N_electrodes] = -U_e + (U_e + U_e) * lhs_samples[:,:N_electrodes]
    
    # Sample Gate Voltages
    if isinstance(U_g, list):
        scaled_samples[:,-1] = U_g[0] + (U_g[1] - U_g[0]) * lhs_samples[:,-1]
    else:
        scaled_samples[:,-1] = -U_g + (U_g + U_g) * lhs_samples[:,-1]

    # Set floating electrodes to 0V
    scaled_samples[:,floating_idx] = 0.0
    scaled_samples[:,target_electrode-1] = 0.0

    return scaled_samples

def logic_gate_sample(U_e : Union[float, List[float]], input_pos : List[int], N_samples : int, topology_parameter : dict,
                      U_i : Union[float, List[float]] = 0.01, U_g : Union[float, List[float]] = 0.0, sample_technique = 'lhs', target_electrode : int = -1) -> np.ndarray:
    """Returns a sample of electrode voltages, with floating electrodes set Zero and Gate electrodes defined individually.
    At input_pos electrode values are set to Boolean logic states defined in U_i

    Parameters
    ----------
    U_e : Union[float, List[float]]
        Electrode Voltage range
    input_pos : List[int]
        Input Electrode positions
    N_samples : int
        Number of Samples
    topology_parameter : dict
        Network topology dictonary
    U_i : Union[float, List[float]], optional
        Input Voltages, by default 0.01
    U_g : Union[float, List[float]], optional
        Gate voltage range, by default 0.0, by default 0.0
    sample_technique : str, optional
        Sampling technique, either lhs or uniform, by default 'lhs'
    target_electrode : int
        Electrode set to be grounded at start

    Returns
    -------
    np.array
        logic gate sample

    Raises
    ------
    ValueError
        If sample_technique isn't either 'lhs' or 'uniform'
    """

    # Define sampling technique
    if sample_technique == 'lhs':
        sample = lhs_sample(U_e=U_e, N_samples=N_samples, topology_parameter=topology_parameter, U_g=U_g, target_electrode=target_electrode)
    elif sample_technique == 'uniform':
        sample = uniform_sample(U_e=U_e, N_samples=N_samples, topology_parameter=topology_parameter, U_g=U_g, target_electrode=target_electrode)
    else:
        raise ValueError("Sample technique 'lhs' or 'uniform' supported")

    # Repeat Sample 4 times for each logic state
    sample = np.repeat(a=sample, repeats=4, axis=0)

    # Put logic states into sample
    if isinstance(U_i, list):
        sample[:,input_pos[0]] = np.tile([U_i[0],U_i[0],U_i[1],U_i[1]], N_samples)
        sample[:,input_pos[1]] = np.tile([U_i[0],U_i[1],U_i[0],U_i[1]], N_samples)
    else:
        sample[:,input_pos[0]] = np.tile([0.0,0.0,U_i,U_i], N_samples)
        sample[:,input_pos[1]] = np.tile([0.0,U_i,0.0,U_i], N_samples)

    return sample

def sinusoidal_voltages(N_samples : int, topology_parameter : dict, amplitudes : Union[float, List[float]], frequencies : Union[float, List[float]] = 0.0,
                        phase : Union[float, List[float]]=0.0, offset : Union[float, List[float]] = 0.0, time_step : float = 1e-10)->Tuple[np.array,np.array]:
    """Return voltage array containing sinusoidal signals of given frequencies and amplitudes

    Parameters
    ----------
    N_samples : int
        Number of voltage values
    topology_parameter : dict
        Network topology dictonary
    amplitudes : Union[float, List[float]]
        Single amplitude for all constant electrodes or individual amplitude for each electrode
    frequencies : Union[float, List[float]]
        Single frequency for all constant electrodes or individual frequency for each electrode in Hz, if set to zero voltages are constant
    phase: Union[float, List[float]]
        Single phase for all constant electrodes or individual phase for each electrode
    offset: Union[float, List[float]]
        Single offset for all constant electrodes or individual offset for each electrode
    time_step : float, optional
        Time step size in seconds, by default 1e-10

    Returns
    -------
    Tuple[np.array,np.array]
        Time and Voltages
    """
    # Voltages and Time Scale
    N_electrodes = len(topology_parameter["electrode_type"])
    voltages = np.zeros(shape=(N_samples, N_electrodes+1))
    time_steps = time_step*np.arange(N_samples)
    
    # Parameter based on electrode specification
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx = np.where(electrode_types=="floating")[0]

    # Signal properties
    frequencies = N_electrodes*[frequencies] if isinstance(frequencies, (int, float)) else frequencies
    amplitudes = N_electrodes*[amplitudes] if isinstance(amplitudes, (int, float)) else amplitudes
    phase = N_electrodes*[phase] if isinstance(phase, (int, float)) else phase
    offset = N_electrodes*[offset] if isinstance(offset, (int, float)) else offset

    # Voltages for each electrode
    for i in range(N_electrodes):
        voltages[:,i] = amplitudes[i]*np.sin(2*np.pi*frequencies[i]*time_steps+phase[i]) + offset[i]
    
    # Set floating electrodes to 0V
    voltages[:,floating_idx] = 0.0
    
    return time_steps, voltages

def generate_band_limited_noise(duration_s: float, max_amplitude: float = 20e-3, bandwidth_hz: float = 4.35e9, dt_s: float = 1e-11) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates time and white noise arrays with a specified sampling rate (dt) and bandwidth.

    Parameters
    ----------
    duration_s: float
        Total length of the simulation time series (seconds).
    bandwidth_hz: float
        The maximum frequency (Hz) to represent in the signal. (Ensures adequate sampling rate for the physical signal).
    dt_s: float, optional
        The discrete time step (seconds). Must be small enough to satisfy Nyquist. Default: 1e-11
    
    Returns:
        A tuple: (time_series, noise_series)
    """

    # 1. Setup Time
    n_samples   = int(duration_s / dt_s)
    time_series = np.linspace(0, duration_s, n_samples, endpoint=False)

    # 2. Generate Raw White Noise (Beta=0)
    raw_noise = cn.powerlaw_psd_gaussian(0, n_samples)

    # 3. Apply Low-Pass Filter (To enforce 4.35 GHz limit)
    # Nyquist frequency
    nyquist = 0.5 / dt_s

    # Design 4th order Butterworth filter
    # normalized_cutoff = cutoff_hz / nyquist
    sos = signal.butter(4, bandwidth_hz / nyquist, btype='low', output='sos')

    # Apply filter
    filtered_noise = signal.sosfiltfilt(sos, raw_noise)
    
    # 4. Normalize (Robust Method)
    # Instead of dividing by random max, we scale so that 3*Sigma = Max Amplitude.
    # This ensures 99.7% of data is within range, and power is constant.
    sigma = np.std(filtered_noise)
    target_sigma = max_amplitude / 3.0
    
    scaled_noise = filtered_noise * (target_sigma / sigma)
    
    # 5. Hard Clip (Safety)
    # Strictly enforce that no outlier exceeds the voltage limit
    final_noise = np.clip(scaled_noise, -max_amplitude, max_amplitude)
    
    return time_series, final_noise