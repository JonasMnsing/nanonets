import numpy as np

def get_frequency_spectrum(signal, dt):
    """
    Computes the calibrated one-sided amplitude spectrum.
    
    Parameters:
    - signal: 1D numpy array (time domain)
    - dt: Time step in seconds
    
    Returns:
    - freqs: Array of frequencies [Hz]
    - amplitudes: Array of physical amplitudes [A] (or whatever unit signal is in)
    """
    N_samples = len(signal)
    
    # 1. Apply Window (Consistent with your previous analysis)
    window = np.blackman(N_samples)
    y_val = signal - np.mean(signal)
    y_windowed = y_val * window
    
    # Window Coherent Gain Correction
    w_gain = np.sum(window) / N_samples
    
    # 2. Compute FFT
    fft_vals = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(N_samples, dt)
    
    # 3. Normalize Magnitude
    # Scale for one-sided (*2), normalize by N, correct for window loss
    amplitudes = (np.abs(fft_vals) * 2 / N_samples) / w_gain
    
    return freqs, amplitudes

def extract_harmonic_features(y_val, n_vals, N_periods=20, search_range=3, mode='complex', pad_len=None, interpolate=True):
    """
    Extracts harmonic features from a time series using FFT with optional zero-padding
    and parabolic interpolation for high-precision peak estimation.

    Parameters
    ----------
    y_val : array-like
        Input time series (e.g., current or voltage).
    n_vals : list or array-like of int
        List of harmonic orders to extract (e.g., [1, 2, 3] or [3, 5, 7]).
    N_periods : float, optional (default=20)
        Number of full periods of the fundamental frequency contained in y_val.
        Used to identify the theoretical FFT bin index of the fundamental.
    search_range : int, optional (default=3)
        Number of FFT bins to search left/right of the ideal index to account for 
        spectral leakage or small frequency deviations.
    mode : str, optional (default='complex')
        Format of the output features:
        - 'complex':   [Re, Im] pairs relative to fundamental phase. (Length = 2 * len(n_vals))
                       Preserves full shape info; ideal for Volumetric analysis.
        - 'abs':       [Magnitude] only. (Length = len(n_vals))
                       Ideal for THD or Centroid analysis.
        - 'projected': [Magnitude * cos(delta_phi)]. (Length = len(n_vals))
                       Scalar projection onto fundamental axis (lossy).
        - 'phase':     [Phase Difference] in normalized radians/pi [-1, 1].
    pad_len : int or None, optional (default=None)
        Length of the FFT padding. If None, uses len(y_val). 
        Higher values (e.g., 4096) increase spectral density, reducing discretization error.
    interpolate : bool, optional (default=True)
        If True, uses parabolic interpolation on the spectral peak to estimate 
        magnitude and frequency more accurately than the discrete FFT bin resolution.
        Recommended to avoid "bimodality" in error distributions.

    Returns
    -------
    np.ndarray
        Array of feature values. Structure and length depend on 'mode'.
    """
    
    # --- 1. Preprocessing: Detrend and Window ---
    y_val = np.array(y_val)
    # Remove DC offset to prevent 0-Hz peak leakage
    y_val = y_val - np.mean(y_val)
    
    N_samples = len(y_val)
    
    # Blackman window minimizes spectral leakage (low sidelobes)
    window = np.blackman(N_samples)
    y_windowed = y_val * window
    
    # Coherent Gain Correction: compensates for energy amplitude lost due to windowing
    w_gain = np.sum(window) / N_samples
    
    # --- 2. FFT Calculation ---
    # Determine padding
    if pad_len is None:
        n_fft = N_samples
        pad_ratio = 1.0
    else:
        n_fft = pad_len
        pad_ratio = n_fft / N_samples
    
    # rfft is efficient for real-valued inputs; returns positive freqs only
    fft_vals = np.fft.rfft(y_windowed, n=n_fft)

    # Helper: Peak Extraction Logic
    def get_peak_metrics(neighborhood):
        """
        Internal helper to find peak magnitude and phase from a spectral slice.
        Handles both discrete argmax and parabolic interpolation.
        """
        # Safety check for empty or zero-signal neighborhoods
        if len(neighborhood) == 0 or np.max(np.abs(neighborhood)) < 1e-15:
            return 0.0, 0.0

        # Magnitude spectrum of the neighborhood
        mags = np.abs(neighborhood)
        idx_local = np.argmax(mags)
        
        # 1. Discrete Estimates (Base)
        mag_peak = mags[idx_local]
        # Phase at the discrete peak bin
        phase_peak = np.angle(neighborhood[idx_local])
        
        # 2. Parabolic Interpolation (Refinement)
        if interpolate:
            # We need 3 points: (left, center, right)
            # Check bounds to ensure we have neighbors
            if 0 < idx_local < len(mags) - 1:
                alpha = mags[idx_local - 1]
                beta  = mags[idx_local]
                gamma = mags[idx_local + 1]
                
                # Calculate fractional peak shift 'delta' (-0.5 to 0.5)
                # Formula: Parabolic peak location
                denominator = (alpha - 2 * beta + gamma)
                if denominator != 0:
                    delta = 0.5 * (alpha - gamma) / denominator
                    
                    # Refined Magnitude Estimate
                    mag_peak = beta - 0.25 * (alpha - gamma) * delta
                    
                    # Optional: We stick to the discrete bin phase for robustness, 
                    # but the magnitude is now "physics-accurate".

        return mag_peak, phase_peak

    # --- 3. Find Fundamental Phase (Reference Frame) ---
    k_fund_ideal = int(round(1.0 * N_periods * pad_ratio))
    
    # Define search window (scaled by pad_ratio)
    s_width = int(search_range * pad_ratio)
    f_start = max(0, k_fund_ideal - s_width)
    f_end   = min(len(fft_vals), k_fund_ideal + s_width + 1)
    
    fund_neighborhood = fft_vals[f_start:f_end]
    
    # Get Fundamental Metrics
    amp_fund_raw, phi_1 = get_peak_metrics(fund_neighborhood)
    
    # Note: amp_fund_raw is currently in "FFT units". 
    # We don't convert it to physical units yet because we only need phi_1 here.
    # (Though for consistency, it's good to know amp_fund is available).

    features = []

    # Scaling Factor: Convert FFT amplitude to Physical Amplitude
    #   * 2.0: Accounts for negative frequencies dropped by rfft
    #   * / N_samples: Normalization by original signal length (not padded length)
    #   * / w_gain: Correction for window attenuation
    phys_scale = (2.0 / N_samples) / w_gain

    # --- 4. Extract Harmonics ---
    for m in n_vals:
        # Ideal bin for m-th harmonic
        k_ideal = int(round(m * N_periods * pad_ratio))
        
        # Search window
        start = max(0, k_ideal - s_width)
        end   = min(len(fft_vals), k_ideal + s_width + 1)
        
        # Check if harmonic is within Nyquist limit
        if start < len(fft_vals) and start < end:
            neighborhood = fft_vals[start:end]
            
            # Get Harmonic Metrics
            mag_raw, phi_m = get_peak_metrics(neighborhood)
            
            # Convert to Physical Magnitude
            mag = mag_raw * phys_scale
            
            # Phase relative to fundamental (Shift Invariant)
            delta_phi = phi_m - (m * phi_1)
            
            # --- Feature Formatting ---
            if mode == 'complex':
                # Map to 2D plane: (x, y) coordinates
                features.append(mag * np.cos(delta_phi))
                features.append(mag * np.sin(delta_phi))

            elif mode == 'phase':
                # Normalized Wrapped Phase [-1, 1]
                wrapped = (delta_phi + np.pi) % (2 * np.pi) - np.pi
                features.append(wrapped / np.pi)
                
            elif mode == 'projected':
                # Scalar projection onto Real axis
                features.append(mag * np.cos(delta_phi))
                
            else: # mode == 'abs'
                # Pure Magnitude
                features.append(mag)
                
        else:
            # Handle out-of-bounds (e.g., frequencies > Nyquist)
            if mode == 'complex':
                features.extend([0.0, 0.0])
            else:
                features.append(0.0)
                
    return np.array(features)

def compute_thd(amplitudes):
    """
    Calculates Total Harmonic Distortion (THD).
    Input: Array of amplitudes [A_fund, A_harm1, A_harm2, ...]
    """
    if len(amplitudes) < 2 or amplitudes[0] == 0:
        return 0.0
        
    power_fund = amplitudes[0]**2
    power_harm = np.sum(amplitudes[1:]**2)
    
    return np.sqrt(power_harm) / np.sqrt(power_fund)