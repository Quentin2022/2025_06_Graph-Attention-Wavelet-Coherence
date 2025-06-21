import numpy as np
import pywt
from scipy.stats import pearsonr
from scipy.signal import coherence
import os

# Import necessary modules and configuration from train.py
try:
    from utils import create_trainer
except ImportError:
    print("Warning: Unable to import utils module, please ensure utils.py is in the current directory")
    create_trainer = None

# Configuration copied from train.py
CONFIG = {
    'subject': 'A01',              # 'all' or 'A01'-'A09'
    'n_splits': 10,                # Number of folds for K-fold cross validation (not used for splitting now, integrate all data)
    'random_seed': 42,             # Random seed
    'normalize': False,            # Whether to normalize
    'num_corrupt': 0,              # No need to corrupt channels
    'ratio_corrupt': 0,            # No need for corruption
    'if_train_corrupt': False,     # No need for corruption
    'data_dir': '---YOUR PATH---\\eeg_raw_dataset_0.5-40',  # Dataset directory
    'Connec_type': 'MSC'          # Connectivity metric type: 'PCC' or 'MSC'
}

# EEG channel names
CHANNEL_NAMES = [
    'Fz', '2', '3', '4', '5', '6', '7', 
    'C3', '9', 'Cz', '11', 'C4', '13', '14', 
    '15', '16', '17', '18', '19', 'Pz', '21', '22'
]

# Wavelet subband definitions (five-level decomposition)
WAVELET_SUBBANDS = ['D1', 'D2', 'D3', 'D4', 'D5', 'A5']

# Sampling parameters
SAMPLING_RATE = 250  # Hz
N_TIMEPOINTS = 1000
N_CHANNELS = 22
WAVELET_TYPE = 'db4'  # Daubechies 4 wavelet

def load_real_eeg_data():
    """
    Load real EEG data using the method from train.py
    Only use first fold data (already covers the entire dataset)
    """
    if create_trainer is None:
        print("Error: Unable to load utils module, please ensure utils.py is in the current directory")
        return None, None
    
    print("Loading real EEG data...")
    print(f"Subject: {CONFIG['subject']}")
    print(f"Data directory: {CONFIG['data_dir']}")
    
    # Create trainer
    trainer = create_trainer(
        subject=CONFIG['subject'],
        n_splits=CONFIG['n_splits'],
        random_seed=CONFIG['random_seed'],
        normalize=CONFIG['normalize'],
        num_corrupt=CONFIG['num_corrupt'],
        ratio_corrupt=CONFIG['ratio_corrupt'],
        if_train_corrupt=CONFIG['if_train_corrupt'],
        data_dir=CONFIG['data_dir']
    )
    
    # Run data preparation
    success = trainer.run_data_preparation()
    
    if not success:
        print("Data preparation failed!")
        return None, None
    
    # Get cross-validation data
    cv_data, scalers, cv_indices = trainer.get_data()
    
    # Only use first fold data (training set + test set = entire dataset)
    X_train, X_test, y_train, y_test = cv_data[0]
    
    # Merge training and test sets
    final_X = np.concatenate([X_train, X_test], axis=0)
    final_y = np.concatenate([y_train, y_test], axis=0)
    
    print(f"Using first fold data (already covers entire dataset):")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples") 
    print(f"  Total: {final_X.shape[0]} samples")
    print(f"  Data shape: {final_X.shape}")
    print(f"  Label range: {np.min(final_y)} - {np.max(final_y)}")
    print(f"  Samples per class: {np.bincount(final_y.astype(int))}")
    
    return final_X, final_y

def dwt_decompose_signal(signal, wavelet=WAVELET_TYPE, levels=5):
    """
    Perform five-level wavelet decomposition on signal
    
    Parameters:
    -----------
    signal : array
        Input signal
    wavelet : str
        Wavelet type, default 'db4'
    levels : int
        Decomposition levels, default 5
        
    Returns:
    --------
    coeffs : dict
        Dictionary containing coefficients for each subband
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    
    # Organize into dictionary format: [A5, D5, D4, D3, D2, D1]
    decomposed = {}
    decomposed['A5'] = coeffs[0]  # Approximation coefficients (low frequency)
    
    # Detail coefficients (high frequency to low frequency)
    for i in range(1, levels + 1):
        decomposed[f'D{levels + 1 - i}'] = coeffs[i]
    
    return decomposed

def decompose_data_by_dwt(data, wavelet=WAVELET_TYPE, levels=5):
    """
    Decompose data into multiple subbands using discrete wavelet transform
    
    Parameters:
    -----------
    data : array, shape (n_samples, n_channels, n_timepoints)
        Original EEG data
    wavelet : str
        Wavelet type
    levels : int
        Decomposition levels
        
    Returns:
    --------
    decomposed_data : dict
        Dictionary containing decomposed data for each subband
    """
    n_samples, n_channels, n_timepoints = data.shape
    decomposed_data = {}
    
    print(f"Performing wavelet decomposition ({wavelet} wavelet, {levels}-level decomposition)...")
    print(f"Will obtain subbands: {WAVELET_SUBBANDS}")
    
    # Initialize storage space for each subband
    for subband in WAVELET_SUBBANDS:
        decomposed_data[subband] = []
    
    # Perform wavelet decomposition for each sample and channel
    for sample in range(n_samples):
        if sample % 50 == 0:
            print(f"    Processing sample {sample+1}/{n_samples}")
        
        sample_decomp = {}
        for subband in WAVELET_SUBBANDS:
            sample_decomp[subband] = []
        
        for ch in range(n_channels):
            # Perform wavelet decomposition on single channel signal
            signal = data[sample, ch, :]
            coeffs = dwt_decompose_signal(signal, wavelet, levels)
            
            # Store coefficients for each subband
            for subband in WAVELET_SUBBANDS:
                sample_decomp[subband].append(coeffs[subband])
        
        # Add current sample's decomposition results to total data
        for subband in WAVELET_SUBBANDS:
            decomposed_data[subband].append(np.array(sample_decomp[subband]))
    
    # Convert to numpy arrays
    for subband in WAVELET_SUBBANDS:
        decomposed_data[subband] = np.array(decomposed_data[subband])
        print(f"    {subband} subband shape: {decomposed_data[subband].shape}")
    
    print("Wavelet decomposition completed!")
    return decomposed_data

def compute_pearson_correlation(x, y):
    """
    Compute Pearson correlation coefficient between two signals
    
    Parameters:
    -----------
    x, y : array
        Two signals
        
    Returns:
    --------
    correlation : float
        Pearson correlation coefficient raw value (-1 to 1)
    """
    try:
        # Compute Pearson correlation coefficient
        corr, p_value = pearsonr(x, y)
        # Return raw value, preserving positive/negative correlation information
        return corr
    except:
        # Return 0 if computation fails
        return 0.0

def compute_magnitude_squared_coherence(x, y, fs=SAMPLING_RATE, nperseg=None):
    """
    Compute magnitude squared coherence between two signals
    
    Parameters:
    -----------
    x, y : array
        Two signals
    fs : float
        Sampling frequency
    nperseg : int
        Length of each segment, automatically chosen when None
        
    Returns:
    --------
    msc : float
        Average magnitude squared coherence value (0 to 1)
    """
    try:
        # Automatically choose nperseg
        if nperseg is None:
            signal_length = len(x)
            nperseg = min(256, signal_length // 4)
            nperseg = max(nperseg, 8)  # Minimum value
        
        # Compute coherence spectrum
        f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg)
        
        # Return average coherence value
        return np.mean(Cxy)
    except:
        # Return 0 if computation fails
        return 0.0

def compute_connectivity_metric(x, y, metric_type, fs=SAMPLING_RATE):
    """
    Compute connectivity metric based on specified type
    
    Parameters:
    -----------
    x, y : array
        Two signals
    metric_type : str
        Connectivity metric type: 'PCC' or 'MSC'
    fs : float
        Sampling frequency
        
    Returns:
    --------
    connectivity : float
        Connectivity value
    """
    if metric_type == 'PCC':
        return compute_pearson_correlation(x, y)
    elif metric_type == 'MSC':
        return compute_magnitude_squared_coherence(x, y, fs)
    else:
        raise ValueError(f"Unsupported connectivity metric type: {metric_type}. Supported types: 'PCC', 'MSC'")

def get_connectivity_info(metric_type):
    """
    Get information about connectivity metric
    
    Parameters:
    -----------
    metric_type : str
        Connectivity metric type
        
    Returns:
    --------
    info : dict
        Dictionary containing metric information
    """
    info_dict = {
        'PCC': {
            'name': 'Pearson Correlation Coefficient',
            'short_name': 'PCC',
            'range': '(-1, 1)',
            'description': 'Pearson correlation coefficient, measures linear correlation'
        },
        'MSC': {
            'name': 'Magnitude Squared Coherence',
            'short_name': 'MSC', 
            'range': '(0, 1)',
            'description': 'Magnitude squared coherence, measures frequency domain coherence'
        }
    }
    
    if metric_type not in info_dict:
        raise ValueError(f"Unsupported connectivity metric type: {metric_type}")
    
    return info_dict[metric_type]

def compute_connectivity_matrices(decomposed_data, metric_type='PCC'):
    """
    Compute connectivity matrices for each wavelet subband
    
    Parameters:
    -----------
    decomposed_data : dict
        Decomposed data for each subband
    metric_type : str
        Connectivity metric type: 'PCC' or 'MSC'
        
    Returns:
    --------
    connectivity_matrices : dict
        Connectivity matrices for each subband
    """
    connectivity_matrices = {}
    n_samples = next(iter(decomposed_data.values())).shape[0]
    
    # Get connectivity metric information
    metric_info = get_connectivity_info(metric_type)
    
    print(f"\nComputing connectivity matrices...")
    print(f"Connectivity metric: {metric_info['name']} ({metric_info['short_name']})")
    print(f"Value range: {metric_info['range']}")
    print(f"Description: {metric_info['description']}")
    
    for subband in WAVELET_SUBBANDS:
        print(f"  Processing {subband} subband")
        
        subband_data = decomposed_data[subband]
        # Initialize connectivity matrix (n_samples x n_channels x n_channels)
        conn_matrices = np.zeros((n_samples, N_CHANNELS, N_CHANNELS))
        
        for sample in range(n_samples):
            if sample % 50 == 0:
                print(f"    Processing sample {sample+1}/{n_samples}")
            
            # Set diagonal to 1
            np.fill_diagonal(conn_matrices[sample], 1.0)
            
            # Compute upper triangular part
            for i in range(N_CHANNELS):
                for j in range(i+1, N_CHANNELS):
                    # Compute connectivity based on specified type
                    connectivity = compute_connectivity_metric(
                        subband_data[sample, i, :], 
                        subband_data[sample, j, :],
                        metric_type,
                        fs=SAMPLING_RATE
                    )
                    
                    # Fill upper and lower triangular parts
                    conn_matrices[sample, i, j] = connectivity
                    conn_matrices[sample, j, i] = connectivity
        
        connectivity_matrices[subband] = conn_matrices
        print(f"    Completed {subband} ({conn_matrices.shape[0]} samples)")
    
    print("Connectivity computation completed!")
    return connectivity_matrices

def analyze_subband_characteristics(decomposed_data, fs=SAMPLING_RATE):
    """
    Analyze frequency characteristics of each wavelet subband
    
    Parameters:
    -----------
    decomposed_data : dict
        Decomposed data for each subband
    fs : float
        Sampling frequency
    """
    print(f"\nWavelet subband frequency characteristic analysis:")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Wavelet type: {WAVELET_TYPE}")
    
    # Theoretical frequency ranges (based on 5-level decomposition)
    freq_ranges = {
        'D1': (fs/4, fs/2),      # Highest frequency band
        'D2': (fs/8, fs/4),      
        'D3': (fs/16, fs/8),     
        'D4': (fs/32, fs/16),    
        'D5': (fs/64, fs/32),    
        'A5': (0, fs/64)         # Lowest frequency band (approximation)
    }
    
    for subband in WAVELET_SUBBANDS:
        low_freq, high_freq = freq_ranges[subband]
        coeff_length = decomposed_data[subband].shape[2]
        print(f"  {subband}: Frequency range ≈ {low_freq:.1f}-{high_freq:.1f} Hz, Coefficient length: {coeff_length}")

def save_connectivity_data(connectivity_matrices, subject_id, metric_type='PCC'):
    """
    Save connectivity data to .npy file
    
    Parameters:
    -----------
    connectivity_matrices : dict
        Connectivity matrices for each subband
    subject_id : str
        Subject ID
    metric_type : str
        Connectivity metric type
    """
    output_dir = "connectivity_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{output_dir}/{subject_id}_{metric_type}.npy"
    
    print(f"\nSaving connectivity data to: {filename}")
    
    # Create save data with metadata
    save_data = {
        'connectivity_matrices': connectivity_matrices,
        'metric_type': metric_type,
        'metric_info': get_connectivity_info(metric_type),
        'subject_id': subject_id,
        'channel_names': CHANNEL_NAMES,
        'wavelet_subbands': WAVELET_SUBBANDS,
        'sampling_rate': SAMPLING_RATE,
        'wavelet_type': WAVELET_TYPE
    }
    
    # Save data
    np.save(filename, save_data)
    
    print(f"Data saved successfully!")
    print(f"File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    
    # Print saved data information
    metric_info = get_connectivity_info(metric_type)
    print(f"\nSaved data information:")
    print(f"  Connectivity metric: {metric_info['name']} ({metric_info['short_name']})")
    print(f"  Value range: {metric_info['range']}")
    print(f"  Subject: {subject_id}")
    for subband in WAVELET_SUBBANDS:
        shape = connectivity_matrices[subband].shape
        print(f"  {subband}: {shape} (n_samples x n_channels x n_channels)")

def main():
    """Main function - computation part"""
    print("EEG Wavelet Subband Connectivity Computation (using Discrete Wavelet Transform)")
    print("=" * 70)
    
    # Validate connectivity metric type
    metric_type = CONFIG['Connec_type']
    if metric_type not in ['PCC', 'MSC']:
        print(f"Error: Unsupported connectivity metric type '{metric_type}'")
        print("Supported types: 'PCC' (Pearson Correlation Coefficient), 'MSC' (Magnitude Squared Coherence)")
        return None
    
    metric_info = get_connectivity_info(metric_type)
    print(f"Connectivity metric: {metric_info['name']} ({metric_info['short_name']})")
    print(f"Value range: {metric_info['range']}")
    print(f"Description: {metric_info['description']}")
    
    # 1. Load real EEG data (only use one fold, already covers entire dataset)
    print("\n1. Loading real EEG data...")
    data, labels = load_real_eeg_data()
    
    if data is None:
        print("Data loading failed, please check:")
        print("- Whether utils.py is in the current directory")
        print("- Whether data path is correct")
        print("- Whether required dependencies are installed")
        print("- Whether pywt library is installed: pip install PyWavelets")
        return None
    
    print(f"   Data shape: {data.shape}")
    print(f"   Number of channels: {data.shape[1]}")
    print(f"   Number of time points: {data.shape[2]}")
    print(f"   Sampling rate: {SAMPLING_RATE} Hz")
    
    # 2. Wavelet decomposition
    print(f"\n2. Discrete wavelet transform decomposition...")
    decomposed_data = decompose_data_by_dwt(data, wavelet=WAVELET_TYPE, levels=5)
    
    # 3. Analyze subband characteristics
    analyze_subband_characteristics(decomposed_data)
    
    # 4. Compute connectivity matrices
    print(f"\n3. Computing connectivity matrices...")
    connectivity_matrices = compute_connectivity_matrices(decomposed_data, metric_type)
    
    # 5. Save results
    print(f"\n4. Saving connectivity data...")
    save_connectivity_data(connectivity_matrices, CONFIG['subject'], metric_type)
    
    print(f"\nComputation completed!")
    print(f"Connectivity data has been saved, you can run visualization code for plotting.")
    print(f"Filename: connectivity_matrices_{CONFIG['subject']}_{metric_type}.npy")
    
    return connectivity_matrices

if __name__ == "__main__":
    # Modify configuration here if needed
    CONFIG['subject'] = 'all'           # Change subject
    # CONFIG['data_dir'] = 'your_data_path'   # Change data path
    CONFIG['Connec_type'] = 'MSC'       # Change connectivity metric: 'PCC' or 'MSC'
    
    print(f"Current configuration:")
    print(f"  Subject: {CONFIG['subject']}")
    print(f"  Connectivity metric: {CONFIG['Connec_type']}")
    print(f"  Data directory: {CONFIG['data_dir']}")
    
    # Run computation
    connectivity_matrices = main()