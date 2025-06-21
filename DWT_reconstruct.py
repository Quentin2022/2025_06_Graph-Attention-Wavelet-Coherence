import numpy as np
import torch
import torch.nn.functional as F
import pywt
import os

# DWT-related configuration
DWT_CONFIG = {
    'wavelet_type': 'db4',      # Wavelet type
    'decomp_level': 5,          # Decomposition level
    'wavelet_subbands': ['D1', 'D2', 'D3', 'D4', 'D5', 'A5']  # Subband names
}

def load_connectivity_matrices(subject, metric_type='MSC'):
    """
    Load connectivity matrices computed by DWT.py and calculate average
    
    Args:
        subject: Subject ID
        metric_type: Connectivity metric type
        
    Returns:
        avg_conn_matrices: dict containing 6 average connectivity matrices
    """
    filepath = f"connectivity_results/{subject}_{metric_type}.npy"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Connectivity matrix file does not exist: {filepath}")
    
    print(f"Loading connectivity matrices: {filepath}")
    
    # Load data
    data = np.load(filepath, allow_pickle=True).item()
    connectivity_matrices = data['connectivity_matrices']
    
    # Calculate average connectivity matrix for each subband
    avg_conn_matrices = {}
    for subband in DWT_CONFIG['wavelet_subbands']:
        # Shape: [n_samples, n_channels, n_channels] -> [n_channels, n_channels]
        avg_conn_matrices[subband] = np.mean(connectivity_matrices[subband], axis=0)
        print(f"  {subband} average connectivity matrix shape: {avg_conn_matrices[subband].shape}")
    
    return avg_conn_matrices

def precompute_dwt_coefficients(data, wavelet='db4', level=5):
    """
    Precompute DWT decomposition coefficients for all data
    
    Args:
        data: numpy array [n_samples, n_channels, n_timepoints]
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        all_coeffs: dict containing coefficients for all samples and subbands
    """
    n_samples, n_channels, n_timepoints = data.shape
    
    print(f"Precomputing DWT coefficients (number of samples: {n_samples})...")
    
    # First perform one decomposition to get accurate subband lengths
    test_coeffs = pywt.wavedec(data[0, 0, :], wavelet, level=level)
    subband_lengths = {
        'A5': len(test_coeffs[0]),
        'D5': len(test_coeffs[1]),
        'D4': len(test_coeffs[2]),
        'D3': len(test_coeffs[3]),
        'D2': len(test_coeffs[4]),
        'D1': len(test_coeffs[5])
    }
    
    # Initialize storage
    all_coeffs = {}
    for subband, length in subband_lengths.items():
        all_coeffs[subband] = np.zeros((n_samples, n_channels, length))
    
    # Decompose each sample and channel
    for sample_idx in range(n_samples):
        if sample_idx % 100 == 0:
            print(f"  Processing sample {sample_idx}/{n_samples}")
        
        for ch_idx in range(n_channels):
            # Perform DWT decomposition
            coeffs = pywt.wavedec(data[sample_idx, ch_idx, :], wavelet, level=level)
            
            # Store coefficients
            all_coeffs['A5'][sample_idx, ch_idx, :] = coeffs[0]
            for i in range(1, level + 1):
                subband = f'D{level + 1 - i}'
                all_coeffs[subband][sample_idx, ch_idx, :] = coeffs[i]
    
    print("DWT precomputation completed!")
    return all_coeffs

def dwt_reconstruct_signal(coeffs_list, wavelet='db4'):
    """
    Reconstruct signal using DWT coefficients
    
    Args:
        coeffs_list: list containing [A5, D5, D4, D3, D2, D1]
        wavelet: Wavelet type
        
    Returns:
        reconstructed: Reconstructed signal
    """
    reconstructed = pywt.waverec(coeffs_list, wavelet)
    return reconstructed

def generate_corruption_mask(n_samples, n_channels, num_corrupt, ratio_corrupt, random_seed):
    """
    Generate fixed corruption channel mask
    
    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        num_corrupt: Number of corrupted channels per sample
        ratio_corrupt: Ratio of corrupted samples (0-100)
        random_seed: Random seed
        
    Returns:
        corruption_mask: numpy array [n_samples, n_channels], True indicates corruption
    """
    np.random.seed(random_seed)
    corruption_mask = np.zeros((n_samples, n_channels), dtype=bool)
    
    if num_corrupt <= 0 or ratio_corrupt <= 0:
        return corruption_mask
    
    # Ensure num_corrupt doesn't exceed number of channels
    actual_num_corrupt = min(num_corrupt, n_channels - 1)
    
    # Calculate number of samples to corrupt
    num_samples_to_corrupt = int(n_samples * ratio_corrupt / 100)
    if num_samples_to_corrupt == 0 and ratio_corrupt > 0:
        num_samples_to_corrupt = 1
    
    # Randomly select samples to corrupt
    samples_to_corrupt = np.random.choice(n_samples, num_samples_to_corrupt, replace=False)
    
    # Randomly select channels to corrupt for each selected sample
    for sample_idx in samples_to_corrupt:
        channels_to_corrupt = np.random.choice(n_channels, actual_num_corrupt, replace=False)
        corruption_mask[sample_idx, channels_to_corrupt] = True
    
    print(f"Generated corruption mask: {num_samples_to_corrupt} samples, {actual_num_corrupt} channels per sample")
    return corruption_mask

def apply_corruption_with_mask(data, corruption_mask, noise_std=0.01):
    """
    Apply channel corruption according to predefined mask
    
    Args:
        data: numpy array [n_samples, n_channels, n_timepoints]
        corruption_mask: numpy array [n_samples, n_channels], True indicates corruption
        noise_std: White noise standard deviation
        
    Returns:
        corrupted_data: Corrupted data
    """
    corrupted_data = data.copy()
    n_samples, n_channels, n_timepoints = data.shape
    
    # Generate noise for each channel marked as corrupted
    for sample_idx in range(n_samples):
        for ch_idx in range(n_channels):
            if corruption_mask[sample_idx, ch_idx]:
                # Generate small amplitude white noise
                noise = np.random.normal(0, noise_std, n_timepoints)
                corrupted_data[sample_idx, ch_idx, :] = noise
    
    return corrupted_data

def reconstruct_all_corrupted_channels(data, corruption_mask, avg_conn_matrices, precomputed_coeffs, 
                                      wavelet='db4', level=5, device='cpu'):
    """
    Batch reconstruct all corrupted channels for all data
    
    Args:
        data: numpy array [n_samples, n_channels, n_timepoints]
        corruption_mask: numpy array [n_samples, n_channels], True indicates corruption
        avg_conn_matrices: dict, average connectivity matrices
        precomputed_coeffs: dict, precomputed DWT coefficients
        wavelet: Wavelet type
        level: Decomposition level
        device: Computing device
        
    Returns:
        reconstructed_data: Reconstructed data
    """
    n_samples, n_channels, n_timepoints = data.shape
    reconstructed_data = data.copy()
    
    # Calculate number of samples that need reconstruction
    samples_with_corruption = np.any(corruption_mask, axis=1)
    n_corrupted_samples = np.sum(samples_with_corruption)
    
    print(f"Starting corrupted channel reconstruction (total {n_corrupted_samples} samples need reconstruction)...")
    
    # Reconstruct each sample
    for sample_idx in range(n_samples):
        if not corruption_mask[sample_idx].any():
            continue  # Skip samples without corruption
        
        if sample_idx % 100 == 0 and samples_with_corruption[sample_idx]:
            print(f"  Reconstructing sample {sample_idx}/{n_samples}")
        
        # Get corrupted and intact channels for current sample
        corrupted_channels = np.where(corruption_mask[sample_idx])[0]
        good_channels = np.where(~corruption_mask[sample_idx])[0]
        
        if len(good_channels) == 0:
            continue  # Skip if all channels are corrupted
        
        # Reconstruct each corrupted channel
        for corrupt_ch_idx in corrupted_channels:
            # Store reconstructed subband coefficients
            reconstructed_coeffs = []
            
            # Organize coefficients in order required by pywt.waverec: [A5, D5, D4, D3, D2, D1]
            subband_order = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
            
            for subband in subband_order:
                # Get connectivity values for current subband
                conn_matrix = avg_conn_matrices[subband]
                conn_values = conn_matrix[corrupt_ch_idx, good_channels]
                
                # Calculate attention scores (softmax normalization)
                conn_values_tensor = torch.from_numpy(conn_values).float().to(device)
                attention_scores = F.softmax(conn_values_tensor, dim=0).cpu().numpy()
                
                # Get coefficients of intact channels from precomputed coefficients
                good_coeffs = precomputed_coeffs[subband][sample_idx, good_channels, :]
                
                # Use attention scores for weighted sum
                reconstructed_subband = np.sum(good_coeffs * attention_scores[:, np.newaxis], axis=0)
                reconstructed_coeffs.append(reconstructed_subband)
            
            # Reconstruct signal using inverse DWT
            reconstructed_signal = dwt_reconstruct_signal(reconstructed_coeffs, wavelet)
            
            # Ensure length matches
            if len(reconstructed_signal) > n_timepoints:
                reconstructed_signal = reconstructed_signal[:n_timepoints]
            elif len(reconstructed_signal) < n_timepoints:
                # Pad to original length
                pad_length = n_timepoints - len(reconstructed_signal)
                reconstructed_signal = np.pad(reconstructed_signal, (0, pad_length), mode='edge')
            
            # Update reconstructed channel
            reconstructed_data[sample_idx, corrupt_ch_idx, :] = reconstructed_signal
    
    print("Channel reconstruction completed!")
    return reconstructed_data

def preprocess_data_with_dwt_reconstruction(data, config, avg_conn_matrices, device='cpu'):
    """
    Preprocess data: apply corruption and perform DWT reconstruction
    
    Args:
        data: numpy array [n_samples, n_channels, n_timepoints]
        config: Configuration dictionary
        avg_conn_matrices: Average connectivity matrices
        device: Computing device
        
    Returns:
        processed_data: Processed data
        corruption_mask: Corruption mask
    """
    n_samples, n_channels, n_timepoints = data.shape
    
    # Generate fixed corruption mask
    corruption_mask = generate_corruption_mask(
        n_samples, n_channels, 
        config['num_corrupt'], 
        config['ratio_corrupt'],
        config['random_seed']
    )
    
    # Apply corruption
    corrupted_data = apply_corruption_with_mask(data, corruption_mask)
    
    # If DWT reconstruction is enabled
    if config['if_DWT_reconstruct'] and avg_conn_matrices is not None:
        # Precompute DWT coefficients (using original uncorrupted data)
        precomputed_coeffs = precompute_dwt_coefficients(
            data, DWT_CONFIG['wavelet_type'], DWT_CONFIG['decomp_level']
        )
        
        # Reconstruct corrupted channels
        processed_data = reconstruct_all_corrupted_channels(
            corrupted_data, corruption_mask, avg_conn_matrices, precomputed_coeffs,
            DWT_CONFIG['wavelet_type'], DWT_CONFIG['decomp_level'], device
        )
    else:
        processed_data = corrupted_data
    
    return processed_data, corruption_mask