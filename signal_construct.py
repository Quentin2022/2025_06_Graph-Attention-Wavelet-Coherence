import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings('ignore')

class EEGRawSignalProcessor:
    def __init__(self, data_file='---YOUR PATH---\\all_subjects_eeg_data_filted_0.5-40.mat', fs=250):
        """
        Initialize EEG raw signal processor
        
        Args:
            data_file: Path to MATLAB data file
            fs: Sampling frequency in Hz
        """
        self.data_file = data_file
        self.fs = fs
        self.subjects = [f'A{i:02d}' for i in range(1, 10)]  # A01 to A09
        self.load_data()
        
        print(f"Initialized processor for raw EEG signals")
        
    def load_data(self):
        """Load EEG data from MATLAB file"""
        print("Loading EEG data...")
        try:
            # Try scipy.io first
            try:
                mat_data = sio.loadmat(self.data_file)
                self.all_subjects_data = mat_data['all_subjects_data']
                self.use_h5py = False
                print("✓ Data loaded successfully with scipy.io")
            except NotImplementedError:
                # Use h5py for v7.3 files
                self.h5_file = h5py.File(self.data_file, 'r')
                self.all_subjects_data = self.h5_file['all_subjects_data']
                self.use_h5py = True
                print("✓ Data loaded successfully with h5py")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.all_subjects_data = None
    
    def get_subject_data(self, subject):
        """Extract data for specific subject"""
        if self.all_subjects_data is None:
            return None, None
            
        try:
            if self.use_h5py:
                if subject in self.all_subjects_data:
                    subject_group = self.all_subjects_data[subject]
                    data = subject_group['Data'][()]
                    labels = subject_group['Label'][()].flatten()
                else:
                    return None, None
            else:
                if hasattr(self.all_subjects_data, 'dtype') and self.all_subjects_data.dtype.names:
                    if subject in self.all_subjects_data.dtype.names:
                        subject_struct = self.all_subjects_data[subject][0, 0]
                        data = subject_struct['Data']
                        labels = subject_struct['Label'].flatten()
                    else:
                        return None, None
                else:
                    subject_struct = self.all_subjects_data[subject][0, 0]
                    data = subject_struct['Data']
                    labels = subject_struct['Label'].flatten()
            
            # Determine correct transpose based on which dimension matches labels
            if len(data.shape) == 3:
                if data.shape[2] == len(labels):
                    # (timepoints, channels, samples) -> (samples, channels, timepoints)
                    data = np.transpose(data, (2, 1, 0))
                elif data.shape[1] == len(labels):
                    # (channels, samples, timepoints) -> (samples, channels, timepoints)  
                    data = np.transpose(data, (1, 0, 2))
                elif data.shape[0] == len(labels):
                    # Already (samples, channels, timepoints)
                    pass
            
            return data, labels
            
        except Exception as e:
            print(f"Error extracting {subject} data: {e}")
            return None, None
    
    def process_all_subjects(self, output_dir='eeg_raw_dataset_0.5-40'):
        """
        Process all subjects and organize raw EEG signals
        Save in subject-organized format maintaining original signal structure
        
        Args:
            output_dir: Directory to save the processed dataset
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize storage - organized by subject
        subjects_data = {}
        
        print(f"\nProcessing {len(self.subjects)} subjects...")
        
        for subject in tqdm(self.subjects, desc="Processing subjects"):
            # Load subject data
            data, labels = self.get_subject_data(subject)
            
            if data is None:
                print(f"Skipping {subject} (no data)")
                continue
            
            n_samples, n_channels, n_timepoints = data.shape
            print(f"\n{subject}: {n_samples} samples, {n_channels} channels, {n_timepoints} timepoints")
            
            # Store raw signals directly - no feature extraction
            # Shape remains: [n_samples, n_channels, n_timepoints]
            subject_signals = data.astype(np.float32)
            subject_labels = labels.astype(np.int32)
            
            # Store in subjects dictionary
            subjects_data[subject] = {
                'features': subject_signals,  # Shape: [n_samples, n_channels, n_timepoints] - keeping 'features' key for consistency
                'labels': subject_labels,     # Shape: [n_samples]
                'n_samples': n_samples,
                'n_channels': n_channels,
                'n_timepoints': n_timepoints  # Changed from n_features_per_channel to n_timepoints
            }
            
            print(f"✓ {subject}: {subject_signals.shape} raw signals, {len(subject_labels)} labels")
        
        # Calculate overall statistics
        total_samples = sum([subjects_data[subj]['n_samples'] for subj in subjects_data.keys()])
        all_labels = np.concatenate([subjects_data[subj]['labels'] for subj in subjects_data.keys()])
        
        print(f"\n=== Dataset Summary ===")
        print(f"Processed subjects: {list(subjects_data.keys())}")
        print(f"Total samples across all subjects: {total_samples}")
        print(f"Timepoints per channel: {n_timepoints}")
        print(f"Channels per sample: {n_channels}")
        print(f"Data shape per subject: [n_samples, {n_channels}, {n_timepoints}]")
        print(f"Sampling frequency: {self.fs} Hz")
        
        # Class distribution
        print(f"\nOverall class distribution:")
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({100*count/len(all_labels):.1f}%)")
        
        # Save dataset
        print(f"\nSaving dataset to {output_dir}/...")
        np.save(os.path.join(output_dir, 'subjects_data.npy'), subjects_data)
        
        # Save metadata
        metadata = {
            'n_subjects': len(subjects_data),
            'total_samples': int(total_samples),
            'n_timepoints': int(n_timepoints),  # Changed from n_features_per_channel
            'n_channels': int(n_channels),
            'n_classes': len(np.unique(all_labels)),
            'sampling_rate': self.fs,
            'subjects': list(subjects_data.keys()),
            'data_type': 'raw_signals',  # Added to indicate this contains raw signals
            'class_distribution': {int(k): int(v) for k, v in zip(unique_labels, counts)}
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Dataset saved successfully!")
        print(f"Data structure: subjects_data[subject]['features']: [n_samples, {n_channels}, {n_timepoints}]")
        print("Note: 'features' key now contains raw EEG signals instead of extracted features")
        
        return subjects_data

if __name__ == "__main__":
    # Initialize and run processor
    processor = EEGRawSignalProcessor('---YOUR PATH---\\all_subjects_eeg_data_filted_0.5-40.mat')
    subjects_data = processor.process_all_subjects()
    
    print("\n=== Processing Complete ===")
    print("Raw EEG dataset ready for deep learning!")
    
    # Example: Load and use the data
    # subjects_data = np.load('eeg_raw_dataset_segmented/subjects_data.npy', allow_pickle=True).item()
    # X_subj = subjects_data['A01']['features']  # Shape: [n_samples, n_channels, n_timepoints] - raw signals
    # y_subj = subjects_data['A01']['labels']    # Shape: [n_samples]