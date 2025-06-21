import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

class EEGRawDataset:
    """
    Utility class for loading EEG raw signal dataset organized by subjects
    
    Data format:
    - subjects_data[subject]['features']: [n_samples, n_channels, n_timepoints]
    - subjects_data[subject]['labels']: [n_samples]
    """
    def __init__(self, data_dir='eeg_raw_dataset_segmented'):
        self.data_dir = data_dir
        self.load_dataset()
    
    def load_dataset(self):
        """Load the complete dataset"""
        # Load the main data structure
        subjects_data_file = os.path.join(self.data_dir, 'subjects_data.npy')
        self.subjects_data = np.load(subjects_data_file, allow_pickle=True).item()
        
        # Load metadata
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Extract basic info
        self.subjects = list(self.subjects_data.keys())
        self.n_channels = self.metadata['n_channels']
        self.n_timepoints = self.metadata['n_timepoints']  # Changed from n_features_per_channel
        
        print(f"Loaded dataset: {len(self.subjects)} subjects")
        print(f"Available subjects: {self.subjects}")
        print(f"Data shape per subject: [n_samples, {self.n_channels}, {self.n_timepoints}]")
        print(f"Data type: {self.metadata.get('data_type', 'raw_signals')}")
        
        # Calculate total samples
        total_samples = sum([self.subjects_data[subj]['n_samples'] for subj in self.subjects])
        print(f"Total samples across all subjects: {total_samples}")
    
    def get_subject_data(self, subject_id):
        """
        Get data for specific subject
        
        Args:
            subject_id: Subject ID (e.g., 'A01')
            
        Returns:
            features: [n_samples, n_channels, n_timepoints]
            labels: [n_samples]
        """
        if subject_id not in self.subjects_data:
            raise ValueError(f"Subject '{subject_id}' not found. Available: {self.subjects}")
        
        return self.subjects_data[subject_id]['features'], self.subjects_data[subject_id]['labels']
    
    def get_all_data(self):
        """
        Get all data concatenated across subjects
        
        Returns:
            features: [total_samples, n_channels, n_timepoints]
            labels: [total_samples]
            subject_ids: [total_samples] - array indicating which subject each sample belongs to
        """
        all_features = []
        all_labels = []
        all_subject_ids = []
        
        for subject_id in self.subjects:
            subj_features = self.subjects_data[subject_id]['features']
            subj_labels = self.subjects_data[subject_id]['labels']
            
            all_features.append(subj_features)
            all_labels.append(subj_labels)
            all_subject_ids.extend([subject_id] * len(subj_labels))
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_subject_ids = np.array(all_subject_ids)
        
        return all_features, all_labels, all_subject_ids
    
    def prepare_data_for_cv(self, subject_choice='all', n_splits=10, random_state=42, 
                           normalize=True):
        """
        Prepare data for K-fold cross validation with stratification
        
        Args:
            subject_choice: 'all' for all subjects, or specific subject ID (e.g., 'A01')
            n_splits: Number of folds for cross validation (default: 10)
            random_state: Random seed for reproducibility
            normalize: Whether to apply channel-wise standardization using full dataset
            
        Returns:
            cv_data: List of (X_train, X_test, y_train, y_test) tuples for each fold
            scalers: Dictionary of scalers for each channel (if normalize=True)
            cv_indices: StratifiedKFold indices for reproducibility
        """
        # Get data based on subject choice
        if subject_choice == 'all':
            print("Using all subjects")
            X, y, subject_ids = self.get_all_data()
            print(f"Combined data shape: {X.shape}")
        else:
            print(f"Using subject: {subject_choice}")
            X, y = self.get_subject_data(subject_choice)
            subject_ids = np.array([subject_choice] * len(y))
            print(f"Subject data shape: {X.shape}")
        
        # Print class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")
        
        # Normalize data if requested using FULL DATASET parameters
        scalers = None
        if normalize:
            print("Applying channel-wise standardization using full dataset parameters...")
            X_normalized, scalers = self._normalize_data_full(X)
            X = X_normalized
            print("✓ Standardization completed")
        
        # Set up stratified K-fold cross validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_data = []
        cv_indices = list(skf.split(X, y))
        
        print(f"\nPreparing {n_splits}-fold cross validation...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_indices):
            X_train_fold = X[train_idx].copy()
            X_test_fold = X[test_idx].copy()
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]
            
            cv_data.append((X_train_fold, X_test_fold, y_train_fold, y_test_fold))
            
            # Print fold statistics
            train_unique, train_counts = np.unique(y_train_fold, return_counts=True)
            test_unique, test_counts = np.unique(y_test_fold, return_counts=True)
            print(f"  Fold {fold_idx+1}: Train {len(y_train_fold)} samples {dict(zip(train_unique, train_counts))}, "
                  f"Test {len(y_test_fold)} samples {dict(zip(test_unique, test_counts))}")
        
        print(f"\n✓ {n_splits}-fold cross validation data prepared")
        
        return cv_data, scalers, cv_indices
    
    def _normalize_data_full(self, X):
        """
        Normalize data channel-wise using the FULL DATASET for parameters
        Each channel is normalized independently across all timepoints and samples
        
        Args:
            X: [n_samples, n_channels, n_timepoints]
            
        Returns:
            X_norm: Normalized data
            scalers: Dictionary of scalers for each channel
        """
        n_samples, n_channels, n_timepoints = X.shape
        
        # Initialize normalized array
        X_norm = np.zeros_like(X)
        
        # Store scalers for each channel
        scalers = {}
        
        print(f"Processing {n_channels} channels using full dataset parameters...")
        
        # Process each channel independently
        for ch_idx in range(n_channels):
            # Get all data for this channel: flatten across samples and timepoints
            # Shape: [n_samples * n_timepoints]
            channel_data = X[:, ch_idx, :].flatten().reshape(-1, 1)
            
            # Create and fit scaler on full dataset
            scaler = StandardScaler()
            scaler.fit(channel_data)
            
            # Transform data
            normalized_data = scaler.transform(channel_data)
            X_norm[:, ch_idx, :] = normalized_data.reshape(n_samples, n_timepoints)
            
            # Store scaler
            scalers[f'channel_{ch_idx}'] = scaler
            
            # Print statistics for this channel
            print(f"  Channel {ch_idx}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
        
        print(f"Channel-wise normalization completed for {n_channels} channels using full dataset")
        
        return X_norm, scalers
    
    def get_info(self):
        """Print dataset information"""
        print("=== EEG Raw Signal Dataset Info ===")
        print(f"Subjects: {len(self.subjects)} ({', '.join(self.subjects)})")
        print(f"Channels: {self.n_channels}")
        print(f"Timepoints per sample: {self.n_timepoints}")
        print(f"Classes: {self.metadata['n_classes']}")
        print(f"Sampling rate: {self.metadata['sampling_rate']} Hz")
        duration = self.n_timepoints / self.metadata['sampling_rate']
        print(f"Sample duration: {duration:.2f} seconds")
        
        # Per-subject info
        print("\nPer-subject details:")
        for subject_id in self.subjects:
            subj_data = self.subjects_data[subject_id]
            labels = subj_data['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  {subject_id}: {len(labels)} samples, classes {unique_labels} with counts {counts}")
        
        print(f"\nData format:")
        print(f"  3D: [n_samples, {self.n_channels}, {self.n_timepoints}] - Raw EEG signals")


class EEGTrainer:
    """
    Complete training pipeline for EEG classification using raw signals with K-fold CV
    """
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary with training parameters
        """
        self.config = config
        self.dataset = None
        self.cv_data = None
        self.scalers = None
        self.cv_indices = None
        
    def validate_subject_choice(self, subject, available_subjects):
        """Validate subject choice"""
        if subject == 'all':
            return True
        elif subject in available_subjects:
            return True
        else:
            print(f"Error: Subject '{subject}' not found.")
            print(f"Available subjects: {available_subjects}")
            print("Use 'all' to use all subjects together.")
            return False
    
    def load_and_prepare_data(self):
        """Load dataset and prepare K-fold cross validation splits"""
        print("="*60)
        print("EEG MOTOR IMAGERY CLASSIFICATION - K-FOLD CROSS VALIDATION DATA PREPARATION")
        print("="*60)
        
        # Display current configuration
        print("CURRENT CONFIGURATION:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        # Initialize dataset
        print(f"Loading dataset from: {self.config['data_dir']}")
        self.dataset = EEGRawDataset(data_dir=self.config['data_dir'])
        
        # Validate subject choice
        if not self.validate_subject_choice(self.config['subject'], self.dataset.subjects):
            return False
        
        # Display dataset info
        print(f"\nDataset Information:")
        self.dataset.get_info()
        
        # Prepare data with configuration parameters
        print(f"\n" + "="*50)
        print("PREPARING DATA FOR K-FOLD CROSS VALIDATION")
        print(f"Subject choice: {self.config['subject']}")
        print(f"Number of folds: {self.config['n_splits']}")
        print(f"Random seed: {self.config['random_seed']}")
        print(f"Channel-wise normalization: {self.config['normalize']}")
        print(f"Dynamic corruption: {self.config['num_corrupt']} channels in {self.config['ratio_corrupt']}% of samples")
        print(f"Train set corruption: {self.config['if_train_corrupt']}")
        print("="*50)
        
        self.cv_data, self.scalers, self.cv_indices = self.dataset.prepare_data_for_cv(
            subject_choice=self.config['subject'],
            n_splits=self.config['n_splits'],
            random_state=self.config['random_seed'],
            normalize=self.config['normalize']
        )
        
        # Display final data statistics
        print(f"\n" + "="*50)
        print("DATA PREPARATION COMPLETE")
        print("="*50)
        
        # Show statistics for first fold as example
        X_train_0, X_test_0, y_train_0, y_test_0 = self.cv_data[0]
        print(f"Example (Fold 1):")
        print(f"  Training set:")
        print(f"    Shape: {X_train_0.shape}")
        print(f"    Data type: {X_train_0.dtype}")
        print(f"    Value range: [{X_train_0.min():.4f}, {X_train_0.max():.4f}]")
        print(f"    Labels: {np.unique(y_train_0, return_counts=True)}")
        
        print(f"  Test set:")
        print(f"    Shape: {X_test_0.shape}")
        print(f"    Data type: {X_test_0.dtype}")
        print(f"    Value range: [{X_test_0.min():.4f}, {X_test_0.max():.4f}]")
        print(f"    Labels: {np.unique(y_test_0, return_counts=True)}")
        
        if self.config['normalize'] and self.scalers:
            print(f"\nNormalization scalers: {len(self.scalers)} channel scalers stored")
        
        return True
    
    def display_model_info(self):
        """Display information for model development"""
        print(f"\n" + "="*60)
        print("MODEL DEVELOPMENT INFORMATION")
        print("="*60)
        print("Raw EEG signal data is ready for K-fold cross validation training!")
        print(f"Available data:")
        print(f"  {self.config['n_splits']} folds prepared")
        print(f"  Data format: [n_samples, n_channels, n_timepoints]")
        print(f"  scalers: Channel-wise normalization parameters (if normalization was applied)")
        print("Ready for CNN, RNN, or other deep learning models with K-fold CV!")
        if self.config['num_corrupt'] > 0 and self.config['ratio_corrupt'] > 0:
            print("Note: Channel corruption will be applied dynamically during training/testing.")
            print(f"  - {self.config['num_corrupt']} channels will be corrupted in {self.config['ratio_corrupt']}% of samples")
            if self.config['if_train_corrupt']:
                print("  - Training data will also be corrupted dynamically")
            else:
                print("  - Only test data will be corrupted during evaluation")
        else:
            print("Note: No channel corruption will be applied.")
    
    def get_data(self):
        """
        Get prepared cross validation data
        
        Returns:
            cv_data: List of (X_train, X_test, y_train, y_test) tuples
            scalers: Dictionary of scalers
            cv_indices: Cross validation indices
        """
        if self.cv_data is None:
            raise ValueError("Data not prepared yet. Call load_and_prepare_data() first.")
        
        return self.cv_data, self.scalers, self.cv_indices
    
    def run_data_preparation(self):
        """
        Complete data preparation pipeline
        
        Returns:
            success: Boolean indicating if preparation was successful
        """
        success = self.load_and_prepare_data()
        if success:
            self.display_model_info()
        
        return success


def create_trainer(subject='A01', n_splits=10, random_seed=42, normalize=True, 
                  num_corrupt=0, ratio_corrupt=0, if_train_corrupt=False,
                  data_dir='eeg_raw_dataset_segmented'):
    """
    Create and initialize EEG trainer with default configuration for raw signals and K-fold CV
    
    Args:
        subject: 'all' or specific subject ID ('A01'-'A09')
        n_splits: Number of folds for cross validation
        random_seed: Random seed for reproducibility
        normalize: Whether to apply channel-wise standardization
        num_corrupt: Number of channels to corrupt with noise (applied dynamically)
        ratio_corrupt: Percentage of samples to corrupt (0-100), e.g., 10 means 10% of samples
        if_train_corrupt: Whether to also corrupt training set dynamically
        data_dir: Directory containing the raw signal dataset
        
    Returns:
        trainer: EEGTrainer instance
    """
    config = {
        'subject': subject,
        'n_splits': n_splits,
        'random_seed': random_seed,
        'normalize': normalize,
        'num_corrupt': num_corrupt,
        'ratio_corrupt': ratio_corrupt,
        'if_train_corrupt': if_train_corrupt,
        'data_dir': data_dir
    }
    
    return EEGTrainer(config)