1. Download the BCI Competition IV 2a raw data from "https://www.bbci.de/competition/iv/", including the compressed packages of training set data, training set labels, and test set labels. Extract them to 'your directory\preprocessing'.
2. Run preprocessing_with_filter.m and obtain the preliminarily processed dataset all_subjects_eeg_data_filted_0.5-40.mat.
3. Run signal_construct.py to get the further processed dataset, which will be located in the eeg_raw_dataset_0.5-40 folder in your directory (automatically created).
4. Finally run train.ipynb.
