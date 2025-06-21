clc; clear;

eeglab;

%% Configuration
% Subject list
subjects = {'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'};
num_subjects = length(subjects);

% Initialize storage structure
all_subjects_data = struct();

% Processing parameters
time_window = 251:1250;

% Filter parameters
fs = 250;
low_freq = 0.5;
high_freq = 40;
filter_order = 4;

%% Process each subject
fprintf('Starting multi-subject data loading...\n');

for sub_idx = 1:num_subjects
    current_subject = subjects{sub_idx};
    fprintf('Processing Subject %s (%d/%d)...\n', current_subject, sub_idx, num_subjects);
    
    try
        % Load training data
        train_file = [current_subject 'T.gdf'];
        [Data_train, Label_train] = load_data_1subject(train_file);
        
        % Load testing data
        test_file = [current_subject 'E.gdf'];
        [Data_test, ~] = load_data_1subject(test_file);
        
        % Load testing labels
        label_file = [current_subject 'E.mat'];
        load(label_file, 'classlabel');
        Label_test = classlabel;
        
        % Combine all data (no distinction between train/test)
        Data_combined = [Data_train; Data_test];
        Label_combined = [Label_train; Label_test];
        
        % Find and remove missing values (MANDATORY removal)
        missing_value_indices = [];
        num_samples = size(Data_combined, 1);
        
        for i = 1:num_samples
            sample_data = squeeze(Data_combined(i, :, :));
            if any(isnan(sample_data(:))) || any(isinf(sample_data(:)))
                missing_value_indices = [missing_value_indices, i];
            end
        end
        
        % Statistics for missing values
        num_missing = length(missing_value_indices);
        fprintf('  Found %d samples with missing/invalid values (will be removed)\n', num_missing);
        
        if num_missing > 0
            % Analyze missing value distribution by class before removal
            missing_value_labels = Label_combined(missing_value_indices);
            class_counts = histcounts(missing_value_labels, 1:5);
            
            % Store missing value info
            all_subjects_data.(current_subject).missing_info.indices = missing_value_indices;
            all_subjects_data.(current_subject).missing_info.class_distribution = class_counts(1:4);
            
            % REMOVE samples with missing values (mandatory)
            all_indices = 1:size(Data_combined, 1);
            valid_indices = setdiff(all_indices, missing_value_indices);
            Data_no_missing = Data_combined(valid_indices, :, :);
            Label_no_missing = Label_combined(valid_indices, :);
            
            fprintf('  Removed %d samples with missing values\n', num_missing);
        else
            Data_no_missing = Data_combined;
            Label_no_missing = Label_combined;
            all_subjects_data.(current_subject).missing_info.indices = [];
            all_subjects_data.(current_subject).missing_info.class_distribution = zeros(1, 4);
            fprintf('  No missing values found\n');
        end
        
        % Apply bandpass filter (0.5-100 Hz)
        fprintf('  Applying bandpass filter (%.1f-%.1f Hz)...\n', low_freq, high_freq);
        [n_samples, n_channels, n_timepoints] = size(Data_no_missing);
        Data_filtered = zeros(size(Data_no_missing));
        
        % Design Butterworth bandpass filter
        nyquist = fs / 2;
        low_norm = low_freq / nyquist;
        high_norm = high_freq / nyquist;
        [b, a] = butter(filter_order, [low_norm, high_norm], 'bandpass');
        
        % Apply filter to each sample and channel
        for sample_idx = 1:n_samples
            for ch = 1:n_channels
                signal = squeeze(Data_no_missing(sample_idx, ch, :));
                filtered_signal = filtfilt(b, a, signal);
                Data_filtered(sample_idx, ch, :) = filtered_signal;
            end
        end
        
        fprintf('  Bandpass filtering completed\n');
        
        % Apply time window (掐头去尾) - 移到滤波后执行
        fprintf('  Applying time window (%d:%d)...\n', time_window(1), time_window(end));
        Data_final = Data_filtered(:, :, time_window);
        Label_final = Label_no_missing;
        
        % Convert labels to single precision
        Label_final = single(Label_final);
        
        % Store processed data
        all_subjects_data.(current_subject).Data = Data_final;
        all_subjects_data.(current_subject).Label = Label_final;
        all_subjects_data.(current_subject).total_original_samples = size(Data_combined, 1);
        all_subjects_data.(current_subject).num_valid_samples = size(Data_final, 1);
        all_subjects_data.(current_subject).num_removed_samples = num_missing;
        
        % Store processing parameters
        all_subjects_data.(current_subject).processing_info.filter_freq = [low_freq, high_freq];
        all_subjects_data.(current_subject).processing_info.filter_order = filter_order;
        all_subjects_data.(current_subject).processing_info.sampling_rate = fs;
        all_subjects_data.(current_subject).processing_info.time_window = time_window;
        
        % Store data dimensions
        [n_samples, n_channels, n_timepoints] = size(Data_final);
        all_subjects_data.(current_subject).dimensions.samples = n_samples;
        all_subjects_data.(current_subject).dimensions.channels = n_channels;
        all_subjects_data.(current_subject).dimensions.timepoints = n_timepoints;
        
        fprintf('  Successfully processed: %d valid samples (%d channels, %d timepoints)\n', ...
                n_samples, n_channels, n_timepoints);
        fprintf('  Data integrity: 100%% valid (all missing values removed, filtered & windowed)\n');
        
    catch ME
        fprintf('  Error processing subject %s: %s\n', current_subject, ME.message);
        all_subjects_data.(current_subject).error = ME.message;
        continue;
    end
end

%% Save all data
save_filename = 'all_subjects_eeg_data.mat';
fprintf('\nSaving all subjects data to %s...\n', save_filename);
save(save_filename, 'all_subjects_data', 'subjects', 'time_window', 'fs', 'low_freq', 'high_freq', 'filter_order', '-v7.3');

%% Generate summary report
fprintf('\n=== PROCESSING SUMMARY ===\n');
total_original_samples = 0;
total_valid_samples = 0;
total_removed = 0;

for sub_idx = 1:num_subjects
    current_subject = subjects{sub_idx};
    if isfield(all_subjects_data.(current_subject), 'Data')
        num_original = all_subjects_data.(current_subject).total_original_samples;
        num_valid = all_subjects_data.(current_subject).num_valid_samples;
        num_removed = all_subjects_data.(current_subject).num_removed_samples;
        
        total_original_samples = total_original_samples + num_original;
        total_valid_samples = total_valid_samples + num_valid;
        total_removed = total_removed + num_removed;
        
        fprintf('Subject %s: %d valid samples (removed %d with missing values, %.1f%% retention)\n', ...
                current_subject, num_valid, num_removed, 100*num_valid/num_original);
    else
        fprintf('Subject %s: FAILED TO PROCESS\n', current_subject);
    end
end

fprintf('\nOverall Statistics:\n');
fprintf('Total original samples: %d\n', total_original_samples);
fprintf('Total valid samples: %d\n', total_valid_samples);
fprintf('Total removed samples: %d\n', total_removed);
fprintf('Overall retention rate: %.1f%%\n', 100*total_valid_samples/total_original_samples);
fprintf('Data processing: Missing values removed → Bandpass filtered (%.1f-%.1f Hz) → Time windowed\n', low_freq, high_freq);

%% Visualization: Missing value distribution
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Missing values per subject
subplot(1, 3, 1);
missing_counts = zeros(1, num_subjects);
for sub_idx = 1:num_subjects
    current_subject = subjects{sub_idx};
    if isfield(all_subjects_data.(current_subject), 'missing_info')
        missing_counts(sub_idx) = length(all_subjects_data.(current_subject).missing_info.indices);
    end
end
bar(1:num_subjects, missing_counts);
xlabel('Subject');
ylabel('Number of Missing Value Samples');
title('Missing Value Samples per Subject');
xticks(1:num_subjects);
xticklabels(subjects);
xtickangle(45);

% Plot 2: Sample retention per subject
subplot(1, 3, 2);
retention_rates = zeros(1, num_subjects);
for sub_idx = 1:num_subjects
    current_subject = subjects{sub_idx};
    if isfield(all_subjects_data.(current_subject), 'total_original_samples')
        original = all_subjects_data.(current_subject).total_original_samples;
        valid = all_subjects_data.(current_subject).num_valid_samples;
        retention_rates(sub_idx) = 100 * valid / original;
    end
end
bar(1:num_subjects, retention_rates);
xlabel('Subject');
ylabel('Retention Rate (%)');
title('Data Retention Rate per Subject');
xticks(1:num_subjects);
xticklabels(subjects);
xtickangle(45);
ylim([0, 105]);

% Plot 3: Class distribution of missing values (aggregated)
subplot(1, 3, 3);
total_class_missing = zeros(1, 4);
for sub_idx = 1:num_subjects
    current_subject = subjects{sub_idx};
    if isfield(all_subjects_data.(current_subject), 'missing_info')
        total_class_missing = total_class_missing + all_subjects_data.(current_subject).missing_info.class_distribution;
    end
end
bar(1:4, total_class_missing);
xlabel('Class');
ylabel('Number of Missing Value Samples');
title('Missing Value Distribution by Class (All Subjects)');
xticks(1:4);

sgtitle(sprintf('EEG Data Processing Summary - All Subjects (Filtered %.1f-%.1f Hz)', low_freq, high_freq));

fprintf('\nData loading and processing completed!\n');
fprintf('✓ All missing values have been removed from all subjects\n');
fprintf('✓ Bandpass filter (%.1f-%.1f Hz) applied to all valid samples\n', low_freq, high_freq);
fprintf('✓ Time window (%d:%d) applied after filtering\n', time_window(1), time_window(end));
fprintf('✓ Data integrity guaranteed: 100%% valid, filtered samples\n');
fprintf('Use the following code to access individual subject data:\n');
fprintf('  load(''all_subjects_eeg_data_filtered.mat'');\n');
fprintf('  subject_data = all_subjects_data.A01.Data;  %% Example for subject A01\n');
fprintf('  subject_labels = all_subjects_data.A01.Label;\n');