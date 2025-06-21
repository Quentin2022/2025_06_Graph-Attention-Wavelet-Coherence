function filtered_signal = multi_channel_filter(data, fs, low_cutoff, high_cutoff, bandpass_freqs)
    % Parameters:
    % data: Input multi-channel signal, size [samples, channels, timepoints]
    % fs: Sampling frequency (Hz)
    % low_cutoff: Cutoff frequency for high-pass filter (Hz), removes low-frequency artifacts
    % high_cutoff: Cutoff frequency for low-pass filter (Hz), removes high-frequency noise
    % bandpass_freqs: Frequency bands for bandpass filters, cell array formatted as {[low, high], ...}
    
    [n_samples, n_channels, n_points] = size(data);
    
    % Initialize filtered signal
    filtered_signal = zeros(size(data));
    
    % Filter each channel
    for ch = 1:n_channels
        channel_data = squeeze(data(:, ch, :))';  % Extract channel data, transpose to [timepoints, samples] dimension
        
        % High-pass filter design (remove low-frequency artifacts)
        if ~isempty(low_cutoff)
            [b_hp, a_hp] = butter(2, low_cutoff / (fs / 2), 'high'); % 2nd-order Butterworth high-pass
            channel_data = filtfilt(b_hp, a_hp, channel_data); % Zero-phase filtering
        end
        
        % Low-pass filter design (remove high-frequency noise)
        if ~isempty(high_cutoff)
            [b_lp, a_lp] = butter(2, high_cutoff / (fs / 2), 'low'); % 2nd-order Butterworth low-pass
            channel_data = filtfilt(b_lp, a_lp, channel_data); % Zero-phase filtering
        end
        
        % Band-pass filter design (select specified frequency bands)
        for i = 1:length(bandpass_freqs)
            band = bandpass_freqs{i};
            [b_bp, a_bp] = butter(2, [band(1) band(2)] / (fs / 2), 'bandpass'); % 2nd-order Butterworth band-pass
            channel_data = filtfilt(b_bp, a_bp, channel_data); % Zero-phase filtering
        end
        
        % Store filtered signal back into matrix
        filtered_signal(:, ch, :) = channel_data';
    end
end