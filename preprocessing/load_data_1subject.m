function [data_set, label] = load_data_1subject(filepath)
    [s, h] = sload(filepath);
    s = s(:, 1:22);

    num_sample = size(h.Classlabel, 1);
    data_set = zeros(num_sample, 22, 1500);

    TRIG_indice = find(h.EVENT.TYP == 768);  % code768 corresponds the start of a trial
    TRIG_position = h.EVENT.POS(TRIG_indice)+250;  % 1s before the start

    for i = 1:num_sample
        begin_time = TRIG_position(i);
        data_set(i, :, :) = s(begin_time+1:begin_time+1500, :)';
    end

    label = h.Classlabel;
end