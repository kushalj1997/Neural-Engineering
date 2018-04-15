function [all_data, all_label, all_result_matrix] = GetAllData(name)
    % Read data from saved file.
    % Input: name: name of the file that has the saved data
    % Output: all_data: All the collected data. Its size is
    %                   (channel_num + 1) * sample_size
    %         all_label: The labels that correspond to each sample. Its 
    %                    size is 1 * sample_size
    %         all_result_matrix: The matrix that uses one-hot encoding to
    %                            encode labels. Its size is
    %                            num_possible_labels * sample_size
    load(name)

    % Process and normalize input data
    channel = size(firingrate, 1);
    number_samples_per_direction = size(firingrate, 2);
    number_directions = size(firingrate, 3);
    all_data = ones(channel + 1, number_samples_per_direction *...
        number_directions);
    all_label = zeros(1, number_samples_per_direction * number_directions);
    all_result_matrix = zeros(number_directions,...
        number_samples_per_direction * number_directions);
    count = 1;
    for j = 1:number_samples_per_direction
        for k = 1:number_directions
            all_data(1:channel, count) = firingrate(:, j, k);
            all_result_matrix(k, count) = 1;
            all_label(1, count) = k;
            count = count + 1;
        end
    end
    all_data = all_data / sqrt(count - 1);
end