close all
clear all
tic
[all_data, all_label, all_result_matrix] = GetAllData('firingrate.mat');
[test_accuracy, training_accuracy] = TrainAndTest(all_data, all_label, all_result_matrix, 5);
toc