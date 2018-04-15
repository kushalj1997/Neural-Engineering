%% Lab 11 - Epilepsy Prediction
close all
load('Patient_A.mat')
Pat_A = data;
load('Patient_B.mat')
Pat_B = data;
% set sampling frequency and time domains
samp_freq = 3*1000; % [Hz]
time_pt = 1/samp_freq; % time between each point [s] windows or 1000 points
end_time = length(Pat_A(1,:))*time_pt;
windowsize = 1000; %size of the bins
time_pbin = windowsize*time_pt;
time = linspace(0,end_time,length(Pat_A(1,:)));
trainA_dat = Pat_A(3,:);
testA_dat = Pat_B(2,:);
% train the decoder by determine thresholds
rms_func = @rms;
zc_func = @zc_calc;
ll_func = @ll_calc;

rms_results = bin_operate(trainA_dat, windowsize, rms_func);
trainA_rms = smooth(rms_results(2,:),0.0056,'moving');
zero_cross_results= bin_operate(trainA_dat, windowsize, zc_func);
trainA_zc= smooth(zero_cross_results(2,:),0.0056,'moving');
line_len_results= bin_operate(trainA_dat, windowsize, ll_func);
trainA_ll = smooth(line_len_results(2,:),0.0056,'moving');
ll_time = time_pt*line_len_results(1,:)';
bin_time = ll_time;
% After determining thresholds - apply them here
rmsA_th = 400;
% zcA_th = 10;
llA_th = 12000;
% Test on the training data first
seizA_train = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if trainA_rms(i) > rmsA_th && trainA_ll(i) > llA_th
        seizA_train(i) = 1;
    end
end
% Test on the testing data now
rms_results = bin_operate(testA_dat, windowsize, rms_func);
testA_rms = smooth(rms_results(2,:),0.0056,'moving');
zero_cross_results= bin_operate(testA_dat, windowsize, zc_func);
testA_zc = smooth(zero_cross_results(2,:),0.0056,'moving');
line_len_results = bin_operate(testA_dat, windowsize, ll_func);
testA_ll = smooth(line_len_results(2,:),0.0056,'moving');
seizA_test = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if testA_rms(i) > rmsA_th && testA_ll(i) > llA_th
        seizA_test(i) = 1;
    end
end

% plot the training data and features
figure(1)
subplot(3,1,1)
plot(bin_time, trainA_rms,'r','linewidth',2)
hold on
hold off

subplot(3,1,2)
plot(bin_time, trainA_zc,'r','linewidth',2)
hold on
hold off
subplot(3,1,3)
plot(bin_time, trainA_ll,'r','linewidth',2)
hold on
hold off


figure(3)
subplot(2,1,1)
plot(time,trainA_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizA_train,'linewidth',2)
ylim([0 1.1])
hold on
hold off
% plot the testing data and features
figure(2)
subplot(3,1,1)
plot(bin_time, testA_rms,'r','linewidth',2)
hold on
hold off
subplot(3,1,2)
plot(bin_time, testA_zc,'r','linewidth',2)
hold on
hold off
subplot(3,1,3)
plot(bin_time, testA_ll,'r','linewidth',2)
hold on
hold off

figure(4)
subplot(2,1,1)
plot(time,testA_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizA_test,'linewidth',2)
ylim([0 1.1])
hold on

hold off
%% Do the same now using Patient B as the training and Pat
A as the testing
trainB_dat = Pat_B(2,:);
testB_dat = Pat_A(3,:);
rms_results = bin_operate(trainB_dat, windowsize, rms_func);
trainB_rms = smooth(rms_results(2,:),0.0056,'moving');
% zero_cross_results= bin_operate(trainB_dat, windowsize, zc_func);
% trainB_zc = smooth(zero_cross_results(2,:),0.0056,'moving'); line_len_results= bin_operate(trainB_dat, windowsize, ll_func);
trainB_ll = smooth(line_len_results(2,:),0.0056,'moving');
% After determining thresholds - apply them here
rmsB_th = 600;
% zcB_th = 18;
llB_th = 15000;
% Test on the training data first
seizB_train = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if trainB_rms(i) > rmsB_th && trainB_ll(i) > llB_th
        seizB_train(i) = 1;
    end
end

% Test on the testing data now

rms_results = bin_operate(testB_dat, windowsize, rms_func);
testB_rms = smooth(rms_results(2,:),0.0056,'moving');
% zero_cross_results= bin_operate(testB_dat, windowsize, zc_func);
% testB_zc = smooth(zero_cross_results(2,:),0.0056,'moving'); line_len_results = bin_operate(testB_dat, windowsize, ll_func);
testB_ll = smooth(line_len_results(2,:),0.0056,'moving');

seizB_test = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if testB_rms(i) > rmsB_th && testB_ll(i) > llB_th
        seizB_test(i) = 1;
    end
end

% plot the training data
figure(5)
subplot(2,1,1)
plot(time,trainB_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizB_train,'linewidth',2)
ylim([0 1.1])
hold on
hold off
% plot the testing data
figure(6)
subplot(2,1,1)
plot(time,testB_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizB_test,'linewidth',2)
ylim([0 1.1])
hold on
hold off
%% LDA decoder - train and test with these features and the
seizures binary predictors
seizA = zeros(1,length(bin_time)); seizB = zeros(1,length(bin_time));
seizA_start = find(bin_time > 818.4,1);
seizA_end = find(bin_time > 924.3,1);
seizB_start = find(bin_time > 931,1);
seizB_end = find(bin_time > 1025,1);
seizA(seizA_start:seizA_end) = 1;
seizB(seizB_start:seizB_end) = 1;
seiz_start = [seizB_start,seizA_start];

seiz_end = [seizB_end,seizA_end];
% LDA decoder
correct = zeros(1,2);
correct_seiz = zeros(1,2);
pred = zeros(1,2);
pred_seiz = zeros(1,2);
trainA_class = trainA_rms;
trainB_class = trainB_rms;
testA_class = testA_rms;
testB_class = testB_rms;
ttruth = [seizA ; seizB];
gtruth = [seizB ; seizA];
seiz_decoder = zeros(2,length(bin_time));
seiz_decoder(1,:) = classify(testA_class, trainA_class, ttruth(1,:)', 'linear');
seiz_decoder(2,:) = classify(testB_class, trainB_class, ttruth(2,:)', 'linear');
for i=1:2
    for j=1:length(bin_time)
        if j >= seiz_start(i) && j <= seiz_end(i)
            if seiz_decoder(i,j) == gtruth(i,j)
                correct(i)=correct(i)+1;
                correct_seiz(i)=correct_seiz(i)+1;
            end
            pred_seiz(i) = pred_seiz(i)+1;
        else
            if seiz_decoder(i,j) == gtruth(i,j)
                correct(i)=correct(i)+1;
            end
        end
        pred(i) = pred(i)+1;
    end
end
Pat = ['Patient B' ; 'Patient A'];
for i=1:2
    LDA_accu_all(i) = correct(i)/pred(i);
    LDA_accu_seiz(i) = correct_seiz(i)/pred_seiz(i);
    fprintf('LDA Accuracy on %s: %.2f%%\n',Pat(i,:),LDA_accu_all(i)*100)
    fprintf('LDA Accuracy for seizure on %s: %.2f%%\n',Pat(i,:),LDA_accu_seiz(i)*100)
end

% plot LDA results

figure(7)
subplot(2,1,1)
plot(time,testA_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizB,'linewidth',2)
hold on
plot(bin_time,seiz_decoder(1,:),'r--','linewidth',1.3)
legend('Seizure Event','LDA Predicted Seizure')
ylim([0 1.1])
hold off
figure(8)
subplot(2,1,1)
plot(time,testB_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizA,'linewidth',2)
hold on
plot(bin_time,seiz_decoder(2,:),'r--','linewidth',1.3)
legend('Seizure Event','LDA Predicted Seizure')
ylim([0 1.1])
hold off

%% HFOs - filtered data
[filt_z,filt_p,filt_k]=butter(2,80/3000,'low');
filt_sos = zp2sos(filt_z,filt_p,filt_k);
filt_Pat_A = filtfilt(filt_sos,1,Pat_A(3,:));
filt_Pat_B = filtfilt(filt_sos,1,Pat_B(2,:));
trainA_dat = filt_Pat_A;
testA_dat = filt_Pat_B;
% train the decoder by determine thresholds
rms_results = bin_operate(trainA_dat, windowsize, rms_func);
trainA_rms = smooth(rms_results(2,:),0.0112,'moving');
zero_cross_results= bin_operate(trainA_dat, windowsize, zc_func);
trainA_zc = smooth(zero_cross_results(2,:),0.0112,'moving');

line_len_results= bin_operate(trainA_dat, windowsize, ll_func);
trainA_ll = smooth(line_len_results(2,:),0.0112,'moving');
% After determining thresholds - apply them here
rmsA_th = 400;
zcA_th = 10;
llA_th = 3933;
% Test on the training data first
seizA_train = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if trainA_rms(i) > rmsA_th && trainA_ll(i) > llA_th
        seizA_train(i) = 1;
    end
end
% Test on the testing data now
rms_results = bin_operate(testA_dat, windowsize, rms_func);
testA_rms = smooth(rms_results(2,:),0.0112,'moving');
zero_cross_results= bin_operate(testA_dat, windowsize, zc_func);
testA_zc = smooth(zero_cross_results(2,:),0.0112,'moving');
line_len_results = bin_operate(testA_dat, windowsize, ll_func);
testA_ll = smooth(line_len_results(2,:),0.0112,'moving');
seizA_test = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if testA_rms(i) > rmsA_th && testA_ll(i) > llA_th
        seizA_test(i) = 1;
    end
end
% plot the training data and features
figure(9)
subplot(3,1,1)
plot(bin_time, trainA_rms,'r','linewidth',2)
hold on
hold off
subplot(3,1,2)
plot(bin_time, trainA_zc,'r','linewidth',2)

hold on
hold off
subplot(3,1,3)
plot(bin_time, trainA_ll,'r','linewidth',2)
hold on
hold off
figure(11)
subplot(2,1,1)
plot(time,trainA_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizA_train,'linewidth',2)
ylim([0 1.1])
hold on
hold off
% plot the testing data and features
figure(10)
subplot(3,1,1)
plot(bin_time, testA_rms,'r','linewidth',2)
hold on
hold off
subplot(3,1,2)
plot(bin_time,testA_zc,'r','linewidth',2)
hold on
hold off
subplot(3,1,3)
plot(bin_time, testA_ll,'r','linewidth',2)
hold on
hold off

figure(12)
subplot(2,1,1)
plot(time,testA_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizA_test,'linewidth',2)
ylim([0 1.1])
hold on
hold off

% testing it with the other data now
trainB_dat = filt_Pat_B;
testB_dat = filt_Pat_A;
rms_results = bin_operate(trainB_dat, windowsize, rms_func);
trainB_rms = smooth(rms_results(2,:),0.0112,'moving');
% zero_cross_results= bin_operate(trainB_dat, windowsize, zc_func);
% trainB_zc = smooth(zero_cross_results(2,:),0.0112,'moving');
line_len_results= bin_operate(trainB_dat, windowsize, ll_func);
trainB_ll = smooth(line_len_results(2,:),0.0112,'moving');
% After determining thresholds - apply them here
rmsB_th = 411;
zcB_th = 18;
llB_th = 5000;
% Test on the training data first
seizB_train = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if trainB_rms(i) > rmsB_th && trainB_ll(i) > llB_th
        seizB_train(i) = 1;
    end
end
% Test on the testing data now
rms_results = bin_operate(testB_dat, windowsize, rms_func);
testB_rms = smooth(rms_results(2,:),0.0112,'moving');
% zero_cross_results= bin_operate(testB_dat, windowsize, zc_func);
% testB_zc = smooth(zero_cross_results(2,:),0.0112,'moving');
line_len_results = bin_operate(testB_dat, windowsize, ll_func);
testB_ll = smooth(line_len_results(2,:),0.0112,'moving');
seizB_test = zeros(1,length(bin_time));
for i = 1:length(bin_time)
    if testB_rms(i) > rmsB_th && testB_ll(i) > llB_th
        seizB_test(i) = 1;
    end
end
% plot the training data
figure(13)
subplot(2,1,1)
plot(time,trainB_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizB_train,'linewidth',2)
ylim([0 1.1])
hold on
hold off
% plot the testing data
figure(14)
subplot(2,1,1)
plot(time,testB_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizB_test,'linewidth',2)
ylim([0 1.1])
hold on
hold off
% LDA
seizA = zeros(1,length(bin_time));
seizB = zeros(1,length(bin_time));
seizA_start = find(bin_time > 818.4,1);
seizA_end = find(bin_time > 924.3,1);
seizB_start = find(bin_time > 931,1);
seizB_end = find(bin_time > 1025,1);
seizA(seizA_start:seizA_end) = 1;
seizB(seizB_start:seizB_end) = 1;
seiz_start = [seizB_start,seizA_start];
seiz_end = [seizB_end,seizA_end];
% LDA decoder
correct = zeros(1,2);
correct_seiz = zeros(1,2);
pred = zeros(1,2);
pred_seiz = zeros(1,2);

trainA_class = [trainA_rms,trainA_ll];
trainB_class = [trainB_rms,trainB_ll];
testA_class = [testA_rms,testA_ll];
testB_class = [testB_rms,testB_ll];
ttruth = [seizA ; seizB];
gtruth = [seizB ; seizA];
seiz_decoder = zeros(2,length(bin_time));
seiz_decoder(1,:) = classify(testA_class, trainA_class, ttruth(1,:)', 'linear');
seiz_decoder(2,:) = classify(testB_class, trainB_class, ttruth(2,:)', 'linear');
for i=1:2
    for j=1:length(bin_time)
        if j >= seiz_start(i) && j <= seiz_end(i)
            if seiz_decoder(i,j) == gtruth(i,j)
                correct(i)=correct(i)+1;
                correct_seiz(i)=correct_seiz(i)+1;
            end
            pred_seiz(i) = pred_seiz(i)+1;
        else
            if seiz_decoder(i,j) == gtruth(i,j)
                correct(i)=correct(i)+1;
            end         
        end
        pred(i) = pred(i)+1;
    end
end
Pat = ['Patient B' ; 'Patient A'];
for i=1:2
    LDA_accu_all(i) = correct(i)/pred(i);
    LDA_accu_seiz(i) = correct_seiz(i)/pred_seiz(i); 
    fprintf('LDA Accuracy on %s: %.2f%%\n',Pat(i,:),LDA_accu_all(i)*100)
    fprintf('LDA Accuracy for seizure on %s: %.2f%%\n',Pat(i,:),LDA_accu_seiz(i)*100)
end
% plot LDA results
figure(15)
subplot(2,1,1)
plot(time,testA_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)

plot(bin_time,seizB,'linewidth',2)
hold on plot(bin_time,seiz_decoder(1,:),'r--','linewidth',1.3) 
legend('Seizure Event','LDA Predicted Seizure') 
ylim([0 1.1])
hold off

figure(16)
subplot(2,1,1)
plot(time,testB_dat,'linewidth',1.3)
hold on
hold off
subplot(2,1,2)
plot(bin_time,seizA,'linewidth',2)
hold on
plot(bin_time,seiz_decoder(2,:),'r--','linewidth',1.3)
legend('Seizure Event','LDA Predicted Seizure')
ylim([0 1.1])
hold off

%{
%% Fix figures
% figure(1)
% subplot(3,1,1)
% title1 = title('Patient A Features','fontsize',16); % ylabel1 = ylabel('RMS Amp [uV]','fontsize',14);
% xlim([0 1800]);
% ylim([0 1000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick' ...
%   'LineWidth'
% subplot(3,1,2)
% ylabel1 = ylabel('Zero Crossings','fontsize',14); % xlim([0 1800]);
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,500,1000],'FontName','Times-Roman',
, 1.3);

% ylim([0 10]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
% subplot(3,1,3)
% ylabel1 = ylabel('Line Length','fontsize',14);
% xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% ylim([0 25000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,5,10],'FontName','Times-Roman', ...
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(2)
% subplot(3,1,1)
% title1 = title('Patient B Features','fontsize',16); % ylabel1 = ylabel('RMS Amp [uV]','fontsize',14);
% xlim([0 1800]);
% ylim([0 4000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'      , ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,25000],'FontName','Times-Roman', ...
, 1.3);

%   'YMinorTick'  , 'off'      , ...
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick' ...
%   'LineWidth'
% subplot(3,1,2)
% ylabel1 = ylabel('Zero
% xlim([0 1800]);
% ylim([0 10]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'      , ...
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,2000,4000],'FontName','Times-Roman', , 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
% subplot(3,1,3)
% ylabel1 = ylabel('Line Length','fontsize',14);
% xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% ylim([0 60000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
, ...
, ...
, ...
Crossings','fontsize',14);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,5,10],'FontName','Times-Roman', ...
, 1.3);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,60000],'FontName','Times-Roman', ...
, 1.3);

%
% figure(9)
% subplot(3,1,1)
% title1 = title('Patient A HFO Features','fontsize',16); % ylabel1 = ylabel('RMS Amp [uV]','fontsize',14);
% xlim([0 1800]);
% ylim([0 1000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick' ...
%   'LineWidth'
% subplot(3,1,2)
% ylabel1 = ylabel('Zero
% xlim([0 1800]);
% ylim([0 10]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
% subplot(3,1,3)
% ylabel1 = ylabel('Line Length','fontsize',14);
% xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% ylim([0 25000]);
% set(gca,...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,500,1000],'FontName','Times-Roman',
, 1.3);
Crossings','fontsize',14);
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,5,10],'FontName','Times-Roman', ...
, 1.3);

%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(10)
% subplot(3,1,1)
% title1 = title('Patient B HFO Features','fontsize',16); % ylabel1 = ylabel('RMS Amp [uV]','fontsize',14);
% xlim([0 1800]);
% ylim([0 4000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick' ...
%   'LineWidth'
% subplot(3,1,2)
% ylabel1 = ylabel('Zero Crossings','fontsize',14); % xlim([0 1800]);
% ylim([0 10]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
%   'YGrid'       , 'off'
, ...
, ...
, ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,25000],'FontName','Times-Roman', ...
, 1.3);
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,2000,4000],'FontName','Times-Roman', , 1.3);

% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
% subplot(3,1,3)
% ylabel1 = ylabel('Line Length','fontsize',14);
% xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% ylim([0 60000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,5,10],'FontName','Times-Roman', ...
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(3)
% subplot(2,1,1)
% title1 = title('train on Patient A','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-3500 3500]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
, ...
, ...
, ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,60000],'FontName','Times-Roman', ...
, 1.3);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-3500,0,3500],'FontName','Times-

% 'LineWidth' , 1.3);
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(6)
% subplot(2,1,1)
% title1 = title('test on Patient A','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-3500 3500]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-3500,0,3500],'FontName','Times-
, 1.3);

%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(8)
% subplot(2,1,1)
% title1 = title('LDA decoder test on Patient
A','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-3500 3500]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'      , ...
%   'YMinorTick'  , 'off'      , ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-3500,0,3500],'FontName','Times-
, 1.3);

% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(11)
% subplot(2,1,1)
% title1 = title('HFO train on Patient A','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-3500 3500]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, 'off'      , ...
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
, ...
, ...
, ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-3500,0,3500],'FontName','Times-
, 1.3);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...

%   'LineWidth'   , 1.3);
%
% figure(14)
% subplot(2,1,1)
% title1 = title('HFO test on Patient A','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-3500 3500]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-3500,0,3500],'FontName','Times-
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(16)
% subplot(2,1,1)
% title1 = title('LDA decoder test on HFO Patient
A','fontsize',16);
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);

% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-3500 3500]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-3500,0,3500],'FontName','Times-
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(4)
% subplot(2,1,1)
% title1 = title('test on Patient B','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-15000 15000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);

%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(5)
% subplot(2,1,1)
% title1 = title('train on Patient B','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-15000 15000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
, ...
, ...
, ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-15000,0,15000],'FontName','Times-
, 1.3);
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...

% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, [0,900,1800], 'FontName','Times-Roman',
, [-15000,0,15000],'FontName','Times-
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(7)
% subplot(2,1,1)
% title1 = title('LDA decoder test on Patient
B','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-15000 15000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
%   'LineWidth'
, ...
, ...
, ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-15000,0,15000],'FontName','Times-
, 1.3);

% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(12)
% subplot(2,1,1)
% title1 = title('HFO test on Patient B','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-15000 15000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
% 'Box' , 'off', ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-15000,0,15000],'FontName','Times-
, 1.3);

%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
% figure(13)
% subplot(2,1,1)
% title1 = title('HFO train on Patient B','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-15000 15000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
%   'YGrid'       , 'off'
%   'XColor'      , [.3 .3 .3], ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-15000,0,15000],'FontName','Times-
, 1.3);
, ...
, ...
, ...

% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
%
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
% figure(15)
% subplot(2,1,1)
% title1 = title('LDA decoder test on HFO Patient B','fontsize',16);
% ylabel1 = ylabel('Voltage trace [uV]','fontsize',14); % xlim([0 1800]);
% ylim([-15000 15000]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
%   'YTick'
Roman', ...
% 'LineWidth'
% subplot(2,1,2)
% ylabel1 = ylabel('Seizure Detector','fontsize',14); % xlabel1 = xlabel('Time [s]','fontsize',14);
% xlim([0 1800]);
% set(gca,...
%   'Box' , 'off', ...
%   'TickDir','out', ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'off'
%   'YMinorTick'  , 'off'
% 'YGrid'
% 'XColor'
% 'YColor'
% 'XTick' ...
% 'YTick'
%   'LineWidth'
, ...
, ...
, ...
, ...
, ...
, ...
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [-15000,0,15000],'FontName','Times-
, 1.3);
, 'off'
, [.3 .3 .3], ...
, [.3 .3 .3], ...
, [0,900,1800], 'FontName','Times-Roman',
, [0,1],'FontName','Times-Roman', ...
, 1.3);
%}