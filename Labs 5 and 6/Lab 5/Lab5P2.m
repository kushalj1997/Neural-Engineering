% BIOMEDE 517 - Neural Engineering
% Lab 5 Part 2
% Kushal Jaligama

% Part 2
% Apply common average referencing to channel 29, only using channels
% in refChannels
refIndex = 1;
refzero = 1;

% refEcogData vector will contain all of the channels that are 1 in
% refChannels
for i=1:128
    if (refChannels(i) == 1)
        row = ecogData(i,:);
        refEcogData(refIndex, :) = row;
        refIndex = refIndex + 1;
    end
    if (refChannels(i) == 0)
        refzero = refzero + 1;
    end
end

% Gather the common average of the channels
average = mean(refEcogData);
% Then subtract the average from channel 29 to reference it
CAR_row = ecogData(29, :) - average;

% Now create filters that will grab the 3 bands of data specified in
% Pistohl et. al. 2011
low_freq = 2; % Hz
high_freq = 6; % Hz
low_band = designfilt('bandpassiir','FilterOrder',20,'HalfPowerFrequency1',low_freq,'HalfPowerFrequency2',high_freq,'SampleRate',1000);
low_freq = 14; % Hz
high_freq = 46 % Hz
intermediate_band = designfilt('bandpassiir','FilterOrder',20,'HalfPowerFrequency1',low_freq,'HalfPowerFrequency2',high_freq,'SampleRate',1000);
low_freq = 54; % Hz
high_freq = 114; % Hz
high_band = designfilt('bandpassiir','FilterOrder',20,'HalfPowerFrequency1',low_freq,'HalfPowerFrequency2',high_freq,'SampleRate',1000);

% Apply the filters to gather the waveforms of each frequency range
low_out = filter(low_band, CAR_row);
inter_out = filter(intermediate_band, CAR_row);
high_out = filter(high_band, CAR_row);

% Plot the data
% We have 8003 snippets of data recorded at 1000Hz (each snippet is 1 ms)
figure(1);
subplot(3, 1, 1);
plot(smooth(low_out.^2, 100));
subplot(3, 1, 2);
plot(smooth(inter_out.^2, 100));
subplot(3, 1, 3);
plot(smooth(high_out.^2, 100));
