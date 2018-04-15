%% Part 1: Run ICA

%Set path
addpath('extraFunctions'); %add path to topoplot and ICA functions
load('eegPhantomDataSnippet.mat'); %Load 5 minutes of 128-channel EEG data

%Run ICA with PCA reduction down to 60 channels
tic
[wts,sph] = runica(eegData,'extended',0,'pca',60,'maxsteps',512);
toc

%extended off: average step time = 0.148 sec
%extended on: average step time = 22 sec

%ICA_activations = wts * sph * data;
W=wts * sph;
ICs= W* eegData;

%% Part 2: Look at weights for first 6 components
load('phantomDataChanlocs.mat');
invW=pinv(W);
figure('name','Component topoplots');
for i=1:6
    subplot(2,3,i);
    topoplot(invW(:,i), chanlocs);
    title(['Component ' num2str(i)]);
end


%% Part 3: Run FFT on components
Fs=256; %sampling rate (Hz)
figure('name','Power spectra');
for i=1:6
    subplot(2,3,i);
    % Run Fast Fourier Transform (FFT) on first 6 components
    waveform = fft(ICs(i, :));
%     plot(waveform);
    %Use pwelch with hamming window (see matlab documentation for pwelch)
    [spectrum, f] = pwelch(ICs(i,:), hamming(length(ICs)), [],[],Fs);
    plot(f, spectrum);
    xlim([0 60]); %only look at frequencies below 60 Hz
    ylim([0 70]); %optional y-axis setting
    title(['Component ' num2str(i)]);
end
figure('name', 'Fourier')
plot(waveform)

%% Part 4: Look at AMICA results (compare to ICA run performed)
ICs_orig=ICs;
load('icaweights_amica.mat');
load('icasphere_amica.mat');
W=icaweights*icasphere;
ICs=W*eegData;


%Run FFT and topoplot on the resulting components and tell which antenna 
%corresponds to each frequency
load('phantomDataChanlocs.mat');
invW=pinv(W);
figure('name','Component topoplots');
for i=1:6
    subplot(2,3,i);
    topoplot(invW(:,i), chanlocs);
    title(['Component ' num2str(i)]);
end

Fs=256; %sampling rate (Hz)
figure('name','Power spectra');
for i=1:6
    subplot(2,3,i);
    % Run Fast Fourier Transform (FFT) on first 6 components
    waveform = fft(ICs(i, :));
%     plot(waveform);
    %Use pwelch with hamming window (see matlab documentation for pwelch)
    [spectrum, f] = pwelch(ICs(i,:), hamming(length(ICs)), [],[],Fs);
    plot(f, spectrum);
    xlim([0 60]); %only look at frequencies below 60 Hz
    ylim([0 70]); %optional y-axis setting
    title(['Component ' num2str(i)]);
end
