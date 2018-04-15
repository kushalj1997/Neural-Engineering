% BIOMEDE 517 - Neural Engineering
% Lab 4
% Kushal Jaligama

% Part 1
% Load currents_<big/small>.mat
importfile('currents_big.mat');

% Calculate voltage at a point 50um perpendicularly away from neuron.
% This is based on the current traveling through each compartment. This
% external voltage varies over time, as action potential propogates.
% The external voltage is the sum of the v measurements of every
% compartment, at each time frame. 

% Time Axis
times_axis = linspace(0, 18.95, 758);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 50 0];
voltageSuperposition50 = calcVext(currents,XYZ,electrodeXYZ);
hold on
plot(times_axis, voltageSuperposition50);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 100 0];
voltageSuperposition100 = calcVext(currents,XYZ,electrodeXYZ);
plot(times_axis, voltageSuperposition100);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 200 0];
voltageSuperposition200 = calcVext(currents,XYZ,electrodeXYZ);
plot(times_axis, voltageSuperposition200);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 300 0];
voltageSuperposition300 = calcVext(currents,XYZ,electrodeXYZ);
plot(times_axis, voltageSuperposition300);
hold off

% Repeat for smaller neuron

% Load currents_<big/small>.mat
importfile('currents_small.mat');

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 50 0];
voltageSuperposition50 = calcVext(currents,XYZ,electrodeXYZ);
figure(2);
hold on
plot(times_axis, voltageSuperposition50);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 100 0];
voltageSuperposition100 = calcVext(currents,XYZ,electrodeXYZ);
plot(times_axis, voltageSuperposition100);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 200 0];
voltageSuperposition200 = calcVext(currents,XYZ,electrodeXYZ);
plot(times_axis, voltageSuperposition200);

% currents mat object is organized as time on X axis, compartment on Y axis
electrodeXYZ = [0 300 0];
voltageSuperposition300 = calcVext(currents,XYZ,electrodeXYZ);
plot(times_axis, voltageSuperposition300);
hold off


% Part 2 - Fire 10 neurons for 1s between 2-50Hz, dt = 0.025ms
% Put the ten neurons at intervals along this variation
% inside the 100um cube
variation_coordinates = [zeros(10,1), transpose(linspace(-40, 40, 10)), zeros(10,1)];
electrodeXYZ = [0 0 50];
new_coordinates = zeros(5310, 3);
for i=1:10
    for j=1:531
        % Place a neuron's compartment at new coordinate
        new_coordinates(j + 531 * (i - 1),:) = XYZ(j,:) + variation_coordinates(i,:);
    end
end

% Generate a current at 40Hz for 1s, this lines up with dt = 0.025
% This is good because the currents_<big/small> objects are at dt = 0.025s
% Do this by creating an array of neuron currents for 40,000 timesteps at 0.025 dt
new_currents = zeros(5310, 40000);
% for i=1:10
%     offset = (i-1) * 1000; % Time steps * 0.025ms = 5ms offset for each neuron
%     for t=1+offset:758:40000
%         % Put the currents for each neuron, separated by some offset so
%         % that no neuron has an action potential at the same time         
%         new_currents(531 * i - 530 : i*531, t:t+757) = currents;
%     end
% end

t = 1; % ms
while t<40000
    % Randomly pick which neuron fires next
    next_neuron = randi(10);
    % Check that the neuron is not firing
    while (new_currents(next_neuron*531,t) ~= 0)
        next_neuron = randi(10);
    end
    fire_this = next_neuron;
    new_currents((fire_this*531-530):(fire_this*531),t:t+757) = currents;
    % Randomly decide how long to wait before the neuron fires again
    delay = randi(20*40);
    t = t + delay;
end

new_currents = new_currents(:, 1:40000);

figure(5)
plot(linspace(1,20, 40000), new_currents);

% Account for the dead zone of 50um around the electrode
new_coordinate_distances = sqrt((new_coordinates(:,1)-50).^2 + (new_coordinates(:,1)-50).^2 + (new_coordinates(:,1)-50).^2);
% Store indices which of the coordinates are further than 50 um away
alive_zone = find(new_coordinate_distances>=50);

% Calculate the voltage at the electrode for compartments only outside
% deadzone
electrode_voltage_with_dead_zone = calcVext(new_currents(alive_zone,:), new_coordinates(alive_zone,:), (100)*[.5 .5 .5]);
% Calculate the voltage at the electrode for all compartments
electrode_voltage = calcVext(new_currents, new_coordinates, (100)*[.5 .5 .5]);

% Limit the bounds for plotting
electrode_voltage = electrode_voltage(1, 1:40000);
electrode_voltage_with_dead_zone = electrode_voltage_with_dead_zone(1, 1:40000);

% Generate pink noise
cn = dsp.ColoredNoise('Color','pink','SamplesPerFrame',40000);
recordedNoise = cn();
recordedNoise = recordedNoise / max(abs(recordedNoise));
recordedNoise = recordedNoise * 5e-6; % scale to 10V p2p noise

% Plot the data for electrode voltage without deadzone
figure(3)
subplot(1,2,1);
hold on;
time_axis_part2 = linspace(0, 1, 40000); % Increment in 0.025 ms
plot(time_axis_part2,electrode_voltage);
% Add the pink noise to plot
plot(time_axis_part2,recordedNoise);
axis([0 1 min(electrode_voltage) max(electrode_voltage)]);

% Plot the data for electrode voltage with deadzone
subplot(1,2,2);
hold on;
plot(time_axis_part2,electrode_voltage_with_dead_zone);
% Add the pink noise in to the plot
plot(time_axis_part2,recordedNoise);
axis([0 1 min(electrode_voltage) max(electrode_voltage)]); % consistent axes


% Part 3
L3P2D = importdata('lab3part2export.txt');
% Place a cell with hillock 50 um from origin using currents mat object
p3_neuron_offset = [0 50 0]; % Y direction
p3_interpolant = scatteredInterpolant(L3P2D(:,1), L3P2D(:,2), L3P2D(:,3), L3P2D(:,4));
% Grab the voltage at the coordinates of the compartments (K matrix)
K_matrix = p3_interpolant(XYZ + p3_neuron_offset);
% V_ext(x,y,z) = I_0(x,y,z) * K(x,y,z)
p3_Vext = currents .* K_matrix;
% Get the superposition of voltage by summing all the voltages
for i= 1:758,
    superposition(1,i) = sum(p3_Vext(:,i));
end

figure(4)
plot(linspace(1, 20, 758), superposition);

% Part 4 - Model Neuron Stimulation

% 1 - The current needed to fire the NEURON model is 4 nA
threshold_current = 4 * 10e-3; % Get 4 nA in milliamps

% 2 - Calculate the external voltage seen by every compartment given an
% electrode that is 1 mm away from middle of axon injecting 1 mA
% Create an axon of 5 um with 1000 compartments and scale 2.5 mm to um
axon_x_coords = [transpose(linspace(-2.5 * 10e3, 2.5 * 10e3, 100)), zeros(100, 1), zeros(100,1)];
% Assume the electrode tip is 1mm away from the middle of the axon
electrode_dist = 1 * 10e3; % Convert 1mm to um
electrodeXYZ = [0 electrode_dist 0];
% Give the electrode a current of 1 mA
elec_current = 1; % mA
% Get a new K-Matrix
K_matrix_2 = p3_interpolant(axon_x_coords + electrodeXYZ) ./ 1000; % Scale from 1 A to 1 mA
% Use the K-Matrix to calculate Vext(x,y,z) for each compartment
compartments_ext_v = threshold_current .* K_matrix_2; % This will be along electrode axis
% compartments_ext_v = calcVext(threshold_current, axon_x_coords, electrodeXYZ);

figure(6)
plot(compartments_ext_v);

% Calculate equivalent intracellular current that arises from injected
% extracellular current using Warman equation
% I_int(n) = g_a * (V_e(n-1) + 2V_e(n) + V_e(n+1))
for i=1:length(compartments_ext_v)
   if i == 1
       I_int(i) = compartments_ext_v(i) + 2*compartments_ext_v(i+1) + compartments_ext_v(i+2);
   elseif i == length(compartments_ext_v)
       I_int(i) = compartments_ext_v(i-2) + 2*compartments_ext_v(i-1) + compartments_ext_v(i);
   else
       I_int(i) = compartments_ext_v(i-1) + 2*compartments_ext_v(i) + compartments_ext_v(i+1);
   end
end

minimal_current = max(compartments_ext_v) / threshold_current

g_a = 3*10^-5; %[S]
injection_current = g_a * I_int;
