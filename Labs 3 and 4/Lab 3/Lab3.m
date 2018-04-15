% BIOMEDE 517 - Lab 3
% Kushal Jaligama

% Part 2 - Create a plot of voltage versus distance from stimulus for
% single point current source
Part2Data = importdata('part2export.txt', ' ');
Voltage = zeros(size(Part2Data(:,1)));
DistanceFromStimulus = zeros(size(Part2Data(:,1)));

for i = 1:size(Part2Data)
    Voltage(i) = Part2Data(i,4);
    X = Part2Data(i,1);
    Y = Part2Data(i,2);
    Z = Part2Data(i,3);
    DistanceFromStimulus(i) = sqrt((X-0)^2 + (Y-0)^2 + (Z-0)^2);
end

figure(1)
hold on
syms f(x)
f(x) = 1/x;
fplot(f, 'r')
scatter(DistanceFromStimulus, Voltage)
hold off


% Part 3 - Create a plot of voltage versus distance from stimulus for a
% Deep Brain Stimulation (DBS) model
Part3Data = importdata('part3export.txt');
DBSVoltages = zeros(size(Part3Data(:,1)));
DBSDistances = zeros(size(Part3Data(:,1)));

for i = 1:size(Part3Data)
   DBSVoltages(i) = Part3Data(i, 4);
   X = Part3Data(i,1);
   Y = Part3Data(i,2);
   Z = Part3Data(i,3);
   DBSDistances(i) = sqrt((X-0)^2 + (Y-0)^2 + (Z-0)^2);
end

% subplot(4,1,2)
% scatter(DBSDistances, DBSVoltages)

% Interpolate the data
interpolated = scatteredInterpolant(Part3Data(:,1), Part3Data(:,2), Part3Data(:,3), Part3Data(:,4));
x_coordinates = [transpose(linspace(-0.1, 0.1, 1000)), zeros(1000, 1), zeros(1000, 1)];
x_voltages = interpolated(x_coordinates);

figure(2)
plot(x_voltages);

% Interpolate data and calculate second spatial derivative of
% voltage in direction going away from the active contact and perpendicular
% to the probe (x-direction)

% [v(i+1)-2v(i)+v(i-1)] / (delta_z)^2)this is the second deriv eq

delta_z = 0.001;
for i = 2:size(x_voltages) - 1
    x_voltages_second_deriv(i) = (x_voltages(i+1) - 2*x_voltages(i) + x_voltages(i-1)) / delta_z^2;
end

figure(3)
plot(linspace(-0.1, 0.1, 999), x_voltages_second_deriv)

% Part 4 - Apply patient-specific anisotropic DBS model
% Export conductivity tensors to a form COMSOL can read, scale data
% importTensorData('tensor_data.mat')
% tensor_scalar = 0.844;
% x = x - thalamus_center(1);
% y = y - thalamus_center(2);
% z = z - thalamus_center(3);
% S11 = S11 * tensor_scalar;
% S12 = S12 * tensor_scalar;
% S13 = S13 * tensor_scalar;
% S22 = S22 * tensor_scalar;
% S23 = S23 * tensor_scalar;
% S33 = S33 * tensor_scalar;
% 
% fileID = fopen('L3P4ScaledOut.txt', 'wt');
% fprintf(fileID, '%x y z S11 S12 S13 S22 S23 S33');
% format_string = "%7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f\n";
% fprintf(fileID, format_string, x, y, z, S11, S12, S13, S22, S23, S33);

% Read the voltage data from COMSOL simulation with anisotropic params
Part4Data = importdata('part4export.txt');

% Interpolate the data
interpolated = scatteredInterpolant(Part4Data(:,1), Part4Data(:,2), Part4Data(:,3), Part4Data(:,4));
anisotropic_x_coordinates = [transpose(linspace(-0.02, 0.02, 1000)), zeros(1000, 1), zeros(1000, 1)];
anisotropic_x_voltages = interpolated(anisotropic_x_coordinates);

figure(4)
plot(linspace(-0.02, 0.02, 1000), anisotropic_x_voltages);

% Interpolate data and calculate second spatial derivative like above
for i = 2:size(x_voltages) - 1
    anisotropic_x_voltages_second_deriv(i) = (anisotropic_x_voltages(i+1) - 2*anisotropic_x_voltages(i) + anisotropic_x_voltages(i-1)) / delta_z^2;
end

figure(5)
plot(linspace(-0.02, 0.02, 999), anisotropic_x_voltages_second_deriv);
