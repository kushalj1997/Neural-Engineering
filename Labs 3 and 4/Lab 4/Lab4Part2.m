rand_coords = 100 * [rand(10,1) rand(10,1) rand(10,1)]; % [um, um, um]

t = 1; %[ms]
%dt = 0.025ms; total t is 1 second
pt2currents = zeros(5310,40000); %output currents array
%adjust new compartment coordinates to be relative to each neuron's position
new_compartmentXYZ = [XYZ+rand_coords(1,:);
    XYZ+rand_coords(2,:);
    XYZ+rand_coords(3,:);
    XYZ+rand_coords(4,:);
    XYZ+rand_coords(5,:);
    XYZ+rand_coords(6,:);
    XYZ+rand_coords(7,:);
    XYZ+rand_coords(8,:);
    XYZ+rand_coords(9,:);
    XYZ+rand_coords(10,:)];

while t<40000,
    ntf_candidate = randi(10); %determine which neuron will fire next (of the 10)
    while (pt2currents(ntf_candidate*531,t) ~= 0), %ensure not already firing
        ntf_candidate = randi(10);
    end
    neuron_toFire = ntf_candidate;
    pt2currents((neuron_toFire*531-530):(neuron_toFire*531),t:t+757) = currents;
    delay = randi(20*40); %determine how long before the next firing; (0,20) ms exclusive
    t = t + delay;
end

pt2currents = pt2currents(1:5310,1:40000); %trim to time frame

hold on;

%{
for i=1:10,
    Vi = Vext(pt2currents((i-1)*531+1:(i)*531,:),new_compartmentXYZ((i-1)*531+1:(i)*531,:), (100)*[.5 .5 .5]);
    subplot(10,1,i);
    plot(Vi);
end
%}

% DEAD ZONE
new_compartmentDists = sqrt((new_compartmentXYZ(:,1)-50).^2 + (new_compartmentXYZ(:,1)-50).^2 + (new_compartmentXYZ(:,1)-50).^2);
not_dead = find(new_compartmentDists>=50);

V_electrodeDZ = calcVext(pt2currents(not_dead,:), new_compartmentXYZ(not_dead,:), (100)*[.5 .5 .5]);
V_electrode = calcVext(pt2currents, new_compartmentXYZ, (100)*[.5 .5 .5]);

% Generate pink noise
cn = dsp.ColoredNoise('Color','pink','SamplesPerFrame',40000);
recordedNoise = cn();
recordedNoise = recordedNoise / max(abs(recordedNoise));
recordedNoise = recordedNoise * 5e-6; % scale to 10V p2p noise

% Plotting
subplot(1,2,1);
hold on;
plot([.000025:.000025:1],V_electrode,'b'); %x=t=[s]; y=V=[V]
plot([.000025:.000025:1],recordedNoise,'r'); % Add pink noise to plot
axis([0 1 min(V_electrode) max(V_electrode)]);
subplot(1,2,2);
hold on;
plot([.000025:.000025:1],V_electrodeDZ,'b'); %x=t=[s]; y=V=[V]
plot([.000025:.000025:1],recordedNoise,'r'); % Add pink noise to plot
axis([0 1 min(V_electrode) max(V_electrode)]); % consistent axes














