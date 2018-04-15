% Part 1
% Problem 1
z_K = 1;
Co_K = 4;
Ci_K = 155;

E_K = 25/z_K*log(Co_K/Ci_K);

z_Na = 1;
Co_Na = 145;
Ci_Na = 12;

E_Na = 25/z_Na*log(Co_Na/Ci_Na);

% Problem 2
P_K = 1.0;
P_Na = 0.04;
P_Cl = 0.45;

Co_Cl = 120;
Ci_Cl = 4;

Vm_rest = 25*log((P_K * Co_K + P_Na * Co_Na + P_Cl * Ci_Cl) / (P_K * Ci_K + P_Na * Ci_Na + P_Cl * Co_Cl)) 

% Problem 3
% Calculate the resting point for all dynamic channel variables
% These will be simulated over time to generate the action potential,
% m(inf), n(inf), and h(inf)
% V_rest corresponds to Vm (resting potential) of cell membrane

% Sodium Gate Probabilities
m_gate = gate_open_probability(Vm_rest, 0.182, -35, 9, 0.124, -35, 9)
h_gate = gate_open_probability(Vm_rest, 0.024, -50, 5, 0.0091, -75, 5)

% Potassium Gate Probability
n_gate = gate_open_probability(Vm_rest, 0.02, 20, 9, 0.002, 20, 9)

% Part 2 - Outputs the graphs for action potential, m, h, n gate
% probabilities
action_potential(Vm_rest, m_gate, h_gate, n_gate, E_Na, E_K)

% Part 3
% Problem 1 - threshold voltage = -45.48mV @ current of 250mA (obtained by
% looking at the graph and noticing where the voltage begins to spike)

% Problem 2 - calculate length constant associated with myelinated part of
% axon compartment
axon_radius = 1.5e-6 % m
R_m = 40000e-4 % ohm-m^2
R_i = 200e-2 % ohm-m
r_m = R_m / (pi * (axon_radius*2)) % ohm-m
r_i = R_i / (pi * (axon_radius)^2) % ohm/m
length_constant = sqrt(r_m / r_i) % in m (lambda)
length_constant = length_constant*10^(6) % um

% Problem 3 - how far can myelin extend down cable and still enable action
% potential to be generated downstream
% Normalize the action potential voltages
peak_voltage = 60
V_0 = peak_voltage - Vm_rest
V_thresh = -45.8
V_x = V_thresh - Vm_rest
critical_length = log(V_x / V_0) * -length_constant % um
critical_length = critical_length * 10^(-3) % mm
