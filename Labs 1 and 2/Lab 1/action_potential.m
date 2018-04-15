% Part 2 of Lab 1
% Get the actional potential curve based on stimulus current and gate probs
function action_potential(V_m, m_gate, h_gate, n_gate, E_Na, E_K)
% Set up constants
C_m = 1; % uF/cm^2
g_bar_Na = 100; % mS/cm^2
g_bar_K = 50; % mS/cm^2
g_leak = 0.5; % ms/cm^2
E_leak = -72.5; % mV
time_step = 0.01;

total_time_steps = 500;

I_M = zeros(total_time_steps, 1);
I_M(10:20) = 100
for t=1:total_time_steps
    % Sodium
    g_Na = g_bar_Na * m_gate^3 * (1-h_gate);
    I_Na = g_Na * (V_m - E_Na);
    
    % Potassium
    g_K = g_bar_K * n_gate^4;
    I_K = g_K * (V_m - E_K);
    
    % Leakage
    I_L = g_leak * (V_m - E_leak);
    
    % Update your deltas
    % Voltage
    delta_V_m = time_step * (I_M(t) - I_Na - I_K - I_L) / C_m;
    % Sodium
    delta_m_gate = gate_differential(time_step, m_gate, alpha(V_m, 0.182, -35, 9), beta(V_m, 0.124, -35, 9));
    delta_h_gate = gate_differential(time_step, h_gate, alpha(V_m, 0.024, -50, 5), beta(V_m, 0.0091, -75, 5));
    % Potassium
    delta_n_gate = gate_differential(time_step, n_gate, alpha(V_m, 0.02, 20, 9), beta(V_m, 0.002, 20, 9));
    
    % Update the actual values
    V_m = V_m + delta_V_m;
    m_gate = m_gate + delta_m_gate;
    h_gate = h_gate + delta_h_gate;
    n_gate = n_gate + delta_n_gate;
    
    % Put into vectors for plotting purposes
    Voltage_Vector(t) = V_m;
    M_Gate_Vector(t) = m_gate;
    H_Gate_Vector(t) = h_gate;
    N_Gate_Vector(t) = n_gate;
end

figure(1)
plot(Voltage_Vector)
figure(2)
hold on
plot(M_Gate_Vector)
% figure(3)
plot(H_Gate_Vector)
% figure(4)
plot(N_Gate_Vector)