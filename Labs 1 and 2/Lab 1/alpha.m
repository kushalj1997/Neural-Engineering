function alpha_out = alpha(V_m, A, V_half, k)
alpha_out = (A * (V_m - V_half)) / (1 - exp((-(V_m - V_half) / k)));
end