function beta_out = beta(V_m, A, V_half, k)
beta_out = (-A * (V_m - V_half)) / (1 - exp((V_m - V_half) / k));
end