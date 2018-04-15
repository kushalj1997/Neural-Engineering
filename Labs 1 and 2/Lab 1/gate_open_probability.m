function gate_prob = gate_open_probability(V_rest, A_alpha, V_half_alpha, k_alpha, A_beta, V_half_beta, k_beta)
gate_prob = alpha(V_rest, A_alpha, V_half_alpha, k_alpha) / (alpha(V_rest, A_alpha, V_half_alpha, k_alpha) + beta(V_rest, A_beta, V_half_beta, k_beta));
end