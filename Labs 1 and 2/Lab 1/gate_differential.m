function diff = gate_differential(timestep, gate_prob, alpha, beta)
diff = timestep * (alpha * (1 - gate_prob) - (beta * gate_prob));
end