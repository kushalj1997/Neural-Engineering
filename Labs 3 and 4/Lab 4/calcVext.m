function voltageTrace = calcVext(currentTracesOverTime, currentXYZ, electrodeXYZ)
    % Cindy's Favorite Equation
    % V = I_0 / (4*pi*r*small_sigma)
    % Scale the input distances to meters
    currentXYZ = currentXYZ * 1e-6;
    electrodeXYZ = electrodeXYZ * 1e-6;
    
    r = sqrt((currentXYZ(:,1)-electrodeXYZ(1)).^2 + (currentXYZ(:,2)-electrodeXYZ(2)).^2 + (currentXYZ(:,3)-electrodeXYZ(3)).^2);
    small_sigma = 0.3333; % S/m => Conductivity
    
    size(r)
    size(currentTracesOverTime)
    
    % Perform Cindy's Equation for all of the currents for all compartments
    voltageTraces(:,:) = currentTracesOverTime(:,:) ./ (4 .* pi .* r(:,:) .* small_sigma);
    % Results in voltage traces at a distance r away from the compartment
    
    voltageTrace(1,:) = sum(voltageTraces(:,:));
end