% Random variable generator based on inverse transform sampling
function x=genLaplacianSamples(N)
    u = rand(N,1);
    % Inverse transform of trancuted Laplacian
    sigma_phi = 50; % standard deviation of the power azimuth spectrum (PAS)
    beta = 1/(1-exp(-sqrt(2)*pi/sigma_phi));
    x = beta*(exp(-sqrt(2)/sigma_phi*pi) - cosh(u));
end