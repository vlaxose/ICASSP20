% Generate the transmit and receive array responces
function vectors_of_angles=angle(phi, M)

    % For Uniform Linear Arrays (ULA) compute the phase shift
    Ghz = 28;
    wavelength = 30/Ghz; % w=c/lambda
    array_element_spacing = 0.5*wavelength;
    wavenumber = 2*pi/wavelength; % k = 2pi/lambda
    phi0 = 0; % mean AOA
    phase_shift = wavenumber*array_element_spacing*sin(phi0-phi)*(0:M-1).';
    vectors_of_angles = 1/sqrt(M)*exp(-1j*phase_shift);
end
