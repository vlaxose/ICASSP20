function [Y_conventional_hbf, Psi_bar, Y] = hbf(H, N, Psi_i, T, W)

   %% Parameter initialization
   [~, Nt, L] = size(H);

   %% Variables initialization
   Psi_bar = zeros(Nt, T, L);

   %% Wideband channel modeling

   % Construct the received signal
   Y = zeros(size(N));
   for l=1:L
    for k=1:Nt
     Psi_bar(k,:,l) = Psi_i(l,:,k);
    end
    Y = Y + H(:,:,l)*Psi_bar(:,:,l);
   end
     
   R = Y + N;

   %% Conventional HBF architecture
   Y_conventional_hbf = W'*R;
   
end
