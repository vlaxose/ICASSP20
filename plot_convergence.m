    
%% Environment init
clear;
clc;

addpath([pwd,'/basic_system_functions']);
addpath(genpath([pwd, '/benchmark_algorithms']));

%% Parameters init
maxRealizations = 1;
Nt = 4; Nr = 32; Mr = 8; L = 4;
total_num_of_clusters = 1; total_num_of_rays = 2;
maxFrames = 500; snr_db = 5;


% Set an upper length for generating the training samples.
T = 20*Nt;

% For the proposed HBF architecture we assume that the RX gathers
% simultaneously Mr samples over the period of T_prop. So the total length
% of the training period is Mr*T_prop.
T_prop = 5*Nt;

% For the conventional HBF architecutre we assume that the RX gathers Nr
% samples over the period of Mr*T_prop/Nr. So the total length of the
% training sequence is Mr*T_prop.
T_hbf = round(T_prop/(Nr/Mr)); 

square_noise_variance = 10^(-snr_db/10);
gamma = 0.999;
gamma_ar = 0.8;
numOfnz = min(Nr*Nt*L, 10);        

%% Variables definitions
convergence_error = zeros(maxFrames, 4); % convergence error over all frames
ase = zeros(maxFrames, 4); % convergence error over all frames
mean_convergence_error = zeros(maxFrames, 4, maxRealizations);
mean_ase = zeros(maxFrames, 4, maxRealizations);


%% Monte-Carlo realizations
for r=1:maxRealizations
    
    Psi_i = zeros(T, T, Nt);
    Zbar_u_proposed = zeros(Nr, Nt*L);
    Zbar_proposed = zeros(Nr, Nt*L);
    
    Zbar_ada_omp = zeros(Nr, Nt*L);
    res = zeros(Nr*T_hbf, 1);
    
    X = zeros(Nr, T_prop);
    V1 = zeros(Nr, T_prop);
    V2 = zeros(Nr, T_prop);
    C = zeros(Nr, T_prop);
    s = zeros(Nr*Nt*L, 1);
    v = zeros(Nr*Nt*L, 1);
    Yp = zeros(Nr, T_prop);

    %% TX: Generate the training symbols as a Toeplitz matrix
    for k=1:Nt
        sourceSymbols = 1/sqrt(2)*(randn(T, 1) + 1j*randn(T, 1));
        Psi_i(:,:,k) =  toeplitz(sourceSymbols);
    end

    %% Wideband channel modeling
    H = zeros(Nr, Nt, L);
    Z = zeros(Nr, Nt, L);
    rayleigh_coeff = zeros(L, total_num_of_clusters*total_num_of_rays);
    phi_r = zeros(L, total_num_of_clusters*total_num_of_rays);
    phi_t = zeros(L, total_num_of_clusters*total_num_of_rays);
    Ar = zeros(Nr, total_num_of_clusters*total_num_of_rays, L);
    At = zeros(Nt, total_num_of_clusters*total_num_of_rays, L);
    Dr = 1/sqrt(Nr)* exp(-1j*(0:Nr-1)'*2*pi*(0:Nr-1)/Nr);
    Dt = 1/sqrt(Nt)* exp(-1j*(0:Nt-1)'*2*pi*(0:Nt-1)/Nt);
    for l=1:L
        index = 1;
        Hl = zeros(Nr, Nt);            
        for tap = 1:total_num_of_clusters
            for ray=1:total_num_of_rays
                rayleigh_coeff(l, index) = 1/sqrt(2)*(randn(1)+1j*randn(1));
                phi_r(l, index) = genLaplacianSamples(1);
                Ar(:, index, l) = angle(phi_r(l, index), Nr);
                phi_t(l, index) = genLaplacianSamples(1);
                At(:, index, l) = angle(phi_t(l, index), Nt);
                Hl = Hl + rayleigh_coeff(l, index)*Ar(:, index)*At(:, index)';
                index = index + 1;                    
            end
            H(:,:,l) = H(:,:,l) + Hl;

        end
        H(:,:,l) = sqrt(Nr*Nt)/sqrt(total_num_of_rays*total_num_of_clusters)*H(:,:,l);
        Z(:,:,l) = Dr'*H(:,:,l)*Dt;
    end
    Zbar = reshape(Z, Nr, L*Nt);

    %% RX
    W = createBeamformer(Nr, 'ps');

    for n=1:maxFrames


        N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
        [Y_proposed_hbf, W_tilde, Psi_bar, Omega] = proposed_hbf(H, N(:,1:T_prop), Psi_i(1:T_prop, 1:T_prop,:), T_prop, Nr, Mr, W);
        A = W_tilde'*Dr;
        B = zeros(L*Nt, T_prop);
        for l=1:L
          B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
        end
        tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
        tau_Z = 1/norm(pinv(A)*Y_proposed_hbf*pinv(B), 'fro')^2/2;
        eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
        rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));


        %% Channel recovery    

        %%% Proposed technique %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(n==1)
            Omega_S = ones(Nr, Nt*L);
            Zbar_proposed = pinv(A)*Y_proposed_hbf*pinv(B);
        else
            [~, indx_S] = sort(abs(vec(Zbar_proposed)), 'descend');
            Omega_S = zeros(Nr, Nt*L);
            Omega_S(indx_S(1:numOfnz)) = 1;
        end

        Yp = A*Zbar_proposed*B;
        
        K1 = zeros(Nr*T_prop);
        for i=1:Nr
            Eii = zeros(Nr);
            Eii(i,i) = 1;
            K1 = K1 + kron(diag(Omega(i, :))', Eii);
        end
        iK1 = sparse(diag(1./diag(K1+2*rho*eye(Nr*T_prop))));

        K2 = kron(B.', A);
%         R = K2'*K2;
                  
%         K3 = 0*spones(Nr*Nt*L);
%         for ii=1:size(Omega_S, 1)
%             Eii = zeros(size(Omega_S, 1));
%             Eii(ii,ii) = 1;
%             K3 = K3 + kron(diag(Omega_S(ii, :))', Eii);
%         end


        % sub 1
        Y = svt(X-1/rho*V1, tau_Y/rho);

        % sub 2
        b = (vec(V1) + rho*vec(Y) + vec(Y_proposed_hbf) + vec(V2) + rho*vec(C));

        x = iK1*b;
        X = reshape(x, Nr, T_prop);

        % sub 3
        k = (vec(X)-1/rho*vec(V2)-vec(C)-vec(gamma*Yp));


%         res = K2'*k - R*v;
%         alpha = res'*res/(res'*R*res);
%         v = v + alpha*res;
        v = K2\k;
        
        s = max(abs(real(v))-tau_Z/rho,0).*sign(real(v)) + 1j* max(abs(imag(v))-tau_Z/rho,0).*sign(imag(v));
%         s = K3*s;
        Zbar_u_proposed = reshape(s, Nr, Nt*L);
        Xs = A*(Omega_S.*Zbar_u_proposed)*B;

        % sub 4    
        C = rho/(rho+1)*(X - gamma*Yp - sqrt(1-gamma^2)*Xs - V2/rho);

        % dual update
        V1 = V1 + rho*(Y - X);
        V2 = V2 + rho*(C - X + gamma*Yp + sqrt(1-gamma^2)* Xs);

        Zbar_proposed = gamma*Zbar_proposed + sqrt(1-gamma^2)*(Omega_S.*Zbar_u_proposed);

        convergence_error(n, 1) = norm(Zbar_proposed - Zbar, 'fro')^2/norm(Zbar, 'fro')^2;
        ase(n, 1) = log2(real(det(eye(Nr) + 1/(Nt*Nr)*Zbar*Zbar'*1/(square_noise_variance+norm(Zbar-Zbar_proposed)^2/norm(Zbar)^2))));

%         %%% Genie-aided technique %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
%         numOfnz = min(Nr*Nt*L, 500);
%         Omega_S = zeros(Nr, Nt*L);
%         Omega_S(indx_S(1:numOfnz)) = 1;
%         Zbar_ga = proposed_algorithm(Y_proposed_hbf, Omega, A, B, 100, tau_Y, tau_Z, rho, 'approximate');
%         convergence_error(n, 2) = norm(Zbar_ga - Zbar, 'fro')^2/norm(Zbar, 'fro')^2;

 
        %% OMP with MMV base  %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Conventional HBF with ZC, which gathers Nr RF chains by assuming longer channel coherence time
        [Y_hbf_nr, Psi_bar] = hbf(H, N(:, 1:T_hbf), Psi_i(1:T_hbf,1:T_hbf,:), T_hbf, W);
        A = W'*Dr;
        B = zeros(L*Nt, T_hbf);
        for l=1:L
          B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
        end    
        omp_mmv_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, numOfnz);
        S_omp_mmv = omp_mmv_solver.solve(Y_hbf_nr*pinv(B));
        Zbar_ompmmv = S_omp_mmv.Z;
        convergence_error(n, 3) = norm(Zbar_ompmmv - Zbar, 'fro')^2/norm(Zbar, 'fro')^2;
        ase(n, 3) = log2(real(det(eye(Nr) + 1/(Nt*Nr)*Zbar*Zbar'*1/(square_noise_variance+norm(Zbar-Zbar_ompmmv)^2/norm(Zbar)^2))));

        %% Adaptive OMP       %%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ada_omp_rhs = vec(Y_hbf_nr);
        R = kron(B.', A);
        targetMatrix=[];
        indexSet = cell(1,numOfnz);
        for t=1:numOfnz
            [~, indexSet{t}] = max(abs(R'*res));
            targetMatrix = [ targetMatrix, R(:,indexSet{t})];
            x_ada_omp = targetMatrix\ada_omp_rhs;
            a = targetMatrix*x_ada_omp;
            res = ada_omp_rhs - a;    
        end

        % return the estimation vecotr
        z_ada_omp = zeros(Nr*Nt*L,1);
        i=1;
        for t=1:length(indexSet)
            z_ada_omp(indexSet{t}) = x_ada_omp(i);
            i= i + 1;
        end

        Zbar_ada_omp = reshape(z_ada_omp, Nr, Nt*L);

        convergence_error(n, 2) = norm(Zbar_ada_omp - Zbar, 'fro')^2/norm(Zbar, 'fro')^2;
        ase(n, 2) = log2(real(det(eye(Nr) + 1/(Nt*Nr)*Zbar*Zbar'*1/(square_noise_variance+norm(Zbar-Zbar_ada_omp)^2/norm(Zbar)^2))));

        
        %% OMP with MMV based with large Training Sequence  %%% %%%%%%%%%%%
       % Conventional HBF with ZC, which gathers Nr RF chains by assuming longer channel coherence time
        [Y_hbf_nr, Psi_bar] = hbf(H, N(:, 1:T), Psi_i(1:T,1:T,:), T, W);
        A = W'*Dr;
        B = zeros(L*Nt, T);
        for l=1:L
          B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
        end    
        omp_mmv_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, numOfnz);
        S_omp_mmv = omp_mmv_solver.solve(Y_hbf_nr*pinv(B));
        Zbar_ompmmv = S_omp_mmv.Z;
        convergence_error(n, 4) = norm(Zbar_ompmmv - Zbar, 'fro')^2/norm(Zbar, 'fro')^2;
        ase(n, 4) = log2(real(det(eye(Nr) + 1/(Nt*Nr)*Zbar*Zbar'*1/(square_noise_variance+norm(Zbar-Zbar_ompmmv)^2/norm(Zbar)^2))));
        
        H = zeros(Nr, Nt, L);
        for l=1:L
            index = 1;
            Hl = zeros(Nr, Nt);            
            for tap = 1:total_num_of_clusters
                for ray=1:total_num_of_rays
                    rayleigh_coeff(l, index) = gamma_ar*rayleigh_coeff(l, index) + sqrt(1-gamma_ar^2)*1/sqrt(2)*(randn(1)+1j*randn(1));
                    Hl = Hl + rayleigh_coeff(l, index)*Ar(:, index)*At(:, index)';
                    index = index + 1;                    
                end
                H(:,:,l) = H(:,:,l) + Hl;

            end
            H(:,:,l) = sqrt(Nr*Nt)/sqrt(total_num_of_rays*total_num_of_clusters)*H(:,:,l);
            Z(:,:,l) = Dr'*H(:,:,l)*Dt;
        end
        Zbar = reshape(Z, Nr, L*Nt);


        disp(['Frame: ', num2str(n), ', error:', num2str(convergence_error(n, :))])

    end
    mean_convergence_error(:, :, r) = convergence_error;
    mean_ase(:, :, r) = ase;
end


figure;semilogy(squeeze(mean(mean_convergence_error,3)))
legend('Proposed tracking technique', 'AdaOMP',  'OMP',  'OMP long training', 'Location', 'Best');
grid on;
xlabel('Frame index (n)')
ylabel('NMSE')

figure;semilogy(squeeze(mean(mean_ase,3)))
legend('Proposed tracking technique', 'AdaOMP',  'OMP',  'OMP long training', 'Location', 'Best');
grid on;
xlabel('Frame index (n)')
ylabel('ASE (bits/sec/Hz')