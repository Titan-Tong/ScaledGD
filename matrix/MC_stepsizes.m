clear; close all; clc;
n = 1000;
r = 10;
kappa_list = [1, 5, 10, 20];
p = 0.2;

T = 80;
eta_list = linspace(0.1, 1.2, 60);
thresh_up = 1e2;
errors_final_ScaledGD = zeros(length(kappa_list), length(eta_list));
errors_final_GD = zeros(length(kappa_list), length(eta_list));

U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(V_seed, r);
Omega_seed = rand(n, n);
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star';
    Omega = Omega_seed < p;
    Y = Omega.*X_star;
    %% Spectral initialization
    [U0, Sigma0, V0] = svds(Y/p, r);
    for i_eta = 1:length(eta_list)
        eta = eta_list(i_eta);
        %% Scaled GD
        L = U0*sqrt(Sigma0);
        R = V0*sqrt(Sigma0);
        for t = 1:T
            X = L*R';
            error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
            if error > thresh_up
                error = thresh_up;
                break;
            end
            Z = Omega.*X - Y;
            L_plus = L - eta/p*Z*R/(R'*R);
            R_plus = R - eta/p*Z'*L/(L'*L);
            L = L_plus;
            R = R_plus;
        end
        errors_final_ScaledGD(i_kappa, i_eta) = error;
        %% GD
        L = U0*sqrt(Sigma0);
        R = V0*sqrt(Sigma0);
        for t = 1:T
            X = L*R';
            error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
            if error > thresh_up 
                error = thresh_up;
                break;
            end
            Z = Omega.*X - Y;
            L_plus = L - eta/p/sigma_star(1)*Z*R;
            R_plus = R - eta/p/sigma_star(1)*Z'*L;
            L = L_plus;
            R = R_plus;
        end
        errors_final_GD(i_kappa, i_eta) = error;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_final_ScaledGD(i_kappa, :);
    semilogy(eta_list, errors, 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 12);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_final_GD(i_kappa, :);
    semilogy(eta_list, errors, 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 12);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaGD}~\\kappa=%d$', kappa);
end
xlabel('$\eta$', 'Interpreter', 'latex'); xlim([0.1,1.2]); xticks(0.1:0.1:1.2);
ylabel('Relative error'); ylim([2e-16, 2e2]);
legend(lgd, 'Location', 'southwest', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MC_stepsizes_n=%d_r=%d_p=%g', n, r, p);
