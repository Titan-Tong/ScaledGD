clear; close all; clc;
n = 1000;
r = 10;
kappa_list = [1, 5, 10, 20];
alpha = 0.1;

T = 1000;
eta = 0.5;
thresh_up = 1e3; thresh_low = 1e-14;
errors_ScaledGD = zeros(length(kappa_list), T);
errors_GD = zeros(length(kappa_list), T);

U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(U_seed, r);
S_star = Tproj(randn(n, n), alpha, n, n);
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star';
    Y = X_star + S_star;
    %% Spectral initialization
    [U0, Sigma0, V0] = svds(Y - Tproj(Y, alpha, n, n), r);
    %% Scaled GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_ScaledGD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        S = Tproj(Y - X, 2*alpha, n, n);
        L_plus = L - eta*(X + S - Y)*R/(R'*R);
        R_plus = R - eta*(X + S - Y)'*L/(L'*L);
        L = L_plus;
        R = R_plus;
    end
    %% GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_GD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        S = Tproj(Y - X, 2*alpha, n, n);
        L_plus = L - eta/sigma_star(1)*(X + S - Y)*R;
        R_plus = R - eta/sigma_star(1)*(X + S - Y)'*L;
        L = L_plus;
        R = R_plus;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_ScaledGD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = (2*i_kappa):10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_GD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 10:10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaGD}~\\kappa=%d$', kappa);
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('RPCA_n=%d_r=%d_alpha=%g', n, r, alpha);


function S = Tproj(Z, alpha, n1, n2)
% Keep alpha fraction of largest entries in each row and column of Z
    krow = floor(alpha*n1);
    kcol = floor(alpha*n2);
    [~, loc_row] = maxk(abs(Z'), krow, 1);
    [~, loc_col] = maxk(abs(Z), kcol, 1);
    id_row = repmat(1:n1, krow, 1);
    id_col = repmat(1:n2, kcol, 1);
    mask_row = sparse(id_row(:), loc_row(:), 1, n1, n2);
    mask_col = sparse(loc_col(:), id_col(:), 1, n1, n2);
    S = Z.*mask_row.*mask_col;
end