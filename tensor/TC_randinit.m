>clear; close all; clc;
n = 100;
r = 5;
kappa_list = [1, 2, 5, 10];
p = 0.1;

T = 2000;
eta = 0.3;
thresh_up = 1e3; thresh_low = 1e-14;
errors_ScaledGD = zeros(length(kappa_list), T);
errors_GD = zeros(length(kappa_list), T);

S_seed = tenzeros([r, r, r]);
for j1 = 1:r
    for j2 = 1:r
        for j3 = 1:r
            if mod(j1 + j2 + j3, r) == 0
                S_seed(j1,j2,j3) = 1/sqrt(r);
            end
        end
    end
end
U_star = cell(3, 1);
for k = 1:3
    U_seed = sign(rand(n, r) - 0.5);
    [U_star{k}, ~, ~] = svds(U_seed, r);
end
Omega = tensor(rand(n, n, n) < p);
U0_seed = cell(3, 1);
for k = 1:3
    U0_seed{k} = randn(n, r)/sqrt(n);
end
S0_seed = tensor(randn(r, r, r));
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    S_star = ttm(S_seed, diag(linspace(1, 1/kappa, r)), 1);
    svs = zeros(3, r);
    for k = 1:3
        svs(k, :) = svds(double(tenmat(S_star, k)), r);
    end
    sv_max = max(svs(:, 1));
    sv_min = min(svs(:, r));
    X_star = ttm(S_star, U_star);
    Y = Omega.*X_star;
    %% ScaledGD with random initialization
    S = S0_seed*norm(Y)/sqrt(p*r^3);
    U = U0_seed;
    U_plus = cell(3, 1);
    U_pinv = cell(3, 1);
    for t = 1:T
        X = ttm(S, U);
        error = norm(X - X_star)/norm(X_star);
        errors_ScaledGD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        Z = Omega.*X - Y;
        for k = 1:3
            Zm = double(tenmat(Z, k));
            Um = double(tenmat(ttm(S, U, -k), k, 't'));
            U_plus{k} = U{k} - eta/p*Zm*Um/(Um'*Um);
            U_pinv{k} = U{k}/(U{k}'*U{k});
        end
        S = S - eta/p * ttm(Z, U_pinv, 't');
        U = U_plus;
    end
    %% GD with random initialization
    S = S0_seed*norm(Y)/sqrt(p*r^3);
    U = U0_seed;
    U_plus = cell(3, 1);
    for t = 1:T
        X = ttm(S, U);
        error = norm(X - X_star)/norm(X_star);
        errors_GD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        Z = Omega.*X - Y;
        for k = 1:3
            Zm = double(tenmat(Z, k));
            Um = double(tenmat(ttm(S, U, -k), k, 't'));
            U_plus{k} = U{k} - eta/p/sv_max^2*Zm*Um;
        end
        S = S - eta/p * ttm(Z, U, 't');
        U = U_plus;
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
    semilogy(t_subs, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_GD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 20:20:length(errors);
    semilogy(t_subs, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{GD}~\\kappa=%d$', kappa);
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('TC_randinit_n=%d_r=%d_p=%g', n, r, p);
