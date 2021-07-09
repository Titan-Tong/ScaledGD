clear; close all; clc;
n = 200;
r = 10;
kappa_list = [1, 5, 20];
m = 5*n*r;

T = 1000;
eta = 0.5;
thresh_up = 1e3; thresh_low = 1e-14;
times_ScaledGD = zeros(length(kappa_list), T);
errors_ScaledGD = zeros(length(kappa_list), T);
times_GD = zeros(length(kappa_list), T);
errors_GD = zeros(length(kappa_list), T);
times_ALS = zeros(length(kappa_list), T);
errors_ALS = zeros(length(kappa_list), T);

U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(V_seed, r);
As = cell(m, 1);
for k = 1:m
    As{k} = randn(n, n)/sqrt(m);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star';
    y = zeros(m, 1);
    for k = 1:m
        y(k) = As{k}(:)'*X_star(:);
    end
    %% Spectral initialization
    Y = zeros(n, n);
    for k = 1:m
        Y = Y + y(k)*As{k};
    end
    [U0, Sigma0, V0] = svds(Y, r);
    %% Scaled GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    time = 0;
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_ScaledGD(i_kappa, t) = error;
        times_ScaledGD(i_kappa, t) = time;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        tic;
        X = L*R';
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end
        L_plus = L - eta*Z*R/(R'*R);
        R_plus = R - eta*Z'*L/(L'*L);
        L = L_plus;
        R = R_plus;
        time = time + toc;
    end
    %% GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    time = 0;
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_GD(i_kappa, t) = error;
        times_GD(i_kappa, t) = time;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        tic;
        X = L*R';
        Z = zeros(n, n);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end
        L_plus = L - eta/sigma_star(1)*Z*R;
        R_plus = R - eta/sigma_star(1)*Z'*L;
        L = L_plus;
        R = R_plus;
        time = time + toc;
    end
    %% AltMin
    AL = zeros(n*r, m);
    AR = zeros(n*r, m); 
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    time = 0;
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro'); 
        errors_ALS(i_kappa, t) = error;
        times_ALS(i_kappa, t) = time;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        tic;
        for k = 1:m
            AR(:, k) = reshape(As{k}*R, [n*r, 1]);
        end
        L = reshape(AR'\y, [n, r]);
        for k = 1:m
            AL(:, k) = reshape(As{k}'*L, [n*r, 1]);
        end
        R = reshape(AL'\y, [n, r]);
        time = time + toc;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 's', 'd'};
%% Figure (a): relative error vs. iteration count
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
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_ALS(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 1:2:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{3}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{AltMin}~\\kappa=%d$', kappa);
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MS_iter_n=%d_r=%d_m=%d', n, r, m);
%% Figure (b): relative error vs. run time
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_ScaledGD(i_kappa, :);
    times = times_ScaledGD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = (2*i_kappa):10:length(errors);
    semilogy(times(t_subs), errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_GD(i_kappa, :);
    times = times_GD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 10:10:length(errors);
    semilogy(times(t_subs), errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    times = times_ALS(i_kappa, :);
    errors = errors_ALS(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 1:2:length(errors);
    semilogy(times(t_subs), errors(t_subs), 'Color', clrs{3}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{AltMin}~\\kappa=%d$', kappa);
end
xlabel('Run time (seconds)');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MS_time_n=%d_r=%d_m=%d', n, r, m);
