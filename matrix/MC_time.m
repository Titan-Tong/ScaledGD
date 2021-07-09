clear; close all; clc;
n = 1000;
r = 10;
kappa_list = [1, 5, 20];
p = 0.2;

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
Omega_seed = rand(n, n);
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star'; 
    Omega = Omega_seed < p;
    Y = Omega.*X_star;
    [omega_row, omega_col, omega_y] = find(Y);
    %% Spectral initialization
    [U0, Sigma0, V0] = svds(Y/p, r);
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
        if p < 0.1 % store Z as a sparse matrix if Omega is sparse enough
            omega_x = sum(L(omega_row, :).*R(omega_col, :), 2);
            Z = sparse(omega_row, omega_col, omega_x - omega_y, n, n);
        else
            X = L*R';
            Z = Omega.*X - Y;
        end
        L_plus = L - eta/p*Z*R/(R'*R);
        R_plus = R - eta/p*Z'*L/(L'*L);
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
        if p < 0.1 % store Z as a sparse matrix if Omega is sparse enough
            omega_x = sum(L(omega_row, :).*R(omega_col, :), 2);
            Z = sparse(omega_row, omega_col, omega_x - omega_y, n, n);
        else
            X = L*R';
            Z = Omega.*X - Y;
        end
        L_plus = L - eta/p/sigma_star(1)*Z*R;
        R_plus = R - eta/p/sigma_star(1)*Z'*L;
        L = L_plus;
        R = R_plus;
        time = time + toc;
    end
    %% AltMin
    rows_id = cell(n, 1); % observed column indices in each row
    rows_y = cell(n, 1); % observed values in each row
    for row = 1:n
        [~, rows_id{row}, rows_y{row}] = find(Y(row, :));
    end
    cols_id = cell(n, 1); % observed row indices in each column    
    cols_y = cell(n, 1); % observed values in each column
    for col = 1:n
        [~, cols_id{col}, cols_y{col}] = find(Y(:, col)');
    end
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
        for row = 1:n
            R_row = R(rows_id{row}, :);
            L(row, :) = rows_y{row}*R_row/(R_row'*R_row);
        end
        for col = 1:n
            L_col = L(cols_id{col}, :);
            R(col, :) = cols_y{col}*L_col/(L_col'*L_col);
        end
        time = time + toc;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
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
    t_subs = 2:2:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{3}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{AltMin}~\\kappa=%d$', kappa);
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MC_iter_n=%d_r=%d_p=%g', n, r, p);
%% Figure (b): relative error vs. run time
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    times = times_ScaledGD(i_kappa, :);
    errors = errors_ScaledGD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = (2*i_kappa):10:length(errors);
    semilogy(times(t_subs), errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    times = times_GD(i_kappa, :);
    errors = errors_GD(i_kappa, :);
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
    t_subs = 2:2:length(errors);
    semilogy(times(t_subs), errors(t_subs), 'Color', clrs{3}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{AltMin}~\\kappa=%d$', kappa);
end
xlabel('Run time (seconds)');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MC_time_n=%d_r=%d_p=%g', n, r, p);
