clear; close all; clc;
n = 100;
r = 10;
kappa_list = [1, 5, 10, 20];
m = 8*n*r;
ps = 0.2;

T = 1000;
thresh_up = 1e3; thresh_low = 1e-12;
errors_ScaledSM = zeros(length(kappa_list), T);
errors_SM = zeros(length(kappa_list), T);

U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
V_seed = sign(rand(n, r) - 0.5);
[V_star, ~, ~] = svds(U_seed, r);
As = cell(m, 1);
for k = 1:m
	As{k} = randn(n, n);
end
outlier_seed = 2*rand(m, 1) - 1;
outlier_support_seed = rand(m, 1);
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(kappa, 1, r);
    L_star = U_star*diag(sqrt(sigma_star));
    R_star = V_star*diag(sqrt(sigma_star));
    X_star = L_star*R_star';
    y_star = zeros(m, 1);
    for k = 1:m
        y_star(k) = As{k}(:)'*X_star(:);
    end
    outlier = 10*norm(y_star, Inf)*outlier_seed.*(outlier_support_seed < ps);
    y = y_star + outlier;
    loss_star = norm(y_star - y, 1);
    %% Spectral initialization
    Y = zeros(n, n);
    [~, loc_y] = mink(abs(y), ceil(m*(1-ps)));
    for k = loc_y'
        Y = Y + y(k)*As{k};
    end
    Y = Y/length(loc_y);
    [U0, Sigma0, V0] = svds(Y, r);
    %% ScaledSM
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_ScaledSM(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        loss = 0;
        Z = zeros(n, n);
        for k = 1:m
            z = As{k}(:)'*X(:) - y(k);
            loss = loss + abs(z);
            Z = Z + sign(z)*As{k};
        end
        ZL = Z'*L; ZLpinv = ZL/(L'*L);
        ZR = Z*R; ZRpinv = ZR/(R'*R);
        eta = (loss - loss_star)/(ZL(:)'*ZLpinv(:) + ZR(:)'*ZRpinv(:));
        L = L - eta*ZRpinv;
        R = R - eta*ZLpinv;
    end
    %% Subgradient method
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        X = L*R';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_SM(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        loss = 0;
        Z = zeros(n, n);
        for k = 1:m
            z = As{k}(:)'*X(:) - y(k);
            loss = loss + abs(z);
            Z = Z + sign(z)*As{k};
        end
        ZL = Z'*L;
        ZR = Z*R;
        eta = (loss - loss_star)/(ZL(:)'*ZL(:) + ZR(:)'*ZR(:));
        L = L - eta*ZR;
        R = R - eta*ZL;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_ScaledSM(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = (2*i_kappa):10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledSM}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_SM(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 10:10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaSM}~\\kappa=%d$', kappa);
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('MS_l1_n=%d_r=%d_m=%d_ps=%g', n, r, m, ps);
