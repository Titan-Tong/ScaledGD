clear; close all; clc;
n = 100;
r = 5;
kappa = 10;
m = 8*n*r;
ps_list = [0.1, 0.2, 0.3];

T = 1000;
thresh_up = 1e3; thresh_low = 1e-12;
errors_ScaledSM = zeros(length(ps_list), T);
errors_SM = zeros(length(ps_list), T);

U_seed = sign(rand(n, r) - 0.5);
[U_star, ~, ~] = svds(U_seed, r);
as = cell(m, 1);
As = cell(m, 1);
for k = 1:m
	as{k} = randn(n, 1);
	As{k} = as{k}*as{k}';
end
outlier_seed = 2*rand(m, 1) - 1;
outlier_support_seed = rand(m, 1);
for i_ps = 1:length(ps_list)
    ps = ps_list(i_ps);
    sigma_star = linspace(kappa, 1, r);
    L_star = U_star*diag(sqrt(sigma_star));
    X_star = L_star*L_star';
    y_star = zeros(m, 1);
    for k = 1:m
        y_star(k) = norm(as{k}'*L_star)^2;
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
    Y = (Y - sum(y(loc_y))*eye(n))/length(loc_y)/2;
    [U0, Sigma0] = eigs(Y, r);
    assert(all(diag(Sigma0) > 0));
    %% ScaledSM
    L = U0*sqrt(Sigma0);
    for t = 1:T
        X = L*L';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_ScaledSM(i_ps, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        loss = 0;
        Z = zeros(n, n);
        for k = 1:m
            z = norm(as{k}'*L)^2 - y(k);
            loss = loss + abs(z);
            Z = Z + 2*sign(z)*As{k};
        end
        ZL = Z*L; ZLpinv = ZL/(L'*L);
        eta = (loss - loss_star)/(ZL(:)'*ZLpinv(:));
        L = L - eta*ZLpinv;
    end
    %% Subgradient method
    L = U0*sqrt(Sigma0);
    for t = 1:T
        X = L*L';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_SM(i_ps, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        loss = 0;
        Z = zeros(n, n);
        for k = 1:m
            z = norm(as{k}'*L)^2 - y(k);
            loss = loss + abs(z);
            Z = Z + 2*sign(z)*As{k};
        end
        ZL = Z*L;
        eta = (loss - loss_star)/(ZL(:)'*ZL(:));
        L = L - eta*ZL;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_ps = 1:length(ps_list)
    ps = ps_list(i_ps);
    errors = errors_ScaledSM(i_ps, :);
    errors = errors(errors > thresh_low);
    t_subs = (2*i_ps):10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_ps}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledSM}~p_s=%g$', ps);
end
for i_ps = 1:length(ps_list)
    ps = ps_list(i_ps);
    errors = errors_SM(i_ps, :);
    errors = errors(errors > thresh_low);
    t_subs = 10:10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_ps}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaSM}~p_s=%g$', ps);
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('QS_l1_outlier_n=%d_r=%d_m=%d', n, r, m);
