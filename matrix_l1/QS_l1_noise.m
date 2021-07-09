clear; close all; clc;
n = 100;
r = 5;
kappa = 10;
m = 8*n*r;
ps = 0.1;
snr_list = [1e2, 1e3, 1e4];

T = 300;
thresh_up = 1e3; thresh_low = 1e-12;
errors_ScaledSM = zeros(length(snr_list), T);
errors_SM = zeros(length(snr_list), T);

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
noise_seed = 2*rand(m, 1) - 1;
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    sigma_star = linspace(kappa, 1, r);
    L_star = U_star*diag(sqrt(sigma_star));
    X_star = L_star*L_star';
    y_star = zeros(m, 1);
    for k = 1:m
        y_star(k) = norm(as{k}'*L_star)^2;
    end
    outlier = 10*norm(y_star, Inf)*outlier_seed.*(outlier_support_seed < ps);
    noise = outlier + norm(y_star, 1)/m/snr*noise_seed;
    y = y_star + outlier + noise;
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
        errors_ScaledSM(i_snr, t) = error;
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
        errors_SM(i_snr, t) = error;
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
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    errors = errors_ScaledSM(i_snr, :);
    t_subs = round(T/150)*i_snr:round(T/50):T;
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_snr}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledSM}~\\mathrm{SNR}=%d\\mathrm{dB}$', 20*log10(snr));
end
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    errors = errors_SM(i_snr, :);
    t_subs = round(T/150)*i_snr:round(T/50):T;
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_snr}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaSM}~\\mathrm{SNR}=%d\\mathrm{dB}$', 20*log10(snr));
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('QS_l1_noise_n=%d_r=%d_m=%d_ps=%g', n, r, m, ps);
