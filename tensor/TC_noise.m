clear; close all; clc;
n = 100;
r = 5;
snr_list = [1e2, 1e3, 1e4];
p = 0.1;

T = 300;
eta = 0.3;
thresh_up = 1e3; thresh_low = 1e-14;
errors_ScaledGD = zeros(length(snr_list), T);
errors_GD = zeros(length(snr_list), T);

U_star = cell(3, 1);
for k = 1:3
    U_seed = sign(rand(n, r) - 0.5);
    [U_star{k}, ~, ~] = svds(U_seed, r);
end
S_star = tensor(randn(r, r, r));
svs = zeros(3, r);
for k = 1:3
    svs(k, :) = svds(double(tenmat(S_star, k)), r);
end
sv_max = max(svs(:, 1));
sv_min = min(svs(:, r));
X_star = ttm(S_star, U_star);
Omega = tensor(rand(n, n, n) < p);
Noise_seed = tensor(randn(n, n, n));
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    Y = Omega.*(X_star + norm(X_star)/n^1.5/snr*Noise_seed);
    %% Spectral initialization
    U0 = cell(3, 1);
    for k = 1:3
        Ym = double(tenmat(Y, k));
        G = Ym*Ym'/p^2;
        [U0{k}, ~] = eigs(G - diag(diag(G)), r);
    end
    S0 = ttm(Y/p, U0, 't');
    %% ScaledGD
    U = U0;
    S = S0;
    U_plus = cell(3, 1);
    U_pinv = cell(3, 1);
    for t = 1:T
        X = ttm(S, U);
        error = norm(X - X_star)/norm(X_star);
        errors_ScaledGD(i_snr, t) = error;
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
        S = S - eta/p*ttm(Z, U_pinv, 't');
        U = U_plus;
    end
    %% GD
    U = U0;
    S = S0;
    U_plus = cell(3, 1);
    for t = 1:T
        X = ttm(S, U);
        error = norm(X - X_star)/norm(X_star);
        errors_GD(i_snr, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        Z = Omega.*X - Y;
        for k = 1:3
            Zm = double(tenmat(Z, k));
            Um = double(tenmat(ttm(S, U, -k), k, 't'));
            U_plus{k} = U{k} - eta/p/sv_max^2*Zm*Um;
        end
        S = S - eta/p*ttm(Z, U, 't');
        U = U_plus;
    end
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    errors = errors_ScaledGD(i_snr, :);
    t_subs = round(T/150)*i_snr:round(T/50):T;
    semilogy(t_subs, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_snr}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\mathrm{SNR}=%d\\mathrm{dB}$', 20*log10(snr));
end
for i_snr = 1:length(snr_list)
    snr = snr_list(i_snr);
    errors = errors_GD(i_snr, :);
    t_subs = round(T/150)*i_snr:round(T/50):T;
    semilogy(t_subs, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_snr}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{GD}~\\mathrm{SNR}=%d\\mathrm{dB}$', 20*log10(snr));
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'Northeast', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('TC_noise_n=%d_r=%d_p=%g',n,r,p);
