clear; close all; clc;
n1 = 1000; n2=1000; nd=n1+n2-1;
r = 10;
kappa_list = [1, 5, 10, 20];
p = 0.2;

T = 1000;
eta = 0.5;
thresh_up = 1e3; thresh_low = 1e-12;
errors_ScaledGD = zeros(length(kappa_list), T);
errors_GD = zeros(length(kappa_list), T);

freq_seed = randperm(n1, r)/n1;
omega_seed = rand(nd, 1);
w = zeros(nd, 1);
for k = 1:nd
    w(k) = min([k, n1+n2-k, n1, n2]); %length of skew-diagonals
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);
    x_star = exp(2*pi*1i * (0:(nd-1))' * freq_seed) * sigma_star' / sqrt(n1*n2);
    omega = omega_seed < p;
    y = omega.*x_star;
    %% Spectral initialization via Lanczos algorithm
    [U0, Sigma0, V0] = svds(@(v,tflag)HankelVecMul(y/p, v, tflag, nd, n1, n2), [n1, n2], r);
    %% Scaled GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        x = HankelProj(L, R, nd, w);
        error = norm((x - x_star).*sqrt(w))/norm(x_star.*sqrt(w));
        errors_ScaledGD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        z = (omega.*x-y)/p - x;
        [Lz, Rz] = HankelMul(z, L, R, nd, n1, n2); 
        L_plus = L - eta*(L + Lz/(R'*R));
        R_plus = R - eta*(R + Rz/(L'*L));
        L = L_plus;
        R = R_plus;
    end
    %% GD
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    for t = 1:T
        x = HankelProj(L, R, nd, w);
        error = norm((x - x_star).*sqrt(w))/norm(x_star.*sqrt(w));
        errors_GD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end        
        z = (omega.*x-y)/p - x;
        [Lz, Rz] = HankelMul(z, L, R, nd, n1, n2);
        L_plus = L - eta/sigma_star(1)*(L*(R'*R) + Lz);
        R_plus = R - eta/sigma_star(1)*(R*(L'*L) + Rz);
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
fig_name = sprintf('HankelMC_n=%d_r=%d_p=%g', n1, r, p);


function x = HankelProj(L, R, nd, w)
% Hankel projection H[x]=H[LR']
    n = 2^nextpow2(nd);
    x = sum(ifft(fft(L, n) .* fft(conj(R), n)), 2);
    x = x(1:nd)./w;
end

function [Lx, Rx] = HankelMul(x, L, R, nd, n1, n2)
% Hankel multiplication Lx=H[x]R, Rx=H[x]'L
    n = 2^nextpow2(nd);
    Lx = ifft(bsxfun(@times, fft(flip(R), n), fft(x, n)));
    Lx = Lx(n2:nd, :);  
    Rx = ifft(bsxfun(@times, fft(flip(L), n), fft(conj(x), n)));
    Rx = Rx(n1:nd, :);
end

function z = HankelVecMul(x, v, tflag, nd, n1, n2)
% Hankel multiplication z=H[x]v if tflag='notransp'; z=H[x]'v if tflag='transp'
    n = 2^nextpow2(nd);
    if strcmp(tflag, 'notransp')
        z = ifft(fft(flip(v), n) .* fft(x, n));
        z = z(n2:nd);
    else
        z = ifft(fft(flip(v), n) .* fft(conj(x), n));
        z = z(n1:nd);
    end
end