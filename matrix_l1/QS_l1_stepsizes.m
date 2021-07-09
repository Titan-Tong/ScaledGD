clear; close all; clc;
n = 100;
r = 5;
kappa = 10;
m = 8*n*r;
ps = 0.2;
q_list = 0.8:0.01:0.97;
lmda_list = [1, 2, 5, 10, 20, 50];

T = 1000;
thresh_up = 1e3; thresh_low = 1e-12;
errors_ScaledSM = zeros(1 + length(q_list)*length(lmda_list), T);

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
%% ScaledSM with Polyak stepsizes
L = U0*sqrt(Sigma0);
lmdas_Polyak = zeros(1, T);
for t = 1:T
    X = L*L';
    error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
    errors_ScaledSM(1, t) = error;
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
    Znorm = sqrt(ZL(:)'*ZLpinv(:));
    lmda = (loss - loss_star)/Znorm;
    lmdas_Polyak(t) = lmda;
    L = L - lmda/Znorm*ZLpinv;
end
%% ScaledSM with geometrically decaying stepsizes
parfor i = 2:(1 + length(lmda_list)*length(q_list))
    i_lmda = floor((i - 2)/length(q_list)) + 1;
    i_q = mod(i - 2, length(q_list)) + 1;
    lmda = lmda_list(i_lmda);
    q = q_list(i_q);
    errors = zeros(1, T);
    L = U0*sqrt(Sigma0);
    for t = 1:T
        X = L*L';
        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors(t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end
        Z = zeros(n, n);
        for k = 1:m
            z = norm(as{k}'*L)^2 - y(k);
            Z = Z + 2*sign(z)*As{k};
        end
        ZL = Z*L; ZLpinv = ZL/(L'*L);
        Znorm = sqrt(ZL(:)'*ZLpinv(:));
        L = L - lmda/Znorm*ZLpinv;
        lmda = lmda*q;
    end
    errors_ScaledSM(i, :) = errors;
end

clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1], [0,.5,.5]};
mks = {'o', 'x', 'p', 's', 'd', '>'};
%% Figure (a): final distances
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lgd = {};
for i_lmda = 1:length(lmda_list)
    lmda = lmda_list(i_lmda);
    errors_final = zeros(1, length(q_list));
    for i_q = 1:length(q_list)
        errors = errors_ScaledSM((i_lmda - 1)*length(q_list) + i_q + 1, :);
        errors_final(i_q) = max(errors(end), thresh_low);
    end
    semilogy(q_list, errors_final, 'Color', clrs{i_lmda}, 'Marker', mks{i_lmda}, 'MarkerSize', 12);
    hold on; grid on;
    lgd{i_lmda} = sprintf('$\\lambda=%g$', lmda);
end
xlabel('q'); xlim([0.8,0.97]);
ylabel('Final relative error');  ylim([1e-12, 110]); yticks(10.^(-12:2:2));
legend(lgd, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('QS_l1_final_n=%d_r=%d_m=%d_ps=%g', n, r, m, ps);
%% Figure (b): convergence under various lambda
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lmda_sublist = [1, 2, 5, 10, 20]; 
q = 0.92;
lgd = {};
for j = 1:(1 + length(lmda_sublist))
    if j == 1
        errors = errors_ScaledSM(1, :);
        lgd{j} = 'Polyak''s stepsizes';
    else
        lmda = lmda_sublist(j - 1);
        [~, i_lmda] = min(abs(lmda_list - lmda));
        [~, i_q] = min(abs(q_list - q));
        errors = errors_ScaledSM((i_lmda - 1)*length(q_list) + i_q + 1, :);
        lgd{j} = sprintf('$\\lambda=%g$', lmda);
    end
    errors = errors(errors > thresh_low);
    t_subs = (3*j):20:length(errors);
    semilogy(t_subs, errors(t_subs), 'Color', clrs{j}, 'Marker', mks{j}, 'MarkerSize', 10);
    hold on; grid on;
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('QS_l1_lambda_n=%d_r=%d_m=%d_ps=%g_q=%g', n, r, m, ps, q);
%% Figure (c): convergence under various q
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lmda = 5;
q_sublist = [0.88, 0.9, 0.91, 0.92, 0.95];
lgd = {};
for j = 1:(1 + length(q_sublist))
    if j == 1
        errors = errors_ScaledSM(1, :);
        lgd{j} = 'Polyak''s stepsizes';
    else
        q = q_sublist(j - 1);
        [~, i_lmda] = min(abs(lmda_list - lmda));
        [~, i_q] = min(abs(q_list - q));
        errors = errors_ScaledSM((i_lmda - 1)*length(q_list) + i_q + 1, :);
        lgd{j} = sprintf('$q=%g$', q);
    end    
    errors = errors(errors > thresh_low);
    t_subs = (3*j):20:length(errors);
    semilogy(t_subs, errors(t_subs), 'Color', clrs{j}, 'Marker', mks{j}, 'MarkerSize', 10);
    hold on; grid on;
end
xlabel('Iteration count');
ylabel('Relative error');
legend(lgd, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('QS_l1_q_n=%d_r=%d_m=%d_ps=%g_lambda=%g', n, r, m, ps, lmda);
%% Figure (d): stepsizes
figure('Position', [0,0,800,600], 'DefaultAxesFontSize', 20);
lmdas_Polyak = lmdas_Polyak(lmdas_Polyak > 0);
mdl = fitlm(0:(length(lmdas_Polyak) - 1), log(lmdas_Polyak));
lmda_Polyak = exp(mdl.Coefficients{1, 1});
q_Polyak = exp(mdl.Coefficients{2, 1});
t_subs = 1:10:length(lmdas_Polyak);
semilogy(t_subs-1, lmdas_Polyak(t_subs), 'Color', clrs{1}, 'Marker', mks{1}, 'MarkerSize', 12);
hold on; grid on;
semilogy(t_subs-1, lmda_Polyak*q_Polyak.^(t_subs-1), 'Color', clrs{2}, 'Marker', mks{2}, 'MarkerSize', 12);
xlabel('Iteration count');
ylabel('Stepsizes');
legend({'Polyak''s stepsizes', 'Closest geometrically decaying stepsizes'},  'Location', 'best',  'Interpreter', 'latex', 'FontSize', 24);
fig_name = sprintf('QS_l1_stepsizes_n=%d_r=%d_m=%d_ps=%g', n, r, m, ps);
