clear;
clc;
close all;

% Ref. parameters of Cole model
R0_ref    = 123.60;          % [Ohm]
Rinf_ref  = 84.40;           % [Ohm]
tau_ref   = 3.6460145e-6;    % [s]
alpha_ref = 0.747;           % 0 < alpha <= 1

N = 256;
fmin = 3e3;               % [Hz]
fmax = 1e6;            % [Hz]

% ================================================================
% noise_percent represents the standard deviation of the noise
% as a percentage of the local values of R and X.
% noise_percent = 0;    -> noiseless
% noise_percent = 0.5;  -> 0.5 %
% noise_percent = 1;    -> 1 %
% noise_percent = 2;    -> 2 %
% ================================================================
noise_percent = 1;      % [%]

% ================================================================
alpha_min = 0.1;
alpha_max = 1.00;
alpha_step = [0.1, 0.01, 0.001];
% ================================================================

randn("seed", 1);

% ================================================================
f = logspace(log10(fmin), log10(fmax), N).';
omega = 2*pi*f;

% === NOISELESS AND Cp-free COLE MODEL
c_ref = cos(alpha_ref*pi/2);
s_ref = sin(alpha_ref*pi/2);
x_ref = (omega*tau_ref).^alpha_ref;
den_ref =  1 + 2*c_ref*x_ref + x_ref.^2;
R_clean = Rinf_ref + (R0_ref - Rinf_ref) .*(1 + c_ref*x_ref) ./ den_ref;
X_clean = -(R0_ref - Rinf_ref) .*(s_ref*x_ref) ./ den_ref;
Z_clean = R_clean + 1j*X_clean;

% =====REF. COLE MODEL
Zcole = Rinf_ref + (R0_ref - Rinf_ref) ./ (1 + (1j*omega*tau_ref).^alpha_ref);
Zreal_ref = real(Zcole);
Zimag_ref = imag(Zcole);

% ===HOOK EFFECT:Cp
Cp = 100e-12; % [F]
% Parallel combination Zcole || Cp
Zhook = 1 ./ (1./Zcole + 1j*omega*Cp);
R_hook = real(Zhook);
X_hook = imag(Zhook);

noise_level = noise_percent / 100;
sigma_R = noise_level .* abs(R_hook);
sigma_X = noise_level .* abs(X_hook);
noise_R = sigma_R .* randn(size(R_hook));
noise_X = sigma_X .* randn(size(X_hook));

% "Measured" values used for fitting
Rmeas = R_hook + noise_R;
Xmeas = X_hook + noise_X; % not used, except for ploting
Zmeas = Rmeas + 1j*Xmeas;

% ========Estimation setup
tau_min = 1/(1000*max(omega));
tau_max = 1000/min(omega);
logtau_min = log10(tau_min); %more stable for numerical computation
logtau_max = log10(tau_max);

options = optimset("TolX", 1e-10, "MaxIter", 500, "MaxFunEvals", 2000);

function [SSE, R0, Rinf] = cole_ls(logtau, alpha, omega, Rmeas)
    c = cos(alpha*pi/2);
    x = (omega*(10^logtau)).^alpha;
    g = (1 + c*x) ./ (1 + 2*c*x + x.^2);
    A = sum(g.^2);
    B = sum(g.*(1-g));
    C = sum((1-g).^2);
    D = sum(g.*Rmeas);
    E = sum((1-g).*Rmeas);
    R0 = (D*C - B*E) / (A*C - B^2);
    Rinf = (A*E - B*D) / (A*C - B^2);
    SSE = sum((Rmeas - Rinf - (R0 - Rinf)*g).^2);
endfunction

%Estimation start
fprintf("\n");
fprintf("Estimation started...\n\n");
t_start = tic;
for s = 1:length(alpha_step)

    if s == 1
        alpha_grid = linspace(alpha_min, alpha_max, round((alpha_max - alpha_min)/alpha_step(s)) + 1).';
    else
        alpha_grid = linspace(alpha_best - alpha_step(s-1), alpha_best + alpha_step(s-1), ...
            round(2*alpha_step(s-1)/alpha_step(s)) + 1).';
        alpha_grid = round(alpha_grid/alpha_step(s))*alpha_step(s);
        alpha_grid = alpha_grid(alpha_grid > 0 & alpha_grid <= alpha_max);
    endif

    Nalpha = length(alpha_grid);
    SSE_alpha  = Inf(Nalpha,1);
    tau_alpha  = NaN(Nalpha,1);
    R0_alpha   = NaN(Nalpha,1);
    Rinf_alpha = NaN(Nalpha,1);

    for k = 1:Nalpha
        alpha = alpha_grid(k);

        objective = @(logtau) cole_ls(logtau, alpha, omega, Rmeas);

        [logtau_opt, SSE_opt] = fminbnd(objective, logtau_min, logtau_max, options);

        tau_opt = 10.^logtau_opt;

        [SSE_k, R0_est_k, Rinf_est_k] = cole_ls(logtau_opt, alpha, omega, Rmeas);

    % rejected nonphysical solution
        if ...
            !isfinite(R0_est_k) || ...
            !isfinite(Rinf_est_k) || ...
            !isfinite(SSE_k) || ...
            R0_est_k <= 0 || ...
            Rinf_est_k <= 0 || ...
            R0_est_k <= Rinf_est_k
            SSE_k = Inf;
        endif
    %store optimal values
        SSE_alpha(k)  = SSE_k;
        tau_alpha(k)  = tau_opt;
        R0_alpha(k)   = R0_est_k;
        Rinf_alpha(k) = Rinf_est_k;

    endfor

    [~, idx_stage] = min(SSE_alpha);
    alpha_best = alpha_grid(idx_stage);

endfor
exe_time = toc(t_start);
%determination of the best alpha and other parameters calculation
[SSE_best, idx_best] = min(SSE_alpha);
alpha_est = alpha_grid(idx_best);
tau_est = tau_alpha(idx_best);
R0_est = R0_alpha(idx_best);
Rinf_est = Rinf_alpha(idx_best);

%fitted impedance
c_est = cos(alpha_est*pi/2);
x_est = (omega*tau_est).^alpha_est;
g_est =  (1 + c_est*x_est) ./ (1 + 2*c_est*x_est + x_est.^2);

Rfit =  Rinf_est + (R0_est-Rinf_est).*g_est;

%quality of the fit
residual = Rmeas - Rfit;
SSE_best = sum(residual.^2);
RMSE = sqrt(SSE_best/N);
err_R0 = 100*(R0_est-R0_ref)/R0_ref;
err_Rinf = 100*(Rinf_est-Rinf_ref)/Rinf_ref;
err_tau = 100*(tau_est-tau_ref)/tau_ref;
err_alpha = 100*(alpha_est-alpha_ref)/alpha_ref;

fprintf("========================================================\n");
fprintf("        COLE MODEL - ESTIMATED VALUES USING ONLY Re{Z}\n");
fprintf("========================================================\n");
fprintf("\nREFERENCE VALUES\n");
fprintf("R0_ref       = %.12g Ohm\n", R0_ref);
fprintf("Rinf_ref     = %.12g Ohm\n", Rinf_ref);
fprintf("tau_ref      = %.12g s\n", tau_ref);
fprintf("alpha_ref    = %.12g\n", alpha_ref);
fprintf("\nESTIMATED PARAMETERS\n");
fprintf("R0_est       = %.12g Ohm\n", R0_est);
fprintf("Rinf_est     = %.12g Ohm\n", Rinf_est);
fprintf("tau_est      = %.12g s\n", tau_est);
fprintf("alpha_est    = %.12g\n", alpha_est);
fprintf("\nREL. ERRORS\n");
fprintf("R0 error     = %.8f %%\n", err_R0);
fprintf("Rinf error   = %.8f %%\n", err_Rinf);
fprintf("tau error    = %.8f %%\n", err_tau);
fprintf("alpha error  = %.8f %%\n", err_alpha);
fprintf("\nFIT QUALITY PARAMETERS\n");
fprintf("SSE          = %.12g\n", SSE_best);
fprintf("RMSE         = %.12g Ohm\n", RMSE);
fprintf("noise        = %.6g %%\n", noise_percent);
fprintf("exe time     = %.1f ms\n", 1000*exe_time);
%real part of the impedance
figure;
semilogx(f, Rmeas, "o", "markersize", 5);
hold on;
semilogx(f, R_clean, "--",  "linewidth", 1.5);
semilogx( f, Rfit, "-","linewidth", 2);
grid on;
xlabel("f [Hz]");
ylabel("Re{Z} [Ohm]");
title("Cole model - real part of impedance");
legend("Meas.", "Ref.", "Est.", "location","northeast");

%residuals
figure;
semilogx(f, residual, "o-", "markersize", 4);
grid on;
xlabel("f [Hz]");
ylabel("R_{meas} - R_{fit} [Ohm]");
title("Residuals of estimated R");

% NYQUIST PLOT
c_est = cos(alpha_est*pi/2);
s_est = sin(alpha_est*pi/2);
x_est = (omega*tau_est).^alpha_est;
den_est = 1 + 2*c_est*x_est + x_est.^2;
Zreal_est = Rinf_est + (R0_est-Rinf_est) .*(1 + c_est*x_est) ./ den_est;
Zimag_est = -(R0_est-Rinf_est) .* (s_est*x_est) ./ den_est;
figure;
plot( Zreal_ref, -Zimag_ref,"--","linewidth", 2);
hold on;
plot(real(Zhook), -imag(Zhook),"-","linewidth", 2);
hold on;
plot(real(Zmeas), -imag(Zmeas), ".", "markersize", 9);
hold on;
plot(Zreal_est, -Zimag_est, "-", "linewidth", 2);
grid on;
xlabel("Re{Z} [Ohm]");
ylabel("-Im{Z} [Ohm]");
title("Nyquist plot");
legend("Ref. Cole model", "Cole || C_p", "Meas.", "Est.", "location", "best");

%I USE THIS TO CREATE DATASET FOR MCU-BASED ESTIMATION, TO ENSURE THE SAME DATASET IS USED ON BOTH
arduino = 0;
if arduino == 1
fprintf("const int N_DATA = %d;\n\n", length(f));
values_per_line = 15;
fprintf("const float frequency[N_DATA] = {\n  ");
for i = 1:length(f)
    if i < length(f)
        fprintf("%.9ef, ", f(i));
    else
        fprintf("%.9ef", f(i));
    endif
    if mod(i, values_per_line) == 0 && i < length(f)
        fprintf("\n  ");
    endif

endfor
fprintf("\n};\n\n");
fprintf("const float R_data[N_DATA] = {\n  ");

for i = 1:length(Rmeas)
    if i < length(Rmeas)
        fprintf("%.9ef, ", Rmeas(i));
    else
        fprintf("%.9ef", Rmeas(i));
    endif

    if mod(i, values_per_line) == 0 && i < length(Rmeas)
        fprintf("\n  ");
    endif
endfor
fprintf("\n};\n\n");
fprintf("const float X_data[N_DATA] = {\n  ");
for i = 1:length(Xmeas)
    if i < length(Xmeas)
        fprintf("%.9ef, ", Xmeas(i));
    else
        fprintf("%.9ef", Xmeas(i));
    endif

    if mod(i, values_per_line) == 0 && i < length(Xmeas)
        fprintf("\n  ");
    endif
endfor
fprintf("\n};\n\n");
endif
