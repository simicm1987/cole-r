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
noise_list = [0, 0.5, 1, 2];                       % [%]
Cp_list    = [0, 10e-12, 50e-12, 100e-12, 200e-12]; % [F]
csv_file   = "cole_sweep_cnls_results.csv";
% ================================================================

% ================================================================
alpha_min = 0.1;
alpha_max = 1.00;
% ================================================================

% ================================================================
% CNLS on Re{Z} and Im{Z} (Levenberg-Marquardt), random initial values
% initial values are drawn uniformly from ranges derived from R only:
%   R0    in [max(R), 1.5*max(R)]
%   Rinf  in [0.5*min(R), min(R)]
%   log10(tau) in [logtau_min, logtau_max]  (same bounds as the proposed method)
%   alpha in [alpha_min, alpha_max]          (same range as the proposed method)
cnls_runs  = 1000;
cnls_maxit = 500;
cnls_tol   = 1e-12;
success_rel_tol = 1e-3;   % run is successful if SSE_complex <= best_SSE_complex*(1 + tol) + 1e-9
alpha_low = 0.05;         % nonphysical runs ending below this alpha are counted separately
% ================================================================

% ================================================================
f = logspace(log10(fmin), log10(fmax), N).';
omega = 2*pi*f;

% === NOISELESS AND Cp-free COLE MODEL
c_ref = cos(alpha_ref*pi/2);
x_ref = (omega*tau_ref).^alpha_ref;
den_ref =  1 + 2*c_ref*x_ref + x_ref.^2;
R_clean = Rinf_ref + (R0_ref - Rinf_ref) .*(1 + c_ref*x_ref) ./ den_ref;

% =====REF. COLE MODEL
Zcole = Rinf_ref + (R0_ref - Rinf_ref) ./ (1 + (1j*omega*tau_ref).^alpha_ref);

% ========Estimation setup
tau_min = 1/(1000*max(omega));
tau_max = 1000/min(omega);
logtau_min = log10(tau_min); %more stable for numerical computation
logtau_max = log10(tau_max);

% the same uniform random numbers are used for all combinations
rand("seed", 2);
U = rand(cnls_runs, 4);

% ======================== CNLS on Re{Z} and Im{Z} ========================
% parameters p = [R0; Rinf; log10(tau); alpha]
function r = cnls_res(p, omega, Rmeas, Xmeas)
    c = cos(p(4)*pi/2);
    s = sin(p(4)*pi/2);
    x = (omega*10^p(3)).^p(4);
    den = 1 + 2*c*x + x.^2;
    R = p(2) + (p(1) - p(2))*(1 + c*x) ./ den;
    X = -(p(1) - p(2))*(s*x) ./ den;
    r = [Rmeas - R; Xmeas - X];
endfunction

function [p, SSE, nit, nfev] = cnls_lm(p0, omega, Rmeas, Xmeas, lb, ub, maxit, tol)
    p = min(max(p0(:), lb), ub);
    r = cnls_res(p, omega, Rmeas, Xmeas);
    SSE = r.'*r;
    nfev = 1;
    lambda = 1e-3;
    nit = 0;
    ws = warning();
    warning("off", "Octave:singular-matrix");
    warning("off", "Octave:nearly-singular-matrix");
    for it = 1:maxit
        nit = it;
        J = zeros(length(r), 4);
        for j = 1:4
            h = 1e-6*max(1, abs(p(j)));
            dp = zeros(4,1);
            dp(j) = h;
            J(:,j) = (cnls_res(p + dp, omega, Rmeas, Xmeas) - cnls_res(p - dp, omega, Rmeas, Xmeas)) / (2*h);
        endfor
        nfev = nfev + 8;
        A = J.'*J;
        g = J.'*r;
        improved = false;
        while !improved && lambda < 1e12
            step = -(A + lambda*diag(diag(A))) \ g;
            pn = min(max(p + step, lb), ub);
            rn = cnls_res(pn, omega, Rmeas, Xmeas);
            SSEn = rn.'*rn;
            nfev = nfev + 1;
            if isfinite(SSEn) && SSEn < SSE
                dSSE = SSE - SSEn;
                p = pn;
                r = rn;
                SSE = SSEn;
                lambda = max(lambda/10, 1e-15);
                improved = true;
            else
                lambda = lambda*10;
            endif
        endwhile
        if !improved || norm(step) <= tol*(norm(p) + tol) || dSSE <= tol*max(SSE, eps)
            break;
        endif
    endfor
    warning(ws);
endfunction


%Estimation start
fprintf("\n");
fprintf("Estimation started...\n\n");

nN = length(noise_list);
nC = length(Cp_list);
res = zeros(nN*nC, 17);    % same columns as the proposed-method CSV
resN = zeros(nN*nC, 9);    % CNLS-specific columns
row = 0;

lb = [-Inf; -Inf; logtau_min; 1e-3];
ub = [ Inf;  Inf; logtau_max; 1];

for in = 1:nN
    for ic = 1:nC

        noise_percent = noise_list(in);
        Cp = Cp_list(ic);

        randn("seed", 1);

        % ===HOOK EFFECT:Cp
        Zhook = 1 ./ (1./Zcole + 1j*omega*Cp);
        R_hook = real(Zhook);
        X_hook = imag(Zhook);

        noise_level = noise_percent / 100;
        sigma_R = noise_level .* abs(R_hook);
        sigma_X = noise_level .* abs(X_hook);
        noise_R = sigma_R .* randn(size(R_hook));
        noise_X = sigma_X .* randn(size(X_hook));

        Rmeas = R_hook + noise_R;
        Xmeas = X_hook + noise_X;

        % ---------- CNLS on Re{Z} and Im{Z}, random initial values ----------
        Rmax = max(Rmeas);
        Rmin = min(Rmeas);
        P0 = [Rmax*(1 + 0.5*U(:,1)), ...
              Rmin*(0.5 + 0.5*U(:,2)), ...
              logtau_min + (logtau_max - logtau_min)*U(:,3), ...
              alpha_min + (alpha_max - alpha_min)*U(:,4)];

        Pn = zeros(cnls_runs, 4);
        SSEn = zeros(cnls_runs, 1);
        nitn = zeros(cnls_runs, 1);
        nfevn = zeros(cnls_runs, 1);
        t0 = tic;
        for k = 1:cnls_runs
            [Pn(k,:), SSEn(k), nitn(k), nfevn(k)] = cnls_lm(P0(k,:), omega, Rmeas, Xmeas, lb, ub, cnls_maxit, cnls_tol);
        endfor
        exe_time_n = toc(t0)/cnls_runs;

        physical = isfinite(SSEn) & Pn(:,1) > Pn(:,2) & Pn(:,2) > 0;
        if any(physical)
            SSE_cmin = min(SSEn(physical));
        else
            SSE_cmin = NaN;
        endif
        success = physical & SSEn <= SSE_cmin*(1 + success_rel_tol) + 1e-9;
        success_rate = 100*mean(success);
        physical_rate = 100*mean(physical);
        nonphysical_rate = 100 - physical_rate;
        % nonphysical runs that ended at a very small alpha
        low_alpha_rate = 100*mean(!physical & Pn(:,4) < alpha_low);
        % failed runs usually stop at the iteration limit, so they raise the overall mean
        if any(physical)
            mean_iter_phys = mean(nitn(physical));
        else
            mean_iter_phys = NaN;
        endif

        % reported CNLS result: physical run with the smallest complex SSE
        if any(physical)
            idx_phys = find(physical);
            [~, ib] = min(SSEn(idx_phys));
            pb = Pn(idx_phys(ib),:);
            R0_cnls = pb(1);
            Rinf_cnls = pb(2);
            tau_cnls = 10^pb(3);
            alpha_cnls = pb(4);

            c_n = cos(alpha_cnls*pi/2);
            x_n = (omega*tau_cnls).^alpha_cnls;
            g_n = (1 + c_n*x_n) ./ (1 + 2*c_n*x_n + x_n.^2);
            Rfit_n = Rinf_cnls + (R0_cnls - Rinf_cnls).*g_n;

            % SSE and RMSE on R only, comparable with the proposed method and NLLS
            SSE_cnls = sum((Rmeas - Rfit_n).^2);
            RMSE_cnls = sqrt(SSE_cnls/N);
            RMSE_ref_cnls = sqrt(mean((Rfit_n - R_clean).^2));
        else
            R0_cnls = NaN; Rinf_cnls = NaN; tau_cnls = NaN; alpha_cnls = NaN;
            SSE_cnls = NaN; RMSE_cnls = NaN; RMSE_ref_cnls = NaN;
        endif
        err_R0_n = 100*(R0_cnls - R0_ref)/R0_ref;
        err_Rinf_n = 100*(Rinf_cnls - Rinf_ref)/Rinf_ref;
        err_tau_n = 100*(tau_cnls - tau_ref)/tau_ref;
        err_alpha_n = 100*(alpha_cnls - alpha_ref)/alpha_ref;

        fprintf("========================================================\n");
        fprintf("  noise = %g %%,  Cp = %g pF\n", noise_percent, Cp*1e12);
        fprintf("========================================================\n");
        fprintf("ESTIMATED PARAMETERS (CNLS, best physical of %d runs)\n", cnls_runs);
        fprintf("R0_est       = %.12g Ohm\n", R0_cnls);
        fprintf("Rinf_est     = %.12g Ohm\n", Rinf_cnls);
        fprintf("tau_est      = %.12g s\n", tau_cnls);
        fprintf("alpha_est    = %.12g\n", alpha_cnls);
        fprintf("REL. ERRORS\n");
        fprintf("R0 error     = %.8f %%\n", err_R0_n);
        fprintf("Rinf error   = %.8f %%\n", err_Rinf_n);
        fprintf("tau error    = %.8f %%\n", err_tau_n);
        fprintf("alpha error  = %.8f %%\n", err_alpha_n);
        fprintf("FIT QUALITY PARAMETERS\n");
        fprintf("SSE          = %.12g\n", SSE_cnls);
        fprintf("RMSE         = %.12g Ohm\n", RMSE_cnls);
        fprintf("RMSE vs ref  = %.12g Ohm\n", RMSE_ref_cnls);
        fprintf("SSE complex  = %.12g\n", SSE_cmin);
        fprintf("RMSE complex = %.12g Ohm\n", sqrt(SSE_cmin/N));
        fprintf("ITERATIONS\n");
        fprintf("iterations   = %.1f mean, %.1f mean of physical runs, %d max\n", mean(nitn), mean_iter_phys, max(nitn));
        fprintf("J evals      = %.0f mean per run\n", mean(nfevn));
        fprintf("exe time     = %.4f s per run\n", exe_time_n);
        fprintf("PHYSICALITY OF THE %d RUNS\n", cnls_runs);
        fprintf("physical     = %.1f %%\n", physical_rate);
        fprintf("nonphysical  = %.1f %%\n", nonphysical_rate);
        fprintf("low alpha    = %.1f %%  (nonphysical, alpha < %g)\n", low_alpha_rate, alpha_low);
        fprintf("success      = %.1f %%  (physical and within tolerance of the best complex SSE)\n", success_rate);
        fprintf("\n");

        row = row + 1;
        res(row,:) = [noise_percent, Cp*1e12, R0_cnls, Rinf_cnls, tau_cnls, alpha_cnls, ...
                      err_R0_n, err_Rinf_n, err_tau_n, err_alpha_n, SSE_cnls, RMSE_cnls, RMSE_ref_cnls, exe_time_n, ...
                      NaN, NaN, round(mean(nfevn))];
        resN(row,:) = [SSE_cmin, sqrt(SSE_cmin/N), physical_rate, nonphysical_rate, low_alpha_rate, success_rate, mean(nitn), mean_iter_phys, max(nitn)];

    endfor
endfor

% ================= SUMMARY =================
fprintf("\n");
fprintf("SUMMARY: CNLS, best physical of %d runs (relative errors in %%)\n", cnls_runs);
fprintf("%6s %6s | %9s %9s %9s %9s | %9s | %7s %7s %7s %7s\n", "noise", "Cp", "R0", "Rinf", "tau", "alpha", "RMSEref", ...
        "phys%", "nonph%", "lowa%", "succ%");
for r = 1:row
    fprintf("%6g %6g | %9.3f %9.3f %9.3f %9.3f | %9.4f | %7.1f %7.1f %7.1f %7.1f\n", ...
            res(r,1), res(r,2), res(r,7:10), res(r,13), resN(r,3:6));
endfor

% ================= CSV =================
% the first 17 columns are the same as in the proposed-method CSV
fid = fopen(csv_file, "w");
fprintf(fid, "noise_percent,Cp_pF,R0_est,Rinf_est,tau_est,alpha_est,err_R0,err_Rinf,err_tau,err_alpha,SSE,RMSE,RMSE_ref,exe_time,Nalpha,Ntau_max,Nfun,");
fprintf(fid, "SSE_complex,RMSE_complex,physical_pct,nonphysical_pct,nonphysical_low_alpha_pct,success_pct,mean_iter,mean_iter_physical,max_iter\n");
for r = 1:row
    fprintf(fid, "%g,%g,%.10g,%.10g,%.10g,%.6g,%.6f,%.6f,%.6f,%.6f,%.10g,%.10g,%.10g,%.4f,%g,%g,%d,", res(r,:));
    fprintf(fid, "%.10g,%.10g,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%d\n", resN(r,:));
endfor
fclose(fid);
fprintf("\nResults saved to %s\n", csv_file);

% ================= LATEX TABLE ROWS =================
fprintf("\nLATEX TABLE ROWS\n");
for r = 1:row
    if r > 1 && res(r,1) != res(r-1,1)
        fprintf("\\midrule\n");
    endif
    e = round(res(r,7:10)*100)/100 + 0;
    rm = round(res(r,13)*1000)/1000 + 0;
    fprintf("%g & %g & %.2f & %.2f & %.2f & %.2f & %.3f \\\\\n", res(r,1), res(r,2), e, rm);
endfor
