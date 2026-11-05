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
csv_file   = "cole_sweep_nlls_results.csv";
% ================================================================

% ================================================================
alpha_min = 0.1;
alpha_max = 1.00;
alpha_step = [0.1, 0.01, 0.001];
% ================================================================

% ================================================================
% NLLS on Re{Z} only (Levenberg-Marquardt), random initial values
% initial values are drawn uniformly from ranges derived from R only:
%   R0    in [max(R), 1.5*max(R)]
%   Rinf  in [0.5*min(R), min(R)]
%   log10(tau) in [logtau_min, logtau_max]  (same bounds as the proposed method)
%   alpha in [alpha_min, alpha_max]          (same range as the proposed method)
nlls_runs  = 1000;
nlls_maxit = 500;
nlls_tol   = 1e-12;
success_rel_tol = 1e-3;   % run is successful if SSE <= SSE_proposed*(1 + tol) + 1e-9
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

options = optimset("TolX", 1e-10, "MaxIter", 500, "MaxFunEvals", 2000);

% the same uniform random numbers are used for all combinations
rand("seed", 2);
U = rand(nlls_runs, 4);

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

function [R0_est, Rinf_est, tau_est, alpha_est, SSE_best] = cole_estimate(omega, Rmeas, ...
        alpha_min, alpha_max, alpha_step, logtau_min, logtau_max, options)

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

    [SSE_best, idx_best] = min(SSE_alpha);
    alpha_est = alpha_grid(idx_best);
    tau_est = tau_alpha(idx_best);
    R0_est = R0_alpha(idx_best);
    Rinf_est = Rinf_alpha(idx_best);
endfunction

% ======================== NLLS on Re{Z} ========================
% parameters p = [R0; Rinf; log10(tau); alpha]
function r = nlls_res(p, omega, Rmeas)
    c = cos(p(4)*pi/2);
    x = (omega*10^p(3)).^p(4);
    g = (1 + c*x) ./ (1 + 2*c*x + x.^2);
    r = Rmeas - p(2) - (p(1) - p(2))*g;
endfunction

function [p, SSE, nit, nfev] = nlls_lm(p0, omega, Rmeas, lb, ub, maxit, tol)
    p = min(max(p0(:), lb), ub);
    r = nlls_res(p, omega, Rmeas);
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
            J(:,j) = (nlls_res(p + dp, omega, Rmeas) - nlls_res(p - dp, omega, Rmeas)) / (2*h);
        endfor
        nfev = nfev + 8;
        A = J.'*J;
        g = J.'*r;
        improved = false;
        while !improved && lambda < 1e12
            step = -(A + lambda*diag(diag(A))) \ g;
            pn = min(max(p + step, lb), ub);
            rn = nlls_res(pn, omega, Rmeas);
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
resN = zeros(nN*nC, 7);    % NLLS-specific columns
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

        % ---------- proposed estimator ----------
        t0 = tic;
        [R0_est, Rinf_est, tau_est, alpha_est, SSE_best] = cole_estimate(omega, Rmeas, ...
            alpha_min, alpha_max, alpha_step, logtau_min, logtau_max, options);
        exe_time = toc(t0);

        %fitted impedance
        c_est = cos(alpha_est*pi/2);
        x_est = (omega*tau_est).^alpha_est;
        g_est =  (1 + c_est*x_est) ./ (1 + 2*c_est*x_est + x_est.^2);
        Rfit =  Rinf_est + (R0_est-Rinf_est).*g_est;

        %quality of the fit
        residual = Rmeas - Rfit;
        SSE_best = sum(residual.^2);
        RMSE = sqrt(SSE_best/N);
        RMSE_ref = sqrt(mean((Rfit - R_clean).^2));
        err_R0 = 100*(R0_est-R0_ref)/R0_ref;
        err_Rinf = 100*(Rinf_est-Rinf_ref)/Rinf_ref;
        err_tau = 100*(tau_est-tau_ref)/tau_ref;
        err_alpha = 100*(alpha_est-alpha_ref)/alpha_ref;

        % ---------- NLLS on Re{Z}, random initial values ----------
        Rmax = max(Rmeas);
        Rmin = min(Rmeas);
        P0 = [Rmax*(1 + 0.5*U(:,1)), ...
              Rmin*(0.5 + 0.5*U(:,2)), ...
              logtau_min + (logtau_max - logtau_min)*U(:,3), ...
              alpha_min + (alpha_max - alpha_min)*U(:,4)];

        Pn = zeros(nlls_runs, 4);
        SSEn = zeros(nlls_runs, 1);
        nitn = zeros(nlls_runs, 1);
        nfevn = zeros(nlls_runs, 1);
        t0 = tic;
        for k = 1:nlls_runs
            [Pn(k,:), SSEn(k), nitn(k), nfevn(k)] = nlls_lm(P0(k,:), omega, Rmeas, lb, ub, nlls_maxit, nlls_tol);
        endfor
        exe_time_n = toc(t0)/nlls_runs;

        physical = isfinite(SSEn) & Pn(:,1) > Pn(:,2) & Pn(:,2) > 0;
        success = physical & SSEn <= SSE_best*(1 + success_rel_tol) + 1e-9;
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

        % reported NLLS result: physical run with the smallest SSE
        if any(physical)
            idx_phys = find(physical);
            [~, ib] = min(SSEn(idx_phys));
            pb = Pn(idx_phys(ib),:);
            R0_nlls = pb(1);
            Rinf_nlls = pb(2);
            tau_nlls = 10^pb(3);
            alpha_nlls = pb(4);

            c_n = cos(alpha_nlls*pi/2);
            x_n = (omega*tau_nlls).^alpha_nlls;
            g_n = (1 + c_n*x_n) ./ (1 + 2*c_n*x_n + x_n.^2);
            Rfit_n = Rinf_nlls + (R0_nlls - Rinf_nlls).*g_n;

            SSE_nlls = sum((Rmeas - Rfit_n).^2);
            RMSE_nlls = sqrt(SSE_nlls/N);
            RMSE_ref_nlls = sqrt(mean((Rfit_n - R_clean).^2));
        else
            R0_nlls = NaN; Rinf_nlls = NaN; tau_nlls = NaN; alpha_nlls = NaN;
            SSE_nlls = NaN; RMSE_nlls = NaN; RMSE_ref_nlls = NaN;
        endif
        err_R0_n = 100*(R0_nlls - R0_ref)/R0_ref;
        err_Rinf_n = 100*(Rinf_nlls - Rinf_ref)/Rinf_ref;
        err_tau_n = 100*(tau_nlls - tau_ref)/tau_ref;
        err_alpha_n = 100*(alpha_nlls - alpha_ref)/alpha_ref;

        fprintf("========================================================\n");
        fprintf("  noise = %g %%,  Cp = %g pF\n", noise_percent, Cp*1e12);
        fprintf("========================================================\n");
        fprintf("ESTIMATED PARAMETERS (NLLS, best physical of %d runs)\n", nlls_runs);
        fprintf("R0_est       = %.12g Ohm\n", R0_nlls);
        fprintf("Rinf_est     = %.12g Ohm\n", Rinf_nlls);
        fprintf("tau_est      = %.12g s\n", tau_nlls);
        fprintf("alpha_est    = %.12g\n", alpha_nlls);
        fprintf("REL. ERRORS\n");
        fprintf("R0 error     = %.8f %%\n", err_R0_n);
        fprintf("Rinf error   = %.8f %%\n", err_Rinf_n);
        fprintf("tau error    = %.8f %%\n", err_tau_n);
        fprintf("alpha error  = %.8f %%\n", err_alpha_n);
        fprintf("FIT QUALITY PARAMETERS\n");
        fprintf("SSE          = %.12g\n", SSE_nlls);
        fprintf("RMSE         = %.12g Ohm\n", RMSE_nlls);
        fprintf("RMSE vs ref  = %.12g Ohm\n", RMSE_ref_nlls);
        fprintf("ITERATIONS\n");
        fprintf("iterations   = %.1f mean, %.1f mean of physical runs, %d max\n", mean(nitn), mean_iter_phys, max(nitn));
        fprintf("J evals      = %.0f mean per run\n", mean(nfevn));
        fprintf("exe time     = %.4f s per run\n", exe_time_n);
        fprintf("PHYSICALITY OF THE %d RUNS\n", nlls_runs);
        fprintf("physical     = %.1f %%\n", physical_rate);
        fprintf("nonphysical  = %.1f %%\n", nonphysical_rate);
        fprintf("low alpha    = %.1f %%  (nonphysical, alpha < %g)\n", low_alpha_rate, alpha_low);
        fprintf("success      = %.1f %%  (physical and SSE reached the proposed SSE)\n", success_rate);
        fprintf("Proposed SSE = %.12g, exe time = %.3f s\n\n", SSE_best, exe_time);

        row = row + 1;
        res(row,:) = [noise_percent, Cp*1e12, R0_nlls, Rinf_nlls, tau_nlls, alpha_nlls, ...
                      err_R0_n, err_Rinf_n, err_tau_n, err_alpha_n, SSE_nlls, RMSE_nlls, RMSE_ref_nlls, exe_time_n, ...
                      NaN, NaN, round(mean(nfevn))];
        resN(row,:) = [physical_rate, nonphysical_rate, low_alpha_rate, success_rate, mean(nitn), mean_iter_phys, max(nitn)];

    endfor
endfor

% ================= SUMMARY =================
fprintf("\n");
fprintf("SUMMARY: NLLS, best physical of %d runs (relative errors in %%)\n", nlls_runs);
fprintf("%6s %6s | %9s %9s %9s %9s | %9s | %7s %7s %7s %7s\n", "noise", "Cp", "R0", "Rinf", "tau", "alpha", "RMSEref", ...
        "phys%", "nonph%", "lowa%", "succ%");
for r = 1:row
    fprintf("%6g %6g | %9.3f %9.3f %9.3f %9.3f | %9.4f | %7.1f %7.1f %7.1f %7.1f\n", ...
            res(r,1), res(r,2), res(r,7:10), res(r,13), resN(r,1:4));
endfor

% ================= CSV =================
% the first 17 columns are the same as in the proposed-method CSV
fid = fopen(csv_file, "w");
fprintf(fid, "noise_percent,Cp_pF,R0_est,Rinf_est,tau_est,alpha_est,err_R0,err_Rinf,err_tau,err_alpha,SSE,RMSE,RMSE_ref,exe_time,Nalpha,Ntau_max,Nfun,");
fprintf(fid, "physical_pct,nonphysical_pct,nonphysical_low_alpha_pct,success_pct,mean_iter,mean_iter_physical,max_iter\n");
for r = 1:row
    fprintf(fid, "%g,%g,%.10g,%.10g,%.10g,%.6g,%.6f,%.6f,%.6f,%.6f,%.10g,%.10g,%.10g,%.4f,%g,%g,%d,", res(r,:));
    fprintf(fid, "%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%d\n", resN(r,:));
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
