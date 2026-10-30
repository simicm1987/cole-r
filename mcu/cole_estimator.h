#pragma once

#include <math.h>
#include <float.h>

#define COLE_MAX_N 512

struct ColeResult {
    double R0;
    double Rinf;
    double tau;
    double alpha;
    double SSE;
};

static float cole_lnw[COLE_MAX_N];
static float cole_g[COLE_MAX_N];
static const float* cole_R;
static int cole_n;

static double cole_ls(double logtau, double alpha, double* R0_out, double* Rinf_out)
{
    const float a = (float)alpha;
    const float c = cosf((float)(alpha * M_PI / 2.0));
    const float alnt = (float)(alpha * logtau * M_LN10);

    double A = 0, B = 0, C = 0, D = 0, E = 0;
    for (int i = 0; i < cole_n; i++) {
        float x = expf(a * cole_lnw[i] + alnt);
        float g = (1.0f + c * x) / (1.0f + 2.0f * c * x + x * x);
        cole_g[i] = g;
        double gd = g;
        double hd = 1 - gd;
        double r = cole_R[i];
        A += gd * gd;
        B += gd * hd;
        C += hd * hd;
        D += gd * r;
        E += hd * r;
    }

    double det = (double)A * C - (double)B * B;
    double R0 = ((double)D * C - (double)B * E) / det;
    double Rinf = ((double)A * E - (double)B * D) / det;

    double SSE = 0;
    for (int i = 0; i < cole_n; i++) {
        double res = cole_R[i] - Rinf - (R0 - Rinf) * cole_g[i];
        SSE += res * res;
    }

    if (R0_out) *R0_out = R0;
    if (Rinf_out) *Rinf_out = Rinf;
    return (double)SSE;
}

static double cole_fminbnd(double alpha, double a, double b, double tolx, int maxit, double* fx_out)
{
    const double cg = 0.5 * (3.0 - sqrt(5.0));
    const double sqeps = sqrt(DBL_EPSILON);

    double x = a + cg * (b - a);
    double w = x, v = x;
    double fx = cole_ls(x, alpha, 0, 0);
    double fw = fx, fv = fx;
    double d = 0.0, e = 0.0;

    for (int it = 0; it < maxit; it++) {
        double xm = 0.5 * (a + b);
        double tol1 = sqeps * fabs(x) + tolx / 3.0;
        double tol2 = 2.0 * tol1;

        if (fabs(x - xm) <= tol2 - 0.5 * (b - a)) break;

        if (fabs(e) > tol1) {
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p; else q = -q;
            double etemp = e;
            e = d;
            if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
                e = (x >= xm) ? a - x : b - x;
                d = cg * e;
            } else {
                d = p / q;
                double u = x + d;
                if (u - a < tol2 || b - u < tol2) d = (xm >= x) ? tol1 : -tol1;
            }
        } else {
            e = (x >= xm) ? a - x : b - x;
            d = cg * e;
        }

        double u = (fabs(d) >= tol1) ? x + d : x + ((d > 0.0) ? tol1 : -tol1);
        double fu = cole_ls(u, alpha, 0, 0);

        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }

    if (fx_out) *fx_out = fx;
    return x;
}

static ColeResult cole_estimate(const float* freq, const float* Rmeas, int n)
{
    const double alpha_min = 0.1;
    const double alpha_max = 1.00;
    const double alpha_step[3] = {0.1, 0.01, 0.001};
    const double tolx = 1e-10;
    const int maxit = 500;

    cole_R = Rmeas;
    cole_n = n;

    double wmin = 2.0 * M_PI * freq[0];
    double wmax = wmin;
    for (int i = 0; i < n; i++) {
        double w = 2.0 * M_PI * freq[i];
        cole_lnw[i] = (float)log(w);
        if (w < wmin) wmin = w;
        if (w > wmax) wmax = w;
    }

    const double logtau_min = log10(1.0 / (1000.0 * wmax));
    const double logtau_max = log10(1000.0 / wmin);

    ColeResult best;
    double alpha_best = 0.0;

    for (int s = 0; s < 3; s++) {
        double lo, hi;
        int Nalpha;
        if (s == 0) {
            lo = alpha_min;
            hi = alpha_max;
            Nalpha = (int)lround((alpha_max - alpha_min) / alpha_step[s]) + 1;
        } else {
            lo = alpha_best - alpha_step[s - 1];
            hi = alpha_best + alpha_step[s - 1];
            Nalpha = (int)lround(2.0 * alpha_step[s - 1] / alpha_step[s]) + 1;
        }

        best.SSE = INFINITY;
        best.R0 = NAN;
        best.Rinf = NAN;
        best.tau = NAN;
        best.alpha = NAN;

        for (int k = 0; k < Nalpha; k++) {
            double alpha = lo + (hi - lo) * k / (Nalpha - 1);
            if (s > 0) alpha = round(alpha / alpha_step[s]) * alpha_step[s];
            if (alpha <= 0.0 || alpha > alpha_max + 1e-9) continue;

            double logtau_opt = cole_fminbnd(alpha, logtau_min, logtau_max, tolx, maxit, 0);
            double R0_k, Rinf_k;
            double SSE_k = cole_ls(logtau_opt, alpha, &R0_k, &Rinf_k);

            if (!isfinite(R0_k) || !isfinite(Rinf_k) || !isfinite(SSE_k) ||
                R0_k <= 0.0 || Rinf_k <= 0.0 || R0_k <= Rinf_k) {
                SSE_k = INFINITY;
            }

            if (SSE_k < best.SSE) {
                best.SSE = SSE_k;
                best.R0 = R0_k;
                best.Rinf = Rinf_k;
                best.tau = pow(10.0, logtau_opt);
                best.alpha = alpha;
            }
        }

        alpha_best = best.alpha;
    }

    return best;
}
