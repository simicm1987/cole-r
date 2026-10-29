"use strict";

let coleLnw = [];
let coleG = [];
let coleR = [];
let coleN = 0;

function coleLs(logtau, alpha) {
  const c = Math.cos(alpha * Math.PI / 2);
  const alnt = alpha * logtau * Math.LN10;

  let A = 0, B = 0, C = 0, D = 0, E = 0;
  for (let i = 0; i < coleN; i++) {
    const x = Math.exp(alpha * coleLnw[i] + alnt);
    const g = (1 + c * x) / (1 + 2 * c * x + x * x);
    const h = 1 - g;
    coleG[i] = g;
    A += g * g;
    B += g * h;
    C += h * h;
    D += g * coleR[i];
    E += h * coleR[i];
  }

  const det = A * C - B * B;
  const R0 = (D * C - B * E) / det;
  const Rinf = (A * E - B * D) / det;

  let SSE = 0;
  for (let i = 0; i < coleN; i++) {
    const res = coleR[i] - Rinf - (R0 - Rinf) * coleG[i];
    SSE += res * res;
  }

  return [SSE, R0, Rinf];
}

function coleFminbnd(alpha, a, b, tolx, maxit) {
  const cg = 0.5 * (3 - Math.sqrt(5));
  const sqeps = Math.sqrt(Number.EPSILON);

  let x = a + cg * (b - a);
  let w = x, v = x;
  let fx = coleLs(x, alpha)[0];
  let fw = fx, fv = fx;
  let d = 0, e = 0;

  for (let it = 0; it < maxit; it++) {
    const xm = 0.5 * (a + b);
    const tol1 = sqeps * Math.abs(x) + tolx / 3;
    const tol2 = 2 * tol1;

    if (Math.abs(x - xm) <= tol2 - 0.5 * (b - a)) break;

    if (Math.abs(e) > tol1) {
      const r = (x - w) * (fx - fv);
      let q = (x - v) * (fx - fw);
      let p = (x - v) * q - (x - w) * r;
      q = 2 * (q - r);
      if (q > 0) p = -p;
      else q = -q;
      const etemp = e;
      e = d;
      if (Math.abs(p) >= Math.abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
        e = (x >= xm) ? a - x : b - x;
        d = cg * e;
      } else {
        d = p / q;
        const u = x + d;
        if (u - a < tol2 || b - u < tol2) d = (xm >= x) ? tol1 : -tol1;
      }
    } else {
      e = (x >= xm) ? a - x : b - x;
      d = cg * e;
    }

    const u = (Math.abs(d) >= tol1) ? x + d : x + ((d > 0) ? tol1 : -tol1);
    const fu = coleLs(u, alpha)[0];

    if (fu <= fx) {
      if (u >= x) a = x;
      else b = x;
      v = w; fv = fw;
      w = x; fw = fx;
      x = u; fx = fu;
    } else {
      if (u < x) a = u;
      else b = u;
      if (fu <= fw || w === x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v === x || v === w) {
        v = u; fv = fu;
      }
    }
  }

  return x;
}

function coleEstimate(freqHz, Rmeas) {
  const alphaMin = 0.1;
  const alphaMax = 1.0;
  const alphaStep = [0.1, 0.01, 0.001];
  const tolx = 1e-10;
  const maxit = 500;

  coleN = freqHz.length;
  coleR = Rmeas;
  coleLnw = new Array(coleN);
  coleG = new Array(coleN);

  let wmin = Infinity, wmax = 0;
  for (let i = 0; i < coleN; i++) {
    const w = 2 * Math.PI * freqHz[i];
    coleLnw[i] = Math.log(w);
    wmin = Math.min(wmin, w);
    wmax = Math.max(wmax, w);
  }

  const logtauMin = Math.log10(1 / (1000 * wmax));
  const logtauMax = Math.log10(1000 / wmin);

  let best = null;
  let alphaBest = 0;

  for (let s = 0; s < alphaStep.length; s++) {
    let lo, hi, Nalpha;
    if (s === 0) {
      lo = alphaMin;
      hi = alphaMax;
      Nalpha = Math.round((alphaMax - alphaMin) / alphaStep[s]) + 1;
    } else {
      lo = alphaBest - alphaStep[s - 1];
      hi = alphaBest + alphaStep[s - 1];
      Nalpha = Math.round(2 * alphaStep[s - 1] / alphaStep[s]) + 1;
    }

    best = { R0: NaN, Rinf: NaN, tau: NaN, alpha: NaN, SSE: Infinity };

    for (let k = 0; k < Nalpha; k++) {
      let alpha = lo + (hi - lo) * k / (Nalpha - 1);
      if (s > 0) alpha = Math.round(alpha / alphaStep[s]) * alphaStep[s];
      if (!(alpha > 0) || alpha > alphaMax + 1e-9) continue;

      const logtauOpt = coleFminbnd(alpha, logtauMin, logtauMax, tolx, maxit);
      let [SSEk, R0k, Rinfk] = coleLs(logtauOpt, alpha);

      if (!Number.isFinite(R0k) || !Number.isFinite(Rinfk) || !Number.isFinite(SSEk) ||
          R0k <= 0 || Rinfk <= 0 || R0k <= Rinfk) {
        SSEk = Infinity;
      }

      if (SSEk < best.SSE) {
        best = { R0: R0k, Rinf: Rinfk, tau: Math.pow(10, logtauOpt), alpha: alpha, SSE: SSEk };
      }
    }

    alphaBest = best.alpha;
  }

  return best;
}

function coleRe(w, p) {
  const c = Math.cos(p.alpha * Math.PI / 2);
  const x = Math.pow(w * p.tau, p.alpha);
  return p.Rinf + (p.R0 - p.Rinf) * (1 + c * x) / (1 + 2 * c * x + x * x);
}

function coleAbsIm(w, p) {
  const c = Math.cos(p.alpha * Math.PI / 2);
  const s = Math.sin(p.alpha * Math.PI / 2);
  const x = Math.pow(w * p.tau, p.alpha);
  return (p.R0 - p.Rinf) * s * x / (1 + 2 * c * x + x * x);
}

/* ===================== State ===================== */

const W = 1200, H = 680;

const COL = {
  bg: "rgb(250,250,250)",
  panel: "rgb(255,255,255)",
  panelEdge: "rgb(215,215,215)",
  grid: "rgb(225,225,225)",
  gridMajor: "rgb(200,200,200)",
  gridMinor: "rgb(238,238,238)",
  axis: "rgb(90,90,90)",
  text: "rgb(30,30,30)",
  label: "rgb(110,110,110)",
  hint: "rgb(140,140,140)",
  meas: "rgb(40,110,200)",
  fit: "rgb(220,90,30)",
  dev: "rgb(60,60,60)",
  devLine: "rgb(170,170,170)"
};

const FONT_NORMAL = "14px sans-serif";
const FONT_BOLD = "bold 16px sans-serif";
const FONT_SMALL = "11px sans-serif";

let rawRows = null;
let fileName = "";
let status = "Press \"Open CSV\" to load a file (f, R, |X|)";
let data = null;

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const fileInput = document.getElementById("fileInput");
const openBtn = document.getElementById("openBtn");
const prescalerSel = document.getElementById("prescaler");

/* ===================== Loading ===================== */

openBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileName = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    parseCsv(String(reader.result));
    runEstimation();
  };
  reader.onerror = () => {
    data = null;
    status = "Cannot read " + fileName;
    render();
  };
  reader.readAsText(file);
  fileInput.value = "";
});

prescalerSel.addEventListener("change", () => {
  if (rawRows) runEstimation();
});

document.addEventListener("keydown", (ev) => {
  if ((ev.key === "o" || ev.key === "O") && !ev.ctrlKey && !ev.metaKey && !ev.altKey) fileInput.click();
});

window.addEventListener("resize", render);

function parseCsv(text) {
  rawRows = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const s = line.trim();
    if (s.length === 0) continue;
    let p;
    if (s.indexOf(";") >= 0) p = s.replace(/,/g, ".").split(/[;\t ]+/);
    else p = s.split(/[,\t ]+/);
    p = p.filter((t) => t.length > 0);
    if (p.length < 3) continue;
    const f = Number(p[0]), r = Number(p[1]), x = Number(p[2]);
    if (!Number.isFinite(f) || !Number.isFinite(r) || !Number.isFinite(x)) continue;
    rawRows.push([f, r, Math.abs(x)]);
  }
}

function runEstimation() {
  data = null;
  if (!rawRows || rawRows.length < 5) {
    status = "Not enough valid rows in " + fileName;
    render();
    return;
  }

  const prescaler = Number(prescalerSel.value);
  const N = rawRows.length;
  const fHz = rawRows.map((r) => r[0] * prescaler);
  const Rmeas = rawRows.map((r) => r[1]);
  const Xmeas = rawRows.map((r) => r[2]);

  const t0 = performance.now();
  const est = coleEstimate(fHz, Rmeas);
  const t1 = performance.now();

  if (!Number.isFinite(est.R0)) {
    status = "Estimation failed for " + fileName;
    render();
    return;
  }

  const relDev = new Array(N);
  let meanDev = 0, meanAbsDev = 0;
  for (let i = 0; i < N; i++) {
    const Rfit = coleRe(2 * Math.PI * fHz[i], est);
    relDev[i] = 100 * (Rfit - Rmeas[i]) / Rmeas[i];
    meanDev += relDev[i];
    meanAbsDev += Math.abs(relDev[i]);
  }
  meanDev /= N;
  meanAbsDev /= N;

  const M = 400;
  const fmin = Math.min(...fHz), fmax = Math.max(...fHz);
  const curveR = new Array(M), curveX = new Array(M);
  for (let i = 0; i < M; i++) {
    const lf = Math.log10(fmin) + (Math.log10(fmax) - Math.log10(fmin)) * i / (M - 1);
    const w = 2 * Math.PI * Math.pow(10, lf);
    curveR[i] = coleRe(w, est);
    curveX[i] = coleAbsIm(w, est);
  }

  data = {
    N, fHz, Rmeas, Xmeas, est,
    RMSE: Math.sqrt(est.SSE / N),
    exeMs: t1 - t0,
    relDev, meanDev, meanAbsDev, curveR, curveX
  };
  render();
}

/* ===================== Drawing helpers ===================== */

function setupCanvas() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = canvas.clientWidth || W;
  const scale = (cssW / W) * dpr;
  canvas.width = Math.round(W * scale);
  canvas.height = Math.round(H * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function line(x1, y1, x2, y2, color, lw = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function strokeRect(x, y, w, h, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);
}

function roundRectPath(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function text(str, x, y, font, color, align, baseline) {
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(str, x, y);
}

function textRotated(str, x, y, font, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-Math.PI / 2);
  text(str, 0, 0, font, color, "center", "middle");
  ctx.restore();
}

function circle(x, y, d, strokeColor, fillColor, lw = 1) {
  ctx.beginPath();
  ctx.arc(x, y, d / 2, 0, 2 * Math.PI);
  if (fillColor) {
    ctx.fillStyle = fillColor;
    ctx.fill();
  }
  if (strokeColor) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lw;
    ctx.stroke();
  }
}

function niceStep(range, nTicks) {
  const rough = range / nTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  let st;
  if (norm < 1.5) st = 1;
  else if (norm < 3) st = 2;
  else if (norm < 7) st = 5;
  else st = 10;
  return st * mag;
}

function fmt(v, step) {
  const dec = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  if (Math.abs(v) < step * 1e-6) v = 0;
  return v.toFixed(dec);
}

function freqLabel(f) {
  if (f >= 1e6) return (f / 1e6).toFixed(0) + "M";
  if (f >= 1e3) return (f / 1e3).toFixed(0) + "k";
  return f.toFixed(0);
}

function minOf(a) {
  let m = Infinity;
  for (const v of a) m = Math.min(m, v);
  return m;
}

function maxOf(a) {
  let m = -Infinity;
  for (const v of a) m = Math.max(m, v);
  return m;
}

function render() {
  setupCanvas();

  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  const half = W / 2;
  drawPanelFrame(10, 10, half - 15, H - 20);
  drawPanelFrame(half + 5, 10, half - 15, H - 20);

  if (!data) {
    text(status, W / 2, H / 2, FONT_NORMAL, COL.text, "center", "middle");
    return;
  }

  drawNyquist(10, 10, half - 15, H - 20);
  drawResults(half + 5, 10, half - 15, H - 20);
}

function drawPanelFrame(x, y, w, h) {
  roundRectPath(x, y, w, h, 6);
  ctx.fillStyle = COL.panel;
  ctx.fill();
  ctx.strokeStyle = COL.panelEdge;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawNyquist(px, py, pw, ph) {
  const d = data;
  const left = px + 75, right = px + pw - 25, top = py + 55, bottom = py + ph - 65;
  const plotW = right - left, plotH = bottom - top;

  let xmin = Math.min(minOf(d.Rmeas), minOf(d.curveR));
  let xmax = Math.max(maxOf(d.Rmeas), maxOf(d.curveR));
  let ymax = Math.max(maxOf(d.Xmeas), maxOf(d.curveX));
  const padX = 0.05 * (xmax - xmin);
  xmin -= padX;
  xmax += padX;
  ymax *= 1.10;
  const ymin = 0;

  const sc = Math.min(plotW / (xmax - xmin), plotH / (ymax - ymin));
  const xr = plotW / sc, yr = plotH / sc;
  const xc = 0.5 * (xmin + xmax);
  xmin = xc - 0.5 * xr;
  xmax = xc + 0.5 * xr;
  ymax = ymin + yr;

  const step = niceStep(Math.max(xr, yr), 8);

  for (let v = Math.ceil(xmin / step) * step; v <= xmax + 1e-9; v += step) {
    const sx = left + (v - xmin) * sc;
    line(sx, top, sx, bottom, COL.grid);
    text(fmt(v, step), sx, bottom + 6, FONT_SMALL, COL.text, "center", "top");
  }
  for (let v = Math.ceil(ymin / step) * step; v <= ymax + 1e-9; v += step) {
    const sy = bottom - (v - ymin) * sc;
    line(left, sy, right, sy, COL.grid);
    text(fmt(v, step), left - 6, sy, FONT_SMALL, COL.text, "right", "middle");
  }

  strokeRect(left, top, plotW, plotH, COL.axis);

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, plotW, plotH);
  ctx.clip();

  for (let i = 0; i < d.N; i++) {
    const sx = left + (d.Rmeas[i] - xmin) * sc;
    const sy = bottom - (d.Xmeas[i] - ymin) * sc;
    circle(sx, sy, 5, COL.meas, null, 1.2);
  }

  ctx.strokeStyle = COL.fit;
  ctx.lineWidth = 2.2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 0; i < d.curveR.length; i++) {
    const sx = left + (d.curveR[i] - xmin) * sc;
    const sy = bottom - (d.curveX[i] - ymin) * sc;
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();

  text("Nyquist plot", px + 15, py + 15, FONT_BOLD, COL.text, "left", "top");
  text("Re{Z} [Ω]", (left + right) / 2, bottom + 28, FONT_NORMAL, COL.text, "center", "top");
  textRotated("|Im{Z}| [Ω]", px + 22, (top + bottom) / 2, FONT_NORMAL, COL.text);

  const lx = right - 170, ly = top + 12;
  roundRectPath(lx - 10, ly - 8, 175, 50, 4);
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fill();
  circle(lx + 10, ly + 6, 6, COL.meas, null, 1.2);
  line(lx, ly + 28, lx + 20, ly + 28, COL.fit, 2.2);
  text("Measured", lx + 30, ly + 6, FONT_SMALL, COL.text, "left", "middle");
  text("Estimated Cole model", lx + 30, ly + 28, FONT_SMALL, COL.text, "left", "middle");
}

function drawResults(px, py, pw, ph) {
  const d = data;
  const e = d.est;

  text("Estimated Cole parameters", px + 15, py + 15, FONT_BOLD, COL.text, "left", "top");

  const rows = [
    ["R0", e.R0.toFixed(4) + " Ω"],
    ["R∞", e.Rinf.toFixed(4) + " Ω"],
    ["τ", (e.tau * 1e6).toFixed(5) + " µs"],
    ["fc = 1/(2πτ)", (1e-3 / (2 * Math.PI * e.tau)).toFixed(3) + " kHz"],
    ["α", e.alpha.toFixed(3)],
    ["SSE", Number(e.SSE.toPrecision(6)).toString() + " Ω²"],
    ["RMSE", d.RMSE.toFixed(5) + " Ω"],
    ["N points", String(d.N)],
    ["Estimation time", d.exeMs.toFixed(1) + " ms"],
    ["File", fileName]
  ];

  const ty = py + 50;
  for (let i = 0; i < rows.length; i++) {
    const yy = ty + i * 22;
    text(rows[i][0], px + 25, yy, FONT_NORMAL, COL.label, "left", "top");
    text(rows[i][1], px + 190, yy, FONT_NORMAL, COL.text, "left", "top");
  }

  const left = px + 75, right = px + pw - 25, top = py + 330, bottom = py + ph - 65;
  const plotW = right - left, plotH = bottom - top;

  const fmin = minOf(d.fHz), fmax = maxOf(d.fHz);
  const lx0 = Math.log10(fmin) - 0.03, lx1 = Math.log10(fmax) + 0.03;
  let dmax = 0;
  for (let i = 0; i < d.N; i++) dmax = Math.max(dmax, Math.abs(d.relDev[i]));
  dmax = Math.max(dmax * 1.15, 1e-6);
  const ystep = niceStep(2 * dmax, 6);
  dmax = Math.ceil(dmax / ystep) * ystep;

  const mapX = (f) => left + (Math.log10(f) - lx0) / (lx1 - lx0) * plotW;
  const mapY = (v) => bottom - (v + dmax) / (2 * dmax) * plotH;

  for (let v = -dmax; v <= dmax + 1e-12; v += ystep) {
    const sy = mapY(v);
    line(left, sy, right, sy, COL.grid);
    text(fmt(v, ystep), left - 6, sy, FONT_SMALL, COL.text, "right", "middle");
  }
  for (let dec = Math.floor(lx0); dec <= Math.ceil(lx1); dec++) {
    for (let m = 1; m <= 9; m++) {
      const lv = dec + Math.log10(m);
      if (lv < lx0 || lv > lx1) continue;
      const sx = left + (lv - lx0) / (lx1 - lx0) * plotW;
      line(sx, top, sx, bottom, m === 1 ? COL.gridMajor : COL.gridMinor);
      if (m === 1) text(freqLabel(Math.pow(10, dec)), sx, bottom + 6, FONT_SMALL, COL.text, "center", "top");
    }
  }

  line(left, mapY(0), right, mapY(0), COL.axis);

  ctx.save();
  ctx.setLineDash([5, 5]);
  line(left, mapY(d.meanDev), right, mapY(d.meanDev), COL.fit, 1.5);
  ctx.restore();

  ctx.strokeStyle = COL.devLine;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < d.N; i++) {
    const sx = mapX(d.fHz[i]), sy = mapY(d.relDev[i]);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  for (let i = 0; i < d.N; i++) circle(mapX(d.fHz[i]), mapY(d.relDev[i]), 3.5, null, COL.dev);

  strokeRect(left, top, plotW, plotH, COL.axis);

  text("Relative residuals of Re{Z}: 100*(fit - meas)/meas", px + 15, top - 30, FONT_BOLD, COL.text, "left", "bottom");
  text("Mean = " + d.meanDev.toFixed(4) + " %", px + 15, top - 8, FONT_NORMAL, COL.fit, "left", "bottom");
  text("Mean |dev| = " + d.meanAbsDev.toFixed(4) + " %", px + 200, top - 8, FONT_NORMAL, COL.text, "left", "bottom");

  text("f [Hz]", (left + right) / 2, bottom + 28, FONT_NORMAL, COL.text, "center", "top");
  textRotated("Rel. deviation [%]", px + 22, (top + bottom) / 2, FONT_NORMAL, COL.text);
}

render();
