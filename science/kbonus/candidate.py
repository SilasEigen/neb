#!/usr/bin/env python3
"""Candidate KIC 6691030: transit shape fit, stellar density from the transit, phase-curve check, and the vetting
sheet (one PNG for astronomers).

    python candidate.py      -> results/candidate-kic-006691030.png and .json

Shape: trapezoid (depth, total duration T14, ingress tau, centre) averaged over the 29.4-min Kepler cadence, fitted
on the folded, detrended light curve. Density: Seager and Mallen-Ornelas (2003) from depth, T14, flat time and P.
Phase curve: least-squares sinusoids at P and P/2 out of transit (an eclipsing binary shows ellipsoidal light at P/2).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hunt as H  # noqa: E402
import kb_common as K  # noqa: E402
import vet  # noqa: E402

SRC = "kic-006691030"
TIC = {"id": 63005412, "teff": 3468, "e_teff": 158, "rad": 0.47355, "e_rad": 0.0291551, "mass": 0.471316, "rho": 4.43828,
       "e_rho": 0.509661, "dist": 498.753, "gaia": "2101746884862893952", "G": 17.62}  # TIC v8 (MAST), retrieved 2026-11-08
CAD = 0.0204340  # days
SURFACE, INK, INK2, BLUE, ORANGE, RAW = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#c9c8c2"


def trapezoid(x, depth, t14, tau):
    """x in days from mid-transit; returns the relative flux change (negative in transit)"""
    a = np.abs(x)
    flat = t14 / 2 - tau
    m = np.where(a <= flat, -depth, np.where(a < t14 / 2, -depth * (t14 / 2 - a) / tau, 0.0))
    return m


def model(x, depth, t14, tau):
    sub = np.linspace(-CAD / 2, CAD / 2, 9)
    return np.mean([trapezoid(x + s, depth, t14, tau) for s in sub], axis=0)


def fit_shape(x, f):
    """grid search on T14, tau/T14, centre; depth by linear least squares"""
    best = None
    m = np.abs(x) < 0.25
    x, f = x[m], f[m]
    for t14 in np.arange(0.9, 2.6, 0.05) / 24:
        for frac in np.arange(0.04, 0.51, 0.02):
            tau = frac * t14
            for c in np.arange(-0.3, 0.31, 0.03) / 24:
                shape = model(x - c, 1.0, t14, tau)
                denom = (shape * shape).sum()
                if denom <= 0:
                    continue
                depth = (shape * f).sum() / denom
                chi = ((f - depth * shape) ** 2).sum()
                if best is None or chi < best[0]:
                    best = (chi, depth, t14, tau, c)
    return best


def density(P, depth, t14, tau):
    """Seager and Mallen-Ornelas 2003, circular orbit; returns (b, a/R*, rho* in solar units)"""
    tf = t14 - 2 * tau
    k = np.sqrt(depth)
    s_t, s_f = np.sin(np.pi * t14 / P) ** 2, np.sin(np.pi * tf / P) ** 2
    r = s_f / s_t
    b2 = ((1 - k) ** 2 - r * (1 + k) ** 2) / (1 - r)
    b2 = max(b2, 0.0)
    a_r = np.sqrt(((1 + k) ** 2 - b2 * (1 - s_t)) / s_t)
    G, day = 6.674e-11, 86400.0
    rho = 3 * np.pi / (G * (P * day) ** 2) * a_r ** 3 / 1408.0  # kg/m3 -> solar (1408 kg/m3)
    return np.sqrt(b2), a_r, rho


def phase_curve(t, f, P, t0, dur):
    x = np.mod(t - t0, P) / P
    out = np.abs(np.mod(t - t0 + P / 2, P) - P / 2) > 2 * dur
    A = np.column_stack([np.ones(out.sum())] + [fn(2 * np.pi * k * x[out]) for k in (1, 2) for fn in (np.sin, np.cos)])
    coef, *_ = np.linalg.lstsq(A, f[out], rcond=None)
    resid = f[out] - A @ coef
    err = np.sqrt(np.diag(np.linalg.inv(A.T @ A))) * resid.std()
    amp = lambda i: (np.hypot(coef[i], coef[i + 1]), np.hypot(err[i], err[i + 1]) / np.sqrt(2))
    return amp(1), amp(3)


def shape_test(x, f, P, best_chi, rho_target):
    """best chi2 among shapes whose transit density is within 30 % of rho_target, and the noise per point"""
    m = np.abs(x) < 0.25
    xs, fs = x[m], f[m]
    sigma2 = best_chi / xs.size
    best = None
    for t14 in np.arange(0.9, 2.6, 0.05) / 24:
        for frac in np.arange(0.04, 0.51, 0.02):
            tau = frac * t14
            shape = model(xs, 1.0, t14, tau)
            denom = (shape * shape).sum()
            depth = (shape * fs).sum() / denom
            if depth <= 0:
                continue
            _, _, rho = density(P, depth, t14, tau)
            if abs(rho / rho_target - 1) > 0.3:
                continue
            chi = ((fs - depth * shape) ** 2).sum()
            if best is None or chi < best[0]:
                best = (chi, depth, t14, tau, rho)
    return best, (best[0] - best_chi) / sigma2 if best else None


def phase_null(t, f, P, t0, dur, trials=300, seed=1):
    """amplitudes at P and P/2 for random periods near P: the empirical noise of the phase-curve test"""
    rng = np.random.default_rng(seed)
    a1s, a2s = [], []
    for p in rng.uniform(0.8 * P, 1.2 * P, trials):
        if abs(p / P - 1) < 0.02:
            continue
        (a1, _), (a2, _) = phase_curve(t, f, p, t0, dur)
        a1s.append(a1)
        a2s.append(a2)
    return np.array(a1s), np.array(a2s)


def main():
    row = next(r for r in csv.DictReader(open(H.OUT)) if r["source"] == SRC)
    P, t0, dur = float(row["real_period"]), float(row["real_t0"]), float(row["real_dur_h"]) / 24
    t, f = vet.clean(*K.read_lc(os.path.join(K.LC_DIR, K.lc_name(SRC))))
    # refine the epoch and period on the full baseline: centre of the shape fit
    x = np.mod(t - t0 + P / 2, P) - P / 2
    chi, depth, t14, tau, c = fit_shape(x, f)
    t0 += c
    x = np.mod(t - t0 + P / 2, P) - P / 2
    b, a_r, rho = density(P, depth, t14, tau)
    (a1, e1), (a2, e2) = phase_curve(t, f, P, t0, t14)
    alt, dchi = shape_test(x - c, f, P, chi, TIC["rho"])
    if alt:
        print("best shape with the TIC density (+-30 %%): depth %.0f ppm, T14 %.2f h, ingress %.2f h, rho %.1f; "
              "delta chi2 = %.1f against the free fit" % (1e6 * alt[1], 24 * alt[2], 24 * alt[3], alt[4], dchi))
    n1, n2 = phase_null(t, f, P, t0, t14)
    extra = {"dchi2_tic_density": dchi, "phase_P_frac_random_above": float((n1 >= a1).mean()),
             "phase_P2_frac_random_above": float((n2 >= a2).mean())}
    print("phase curve: at P %.0f ppm (random periods: median %.0f, 95th pct %.0f, fraction above %.2f); at P/2 %.0f ppm "
          "(median %.0f, 95th pct %.0f, fraction above %.2f)" % (1e6 * a1, 1e6 * np.median(n1), 1e6 * np.percentile(n1, 95),
          (n1 >= a1).mean(), 1e6 * a2, 1e6 * np.median(n2), 1e6 * np.percentile(n2, 95), (n2 >= a2).mean()))
    rp = np.sqrt(depth) * TIC["rad"] * 109.08
    e_rp = rp * np.hypot(TIC["e_rad"] / TIC["rad"], 0.5 * 0.25)  # 25 % depth spread between quarters
    a_au = (TIC["mass"] * (P / 365.25) ** 2) ** (1 / 3)
    teq = TIC["teff"] * np.sqrt(TIC["rad"] * 0.00465 / (2 * a_au))
    res = {"period_d": P, "t0_bkjd": t0, "depth_ppm": 1e6 * depth, "t14_h": 24 * t14, "tau_h": 24 * tau,
           "ingress_fraction": tau / t14, "b": b, "a_over_rstar": a_r, "rho_transit_solar": rho, "rho_tic_solar": TIC["rho"],
           "rp_earth": rp, "e_rp_earth": e_rp, "a_au": a_au, "teq_k": teq, "phase_P_ppm": [1e6 * a1, 1e6 * e1],
           "phase_P2_ppm": [1e6 * a2, 1e6 * e2], "mes": float(row["real_mes"]), "n_transits": int(row["real_n_transits"]),
           "odd_even_sigma": float(row["real_odd_even_sigma"]), "secondary_sigma": float(row["real_secondary_sigma"]),
           "model_score": int(row["real_score"])} | extra
    for k, v in res.items():
        print("%-20s %s" % (k, v))
    json.dump(res, open(os.path.join(K.RESULTS, "candidate-%s.json" % SRC), "w"), indent=1, default=float)
    draw(t, f, res)


def draw(t, f, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P, t0, depth, t14, tau = res["period_d"], res["t0_bkjd"], res["depth_ppm"] * 1e-6, res["t14_h"] / 24, res["tau_h"] / 24
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#d9d8d3"})
    fig = plt.figure(figsize=(12, 10), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.98, top=0.9, bottom=0.05, wspace=0.28, hspace=0.5, height_ratios=[1, 1, 0.62])

    def style(ax, title):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#ecebe6", lw=0.7)
        ax.set_axisbelow(True)

    hours = (np.mod(t - t0 + P / 2, P) - P / 2) * 24
    ph = np.mod(t - t0, P) / P

    def binned(xv, yv, lo, hi, n):
        e = np.linspace(lo, hi, n + 1)
        k = np.digitize(xv, e) - 1
        mid = (e[1:] + e[:-1]) / 2
        med = np.array([np.median(yv[k == i]) if (k == i).sum() > 5 else np.nan for i in range(n)])
        return mid, med

    ax = fig.add_subplot(gs[0, 0:2])
    style(ax, "Folded transit (all quarters) with the cadence-averaged trapezoid fit")
    m = np.abs(hours) < 6
    ax.scatter(hours[m], 1e6 * f[m], s=1.5, color=RAW, linewidths=0, rasterized=True)
    mid, med = binned(hours[m], 1e6 * f[m], -6, 6, 72)
    ax.plot(mid, med, "o", ms=3.5, color=BLUE)
    xx = np.linspace(-6, 6, 600)
    ax.plot(xx, 1e6 * model(xx / 24, depth, t14, tau), color=ORANGE, lw=1.6)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2.2 * res["depth_ppm"], 1.2 * res["depth_ppm"])
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("ppm")

    ax = fig.add_subplot(gs[0, 2])
    style(ax, "Odd and even transits")
    n = np.rint((t - t0) / P).astype(int)
    for parity, col, lab in ((1, BLUE, "odd"), (0, ORANGE, "even")):
        mm = m & (n % 2 == parity)
        mid, med = binned(hours[mm], 1e6 * f[mm], -4, 4, 32)
        ax.plot(mid, med, "o-", ms=3, lw=1.2, color=col, label=lab)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("hours from mid-transit")

    ax = fig.add_subplot(gs[1, 0])
    style(ax, "Phase 0.5 (secondary eclipse window)")
    hs = (np.mod(t - t0, P) - P / 2) * 24
    ms = np.abs(hs) < 6
    mid, med = binned(hs[ms], 1e6 * f[ms], -6, 6, 48)
    ax.plot(mid, med, "o", ms=3.5, color=BLUE)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-2.2 * res["depth_ppm"], 1.2 * res["depth_ppm"])
    ax.set_xlabel("hours from phase 0.5")
    ax.set_ylabel("ppm")

    ax = fig.add_subplot(gs[1, 1])
    style(ax, "Full phase, out of transit (binned)")
    out = np.abs(hours) > 2 * t14 * 24
    mid, med = binned(ph[out], 1e6 * f[out], 0, 1, 50)
    ax.plot(mid, med, "o-", ms=3, lw=1, color=BLUE)
    ax.set_xlabel("orbital phase")

    # difference image of the quarter with the strongest pixel signal (Q11), from the target pixel file of KIC 6691035
    import tpf as T
    from astropy.wcs import WCS
    ax = fig.add_subplot(gs[1, 2])
    style(ax, "Pixels, Q11: out minus in transit (SNR)")
    ax.grid(False)
    best = None
    for path in T.tpf_files(6691035):
        r = T.difference(path, P, t0, t14)
        if r and r["quarter"] == 11:
            best = r
    if best:
        w = WCS(best["hdr"])
        snr = best["D"] / np.where(best["E"] > 0, best["E"], np.inf)
        im = ax.imshow(snr, origin="lower", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.85)
        for ra_, dec_, lab, mk in ((291.6735918808428, 42.15004357542636, "KIC 6691030 (G 17.6)", "x"),
                                   (None, None, None, None)):
            if ra_ is None:
                continue
            gx, gy = w.all_world2pix([[ra_, dec_]], 0)[0]
            ax.plot(gx, gy, mk, ms=9, color="#e34948", mew=2)
            ax.annotate(lab, (gx, gy), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=INK)
        tx, ty = w.all_world2pix([[best["hdr"].get("RA_OBJ", 0) or 0, best["hdr"].get("DEC_OBJ", 0) or 0]], 0)[0]
        ax.set_xlabel("pixel column (4 arcsec)")
        ax.set_ylabel("pixel row")

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    txt = [
        "Star  KIC 6691030 = TIC 63005412 = Gaia DR3 %s" % TIC["gaia"],
        "      Kepler background source (not a Kepler target), G = %.2f" % TIC["G"],
        "      M dwarf: Teff %d +- %d K, R = %.3f +- %.3f Rsun, d = %.0f pc (TIC v8)" % (TIC["teff"], TIC["e_teff"], TIC["rad"], TIC["e_rad"], TIC["dist"]),
        "Signal  P = %.5f d, T14 = %.2f h, depth = %.0f ppm" % (P, res["t14_h"], res["depth_ppm"]),
        "        %d transits, MES %.1f, odd/even %.1f sigma, secondary %.1f sigma" % (res["n_transits"], res["mes"], res["odd_even_sigma"], res["secondary_sigma"]),
        "        ingress/T14 = %.2f, b = %.2f, a/R* = %.1f" % (res["ingress_fraction"], res["b"], res["a_over_rstar"]),
        "FLAG    density from transit %.1f vs TIC %.1f +- %.1f solar; TIC density" % (res["rho_transit_solar"], TIC["rho"], TIC["e_rho"]),
        "        forced: delta chi2 = %.1f (about 2.5 sigma)" % res["dchi2_tic_density"],
        "FLAG    phase curve %.0f ppm at P/2 (above %.0f %% of random periods)," % (res["phase_P2_ppm"][0], 100 * (1 - res["phase_P2_frac_random_above"])),
        "        %.0f ppm at P (above %.0f %%): ellipsoidal light of a binary?" % (res["phase_P_ppm"][0], 100 * (1 - res["phase_P_frac_random_above"])),
        "Pixels  dip centroid 1.4 arcsec from the star (median of 16 quarters);",
        "        nearest other Gaia source >= 8 arcsec; G=12.7 neighbour shows no dip",
        "If on target: Rp = %.1f +- %.1f Earth radii, a = %.3f AU, Teq ~ %.0f K" % (res["rp_earth"], res["e_rp_earth"], res["a_au"], res["teq_k"]),
        "NEB chain model (slot 0, block 1428, weights d8d471ac...): planet, score %+d" % res["model_score"],
        "Status: transit candidate with two flags; an eclipsing binary (grazing",
        "or blended within ~2 arcsec) is not excluded. Not validated (no high-",
        "resolution imaging, no FPP, no spectroscopy).",
    ]
    half = 8  # star, signal and the density flag in the left column; the rest on the right
    ax.text(0, 1, "\n".join(txt[:half]), va="top", ha="left", fontsize=8, family="DejaVu Sans Mono", color=INK, linespacing=1.55)
    ax.text(0.52, 1, "\n".join(txt[half:]), va="top", ha="left", fontsize=8, family="DejaVu Sans Mono", color=INK, linespacing=1.55)

    fig.text(0.06, 0.965, "Transit candidate around KIC 6691030 (TIC 63005412), P = %.4f d" % P, fontsize=14, weight="bold", color=INK)
    fig.text(0.06, 0.94, "Found by the NEB hunt in the Kepler Bonus background light curves; vetting sheet for astronomers.", fontsize=9.5, color=INK2)
    fig.text(0.06, 0.012, "Data: Kepler Bonus PSF light curves (Martinez-Palomera, Hedges & Dotson 2023, MAST doi:10.17909/7jbr-w430), "
             "Kepler target pixel files, Gaia DR3, TIC v8. NASA Kepler mission.", fontsize=7.5, color=INK2)
    out = os.path.join(K.RESULTS, "candidate-%s.png" % SRC)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
