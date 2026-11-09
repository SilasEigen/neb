"""Vetting of a transit-like signal, in the order of the Kepler pipeline, before the model of the chain.

Every test works on the unbinned detrended light curve (t, f: relative flux, 0 = median). The noise of a transit is
measured on the light curve averaged over the transit duration (like the Kepler CDPP), so red noise counts.
"""
import numpy as np

CADENCE = 0.020434  # days, Kepler long cadence
ARTIFACT_BANDS = ((12.5, 14.5),)  # days; detections pile up there in Kepler Bonus light curves (instrumental)
MES_MIN = 7.1  # Kepler detection threshold
RHO_MIN = 2.0  # lowest stellar density (solar units) accepted for an M dwarf when bounding the duration
SEASON_SIGMA_MAX = 4.0  # largest depth difference between two Kepler seasons, in sigma (6 pairs of seasons)...
SEASON_RATIO_MAX = 2.0  # ...and largest deepest/shallowest ratio: known planets reach 1.98 in Kepler Bonus PSF light
# curves (median 1.31, 90th percentile 1.79 over 50 planets of check_vet.py), so a ratio alone is no proof of a blend
SHAPE_DCHI2_MAX = 9.0  # largest chi2 cost, red noise included, of the best shape a planet can make around the star


PERIODS = np.exp(np.arange(np.log(0.7), np.log(20.0), 4e-5))
PERIODS = PERIODS[~np.any([(PERIODS > lo) & (PERIODS < hi) for lo, hi in ARTIFACT_BANDS], axis=0)]
DURATIONS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0]) / 24


def clean_mask(t, f):
    """points kept by clean(): drops flares (points above 4 sigma), the day after each gap of more than 0.3 day
    (monthly downlinks every ~31 days and quarter rolls: the thermal settling after them makes dips at ~15.5 and
    ~31 days) and the last quarter day before each gap; transits are dips, so nothing is clipped below"""
    s = 1.4826 * np.median(np.abs(f - np.median(f)))
    keep = f < np.median(f) + 4 * s
    gaps = np.flatnonzero(np.diff(t) > 0.3)
    starts = np.concatenate([[t[0]], t[gaps + 1]])
    ends = np.concatenate([t[gaps], [t[-1]]])
    for a, b in zip(starts, ends):
        keep &= ~(((t >= a) & (t < a + 1.0)) | ((t > b - 0.25) & (t <= b)))
    return keep


def clean(t, f):
    keep = clean_mask(t, f)
    return t[keep], f[keep]


def robust_mes(t, f, period, t0, dur, noise):
    """MES of the events, 0 when fewer than 3 events, when one event dominates or when most events are not dips"""
    ev = events(t, f, period, t0, dur)
    if len(ev) < 3 or not np.isfinite(noise) or noise <= 0:
        return 0.0
    s = np.array([x[1] for x in ev]) / noise
    if (s > 0).mean() < 0.6 or (s.clip(0) ** 2).max() > 0.5 * (s.clip(0) ** 2).sum():
        return 0.0
    return float(s.sum() / np.sqrt(len(s)))


def search(t, f, peaks=50):
    """box search outside the artifact bands; among the strongest distinct peaks, keeps the one with the largest
    robust MES measured on the unbinned light curve (red noise included). Returns period, epoch, duration (days)."""
    from astropy.timeseries import BoxLeastSquares
    t, f = clean(t, f)
    edges = np.arange(t.min(), t.max() + 1 / 24, 1 / 24)
    k = np.digitize(t, edges)
    order = np.argsort(k, kind="stable")
    ks, fs, ts = k[order], f[order], t[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    tb = np.array([x.mean() for x in np.split(ts, cut)])
    fb = np.array([np.median(x) for x in np.split(fs, cut)])
    dy = np.full(fb.size, 1.4826 * np.median(np.abs(fb - np.median(fb))) or 1e-4)
    res = BoxLeastSquares(tb, fb, dy).power(PERIODS, DURATIONS)
    raw = np.nan_to_num(np.asarray(res.power), nan=0.0)
    # the power rises with the period; compare each period with its neighbourhood (median and scatter of blocks of
    # 2000 grid points, about 8 % in period, interpolated), as the usual SDE normalisation does; blocks
    # rather than a sliding window keep the memory per star small
    b = 2000
    nb = max(1, raw.size // b)
    blocks = raw[:nb * b].reshape(nb, b)
    centers = np.arange(nb) * b + b / 2
    bmed = np.median(blocks, axis=1)
    bmad = 1.4826 * np.median(np.abs(blocks - bmed[:, None]), axis=1)
    idx = np.arange(raw.size)
    med = np.interp(idx, centers, bmed)
    mad = np.interp(idx, centers, bmad) + 1e-12
    power = (raw - med) / mad
    inner = (res.period > PERIODS[0] * 1.02) & (res.period < PERIODS[-1] / 1.02)
    chosen = []
    # the strongest peaks of both spectra (raw and normalised) are the candidates; the robust MES picks one
    for spec in (raw, power):
        picked = 0
        for i in np.argsort(np.where(inner, spec, -np.inf))[::-1]:
            p = res.period[i]
            if all(abs(np.log(p / res.period[j])) > 0.005 for j in chosen):
                chosen.append(i)
                picked += 1
                if picked == peaks:
                    break
    noise = {d: duration_noise(t, f, d) for d in DURATIONS}
    best = max(chosen, key=lambda i: robust_mes(t, f, res.period[i], res.transit_time[i], res.duration[i],
                                                noise[min(DURATIONS, key=lambda d: abs(d - res.duration[i]))]))
    return float(res.period[best]), float(res.transit_time[best]), float(res.duration[best])


def duration_noise(t, f, dur):
    """robust scatter of the flux averaged over `dur` days (out of any transit is assumed: transits are rare)"""
    n = max(1, int(round(dur / CADENCE)))
    if f.size < 4 * n:
        return np.inf
    c = np.cumsum(np.insert(f, 0, 0.0))
    means = (c[n:] - c[:-n]) / n
    # keep windows that span no gap
    span = t[n - 1:] - t[:f.size - n + 1]
    means = means[span < (n + 1) * CADENCE]
    if means.size < 10:
        return np.inf
    return 1.4826 * np.median(np.abs(means - np.median(means)))


def events(t, f, period, t0, dur, phase=0.0):
    """depth of each covered event at `phase` (0 transit, 0.5 secondary): list of (epoch number, depth); the depth
    is the mean of the ring dur..3 dur (at most half a period) around the event minus the mean inside it"""
    x = (t - t0) / period - phase
    n = np.rint(x).astype(np.int64)
    dt = np.abs(x - n) * period
    inn = dt < dur / 2
    ring = (dt >= dur) & (dt < min(3 * dur, period / 2))
    n0 = n.min()
    idx = n - n0
    m = idx.max() + 1
    c_in = np.bincount(idx[inn], minlength=m)
    s_in = np.bincount(idx[inn], weights=f[inn], minlength=m)
    c_r = np.bincount(idx[ring], minlength=m)
    s_r = np.bincount(idx[ring], weights=f[ring], minlength=m)
    need = 0.5 * dur / CADENCE
    ok = (c_in >= need) & (c_r >= need)
    k = np.flatnonzero(ok)
    return [(int(e + n0), float(s_r[e] / c_r[e] - s_in[e] / c_in[e])) for e in k]


def t_max_hours(period, rho=RHO_MIN):
    """longest central-transit duration for a circular orbit around a star of density rho (solar units)"""
    return 13.0 * (period / 365.25) ** (1 / 3) * rho ** (-1 / 3)


def season_of(quarter):
    """Kepler season (0 to 3) of each quarter: the telescope rolled every quarter, so quarters 4 apart share a
    pointing and a place on the detector; Q0 shares the pointing of Q1"""
    q = np.asarray(quarter)
    return np.where(q == 0, 1, q % 4)


def season_depths(t, f, quarter, period, t0, dur, noise):
    """(largest depth difference between two Kepler seasons in sigma, deepest/shallowest season ratio) over the
    seasons with at least 3 events; (nan, nan) with fewer than 2 such seasons. A planet blocks the same fraction
    of its star's light in every season; a dip much deeper in some seasons points to another star whose light
    falls on the photometry differently from one season to the next. The PSF photometry of Kepler Bonus also
    changes the depth of real planets with the season, hence the ratio threshold."""
    s = season_of(quarter)
    seasons = []
    for k in range(4):
        m = s == k
        ev = events(t[m], f[m], period, t0, dur) if m.sum() else []
        if len(ev) >= 3:
            d = np.array([x[1] for x in ev])
            seasons.append((d.mean(), noise / np.sqrt(len(d))))
    if len(seasons) < 2:
        return float("nan"), float("nan")
    sigma = max(abs(a[0] - b[0]) / np.hypot(a[1], b[1]) for i, a in enumerate(seasons) for b in seasons[i + 1:])
    deep, shallow = max(x[0] for x in seasons), min(x[0] for x in seasons)
    return float(sigma), float(deep / shallow) if shallow > 0 else float("inf")


def unit_trapezoid(x, t14, tau):
    """transit of depth 1 (negative), total duration t14 and ingress tau (days), averaged over the long cadence"""
    a = np.abs(np.add.outer(x, np.linspace(-CADENCE / 2, CADENCE / 2, 9)))
    flat = t14 / 2 - tau
    m = np.where(a <= flat, -1.0, np.where(a < t14 / 2, -(t14 / 2 - a) / max(tau, 1e-9), 0.0))
    return m.mean(axis=1)


def shape_dchi2(t, f, period, t0, dur, noise, rho_min=RHO_MIN):
    """chi2 cost, in units of the red-noise-corrected point variance, of the best transit that a planet can make
    around a star of density at least rho_min, against the best free trapezoid. A V-shaped dip that lasts nearly
    the longest central transit the star allows needs an object almost as large as the star: a binary, rarely a
    planet. The dip is folded into 4-minute bins; both fits use the same bins and a free centre."""
    x = np.mod(t - t0 + period / 2, period) - period / 2
    w = min(3 * dur, period / 2)
    m = np.abs(x) < w
    if m.sum() < 50:
        return float("nan")
    xs, fs = x[m], f[m]
    edges = np.arange(-w, w + 1 / 360, 1 / 360)
    k = np.clip(np.digitize(xs, edges) - 1, 0, edges.size - 2)
    n_all = np.bincount(k, minlength=edges.size - 1).astype(float)
    mean_all = np.bincount(k, weights=fs, minlength=edges.size - 1) / np.maximum(n_all, 1)
    within = float(((fs - mean_all[k]) ** 2).sum())  # scatter inside the bins, the same for every model
    ok = n_all > 0
    xc, n, fb = (0.5 * (edges[1:] + edges[:-1]))[ok], n_all[ok], mean_all[ok]

    def best(models):
        """lowest chi2 over (t14, tau, centre, lowest depth, highest depth); the depth comes from weighted linear
        least squares, clipped to its allowed range (the chi2 is quadratic in the depth, so the clip is the
        constrained optimum)"""
        lo = np.inf
        for t14, tau, c, dlo, dhi in models:
            u = unit_trapezoid(xc - c, t14, tau)
            den = (n * u * u).sum()
            if den <= 0:
                continue
            d = min(max((n * u * fb).sum() / den, dlo), dhi)
            lo = min(lo, (n * (fb - d * u) ** 2).sum())
        return lo

    centres = np.arange(-0.3, 0.31, 0.05) * dur
    free = [(t14, frac * t14, c, -np.inf, np.inf) for t14 in dur * np.linspace(0.5, 2.0, 31)
            for frac in np.arange(0.04, 0.51, 0.02) for c in centres]
    chi_free = best(free)
    # a planet: radius ratio k up to 0.25 (Jupiter around an early M dwarf); its depth is about k^2 when the whole
    # disc crosses the star (limb darkening allowed for: 0.5 to 1.2 k^2), at most 1.2 k^2 when it grazes the limb
    phys = []
    for rho in rho_min * 2.0 ** np.arange(4):
        a_r = (6.674e-11 * 1408.0 * rho * (period * 86400.0) ** 2 / (3 * np.pi)) ** (1 / 3)
        for kr in np.geomspace(0.01, 0.25, 30):
            for b in np.linspace(0, 1 + kr, 40, endpoint=False):
                t14 = period / np.pi * np.arcsin(min(1.0, np.sqrt((1 + kr) ** 2 - b * b) / a_r))
                t23 = period / np.pi * np.arcsin(min(1.0, np.sqrt(max((1 - kr) ** 2 - b * b, 0.0)) / a_r))
                if t14 < 2 * CADENCE:
                    continue
                dlo = 0.5 * kr * kr if b <= 1 - kr else 0.0
                phys += [(t14, (t14 - t23) / 2, c, dlo, 1.2 * kr * kr) for c in centres]
    chi_phys = best(phys)
    sigma2 = (within + chi_free) / xs.size
    # red noise: the scatter over the transit duration is larger than white noise of that variance would give
    beta = max(1.0, noise ** 2 / (sigma2 / max(1.0, dur / CADENCE)))
    return float((chi_phys - chi_free) / (sigma2 * beta))


def vet(t, f, period, t0, dur, quarter=None, rho_min=RHO_MIN):
    """dict of measurements and the first failed test ('' when all pass). With the quarter of each point, the
    depth of each Kepler season is compared; rho_min is the lowest stellar density allowed for the host (solar
    units) in the duration and shape tests."""
    sig = duration_noise(t, f, dur)
    ev = events(t, f, period, t0, dur)
    r = {"n_transits": len(ev), "noise_dur_ppm": round(1e6 * sig) if np.isfinite(sig) else -1}
    fail = ""
    if any(lo < period < hi for lo, hi in ARTIFACT_BANDS):
        fail = "artifact period"
    elif len(ev) < 3:
        fail = "fewer than 3 transits"
    if not np.isfinite(sig) or sig <= 0:
        return r | {"fail": fail or "no noise estimate"}
    d = np.array([x[1] for x in ev]) if ev else np.zeros(0)
    snr_ev = d / sig
    mes = float(snr_ev.sum() / np.sqrt(len(d))) if len(d) else 0.0
    depth = float(d.mean()) if len(d) else 0.0
    r.update({"depth_ppm": round(1e6 * depth), "mes": round(mes, 1)})
    if not fail and mes < MES_MIN:
        fail = "MES below 7.1"
    if not fail and (snr_ev.clip(0) ** 2).max() > 0.5 * (snr_ev.clip(0) ** 2).sum():
        fail = "one event dominates"
    if not fail and (d > 0).mean() < 0.6:
        fail = "most events are not dips"
    odd = d[np.array([x[0] % 2 for x in ev]) == 1]
    even = d[np.array([x[0] % 2 for x in ev]) == 0]
    if len(odd) and len(even):
        oe = abs(odd.mean() - even.mean()) / (sig * np.sqrt(1 / len(odd) + 1 / len(even)))
        r["odd_even_sigma"] = round(float(oe), 1)
        if not fail and oe > 3:
            fail = "odd and even depths differ"
    sec = events(t, f, period, t0, dur, phase=0.5)
    if sec:
        s = np.array([x[1] for x in sec])
        ssnr = s.mean() / (sig / np.sqrt(len(s)))
        r["secondary_sigma"] = round(float(ssnr), 1)
        if not fail and ssnr > 3:
            fail = "secondary eclipse"
    r["t_max_h"] = round(t_max_hours(period, rho_min), 2)
    if not fail and dur * 24 > 1.5 * t_max_hours(period, rho_min):
        fail = "too long for an M dwarf"
    if not fail and depth > 0.2:
        fail = "too deep"
    if quarter is not None:
        ss, ratio = season_depths(t, f, quarter, period, t0, dur, sig)
        r["season_sigma"] = round(ss, 1) if np.isfinite(ss) else ""
        r["season_ratio"] = round(ratio, 2) if np.isfinite(ratio) else ("inf" if ratio > 0 else "")
        if not fail and ss > SEASON_SIGMA_MAX and ratio > SEASON_RATIO_MAX:
            fail = "depth changes with the season"
    if not fail:  # the costly shape fit only for signals that pass everything else
        dc = shape_dchi2(t, f, period, t0, dur, sig, rho_min)
        r["shape_dchi2"] = round(dc, 1) if np.isfinite(dc) else ""
        if np.isfinite(dc) and dc > SHAPE_DCHI2_MAX:
            fail = "shape needs a larger object"
    return r | {"fail": fail}
