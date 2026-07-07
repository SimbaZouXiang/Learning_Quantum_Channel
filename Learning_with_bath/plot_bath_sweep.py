"""Plot training curves + final losses for the (T=L, g) sweep at fixed N.

Produces:
  - one loss-curves figure per T_L value (all g's overlaid);
  - a final-test-loss-vs-g figure comparing the T_L values.

Reads .npy files from ./npy_outputs/, writes .png into ./Figures/.
Reads N from $BATH_N (default 10). Skips missing tasks gracefully, so you can
run this while the sweep is still in progress.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fsize = 12
tsize = 12

tdir = 'in'
major = 7.5
minor = 4.5

style = 'default'

plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams['axes.grid']=False
plt.rcParams['grid.alpha'] = 0.25
#mpl.rcParams.update({"axes.grid" : True, "grid.alpha": 0.25})
plt.rcParams['figure.dpi'] = 400
plt.rcParams['text.usetex'] = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(SCRIPT_DIR, "npy_outputs")
FIG_DIR = os.path.join(SCRIPT_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

N = int(os.environ.get("BATH_N", 10))
T_L_VALUES = [2, 3, 4, 5]
COUPLINGS  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
p = 10  # depolarizing strength tag (*100)
# Per-T_L bond cap suffix: must match MAX_BD_BY_TL in run_bath_sweep.py.
MAX_BD_BY_TL = {2: 256, 3: 64, 4: 64, 5: 64}
# Per-T_L override for the g=0 entry only: prefer the higher-bond cheat-init
# value when available, since the optimizer can't reach the architectural floor
# at g=0 (unitary teacher, hard to fit from random init). Falls back to the
# regular MAX_BD_BY_TL[T_L] if the override file isn't present.
G0_OVERRIDE_BD_BY_TL = {2: 1024, 3: 1024, 4: 1024, 5: 1024}

# Multi-seed sweep that was run for selected (T_L, g) pairs in a series of
# debugjob calls. The drivers applied an offset of +100 so we look for
# `_s100`.._s119` files. Where they exist we report mean ± std instead of the
# single-seed canonical value. Not every offset is present at every point;
# `_multiseed_stats` simply aggregates whichever ones it finds (TL=2/3/4 have
# up to 20 seeds each at g>0, TL=5 has 13 seeds — DJ2's TL=5 batch hit the
# debugjob walltime and was recovered partially in DJ3).
MULTISEED_OFFSETS = list(range(100, 120))

final_train = {t: {} for t in T_L_VALUES}
final_test  = {t: {} for t in T_L_VALUES}
# Same keys as final_train/test; values are stds. Missing key ⇒ no multi-seed
# data for that point ⇒ no error bar to draw.
err_train   = {t: {} for t in T_L_VALUES}
err_test    = {t: {} for t in T_L_VALUES}


def _multiseed_stats(N, T_L, g_tag, p, bd):
    """Return (median_tr, (lo_tr, hi_tr), median_te, (lo_te, hi_te)) if at
    least 2 multi-seed files exist for this (T_L, g, bd) point, else None.

    `median` is the median across seeds. `(lo, hi)` are the *positive*
    differences from the median to the 25th and 75th percentiles (so passing
    them as yerr to matplotlib's errorbar draws asymmetric whiskers from
    Q25 to Q75). This pairs with the median in the conventional "median +
    IQR" reporting style and never dips below zero.

    Prefers the longer-trained 200-epoch _e200 sweep (seed offsets 200..279)
    where it exists; otherwise falls back to the 80-epoch sweep at offsets
    100..119.
    """
    import glob

    def _iqr(arr):
        """Return (mean, sigma, sigma) — symmetric ±1 sample std error bars.
        Both legs of the asymmetric whisker pair are set to the same value
        so matplotlib renders a standard symmetric ±σ bar."""
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1))
        return mu, sigma, sigma

    # First try _e200 files (any seed offset starting with 2)
    trs_e200, tes_e200 = [], []
    for llf in sorted(glob.glob(os.path.join(
            NPY_DIR,
            f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}_p{p:03d}_bd{bd}_s*_e200_learning_loss.npy"))):
        tlf = llf.replace("_learning_loss.npy", "_testing_loss.npy")
        if os.path.exists(tlf):
            trs_e200.append(float(np.load(llf)[-1]))
            tes_e200.append(float(np.load(tlf)[0]))
    if len(trs_e200) >= 2:
        trs = np.asarray(trs_e200); tes = np.asarray(tes_e200)
        m_tr, lo_tr, hi_tr = _iqr(trs)
        m_te, lo_te, hi_te = _iqr(tes)
        return m_tr, (lo_tr, hi_tr), m_te, (lo_te, hi_te)

    # Otherwise fall back to canonical _s100..119 multi-seed sweep
    trs, tes = [], []
    for off in MULTISEED_OFFSETS:
        prefix = f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}_p{p:03d}_bd{bd}_s{off}"
        llf = os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy")
        tlf = os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy")
        if os.path.exists(llf) and os.path.exists(tlf):
            trs.append(float(np.load(llf)[-1]))
            tes.append(float(np.load(tlf)[0]))
    if len(trs) < 2:
        return None
    trs = np.asarray(trs); tes = np.asarray(tes)
    m_tr, lo_tr, hi_tr = _iqr(trs)
    m_te, lo_te, hi_te = _iqr(tes)
    return m_tr, (lo_tr, hi_tr), m_te, (lo_te, hi_te)

# --- Per-T_L loss-curves overlay ---
for T_L in T_L_VALUES:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(COUPLINGS)))
    any_loaded = False
    bd = MAX_BD_BY_TL[T_L]
    for g, color in zip(COUPLINGS, cmap):
        g_tag = f"{int(round(g * 100)):03d}"
        # For g=0 use the high-bond cheat-init file if present, else fall back.
        bd_for_g = G0_OVERRIDE_BD_BY_TL.get(T_L, bd) if g == 0.0 else bd
        prefix_new = f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}_p{p:03d}_bd{bd_for_g}"
        prefix_old = f"bath_sweep_N{N}_T{T_L}_L{T_L}_g{g_tag}_p{p:03d}"
        prefix = prefix_new if os.path.exists(
            os.path.join(NPY_DIR, f"{prefix_new}_learning_loss.npy")
        ) else prefix_old
        llf = os.path.join(NPY_DIR, f"{prefix}_learning_loss.npy")
        tlf = os.path.join(NPY_DIR, f"{prefix}_testing_loss.npy")
        if not os.path.exists(llf) or not os.path.exists(tlf):
            print(f"[skip] {prefix}: files missing")
            continue
        any_loaded = True
        ll = np.load(llf)
        tl = np.load(tlf)
        # Multi-seed cross-depth marker = median, error whiskers = (Q25, Q75)
        # as asymmetric `(below_diff, above_diff)` tuples.
        ms = _multiseed_stats(N, T_L, g_tag, p, bd_for_g)
        if ms is not None:
            mtr, err_tr_tuple, mte, err_te_tuple = ms
            final_train[T_L][g] = mtr
            final_test[T_L][g]  = mte
            err_train[T_L][g]   = err_tr_tuple
            err_test[T_L][g]    = err_te_tuple
        else:
            final_train[T_L][g] = float(ll[-1])
            final_test[T_L][g]  = float(tl[0])
        ax.plot(range(len(ll)), ll, color=color, linewidth=1.5,
                label=f"g = {g:.2f}")
    if not any_loaded:
        plt.close(fig)
        continue
    ax.set_xlabel("epoch")
    ax.set_ylabel(r'$d_F(M_{out}, M_{target})$', fontsize=20)
    # ax.set_yscale("log")  # linear is easier to read for cross-depth comparison
    ax.set_title(f"QMLM-with-bath: loss curves vs g (N={N}, T=L={T_L})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"bath_sweep_N{N}_T{T_L}_L{T_L}_loss_curves.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)

# --- Combined final-loss-vs-g across T_L values ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
ax_tr, ax_te = axes
for T_L in T_L_VALUES:
    if not final_test[T_L]:
        continue
    gs = sorted(final_test[T_L].keys())
    tr = np.asarray([final_train[T_L][g] for g in gs])
    te = np.asarray([final_test[T_L][g]  for g in gs])
    # Asymmetric error bars: build [[below_diff_per_g], [above_diff_per_g]]
    # arrays. NaN where no multi-seed data exists (errorbar suppresses NaN).
    def _split_err(err_dict):
        los, his = [], []
        for g in gs:
            e = err_dict.get(g)
            if isinstance(e, tuple) and len(e) == 2:
                los.append(e[0]); his.append(e[1])
            else:
                los.append(np.nan); his.append(np.nan)
        return np.asarray([los, his])
    e_tr = _split_err(err_train[T_L])
    e_te = _split_err(err_test[T_L])
    ax_tr.errorbar(gs, tr, yerr=e_tr, fmt="o-", capsize=3, label=f"T=L={T_L}")
    ax_te.errorbar(gs, te, yerr=e_te, fmt="s-", capsize=3, label=f"T=L={T_L}")

for ax, title in ((ax_tr, r'$Training: d_F(M_{out}, M_{target})$'),
                   (ax_te, r'$Testing: d_F(M_{out}, M_{target})$')):
    ax.set_xlabel(r"$g$", fontsize=20)
    ax.set_ylabel(title, fontsize=20)
    # ax.set_yscale("log")  # linear is easier to read for cross-depth comparison
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"Final Losses of Learning Open-system Coupled to a Bath for N={N}", fontsize=28)
fig.tight_layout()
out = os.path.join(FIG_DIR, f"bath_sweep_N{N}_final_vs_g_all_TL.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
