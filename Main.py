#!/usr/bin/env python3
# gait_analysis_madgwick_all.py
#
# Analisi completa dei passi con:
#   – tutti i calcoli originali (range pressioni, Az/|A|, durate, distanze picchi)
#   – orientazione rispetto alla verticale stimata con filtro Madgwick
#
# Dipendenze (installare una volta sola):
#   pip install ahrs matplotlib pandas numpy
# -----------------------------------------------------------------------
IGNORE_EDGE_STEPS = True      # False → analizza tutti i file
EDGE_FULL_SKIP    = 1         # passi interi da saltare a testa/coda
EDGE_HALF_SKIP    = 2         # coppie .1/.2 da saltare a testa/coda
RESAMPLE_POINTS   = 100       # punti per curve tempo-normalizzate
BETA_MADGWICK     = 0.1       # coefficiente del filtro (0.03–0.1 tipico)
# -----------------------------------------------------------------------

import os, glob, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ahrs.filters import Madgwick

_rx_full = re.compile(r"Passo[_ ]?(\d+)\.csv$",    re.I)
_rx_half = re.compile(r"Passo[_ ]?(\d+)\.(1|2)\.csv$", re.I)

# ----------------------------- utility -----------------------------------
def resample(series, n=RESAMPLE_POINTS):
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, series)

# ----------------------- Madgwick orientation ---------------------------
def madgwick_orientation(df, dt):
    """Restituisce l’angolo (°) fra asse Y sensore e verticale globale."""
    gyr = np.deg2rad(df[["Gx","Gy","Gz"]].values)   # rad/s
    acc = df[["Ax","Ay","Az"]].values               # g
    mag = df[["Mx","My","Mz"]].values               # Gauss

    N = len(df)
    q = np.zeros((N, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])           # quaternione iniziale
    mad = Madgwick(sampleperiod=dt, beta=BETA_MADGWICK)

    for t in range(1, N):
        q[t] = mad.updateMARG(q[t-1], gyr[t], acc[t], mag[t])

    cos_theta = np.clip(1.0 - 2.0*(q[:,1]**2 + q[:,3]**2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ------------------- elabora singolo CSV ---------------------------------
def process_single_step(path, baseline, weight):
    df = pd.read_csv(path)

    # baseline pressioni
    for i,c in enumerate(["S0","S1","S2"]):
        df[c] -= baseline[i]

    duration = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    ranges   = {c:(df[c].max()-df[c].min())/weight for c in ["S0","S1","S2"]}

    # rapporto Az / |A|
    a_norm = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
    ratio  = np.where(a_norm==0, 0, df["Az"]/a_norm)
    ratio_res = resample(ratio)

    # orientazione (Madgwick)
    dt   = np.diff(df["Timestamp"]).mean() if len(df) > 1 else 0.01
    orient = madgwick_orientation(df, dt)
    orient_res = resample(orient)

    # timestamp picchi pressioni
    max_t = {c: df.loc[df[c].idxmax(),"Timestamp"] for c in ["S0","S1","S2"]}
    min_t = {c: df.loc[df[c].idxmin(),"Timestamp"] for c in ["S0","S1","S2"]}

    return dict(
        file=path,
        n=len(df),
        duration=duration,
        ranges=ranges,
        ratio_mean=ratio.mean(),  ratio_std=ratio.std(),  ratio_res=ratio_res,
        orient_mean=orient.mean(),orient_std=orient.std(),orient_res=orient_res,
        max_t=max_t, min_t=min_t
    )

# ---------------------- carica file ordinati ----------------------------
def _key_full(p):  m=_rx_full.search(os.path.basename(p)); return int(m.group(1)) if m else 0
def load_full_steps(folder, base, w):
    files = sorted(glob.glob(os.path.join(folder,"*.csv")), key=_key_full)
    if IGNORE_EDGE_STEPS and len(files) > 2*EDGE_FULL_SKIP:
        files = files[EDGE_FULL_SKIP:-EDGE_FULL_SKIP]
    return [process_single_step(p, base, w) for p in files]

def load_half_steps(folder, base, w):
    pairs={}
    for p in glob.glob(os.path.join(folder,"*.csv")):
        m=_rx_half.search(os.path.basename(p))
        if m: pairs.setdefault(int(m.group(1)),{})[int(m.group(2))]=p
    nums = sorted(pairs)
    if IGNORE_EDGE_STEPS and len(nums) > 2*EDGE_HALF_SKIP:
        nums = nums[EDGE_HALF_SKIP:-EDGE_HALF_SKIP]

    first, second = [], []
    for n in nums:
        if 1 in pairs[n]: first .append(process_single_step(pairs[n][1], base, w))
        if 2 in pairs[n]: second.append(process_single_step(pairs[n][2], base, w))
    return first, second

# ----------------------- aggregazione -----------------------------------
def aggregate(steps):
    if not steps: return {}
    n  = np.array([s["n"] for s in steps])
    rm = np.array([s["ratio_mean"]  for s in steps])
    om = np.array([s["orient_mean"] for s in steps])

    ratio_global = float((rm*n).sum()/n.sum())

    ratio_mat  = np.vstack([s["ratio_res"]  for s in steps])
    orient_mat = np.vstack([s["orient_res"] for s in steps])
    ranges = {c:np.array([s["ranges"][c] for s in steps]) for c in ["S0","S1","S2"]}

    dist={}
    for c in ["S0","S1","S2"]:
        max_ts=np.array([s["max_t"][c] for s in steps])
        min_ts=np.array([s["min_t"][c] for s in steps])
        dist[f"{c}_max"]=np.diff(max_ts) if len(max_ts)>1 else np.array([])
        dist[f"{c}_min"]=np.diff(min_ts) if len(min_ts)>1 else np.array([])

    return dict(
        count=len(steps),
        dur_mean=float(np.mean([s["duration"] for s in steps])),
        dur_std =float(np.std ([s["duration"] for s in steps])),
        ratio_mean_of_means=float(rm.mean()), ratio_mean_std=float(rm.std()),
        ratio_mean_all=float(ratio_global),
        orient_mean_of_means=float(om.mean()), orient_mean_std=float(om.std()),
        ratio_curve_mean=ratio_mat.mean(0),  ratio_curve_std=ratio_mat.std(0),
        orient_curve_mean=orient_mat.mean(0),orient_curve_std=orient_mat.std(0),
        ranges_mean={c:float(ranges[c].mean()) for c in ranges},
        ranges_std ={c:float(ranges[c].std())  for c in ranges},
        dist=dist
    )

# ----------------------------- plot helpers ------------------------------
def plot_mean_std(x, m, s, out, ylab, title):
    plt.figure()
    plt.plot(x, m, label="mean")
    plt.fill_between(x, m-s, m+s, alpha=.3, label="std")
    plt.xlabel("time norm (%)"); plt.ylabel(ylab); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_concat(lst, out, ylab, title):
    concat=np.concatenate(lst); seg=len(lst[0])
    plt.figure(); plt.plot(concat)
    for i in range(1,len(lst)):
        plt.axvline(i*seg, color="gray", ls="--", lw=.5)
    plt.xlabel("concatenated samples"); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(out); plt.close()

# ----------------------------- MAIN --------------------------------------
def main():
    root   = input("Main folder path: ").strip()
    weight = float(input("Weight kg: ").strip())

    print("\nBaseline RIGHT foot")
    base_r = tuple(float(input(f"S{i} baseline: ").strip()) for i in range(3))
    print("\nBaseline LEFT foot")
    base_l = tuple(float(input(f"S{i} baseline: ").strip()) for i in range(3))

    feet = {"Piede_Destro":("right",base_r), "Piede_Sinistro":("left",base_l)}
    out_root = os.path.join(root,"calcoli")
    os.makedirs(out_root, exist_ok=True)
    x_norm = np.linspace(0.0, 100.0, RESAMPLE_POINTS)

    for fd,(tag,base) in feet.items():
        fp = os.path.join(root,"passi",fd)
        if not os.path.isdir(fp):
            print("missing", fp); continue

        out_dir = os.path.join(out_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        full_steps   = load_full_steps (os.path.join(fp,"Passi_Interi"), base, weight)
        first_half, second_half = load_half_steps(os.path.join(fp,"Mezzi_Passi"), base, weight)

        agg_full   = aggregate(full_steps)
        agg_first  = aggregate(first_half)
        agg_second = aggregate(second_half)

        # ---- plots ---------------------------------------------------------
        if agg_full:
            plot_mean_std(x_norm, agg_full["ratio_curve_mean"],  agg_full["ratio_curve_std"],
                          os.path.join(out_dir,"ratio_mean_std.png"),
                          "Az/|A|", "Az/|A| mean ± std")

            plot_mean_std(x_norm, agg_full["orient_curve_mean"], agg_full["orient_curve_std"],
                          os.path.join(out_dir,"orient_mean_std.png"),
                          "orientation deg", "Orientation mean ± std")

            plot_concat([s["orient_res"] for s in full_steps],
                        os.path.join(out_dir,"orient_concat.png"),
                        "orientation deg", "Orientation concatenated")

        # ---- results.txt ---------------------------------------------------
        res_path = os.path.join(out_dir,"results.txt")
        with open(res_path,"w") as f:
            f.write(f"RESULTS {tag.upper()}\n\nFULL STEPS\n")
            if agg_full:
                f.write(f"steps analysed            : {agg_full['count']}\n")
                f.write(f"duration mean std         : {agg_full['dur_mean']} {agg_full['dur_std']}\n")
                for c in ["S0","S1","S2"]:
                    f.write(f"{c} range mean std          : {agg_full['ranges_mean'][c]} {agg_full['ranges_std'][c]}\n")
                f.write(f"Az/|A| mean of means std  : {agg_full['ratio_mean_of_means']} {agg_full['ratio_mean_std']}\n")
                f.write(f"Az/|A| mean all samples   : {agg_full['ratio_mean_all']}\n")
                f.write(f"Orient mean std           : {agg_full['orient_mean_of_means']} {agg_full['orient_mean_std']}\n")
                for c in ["S0","S1","S2"]:
                    dmax = agg_full["dist"][f"{c}_max"]; dmin = agg_full["dist"][f"{c}_min"]
                    if len(dmax):
                        f.write(f"distance max {c} mean std : {dmax.mean()} {dmax.std()}\n")
                        f.write(f"distance min {c} mean std : {dmin.mean()} {dmin.std()}\n")
            else:
                f.write("no data\n")

            f.write("\nFIRST HALF STEPS (.1)\n")
            if agg_first:
                f.write(f"count                     : {agg_first['count']}\n")
                f.write(f"Az/|A| mean of means std  : {agg_first['ratio_mean_of_means']} {agg_first['ratio_mean_std']}\n")
            else:
                f.write("no data\n")

            f.write("\nSECOND HALF STEPS (.2)\n")
            if agg_second:
                f.write(f"count                     : {agg_second['count']}\n")
                f.write(f"Az/|A| mean of means std  : {agg_second['ratio_mean_of_means']} {agg_second['ratio_mean_std']}\n")
            else:
                f.write("no data\n")

            f.write("\nSTEP BY STEP DETAILS\n")
            prev_max = {c: None for c in ["S0","S1","S2"]}
            prev_min = {c: None for c in ["S0","S1","S2"]}
            for s in full_steps:
                name=os.path.basename(s["file"])
                f.write("-"*60 + "\n")
                f.write(f"file               : {name}\n")
                f.write(f"duration           : {s['duration']}\n")
                for c in ["S0","S1","S2"]:
                    f.write(f"{c} range          : {s['ranges'][c]}\n")
                f.write(f"Az/|A| mean std     : {s['ratio_mean']} {s['ratio_std']}\n")
                f.write(f"Orient mean std     : {s['orient_mean']} {s['orient_std']}\n")
                for c in ["S0","S1","S2"]:
                    cmx, cmn = s["max_t"][c], s["min_t"][c]
                    f.write(f"{c} max timestamp     : {cmx}\n")
                    f.write(f"{c} min timestamp     : {cmn}\n")
                    if prev_max[c] is not None:
                        f.write(f"{c} dist max to prev  : {cmx-prev_max[c]}\n")
                        f.write(f"{c} dist min to prev  : {cmn-prev_min[c]}\n")
                    prev_max[c], prev_min[c] = cmx, cmn
                f.write("\n")
        print("created", res_path)

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
