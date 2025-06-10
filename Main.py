#!/usr/bin/env python3
# gait_analysis_madgwick_all.py
#
# Dipendenze:
#   pip install ahrs matplotlib pandas numpy
# -----------------------------------------------------------------------
IGNORE_EDGE_STEPS = True
EDGE_FULL_SKIP    = 1
EDGE_HALF_SKIP    = 2
RESAMPLE_POINTS   = 100
BETA_MADGWICK     = 0.4
# -----------------------------------------------------------------------

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

_rx_full = re.compile(r"Passo[_ ]?(\d+)\.csv$", re.I)
_rx_half = re.compile(r"Passo[_ ]?(\d+)\.(1|2)\.csv$", re.I)

# ---------------- util ---------------------------------------------------
def resample(series, n=RESAMPLE_POINTS):
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, series)

# ------------- Madgwick sull’intera sessione ----------------------------
def madgwick_session(df, dt, beta=BETA_MADGWICK):
    gyr = np.deg2rad(df[["Gx","Gy","Gz"]].values)
    acc = df[["Ax","Ay","Az"]].values
    mag = df[["Mx","My","Mz"]].values
    N   = len(df)

    mad = Madgwick(sampleperiod=dt, beta=beta)
    q   = np.zeros((N,4))
    q_prev = np.array([1.0,0.0,0.0,0.0])

    for t in range(N):
        q_prev = mad.updateMARG(q_prev, gyr[t], acc[t], mag[t])
        if np.dot(q_prev, q_prev) == 0:
            q_prev = np.array([1.0,0.0,0.0,0.0])
        q[t] = q_prev

    cos_t = np.clip(1.0 - 2.0*(q[:,1]**2 + q[:,3]**2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))

# ---------------- metriche per finestra ---------------------------------
def step_metrics(df_slice, orient, ratio, weight):
    ranges = {c:(df_slice[c].max()-df_slice[c].min())/weight
              for c in ["S0","S1","S2"]}
    duration = df_slice["Timestamp"].iloc[-1]-df_slice["Timestamp"].iloc[0]
    max_t = {c: df_slice.loc[df_slice[c].idxmax(),"Timestamp"] for c in ["S0","S1","S2"]}
    min_t = {c: df_slice.loc[df_slice[c].idxmin(),"Timestamp"] for c in ["S0","S1","S2"]}
    return dict(
        n=len(df_slice),
        duration=duration,
        ranges=ranges,
        ratio_mean=float(ratio.mean()),  ratio_std=float(ratio.std()),
        ratio_res=resample(ratio),
        orient_mean=float(orient.mean()),orient_std=float(orient.std()),
        orient_res=resample(orient),
        max_t=max_t, min_t=min_t
    )

# ---------------- aggregazione ------------------------------------------
def aggregate(steps):
    if not steps: return {}
    n  = np.array([s["n"] for s in steps])
    rm = np.array([s["ratio_mean"]  for s in steps])
    om = np.array([s["orient_mean"] for s in steps])
    glob_rat = float((rm*n).sum()/n.sum())

    ratio_mat  = np.vstack([s["ratio_res"]  for s in steps])
    orient_mat = np.vstack([s["orient_res"] for s in steps])
    ranges = {c:np.array([s["ranges"][c] for s in steps]) for c in ["S0","S1","S2"]}

    dist={}
    for c in ["S0","S1","S2"]:
        mx=np.array([s["max_t"][c] for s in steps])
        mn=np.array([s["min_t"][c] for s in steps])
        dist[f"{c}_max"] = np.diff(mx) if len(mx)>1 else np.array([])
        dist[f"{c}_min"] = np.diff(mn) if len(mn)>1 else np.array([])

    return dict(
        count=len(steps),
        dur_mean=float(np.mean([s["duration"] for s in steps])),
        dur_std =float(np.std ([s["duration"] for s in steps])),
        ratio_mean_of_means=float(rm.mean()), ratio_mean_std=float(rm.std()),
        ratio_mean_all=float(glob_rat),
        orient_mean_of_means=float(om.mean()), orient_mean_std=float(om.std()),
        ratio_curve_mean=ratio_mat.mean(0),  ratio_curve_std=ratio_mat.std(0),
        orient_curve_mean=orient_mat.mean(0),orient_curve_std=orient_mat.std(0),
        ranges_mean={c:float(ranges[c].mean()) for c in ranges},
        ranges_std ={c:float(ranges[c].std())  for c in ranges},
        dist=dist
    )

# ----------------- plot helpers -----------------------------------------
def plot_mean_std(x,m,s,out,y,t):
    plt.figure(); plt.plot(x,m,label="mean")
    plt.fill_between(x,m-s,m+s,alpha=.3,label="std")
    plt.xlabel("time norm (%)"); plt.ylabel(y); plt.title(t)
    plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_concat(series_list,out,y,t):
    concat=np.concatenate(series_list); seg=len(series_list[0])
    plt.figure(); plt.plot(concat)
    for i in range(1,len(series_list)):
        plt.axvline(i*seg,color="gray",ls="--",lw=.5)
    plt.xlabel("concatenated samples"); plt.ylabel(y); plt.title(t)
    plt.tight_layout(); plt.savefig(out); plt.close()

# ------------------ half-steps (solo pressioni e ratio) ------------------
def load_half_steps(folder, baseline, weight):
    pairs={}
    for p in glob.glob(os.path.join(folder,"*.csv")):
        m=_rx_half.search(os.path.basename(p))
        if m: pairs.setdefault(int(m.group(1)),{})[int(m.group(2))]=p
    nums = sorted(pairs)
    if IGNORE_EDGE_STEPS and len(nums)>2*EDGE_HALF_SKIP:
        nums = nums[EDGE_HALF_SKIP:-EDGE_HALF_SKIP]

    first, second = [], []
    for n in nums:
        for idx, dest in ((1, first),(2, second)):
            if idx in pairs[n]:
                df = pd.read_csv(pairs[n][idx])
                for i,c in enumerate(["S0","S1","S2"]): df[c]-=baseline[i]
                ln=np.sqrt(df["Ax"]**2+df["Ay"]**2+df["Az"]**2)
                ratio=np.where(ln==0,0,df["Az"]/ln)
                orient=np.zeros(len(df))  # orientazione non calcolata per mezzi passi
                dest.append(step_metrics(df, orient, ratio, weight))
    return first, second

# ------------------------------ MAIN -------------------------------------
def main():
    root   = input("Main folder path: ").strip()
    weight = float(input("Weight kg: ").strip())

    print("\nBaseline RIGHT foot")
    base_r = tuple(float(input(f"S{i} baseline: ").strip()) for i in range(3))
    print("\nBaseline LEFT foot")
    base_l = tuple(float(input(f"S{i} baseline: ").strip()) for i in range(3))

    feet = {"Piede_Destro":("right",base_r),
            "Piede_Sinistro":("left",base_l)}

    out_root = os.path.join(root,"calcoli")
    os.makedirs(out_root, exist_ok=True)
    x_norm = np.linspace(0.0, 100.0, RESAMPLE_POINTS)

    for fd,(tag,baseline) in feet.items():
        fp = os.path.join(root,"passi",fd)
        if not os.path.isdir(fp):
            print("missing", fp); continue

        # -------- concatena tutti i Passi_Interi ---------------------------
        files = sorted(glob.glob(os.path.join(fp,"Passi_Interi","*.csv")),
                       key=lambda p:int(_rx_full.search(os.path.basename(p)).group(1)))
        if IGNORE_EDGE_STEPS and len(files) > 2*EDGE_FULL_SKIP:
            files = files[EDGE_FULL_SKIP:-EDGE_FULL_SKIP]
        if not files:
            print("no full steps for", fd); continue

        df_all, slices = [], []
        start = 0
        for p in files:
            d = pd.read_csv(p)
            df_all.append(d)
            end = start + len(d)
            slices.append((start,end,p))
            start = end
        df_all = pd.concat(df_all, ignore_index=True)

        # baseline pressioni
        for i,c in enumerate(["S0","S1","S2"]): df_all[c] -= baseline[i]

        dt = np.diff(df_all["Timestamp"]).mean() if len(df_all)>1 else 0.01
        orient_all = madgwick_session(df_all, dt)
        ln=np.sqrt(df_all["Ax"]**2+df_all["Ay"]**2+df_all["Az"]**2)
        ratio_all=np.where(ln==0,0,df_all["Az"]/ln)

        # metriche per ogni passo
        full_steps=[]
        for st,en,path in slices:
            m = step_metrics(df_all.iloc[st:en],
                             orient_all[st:en],
                             ratio_all [st:en],
                             weight)
            m["file"]=path
            full_steps.append(m)

        # mezzi passi
        first_half, second_half = load_half_steps(os.path.join(fp,"Mezzi_Passi"),
                                                  baseline, weight)

        # aggregazioni
        agg_full   = aggregate(full_steps)
        agg_first  = aggregate(first_half)
        agg_second = aggregate(second_half)

        # -------------------- output grafici ------------------------------
        out_dir = os.path.join(out_root, tag)
        os.makedirs(out_dir, exist_ok=True)

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

        # -------------------- results.txt ---------------------------------
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
            prev_max={c:None for c in ["S0","S1","S2"]}
            prev_min={c:None for c in ["S0","S1","S2"]}
            for s in full_steps:
                name=os.path.basename(s["file"])
                f.write("-"*60+"\n")
                f.write(f"file               : {name}\n")
                f.write(f"duration           : {s['duration']}\n")
                for c in ["S0","S1","S2"]:
                    f.write(f"{c} range          : {s['ranges'][c]}\n")
                f.write(f"Az/|A| mean std     : {s['ratio_mean']} {s['ratio_std']}\n")
                f.write(f"Orient mean std     : {s['orient_mean']} {s['orient_std']}\n")
                for c in ["S0","S1","S2"]:
                    cmx,cmn=s["max_t"][c],s["min_t"][c]
                    f.write(f"{c} max timestamp     : {cmx}\n")
                    f.write(f"{c} min timestamp     : {cmn}\n")
                    if prev_max[c] is not None:
                        f.write(f"{c} dist max to prev  : {cmx-prev_max[c]}\n")
                        f.write(f"{c} dist min to prev  : {cmn-prev_min[c]}\n")
                    prev_max[c],prev_min[c]=cmx,cmn
                f.write("\n")
        print("created", res_path)

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
