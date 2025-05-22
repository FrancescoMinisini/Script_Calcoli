#!/usr/bin/env python3
# gait_analysis.py   –   April 2025
# ---------------------------------------------------------------
# CONFIG
IGNORE_EDGE_STEPS = True   # False -> include all files
EDGE_FULL_SKIP    = 1      # full steps skipped head/tail
EDGE_HALF_SKIP    = 2      # half-step pairs (.1/.2) skipped head/tail
RESAMPLE_POINTS   = 100    # points for normalised curves
# ---------------------------------------------------------------

import os, glob, re, numpy as np, pandas as pd, matplotlib.pyplot as plt

_num_full = re.compile(r"Passo[_ ]?(\d+)\.csv$", re.I)
_num_half = re.compile(r"Passo[_ ]?(\d+)\.(1|2)\.csv$", re.I)

# ------------------ helpers -----------------------------------
def resample(series, n=RESAMPLE_POINTS):
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, series)

def process_single_step(csv_file, baseline, weight):
    df = pd.read_csv(csv_file)

    for i, col in enumerate(["S0","S1","S2"]):
        df[col] -= baseline[i]

    duration = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    ranges   = {c:(df[c].max()-df[c].min())/weight for c in ["S0","S1","S2"]}

    a_norm = np.sqrt(df["Ax"]**2+df["Ay"]**2+df["Az"]**2)
    ratio  = np.where(a_norm==0, 0, df["Az"]/a_norm)
    ratio_res = resample(ratio)

    m_norm = np.sqrt(df["Mx"]**2+df["My"]**2+df["Mz"]**2)
    angle  = np.degrees(np.arccos(np.clip(df["My"]/m_norm, -1.0, 1.0)))
    angle_res = resample(angle)

    max_t = {c: df.loc[df[c].idxmax(),"Timestamp"] for c in ["S0","S1","S2"]}
    min_t = {c: df.loc[df[c].idxmin(),"Timestamp"] for c in ["S0","S1","S2"]}

    return dict(file=csv_file,
                n_samples=len(df),
                duration=duration,
                ranges=ranges,
                ratio_mean=ratio.mean(),
                ratio_std=ratio.std(),
                ratio_res=ratio_res,
                angle_mean=angle.mean(),
                angle_std=angle.std(),
                angle_res=angle_res,
                max_t=max_t,
                min_t=min_t)

def _full_key(p):  m=_num_full.search(os.path.basename(p)); return int(m.group(1)) if m else 0
def load_full_steps(folder, base, w):
    paths = sorted(glob.glob(os.path.join(folder,"*.csv")), key=_full_key)
    if IGNORE_EDGE_STEPS and len(paths)>2*EDGE_FULL_SKIP:
        paths = paths[EDGE_FULL_SKIP:-EDGE_FULL_SKIP]
    return [process_single_step(p, base, w) for p in paths]

def load_half_steps(folder, base, w):
    pairs={}
    for p in glob.glob(os.path.join(folder,"*.csv")):
        m=_num_half.search(os.path.basename(p))
        if m: pairs.setdefault(int(m.group(1)),{})[int(m.group(2))]=p
    nums=sorted(pairs)
    if IGNORE_EDGE_STEPS and len(nums)>2*EDGE_HALF_SKIP:
        nums = nums[EDGE_HALF_SKIP:-EDGE_HALF_SKIP]
    first,second=[],[]
    for n in nums:
        if 1 in pairs[n]: first .append(process_single_step(pairs[n][1], base, w))
        if 2 in pairs[n]: second.append(process_single_step(pairs[n][2], base, w))
    return first,second

def aggregate(steps):
    if not steps: return {}
    n_samp = np.array([s["n_samples"] for s in steps])
    ratio_m= np.array([s["ratio_mean"] for s in steps])
    ratio_global = (ratio_m * n_samp).sum() / n_samp.sum()

    durations = np.array([s["duration"] for s in steps])
    angle_m   = np.array([s["angle_mean"] for s in steps])
    ranges = {c: np.array([s["ranges"][c] for s in steps]) for c in ["S0","S1","S2"]}
    ratio_mat = np.vstack([s["ratio_res"] for s in steps])
    angle_mat = np.vstack([s["angle_res"] for s in steps])

    dist={}
    for c in ["S0","S1","S2"]:
        mx=np.array([s["max_t"][c] for s in steps])
        mn=np.array([s["min_t"][c] for s in steps])
        dist[f"{c}_max"]=np.diff(mx) if len(mx)>1 else np.array([])
        dist[f"{c}_min"]=np.diff(mn) if len(mn)>1 else np.array([])

    return dict(
        n_steps=len(steps),
        dur_mean=durations.mean(), dur_std=durations.std(),
        ratio_mean_of_means=ratio_m.mean(), ratio_mean_std=ratio_m.std(),
        ratio_mean_all_samples=ratio_global,
        angle_mean=angle_m.mean(), angle_mean_std=angle_m.std(),
        ranges_mean={c:ranges[c].mean() for c in ranges},
        ranges_std={c:ranges[c].std() for c in ranges},
        ratio_curve_m=ratio_mat.mean(0), ratio_curve_s=ratio_mat.std(0),
        angle_curve_m=angle_mat.mean(0), angle_curve_s=angle_mat.std(0),
        dist=dist)

def plot_mean_std(x, m, s, out, ylab, title):
    plt.figure(); plt.plot(x,m,label="mean")
    plt.fill_between(x,m-s,m+s,alpha=.3,label="std")
    plt.xlabel("time norm (%)"); plt.ylabel(ylab); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_concat(lst, out, ylab, title):
    c=np.concatenate(lst); seg=len(lst[0])
    plt.figure(); plt.plot(c)
    for i in range(1,len(lst)): plt.axvline(i*seg,color="gray",ls="--",lw=.5)
    plt.xlabel("concatenated samples"); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(out); plt.close()

# ------------------------------- main --------------------------------------
def main():
    root=input("Main folder path: ").strip()
    weight=float(input("Weight kg: ").strip())
    print("\nBaselines RIGHT"); base_r=tuple(float(input(f"S{i}: ").strip()) for i in range(3))
    print("\nBaselines LEFT");  base_l=tuple(float(input(f"S{i}: ").strip()) for i in range(3))

    feet={"Piede_Destro":("right",base_r),"Piede_Sinistro":("left",base_l)}
    out_root=os.path.join(root,"calcoli"); os.makedirs(out_root,exist_ok=True)
    x_norm=np.linspace(0,100,RESAMPLE_POINTS)

    for fd,(tag,base) in feet.items():
        path=os.path.join(root,"passi",fd)
        if not os.path.isdir(path): print("missing",path); continue
        out=os.path.join(out_root,tag); os.makedirs(out,exist_ok=True)

        full=load_full_steps(os.path.join(path,"Passi_Interi"),base,weight)
        first,second=load_half_steps(os.path.join(path,"Mezzi_Passi"),base,weight)

        agg_full  =aggregate(full)
        agg_first =aggregate(first)
        agg_second=aggregate(second)

        if agg_full:
            plot_mean_std(x_norm,agg_full["ratio_curve_m"],agg_full["ratio_curve_s"],
                          os.path.join(out,"ratio_mean_std.png"),
                          "Az/|A|","Az/|A| mean std")
            plot_mean_std(x_norm,agg_full["angle_curve_m"],agg_full["angle_curve_s"],
                          os.path.join(out,"angle_mean_std.png"),
                          "mag angle deg","mag angle mean std")
            plot_concat([s["angle_res"] for s in full],
                        os.path.join(out,"angle_concat.png"),
                        "mag angle deg","mag angle concatenated steps")

        txt=os.path.join(out,"results.txt")
        with open(txt,"w") as f:
            f.write(f"RESULTS {tag.upper()}\n\nFULL STEPS\n")
            if agg_full:
                f.write(f"steps analysed           : {agg_full['n_steps']}\n")
                f.write(f"duration mean std        : {agg_full['dur_mean']} {agg_full['dur_std']}\n")
                for c in ["S0","S1","S2"]:
                    f.write(f"{c} range mean std         : {agg_full['ranges_mean'][c]} {agg_full['ranges_std'][c]}\n")
                f.write(f"Az/|A| mean of means std : {agg_full['ratio_mean_of_means']} {agg_full['ratio_mean_std']}\n")
                f.write(f"Az/|A| mean all samples  : {agg_full['ratio_mean_all_samples']}\n")
                f.write(f"mag angle mean std       : {agg_full['angle_mean']} {agg_full['angle_mean_std']}\n")
                for c in ["S0","S1","S2"]:
                    dmax=agg_full["dist"][f"{c}_max"]; dmin=agg_full["dist"][f"{c}_min"]
                    if len(dmax):
                        f.write(f"distance max {c} mean std: {dmax.mean()} {dmax.std()}\n")
                        f.write(f"distance min {c} mean std: {dmin.mean()} {dmin.std()}\n")
            else: f.write("no data\n")
            f.write("\nFIRST HALF STEPS (.1)\n")
            if agg_first:
                f.write(f"count                    : {agg_first['n_steps']}\n")
                f.write(f"duration mean std        : {agg_first['dur_mean']} {agg_first['dur_std']}\n")
                f.write(f"Az/|A| mean of means std : {agg_first['ratio_mean_of_means']} {agg_first['ratio_mean_std']}\n")
            else: f.write("no data\n")
            f.write("\nSECOND HALF STEPS (.2)\n")
            if agg_second:
                f.write(f"count                    : {agg_second['n_steps']}\n")
                f.write(f"duration mean std        : {agg_second['dur_mean']} {agg_second['dur_std']}\n")
                f.write(f"Az/|A| mean of means std : {agg_second['ratio_mean_of_means']} {agg_second['ratio_mean_std']}\n")
            else: f.write("no data\n")
            f.write("\nSTEP BY STEP DETAILS\n")
            prev_max={c:None for c in ["S0","S1","S2"]}
            prev_min={c:None for c in ["S0","S1","S2"]}
            for s in full:
                name=os.path.basename(s["file"])
                f.write("-"*60+"\nfile                : "+name+"\n")
                f.write(f"duration            : {s['duration']}\n")
                for c in ["S0","S1","S2"]:
                    f.write(f"{c} range           : {s['ranges'][c]}\n")
                f.write(f"Az/|A| mean std     : {s['ratio_mean']} {s['ratio_std']}\n")
                f.write(f"mag angle mean std  : {s['angle_mean']} {s['angle_std']}\n")
                for c in ["S0","S1","S2"]:
                    cmx, cmn = s["max_t"][c], s["min_t"][c]
                    f.write(f"{c} max timestamp     : {cmx}\n")
                    f.write(f"{c} min timestamp     : {cmn}\n")
                    if prev_max[c] is not None:
                        f.write(f"{c} dist max to prev  : {cmx-prev_max[c]}\n")
                        f.write(f"{c} dist min to prev  : {cmn-prev_min[c]}\n")
                    prev_max[c],prev_min[c]=cmx,cmn
                f.write("\n")
        print("created",txt)

if __name__ == "__main__":
    main()
