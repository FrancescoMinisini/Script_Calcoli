#!/usr/bin/env python3
# gait_analysis_gui.py
#
# GUI completa in italiano per l’analisi dei passi.
# Tutti i calcoli sono identici allo script CLI “gait_analysis_madgwick_all.py”.
#
# Dipendenze:
#   pip install PyQt5 ahrs matplotlib pandas numpy
# -------------------------------------------------------------------------

import os, sys, glob, re, traceback
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")               # backend non interattivo (PNG)
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

# ------------------------ Costanti identiche al CLI ---------------------
IGNORE_EDGE_STEPS = True
EDGE_FULL_SKIP    = 1
EDGE_HALF_SKIP    = 2
RESAMPLE_POINTS   = 100
BETA_MADGWICK     = 0.4             # come nello script CLI

_rx_full = re.compile(r"Passo[_ ]?(\d+)\.csv$",      re.I)
_rx_half = re.compile(r"Passo[_ ]?(\d+)\.(1|2)\.csv$", re.I)

# ------------------------- Funzioni di utilità --------------------------
def resample(series, n=RESAMPLE_POINTS):
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, series)

def madgwick_session(df, dt, beta=BETA_MADGWICK):
    """Calcola l’orientazione dell’intera sessione (angolo Y vs verticale)."""
    gyr = np.deg2rad(df[["Gx","Gy","Gz"]].values)
    acc = df[["Ax","Ay","Az"]].values
    mag = df[["Mx","My","Mz"]].values
    N   = len(df)

    mad = Madgwick(sampleperiod=dt, beta=beta)
    q_prev = np.array([1.0, 0.0, 0.0, 0.0])
    quats  = np.zeros((N,4))

    for t in range(N):
        q_prev = mad.updateMARG(q_prev, gyr[t], acc[t], mag[t])
        if np.dot(q_prev, q_prev) == 0:              # fallback improbabile
            q_prev = np.array([1.0,0.0,0.0,0.0])
        quats[t] = q_prev

    cos_t = np.clip(1.0 - 2.0*(quats[:,1]**2 + quats[:,3]**2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))

def step_metrics(df_slice, orient, ratio, weight):
    """Calcoli per un singolo passo o mezzo passo."""
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
        dist[f"{c}_max"]=np.diff(mx) if len(mx)>1 else np.array([])
        dist[f"{c}_min"]=np.diff(mn) if len(mn)>1 else np.array([])

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

# ------------------------ Mezzi passi (solo pressioni) -------------------
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
                ln=np.linalg.norm(df[["Ax","Ay","Az"]].values, axis=1)
                ratio=np.where(ln==0,0,df["Az"]/ln)
                orient=np.zeros(len(df))              # orientazione non calcolata
                dest.append(step_metrics(df, orient, ratio, weight))
    return first, second

# ----------------------------- Worker thread -----------------------------
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class AnalysisWorker(QObject):
    log    = pyqtSignal(str)
    done   = pyqtSignal(str)

    def __init__(self, root, weight, base_r, base_l,
                 gen_plots: bool, calc_ori: bool):
        super().__init__()
        self.root, self.w = root, weight
        self.base_r, self.base_l = base_r, base_l
        self.gen_plots, self.calc_ori = gen_plots, calc_ori

    def _log(self, m): self.log.emit(m)

    def run(self):
        try:
            self._analyse()
        except Exception:
            self.log.emit("*** ERRORE ***\n"+traceback.format_exc())
        self.done.emit(self.root)

    # ------------------- core analisi (identico al CLI) ------------------
    def _analyse(self):
        self._log("Analisi avviata…")
        feet = {"Piede_Destro":("right",self.base_r),
                "Piede_Sinistro":("left", self.base_l)}
        out_root = os.path.join(self.root,"calcoli")
        os.makedirs(out_root, exist_ok=True)
        x_norm = np.linspace(0,100,RESAMPLE_POINTS)

        for fd,(tag,base) in feet.items():
            self._log(f"\n▶ {fd} ({tag})")
            fp = os.path.join(self.root,"passi",fd)
            if not os.path.isdir(fp):
                self._log("   Cartella mancante, salto.")
                continue

            # ---------------- full steps ---------------------------------
            files = sorted(glob.glob(os.path.join(fp,"Passi_Interi","*.csv")),
                           key=lambda p:int(_rx_full.search(os.path.basename(p)).group(1)))
            if IGNORE_EDGE_STEPS and len(files)>2*EDGE_FULL_SKIP:
                files = files[EDGE_FULL_SKIP:-EDGE_FULL_SKIP]
            if not files:
                self._log("   Nessun passo intero, salto.")
                continue

            df_all, slices = [], []
            start=0
            for p in files:
                d=pd.read_csv(p); df_all.append(d)
                end=start+len(d); slices.append((start,end,p)); start=end
            df_all=pd.concat(df_all, ignore_index=True)

            for i,c in enumerate(["S0","S1","S2"]): df_all[c]-=base[i]

            dt = np.diff(df_all["Timestamp"]).mean() if len(df_all)>1 else 0.01
            if self.calc_ori:
                orient_all = madgwick_session(df_all, dt, BETA_MADGWICK)
            else:
                orient_all = np.zeros(len(df_all))

            ln=np.linalg.norm(df_all[["Ax","Ay","Az"]].values, axis=1)
            ratio_all=np.where(ln==0,0,df_all["Az"]/ln)

            full_steps=[]
            for st,en,path in slices:
                m=step_metrics(df_all.iloc[st:en],
                               orient_all[st:en],
                               ratio_all [st:en],
                               self.w)
                m["file"]=path
                full_steps.append(m)

            # ---------------- half steps ---------------------------------
            first_half, second_half = load_half_steps(os.path.join(fp,"Mezzi_Passi"),
                                                      base, self.w)

            agg_full   = aggregate(full_steps)
            agg_first  = aggregate(first_half)
            agg_second = aggregate(second_half)

            # ---------------- output grafici -----------------------------
            out_dir = os.path.join(out_root,tag)
            os.makedirs(out_dir, exist_ok=True)

            if self.gen_plots and agg_full:
                plot_mean_std(x_norm, agg_full["ratio_curve_mean"], agg_full["ratio_curve_std"],
                              os.path.join(out_dir,"ratio_mean_std.png"),
                              "Az/|A|", "Az/|A| mean ± std")
                if self.calc_ori:
                    plot_mean_std(x_norm, agg_full["orient_curve_mean"], agg_full["orient_curve_std"],
                                  os.path.join(out_dir,"orient_mean_std.png"),
                                  "orientation deg", "Orientation mean ± std")
                    plot_concat([s["orient_res"] for s in full_steps],
                                os.path.join(out_dir,"orient_concat.png"),
                                "orientation deg", "Orientation concatenated")

            # ---------------- results.txt (UTF-8) ------------------------
            res = os.path.join(out_dir,"results.txt")
            with open(res,"w",encoding="utf-8") as f:
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
                        dmax=agg_full["dist"][f"{c}_max"]; dmin=agg_full["dist"][f"{c}_min"]
                        if len(dmax):
                            f.write(f"Δ max {c} mean std     : {dmax.mean()} {dmax.std()}\n")
                            f.write(f"Δ min {c} mean std     : {dmin.mean()} {dmin.std()}\n")
                else:
                    f.write("no data\n")

                f.write("\nFIRST HALF STEPS (.1)\n")
                if agg_first:
                    f.write(f"count                     : {agg_first['count']}\n")
                    f.write(f"Az/|A| mean of means std  : {agg_first['ratio_mean_of_means']} {agg_first['ratio_mean_std']}\n")
                else: f.write("no data\n")

                f.write("\nSECOND HALF STEPS (.2)\n")
                if agg_second:
                    f.write(f"count                     : {agg_second['count']}\n")
                    f.write(f"Az/|A| mean of means std  : {agg_second['ratio_mean_of_means']} {agg_second['ratio_mean_std']}\n")
                else: f.write("no data\n")

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
            self._log(f"   Creato: {res}")
        self._log("\nAnalisi completata.")

# ---------------------------- GUI PyQt5 ----------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QAction, QTabWidget, QGroupBox,
    QGridLayout, QDoubleSpinBox, QCheckBox, QPlainTextEdit, QMessageBox,
    QScrollArea
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ResultsTab(QWidget):
    def __init__(self): super().__init__(); self._build()

    def _build(self):
        v=QVBoxLayout(self)
        self.txt=QPlainTextEdit(readOnly=True); v.addWidget(self.txt,2)
        self.scroll=QScrollArea(widgetResizable=True); v.addWidget(self.scroll,3)
        self.img_holder=QWidget(); self.img_layout=QVBoxLayout(self.img_holder)
        self.scroll.setWidget(self.img_holder)

    def clear(self):
        self.txt.clear()
        while self.img_layout.count():
            w=self.img_layout.takeAt(0).widget()
            if w: w.deleteLater()

    def load(self, root):
        self.clear()
        calc=os.path.join(root,"calcoli")
        if not os.path.isdir(calc):
            self.txt.setPlainText("Nessun risultato.")
            return

        # testo
        blocks=[]
        for foot in ("right","left"):
            p=os.path.join(calc,foot,"results.txt")
            if os.path.isfile(p):
                with open(p,encoding="utf-8") as f: blocks.append(f.read())
        self.txt.setPlainText("\n\n".join(blocks) if blocks else "Nessun results.txt.")

        # immagini
        pngs=[]
        for sub,_,fs in os.walk(calc):
            pngs += [os.path.join(sub,f) for f in fs if f.lower().endswith(".png")]
        for p in sorted(pngs):
            cap=QLabel(os.path.basename(p),alignment=Qt.AlignCenter)
            cap.setStyleSheet("font-weight:bold;margin:4px")
            img=QLabel(alignment=Qt.AlignCenter)
            img.setPixmap(QPixmap(p).scaledToWidth(600,Qt.SmoothTransformation))
            self.img_layout.addWidget(cap); self.img_layout.addWidget(img)
        self.img_layout.addStretch(1)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__(); self.setWindowTitle("Gait Analysis – GUI completa")
        self._thread=self._worker=None; self.cur_root=None
        self._build_ui()

    def _build_ui(self):
        self.tabs=QTabWidget(); self.setCentralWidget(self.tabs)
        self.tab_a=QWidget();   self.tabs.addTab(self.tab_a,"Analisi")
        self.tab_r=ResultsTab();self.tabs.addTab(self.tab_r,"Risultati")

        v=QVBoxLayout(self.tab_a)

        # cartella
        hp=QHBoxLayout()
        self.ed_path=QLineEdit(readOnly=True)
        hp.addWidget(QLabel("Cartella principale:")); hp.addWidget(self.ed_path,1)
        bt=QPushButton("Sfoglia…"); bt.clicked.connect(self.choose_folder)
        hp.addWidget(bt); v.addLayout(hp)

        # parametri
        grp=QGroupBox("Parametri"); g=QGridLayout(grp)
        self.sp_w=QDoubleSpinBox(suffix=" kg",minimum=1,maximum=300,value=70)
        self.bas={}
        for s in ("Dx","Sx"):
            for i in range(3):
                k=f"{s}{i}"
                self.bas[k]=QDoubleSpinBox(decimals=3,minimum=-1000,maximum=1000)
        g.addWidget(QLabel("Peso"),0,0); g.addWidget(self.sp_w,0,1)
        g.addWidget(QLabel("Baseline destro S0/1/2"),1,0)
        g.addWidget(self.bas["Dx0"],1,1); g.addWidget(self.bas["Dx1"],1,2); g.addWidget(self.bas["Dx2"],1,3)
        g.addWidget(QLabel("Baseline sinistro S0/1/2"),2,0)
        g.addWidget(self.bas["Sx0"],2,1); g.addWidget(self.bas["Sx1"],2,2); g.addWidget(self.bas["Sx2"],2,3)
        v.addWidget(grp)

        # opzioni
        grp2=QGroupBox("Opzioni"); ho=QHBoxLayout(grp2)
        self.cb_plot=QCheckBox("Genera grafici",checked=True)
        self.cb_ori =QCheckBox("Calcola orientazione",checked=True)
        ho.addWidget(self.cb_plot); ho.addWidget(self.cb_ori); ho.addStretch(1)
        v.addWidget(grp2)

        # run
        self.bt_run=QPushButton("Esegui analisi",styleSheet="font-weight:bold")
        self.bt_run.clicked.connect(self.start)
        v.addWidget(self.bt_run)

        # log
        self.log=QPlainTextEdit(readOnly=True); v.addWidget(self.log,1)

        # menù
        m=self.menuBar().addMenu("&File")
        act_open=QAction("Apri cartella…",self); act_open.triggered.connect(self.choose_folder)
        m.addAction(act_open); m.addSeparator(); m.addAction("Esci",self.close)

    # ---------- funzioni UI ---------------------------------------------
    def choose_folder(self):
        p=QFileDialog.getExistingDirectory(self,"Cartella principale")
        if p: self.ed_path.setText(p)

    def start(self):
        root=self.ed_path.text().strip()
        if not root:
            QMessageBox.warning(self,"Attenzione","Seleziona una cartella valida.")
            return
        self.cur_root=root
        w=self.sp_w.value()
        br=tuple(self.bas[f"Dx{i}"].value() for i in range(3))
        bl=tuple(self.bas[f"Sx{i}"].value() for i in range(3))
        gen=self.cb_plot.isChecked()
        ori=self.cb_ori.isChecked()

        self.setEnabled(False); self.log.clear()
        self._thread=QThread(); self._worker=AnalysisWorker(root,w,br,bl,gen,ori)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.done.connect(self.finished)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def finished(self,root):
        self.setEnabled(True)
        QMessageBox.information(self,"Completato","Analisi terminata.")
        self.tab_r.load(root)
        self.tabs.setCurrentWidget(self.tab_r)

# ------------------------------- main ------------------------------------
def main():
    app=QApplication(sys.argv)
    win=MainWindow(); win.resize(920,720); win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
