#!/usr/bin/env python3
# gait_analysis_gui.py
#
# GUI in italiano per l’analisi dei passi:
#   – tutti i calcoli originali (range pressioni, Az/|A|, durate, distanze picchi)
#   – filtro Madgwick opzionale per l’orientazione
#   – generazione PNG opzionale
#   – visualizzazione integrata dei risultati
#
# ---------------------------------------------------------------------------
# Dipendenze:
#   pip install PyQt5 ahrs matplotlib pandas numpy
# ---------------------------------------------------------------------------

import os, sys, glob, re, traceback
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                    # backend non interattivo (PNG)
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

# -------------------------- Costanti globali ----------------------------
IGNORE_EDGE_STEPS = True
EDGE_FULL_SKIP    = 1
EDGE_HALF_SKIP    = 2
RESAMPLE_POINTS   = 100
BETA_MADGWICK     = 0.1

_rx_full = re.compile(r"Passo[_ ]?(\d+)\.csv$",      re.I)
_rx_half = re.compile(r"Passo[_ ]?(\d+)\.(1|2)\.csv$", re.I)

# ------------------------------ Utility ---------------------------------
def resample(series, n=RESAMPLE_POINTS):
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, series)

# --------------------- Orientazione (Madgwick) --------------------------
def madgwick_orientation(df, dt):
    """Angolo (°) fra asse Y sensore e verticale globale."""
    gyr = np.deg2rad(df[["Gx", "Gy", "Gz"]].values)
    acc = df[["Ax", "Ay", "Az"]].values
    mag = df[["Mx", "My", "Mz"]].values

    N = len(df)
    q = np.zeros((N, 4));  q[0] = np.array([1, 0, 0, 0])
    mad = Madgwick(sampleperiod=dt, beta=BETA_MADGWICK)
    for t in range(1, N):
        q[t] = mad.updateMARG(q[t-1], gyr[t], acc[t], mag[t])

    cos_th = np.clip(1.0 - 2.0*(q[:,1]**2 + q[:,3]**2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))

# ----------------- Analisi di un singolo CSV ----------------------------
def process_single_step(path, baseline, weight, do_ori):
    df = pd.read_csv(path)

    # baseline pressioni
    for i, c in enumerate(["S0", "S1", "S2"]):
        df[c] -= baseline[i]

    duration = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    ranges   = {c: (df[c].max() - df[c].min()) / weight for c in ["S0","S1","S2"]}

    # rapporto Az / |A|
    a_norm = np.linalg.norm(df[["Ax","Ay","Az"]].values, axis=1)
    ratio  = np.where(a_norm == 0, 0, df["Az"] / a_norm)
    ratio_res = resample(ratio)

    # orientazione opzionale
    if do_ori:
        dt      = np.diff(df["Timestamp"]).mean() if len(df) > 1 else 0.01
        orient  = madgwick_orientation(df, dt)
        orient_res  = resample(orient)
        orient_mean = orient.mean();  orient_std = orient.std()
    else:
        orient_res  = np.zeros_like(ratio_res)
        orient_mean = orient_std = 0.0

    # timestamp picchi pressioni
    max_t = {c: df.loc[df[c].idxmax(), "Timestamp"] for c in ["S0","S1","S2"]}
    min_t = {c: df.loc[df[c].idxmin(), "Timestamp"] for c in ["S0","S1","S2"]}

    return dict(
        file=path, n=len(df), duration=duration,
        ranges=ranges,
        ratio_mean=ratio.mean(), ratio_std=ratio.std(), ratio_res=ratio_res,
        orient_mean=orient_mean, orient_std=orient_std, orient_res=orient_res,
        max_t=max_t, min_t=min_t
    )

# ---------------------- Caricamento file -------------------------------
def _key_full(p):
    m = _rx_full.search(os.path.basename(p))
    return int(m.group(1)) if m else 0

def load_full_steps(folder, base, w, do_ori):
    files = sorted(glob.glob(os.path.join(folder,"*.csv")), key=_key_full)
    if IGNORE_EDGE_STEPS and len(files) > 2*EDGE_FULL_SKIP:
        files = files[EDGE_FULL_SKIP:-EDGE_FULL_SKIP]
    return [process_single_step(p, base, w, do_ori) for p in files]

def load_half_steps(folder, base, w, do_ori):
    pairs={}
    for p in glob.glob(os.path.join(folder,"*.csv")):
        m = _rx_half.search(os.path.basename(p))
        if m: pairs.setdefault(int(m.group(1)),{})[int(m.group(2))]=p
    nums=sorted(pairs)
    if IGNORE_EDGE_STEPS and len(nums) > 2*EDGE_HALF_SKIP:
        nums = nums[EDGE_HALF_SKIP:-EDGE_HALF_SKIP]

    first, second = [], []
    for n in nums:
        if 1 in pairs[n]:
            first.append(process_single_step(pairs[n][1], base, w, do_ori))
        if 2 in pairs[n]:
            second.append(process_single_step(pairs[n][2], base, w, do_ori))
    return first, second

# ---------------------- Aggregazioni globali ---------------------------
def aggregate(steps, do_ori):
    if not steps: return {}
    n  = np.array([s["n"] for s in steps])
    rm = np.array([s["ratio_mean"]  for s in steps])
    ratio_global = float((rm*n).sum()/n.sum())

    ratio_mat = np.vstack([s["ratio_res"] for s in steps])
    ranges = {c: np.array([s["ranges"][c] for s in steps]) for c in ["S0","S1","S2"]}

    if do_ori:
        om = np.array([s["orient_mean"] for s in steps])
        orient_mat = np.vstack([s["orient_res"] for s in steps])
    else:
        om = np.zeros_like(rm)
        orient_mat = np.zeros_like(ratio_mat)

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

# ---------------------- Plot helper ------------------------------------
def plot_mean_std(x, m, s, out, ylab, title):
    plt.figure()
    plt.plot(x, m, label="media")
    plt.fill_between(x, m-s, m+s, alpha=.3, label="dev std")
    plt.xlabel("tempo normalizzato (%)"); plt.ylabel(ylab); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_concat(lst, out, ylab, title):
    concat=np.concatenate(lst); seg=len(lst[0])
    plt.figure(); plt.plot(concat)
    for i in range(1,len(lst)):
        plt.axvline(i*seg, linestyle="--", linewidth=.5)
    plt.xlabel("campioni concatenati"); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(out); plt.close()

# ----------------------------- Thread worker ---------------------------
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class AnalysisWorker(QObject):
    progress = pyqtSignal(str)      # log
    done     = pyqtSignal(str)      # root path al termine

    def __init__(self, root, weight, base_r, base_l, do_plots, do_ori):
        super().__init__()
        self.root     = root
        self.weight   = weight
        self.base_r   = base_r
        self.base_l   = base_l
        self.do_plots = do_plots
        self.do_ori   = do_ori

    # slot principale
    def run(self):
        try:
            self._analyse()
        except Exception as e:
            self.progress.emit(f"*** ERRORE ***\n{e}\n{traceback.format_exc()}")
        self.done.emit(self.root)

    # ------------------ log helper -------------------------------------
    def _log(self, msg): self.progress.emit(msg)

    # ------------------ analisi vera e propria -------------------------
    def _analyse(self):
        self._log("Analisi avviata…\n")
        feet = {"Piede_Destro":("right", self.base_r),
                "Piede_Sinistro":("left",  self.base_l)}
        out_root = os.path.join(self.root, "calcoli")
        os.makedirs(out_root, exist_ok=True)
        x_norm = np.linspace(0, 100, RESAMPLE_POINTS)

        for fd,(tag,base) in feet.items():
            self._log(f"\n▶ {fd} ({tag})")
            fp = os.path.join(self.root,"passi",fd)
            if not os.path.isdir(fp):
                self._log("   Cartella mancante, salto.")
                continue

            out_dir = os.path.join(out_root, tag)
            os.makedirs(out_dir, exist_ok=True)

            full_steps   = load_full_steps (os.path.join(fp,"Passi_Interi"),
                                            base, self.weight, self.do_ori)
            first_half, second_half = load_half_steps(os.path.join(fp,"Mezzi_Passi"),
                                            base, self.weight, self.do_ori)

            agg_full   = aggregate(full_steps,   self.do_ori)
            agg_first  = aggregate(first_half,   self.do_ori)
            agg_second = aggregate(second_half,  self.do_ori)

            # -------- grafici -------------------------------------------
            if self.do_plots and agg_full:
                plot_mean_std(x_norm, agg_full["ratio_curve_mean"],
                              agg_full["ratio_curve_std"],
                              os.path.join(out_dir,"ratio_mean_std.png"),
                              "Az/|A|", "Az/|A| media ± std")

                if self.do_ori:
                    plot_mean_std(x_norm, agg_full["orient_curve_mean"],
                                  agg_full["orient_curve_std"],
                                  os.path.join(out_dir,"orient_mean_std.png"),
                                  "orientazione °", "Orientazione media ± std")
                    plot_concat([s["orient_res"] for s in full_steps],
                                os.path.join(out_dir,"orient_concat.png"),
                                "orientazione °", "Orientazione concatenata")

            # -------- results.txt ---------------------------------------
            res_path = os.path.join(out_dir,"results.txt")
            with open(res_path, "w", encoding="utf-8") as f:
                f.write(f"RISULTATI {tag.upper()}\n\nFULL STEPS\n")
                if agg_full:
                    f.write(f"passi analizzati          : {agg_full['count']}\n")
                    f.write(f"durata media ± std        : "
                            f"{agg_full['dur_mean']:.3f}  {agg_full['dur_std']:.3f}\n")
                    for c in ["S0","S1","S2"]:
                        f.write(f"{c} range media ± std       : "
                                f"{agg_full['ranges_mean'][c]:.3f}  "
                                f"{agg_full['ranges_std'][c]:.3f}\n")
                    f.write(f"Az/|A| media delle medie  : "
                            f"{agg_full['ratio_mean_of_means']:.4f}  "
                            f"{agg_full['ratio_mean_std']:.4f}\n")
                    f.write(f"Az/|A| media globale      : {agg_full['ratio_mean_all']:.4f}\n")
                    if self.do_ori:
                        f.write(f"Orientazione media ± std  : "
                                f"{agg_full['orient_mean_of_means']:.3f}  "
                                f"{agg_full['orient_mean_std']:.3f}\n")
                    for c in ["S0","S1","S2"]:
                        dmax = agg_full["dist"][f"{c}_max"]; dmin = agg_full["dist"][f"{c}_min"]
                        if len(dmax):
                            f.write(f"Δ max {c} media ± std     : {dmax.mean():.3f} {dmax.std():.3f}\n")
                            f.write(f"Δ min {c} media ± std     : {dmin.mean():.3f} {dmin.std():.3f}\n")
                else:
                    f.write("nessun dato\n")

                f.write("\nFIRST HALF STEPS (.1)\n")
                if agg_first:
                    f.write(f"conteggio                 : {agg_first['count']}\n")
                    f.write(f"Az/|A| media ± std        : "
                            f"{agg_first['ratio_mean_of_means']:.4f}  "
                            f"{agg_first['ratio_mean_std']:.4f}\n")
                else:
                    f.write("nessun dato\n")

                f.write("\nSECOND HALF STEPS (.2)\n")
                if agg_second:
                    f.write(f"conteggio                 : {agg_second['count']}\n")
                    f.write(f"Az/|A| media ± std        : "
                            f"{agg_second['ratio_mean_of_means']:.4f}  "
                            f"{agg_second['ratio_mean_std']:.4f}\n")
                else:
                    f.write("nessun dato\n")

            self._log(f"   Creato: {res_path}")
        self._log("\nAnalisi completata.")

# ------------------------------- GUI -----------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QAction, QTabWidget, QGroupBox,
    QGridLayout, QDoubleSpinBox, QCheckBox, QPlainTextEdit, QMessageBox,
    QScrollArea, QFormLayout, QFrame
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ResultsWidget(QWidget):
    """Mostra risultati testuali + PNG generati."""
    def __init__(self):
        super().__init__()
        vbox = QVBoxLayout(self)
        self.text = QPlainTextEdit(); self.text.setReadOnly(True)
        vbox.addWidget(self.text, 2)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.img_container = QWidget(); self.img_layout = QVBoxLayout(self.img_container)
        self.scroll.setWidget(self.img_container)
        vbox.addWidget(self.scroll, 3)

    def clear(self):
        self.text.clear()
        while self.img_layout.count():
            w = self.img_layout.takeAt(0).widget()
            if w: w.deleteLater()

    def load_from_root(self, root):
        self.clear()
        calc_root = os.path.join(root, "calcoli")
        if not os.path.isdir(calc_root):
            self.text.setPlainText("Nessun risultato trovato.")
            return

        # text
        blocks=[]
        for foot in ("right","left"):
            txt_path = os.path.join(calc_root, foot, "results.txt")
            if os.path.isfile(txt_path):
                with open(txt_path) as f: blocks.append(f.read())
        self.text.setPlainText("\n\n".join(blocks) if blocks else "Nessun results.txt.")

        # images
        pngs=[]
        for sub, _, files in os.walk(calc_root):
            pngs += [os.path.join(sub,f) for f in files if f.lower().endswith(".png")]
        for p in sorted(pngs):
            lab = QLabel(alignment=Qt.AlignCenter)
            lab.setPixmap(QPixmap(p).scaledToWidth(600, Qt.SmoothTransformation))
            cap = QLabel(os.path.basename(p), alignment=Qt.AlignCenter)
            cap.setStyleSheet("font-weight:bold; margin:4px")
            self.img_layout.addWidget(cap); self.img_layout.addWidget(lab)

        self.img_layout.addStretch(1)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis – GUI completa")

        self._thread  = None
        self._worker  = None
        self.cur_root = None

        # ------------------ Tab widget ----------------------------------
        self.tabs = QTabWidget()
        self.tab_analysis = QWidget(); self.tabs.addTab(self.tab_analysis, "Analisi")
        self.tab_results  = ResultsWidget(); self.tabs.addTab(self.tab_results, "Risultati")
        self.setCentralWidget(self.tabs)

        # ----------- Contenuto scheda Analisi --------------------------
        vbox = QVBoxLayout(self.tab_analysis)

        # Path
        hpath = QHBoxLayout()
        self.ed_path = QLineEdit(); self.ed_path.setReadOnly(True)
        hpath.addWidget(QLabel("Cartella principale:")); hpath.addWidget(self.ed_path,1)
        bt_path = QPushButton("Sfoglia…"); bt_path.clicked.connect(self.choose_folder)
        hpath.addWidget(bt_path)
        vbox.addLayout(hpath)

        # Parametri
        grp = QGroupBox("Parametri")
        grid = QGridLayout(grp)
        self.sp_weight = QDoubleSpinBox(); self.sp_weight.setSuffix(" kg")
        self.sp_weight.setRange(1, 300); self.sp_weight.setValue(70)

        self.base_spin={}
        for s in ("Dx","Sx"):
            for i in range(3):
                key=f"{s}{i}"
                sp=QDoubleSpinBox(); sp.setDecimals(3); sp.setRange(-1000,1000)
                self.base_spin[key]=sp

        grid.addWidget(QLabel("Peso"),0,0); grid.addWidget(self.sp_weight,0,1)

        grid.addWidget(QLabel("Baseline destro S0/1/2"),1,0)
        grid.addWidget(self.base_spin["Dx0"],1,1)
        grid.addWidget(self.base_spin["Dx1"],1,2)
        grid.addWidget(self.base_spin["Dx2"],1,3)

        grid.addWidget(QLabel("Baseline sinistro S0/1/2"),2,0)
        grid.addWidget(self.base_spin["Sx0"],2,1)
        grid.addWidget(self.base_spin["Sx1"],2,2)
        grid.addWidget(self.base_spin["Sx2"],2,3)

        vbox.addWidget(grp)

        # Opzioni
        grp2 = QGroupBox("Opzioni")
        hopt=QHBoxLayout(grp2)
        self.cb_plots = QCheckBox("Genera grafici"); self.cb_plots.setChecked(True)
        self.cb_ori   = QCheckBox("Calcola orientazione"); self.cb_ori.setChecked(True)
        hopt.addWidget(self.cb_plots); hopt.addWidget(self.cb_ori); hopt.addStretch(1)
        vbox.addWidget(grp2)

        # Bottone esecuzione
        self.bt_run = QPushButton("Esegui analisi"); self.bt_run.setStyleSheet("font-weight:bold")
        self.bt_run.clicked.connect(self.start_analysis)
        vbox.addWidget(self.bt_run)

        # Log
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        vbox.addWidget(self.log,1)

        # ------------------ Menù ---------------------------------------
        men = self.menuBar().addMenu("&File")
        act_open = QAction("Apri cartella…",self); act_open.triggered.connect(self.choose_folder)
        men.addAction(act_open)
        men.addSeparator()
        men.addAction("Esci", self.close)

    # ------------------- Slot: scelta cartella -------------------------
    def choose_folder(self):
        p = QFileDialog.getExistingDirectory(self,"Seleziona cartella principale")
        if p: self.ed_path.setText(p)

    # ------------------- Avvio analisi --------------------------------
    def start_analysis(self):
        root = self.ed_path.text().strip()
        if not root:
            QMessageBox.warning(self,"Attenzione","Seleziona una cartella valida.")
            return
        self.cur_root=root

        weight = self.sp_weight.value()
        base_r = tuple(self.base_spin[f"Dx{i}"].value() for i in range(3))
        base_l = tuple(self.base_spin[f"Sx{i}"].value() for i in range(3))
        do_plots = self.cb_plots.isChecked()
        do_ori   = self.cb_ori.isChecked()

        # disabilita UI
        self.setEnabled(False)
        self.log.clear()

        # thread
        self._thread = QThread()
        self._worker = AnalysisWorker(root, weight, base_r, base_l, do_plots, do_ori)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.appendPlainText)
        self._worker.done.connect(self.analysis_finished)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    # ------------------- Fine analisi ----------------------------------
    def analysis_finished(self, root):
        self.setEnabled(True)
        QMessageBox.information(self,"Completato","Analisi terminata.")
        # carica risultati nella seconda scheda
        self.tab_results.load_from_root(root)
        self.tabs.setCurrentWidget(self.tab_results)

# ------------------------------- MAIN -----------------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(900,700); win.show()
    sys.exit(app.exec())

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
