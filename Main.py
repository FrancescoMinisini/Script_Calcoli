#!/usr/bin/env python3
# gait_analysis_gui.py
#
# GUI completa (PyQt5) per l’analisi dei passi con algoritmo originale
# e filtro Madgwick opzionale. Tutte le funzionalità precedenti sono
# disponibili senza uso del terminale.
#
# Dipendenze (una tantum):
#   pip install PyQt5 ahrs matplotlib pandas numpy
# ----------------------------------------------------------------------

import os, sys, glob, re, traceback
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")                 # backend non interattivo per i PNG
import matplotlib.pyplot as plt

from ahrs.filters import Madgwick

# ---------- Costanti configurabili -------------------------------------
IGNORE_EDGE_STEPS = True      # False → analizza tutti i file
EDGE_FULL_SKIP    = 1         # passi interi da saltare a testa/coda
EDGE_HALF_SKIP    = 2         # coppie .1/.2 da saltare a testa/coda
RESAMPLE_POINTS   = 100       # punti per curve tempo-normalizzate
BETA_MADGWICK     = 0.1       # coefficiente del filtro
# -----------------------------------------------------------------------

_rx_full = re.compile(r"Passo[_ ]?(\d+)\.csv$",      re.I)
_rx_half = re.compile(r"Passo[_ ]?(\d+)\.(1|2)\.csv$", re.I)

# ----------------------------- utility ----------------------------------
def resample(series: np.ndarray, n: int = RESAMPLE_POINTS) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, series)

# ----------------------- Madgwick orientation ---------------------------
def madgwick_orientation(df: pd.DataFrame, dt: float) -> np.ndarray:
    """Restituisce l’angolo (°) fra asse Y sensore e verticale globale."""
    gyr = np.deg2rad(df[["Gx", "Gy", "Gz"]].values)  # rad/s
    acc = df[["Ax", "Ay", "Az"]].values              # g
    mag = df[["Mx", "My", "Mz"]].values              # Gauss

    N = len(df)
    q = np.zeros((N, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])            # quaternione iniziale
    mad = Madgwick(sampleperiod=dt, beta=BETA_MADGWICK)

    for t in range(1, N):
        q[t] = mad.updateMARG(q[t - 1], gyr[t], acc[t], mag[t])

    cos_theta = np.clip(1.0 - 2.0 * (q[:, 1]**2 + q[:, 3]**2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ------------------- elabora singolo CSV --------------------------------
def process_single_step(path: str, baseline, weight: float,
                        compute_orientation: bool):
    df = pd.read_csv(path)

    # baseline sensori pressione
    for i, c in enumerate(["S0", "S1", "S2"]):
        df[c] -= baseline[i]

    duration = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    ranges   = {c: (df[c].max() - df[c].min()) / weight for c in ["S0", "S1", "S2"]}

    # rapporto Az / |A|
    a_norm = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
    ratio  = np.where(a_norm == 0, 0, df["Az"] / a_norm)
    ratio_res = resample(ratio)

    # orientazione (Madgwick) opzionale
    if compute_orientation:
        dt      = np.diff(df["Timestamp"]).mean() if len(df) > 1 else 0.01
        orient  = madgwick_orientation(df, dt)
        orient_res = resample(orient)
        orient_mean = orient.mean()
        orient_std  = orient.std()
    else:
        orient_res = np.zeros_like(ratio_res)
        orient_mean = orient_std = 0.0

    # timestamp picchi pressioni
    max_t = {c: df.loc[df[c].idxmax(), "Timestamp"] for c in ["S0", "S1", "S2"]}
    min_t = {c: df.loc[df[c].idxmin(), "Timestamp"] for c in ["S0", "S1", "S2"]}

    return dict(
        file=path,
        n=len(df),
        duration=duration,
        ranges=ranges,
        ratio_mean=ratio.mean(), ratio_std=ratio.std(), ratio_res=ratio_res,
        orient_mean=orient_mean, orient_std=orient_std, orient_res=orient_res,
        max_t=max_t, min_t=min_t
    )

# ---------------------- carica file ordinati ----------------------------
def _key_full(p): m = _rx_full.search(os.path.basename(p)); return int(m.group(1)) if m else 0
def load_full_steps(folder, base, w, compute_orientation):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")), key=_key_full)
    if IGNORE_EDGE_STEPS and len(files) > 2 * EDGE_FULL_SKIP:
        files = files[EDGE_FULL_SKIP:-EDGE_FULL_SKIP]
    return [process_single_step(p, base, w, compute_orientation) for p in files]

def load_half_steps(folder, base, w, compute_orientation):
    pairs = {}
    for p in glob.glob(os.path.join(folder, "*.csv")):
        m = _rx_half.search(os.path.basename(p))
        if m: pairs.setdefault(int(m.group(1)), {})[int(m.group(2))] = p
    nums = sorted(pairs)
    if IGNORE_EDGE_STEPS and len(nums) > 2 * EDGE_HALF_SKIP:
        nums = nums[EDGE_HALF_SKIP:-EDGE_HALF_SKIP]

    first, second = [], []
    for n in nums:
        if 1 in pairs[n]:
            first.append(process_single_step(pairs[n][1], base, w, compute_orientation))
        if 2 in pairs[n]:
            second.append(process_single_step(pairs[n][2], base, w, compute_orientation))
    return first, second

# ----------------------- aggregazione -----------------------------------
def aggregate(steps, compute_orientation):
    if not steps:
        return {}
    n  = np.array([s["n"] for s in steps])
    rm = np.array([s["ratio_mean"]  for s in steps])
    ratio_global = float((rm * n).sum() / n.sum())

    ratio_mat = np.vstack([s["ratio_res"] for s in steps])
    ranges = {c: np.array([s["ranges"][c] for s in steps]) for c in ["S0", "S1", "S2"]}

    if compute_orientation:
        om = np.array([s["orient_mean"] for s in steps])
        orient_mat = np.vstack([s["orient_res"] for s in steps])
    else:
        om = np.zeros_like(rm)
        orient_mat = np.zeros_like(ratio_mat)

    dist = {}
    for c in ["S0", "S1", "S2"]:
        max_ts = np.array([s["max_t"][c] for s in steps])
        min_ts = np.array([s["min_t"][c] for s in steps])
        dist[f"{c}_max"] = np.diff(max_ts) if len(max_ts) > 1 else np.array([])
        dist[f"{c}_min"] = np.diff(min_ts) if len(min_ts) > 1 else np.array([])

    return dict(
        count=len(steps),
        dur_mean=float(np.mean([s["duration"] for s in steps])),
        dur_std=float(np.std([s["duration"] for s in steps])),
        ratio_mean_of_means=float(rm.mean()), ratio_mean_std=float(rm.std()),
        ratio_mean_all=float(ratio_global),
        orient_mean_of_means=float(om.mean()), orient_mean_std=float(om.std()),
        ratio_curve_mean=ratio_mat.mean(0),  ratio_curve_std=ratio_mat.std(0),
        orient_curve_mean=orient_mat.mean(0), orient_curve_std=orient_mat.std(0),
        ranges_mean={c: float(ranges[c].mean()) for c in ranges},
        ranges_std={c: float(ranges[c].std()) for c in ranges},
        dist=dist
    )

# ----------------------------- plot helpers -----------------------------
def plot_mean_std(x, m, s, out, ylab, title):
    plt.figure()
    plt.plot(x, m, label="mean")
    plt.fill_between(x, m - s, m + s, alpha=.3, label="std")
    plt.xlabel("time norm (%)"); plt.ylabel(ylab); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

def plot_concat(lst, out, ylab, title):
    concat = np.concatenate(lst); seg = len(lst[0])
    plt.figure(); plt.plot(concat)
    for i in range(1, len(lst)):
        plt.axvline(i * seg, linestyle="--", linewidth=.5)
    plt.xlabel("concatenated samples"); plt.ylabel(ylab); plt.title(title)
    plt.tight_layout(); plt.savefig(out); plt.close()

# -------------------------- Worker (thread) -----------------------------
from PyQt5.QtCore import QObject, pyqtSignal, QThread

class AnalysisWorker(QObject):
    finished  = pyqtSignal()
    log       = pyqtSignal(str)

    def __init__(self, root, weight, base_r, base_l,
                 gen_plots: bool, compute_ori: bool):
        super().__init__()
        self.root = root
        self.weight = weight
        self.base_r = base_r
        self.base_l = base_l
        self.gen_plots = gen_plots
        self.compute_ori = compute_ori

    def run(self):
        try:
            self._analyse()
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"ERRORE: {e}\n{tb}")
        self.finished.emit()

    # --------------------------------------------------------------------
    def _analyse(self):
        self.log.emit("Analisi avviata…\n")
        feet = {"Piede_Destro": ("right", self.base_r),
                "Piede_Sinistro": ("left",  self.base_l)}
        out_root = os.path.join(self.root, "calcoli")
        os.makedirs(out_root, exist_ok=True)
        x_norm = np.linspace(0.0, 100.0, RESAMPLE_POINTS)

        for fd, (tag, base) in feet.items():
            self.log.emit(f"\n↳ {fd} ({tag})\n")
            fp = os.path.join(self.root, "passi", fd)
            if not os.path.isdir(fp):
                self.log.emit(f"  Cartella mancante: {fp}\n")
                continue

            out_dir = os.path.join(out_root, tag)
            os.makedirs(out_dir, exist_ok=True)

            full_steps = load_full_steps(os.path.join(fp, "Passi_Interi"),
                                         base, self.weight, self.compute_ori)
            first_half, second_half = load_half_steps(
                os.path.join(fp, "Mezzi_Passi"), base,
                self.weight, self.compute_ori)

            agg_full   = aggregate(full_steps, self.compute_ori)
            agg_first  = aggregate(first_half, self.compute_ori)
            agg_second = aggregate(second_half, self.compute_ori)

            # ------------------ grafici -----------------------------------
            if self.gen_plots and agg_full:
                plot_mean_std(x_norm, agg_full["ratio_curve_mean"],
                              agg_full["ratio_curve_std"],
                              os.path.join(out_dir, "ratio_mean_std.png"),
                              "Az/|A|", "Az/|A| mean ± std")

                if self.compute_ori:
                    plot_mean_std(x_norm, agg_full["orient_curve_mean"],
                                  agg_full["orient_curve_std"],
                                  os.path.join(out_dir, "orient_mean_std.png"),
                                  "orientation deg", "Orientation mean ± std")

                    plot_concat([s["orient_res"] for s in full_steps],
                                os.path.join(out_dir, "orient_concat.png"),
                                "orientation deg", "Orientation concatenated")

            # ------------------ results.txt ------------------------------
            res_path = os.path.join(out_dir, "results.txt")
            with open(res_path, "w") as f:
                f.write(f"RESULTS {tag.upper()}\n\nFULL STEPS\n")
                if agg_full:
                    f.write(f"steps analysed            : {agg_full['count']}\n")
                    f.write(f"duration mean std         : "
                            f"{agg_full['dur_mean']} {agg_full['dur_std']}\n")
                    for c in ["S0", "S1", "S2"]:
                        f.write(f"{c} range mean std          : "
                                f"{agg_full['ranges_mean'][c]} "
                                f"{agg_full['ranges_std'][c]}\n")
                    f.write(f"Az/|A| mean of means std  : "
                            f"{agg_full['ratio_mean_of_means']} "
                            f"{agg_full['ratio_mean_std']}\n")
                    f.write(f"Az/|A| mean all samples   : "
                            f"{agg_full['ratio_mean_all']}\n")
                    if self.compute_ori:
                        f.write(f"Orient mean std           : "
                                f"{agg_full['orient_mean_of_means']} "
                                f"{agg_full['orient_mean_std']}\n")
                    for c in ["S0", "S1", "S2"]:
                        dmax = agg_full["dist"][f"{c}_max"]
                        dmin = agg_full["dist"][f"{c}_min"]
                        if len(dmax):
                            f.write(f"distance max {c} mean std : "
                                    f"{dmax.mean()} {dmax.std()}\n")
                            f.write(f"distance min {c} mean std : "
                                    f"{dmin.mean()} {dmin.std()}\n")
                else:
                    f.write("no data\n")

                f.write("\nFIRST HALF STEPS (.1)\n")
                if agg_first:
                    f.write(f"count                     : {agg_first['count']}\n")
                    f.write(f"Az/|A| mean of means std  : "
                            f"{agg_first['ratio_mean_of_means']} "
                            f"{agg_first['ratio_mean_std']}\n")
                else:
                    f.write("no data\n")

                f.write("\nSECOND HALF STEPS (.2)\n")
                if agg_second:
                    f.write(f"count                     : {agg_second['count']}\n")
                    f.write(f"Az/|A| mean of means std  : "
                            f"{agg_second['ratio_mean_of_means']} "
                            f"{agg_second['ratio_mean_std']}\n")
                else:
                    f.write("no data\n")

            self.log.emit(f"  Creato: {res_path}\n")
        self.log.emit("\nAnalisi completata.\n")

# ----------------------------- GUI --------------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QAction, QDoubleSpinBox,
    QGroupBox, QGridLayout, QCheckBox, QPlainTextEdit, QMessageBox
)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis – GUI")
        self._thread = None
        self._worker = None

        # ---- central widget ---------------------------------------------
        central = QWidget(); self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # ---- PATH --------------------------------------------------------
        path_box = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Cartella principale…")
        self.path_edit.setReadOnly(True)
        browse_btn = QPushButton("Sfoglia…")
        browse_btn.clicked.connect(self.browse_folder)
        path_box.addWidget(QLabel("Cartella:"))
        path_box.addWidget(self.path_edit, 1)
        path_box.addWidget(browse_btn)
        vbox.addLayout(path_box)

        # ---- PARAMS ------------------------------------------------------
        params_box = QGroupBox("Parametri")
        grid = QGridLayout(params_box)

        self.weight_spin = QDoubleSpinBox(); self.weight_spin.setSuffix(" kg")
        self.weight_spin.setRange(1.0, 300.0); self.weight_spin.setValue(70.0)

        # baselines
        self.base_spin = {}
        for side in ("Dx", "Sx"):
            for i in range(3):
                key = f"{side}{i}"
                sp = QDoubleSpinBox(); sp.setDecimals(3)
                sp.setRange(-1000.0, 1000.0); sp.setValue(0.0)
                self.base_spin[key] = sp

        grid.addWidget(QLabel("Peso:"), 0, 0)
        grid.addWidget(self.weight_spin, 0, 1)

        grid.addWidget(QLabel("Baseline destro S0/1/2:"), 1, 0)
        grid.addWidget(self.base_spin["Dx0"], 1, 1)
        grid.addWidget(self.base_spin["Dx1"], 1, 2)
        grid.addWidget(self.base_spin["Dx2"], 1, 3)

        grid.addWidget(QLabel("Baseline sinistro S0/1/2:"), 2, 0)
        grid.addWidget(self.base_spin["Sx0"], 2, 1)
        grid.addWidget(self.base_spin["Sx1"], 2, 2)
        grid.addWidget(self.base_spin["Sx2"], 2, 3)

        vbox.addWidget(params_box)

        # ---- OPTIONS -----------------------------------------------------
        opt_box = QGroupBox("Opzioni")
        hopt = QHBoxLayout(opt_box)
        self.cb_plots = QCheckBox("Genera grafici"); self.cb_plots.setChecked(True)
        self.cb_ori   = QCheckBox("Calcola orientazione"); self.cb_ori.setChecked(True)
        hopt.addWidget(self.cb_plots); hopt.addWidget(self.cb_ori); hopt.addStretch()
        vbox.addWidget(opt_box)

        # ---- RUN BUTTON --------------------------------------------------
        run_btn = QPushButton("Esegui analisi")
        run_btn.setStyleSheet("font-weight: bold")
        run_btn.clicked.connect(self.start_analysis)
        vbox.addWidget(run_btn)

        # ---- LOG ---------------------------------------------------------
        self.log_edit = QPlainTextEdit(); self.log_edit.setReadOnly(True)
        vbox.addWidget(self.log_edit, 1)

        # ---- MENU --------------------------------------------------------
        file_menu = self.menuBar().addMenu("&File")
        open_act = QAction("Apri cartella…", self)
        open_act.triggered.connect(self.browse_folder)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        exit_act = QAction("Esci", self); exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    # ----------------------- slot: scegli cartella -----------------------
    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Seleziona cartella principale")
        if path:
            self.path_edit.setText(path)

    # ----------------------- avvia analisi (thread) ----------------------
    def start_analysis(self):
        root = self.path_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "Attenzione", "Seleziona una cartella prima di procedere.")
            return

        weight = self.weight_spin.value()
        base_r = tuple(self.base_spin[f"Dx{i}"].value() for i in range(3))
        base_l = tuple(self.base_spin[f"Sx{i}"].value() for i in range(3))
        gen_plots = self.cb_plots.isChecked()
        compute_ori = self.cb_ori.isChecked()

        # lock UI elements during run
        self.setEnabled(False)
        self.log_edit.clear()

        # thread & worker
        self._thread = QThread()
        self._worker = AnalysisWorker(root, weight, base_r, base_l,
                                      gen_plots, compute_ori)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_edit.appendPlainText)
        self._worker.finished.connect(self.analysis_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    # ----------------------- fine analisi --------------------------------
    def analysis_done(self):
        self.setEnabled(True)
        QMessageBox.information(self, "Completato",
                                "Analisi terminata. Controlla i log e la cartella 'calcoli'.")

# ------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(900, 650); win.show()
    sys.exit(app.exec())

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
