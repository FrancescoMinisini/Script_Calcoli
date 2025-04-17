import os
import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def resample_series(series, N=100):
    """
    Risample un array (o una serie) su una griglia di N punti in tempo normalizzato [0,1].
    """
    x_orig = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, N)
    resampled = np.interp(x_new, x_orig, series)
    return resampled

def process_single_step(file_path, baseline, weight):
    """
    Elabora un singolo file CSV (un passo o un emiciclo) e ritorna un dizionario contenente:
      - duration: durata del passo (Timestamp finale - Timestamp iniziale)
      - s0_range, s1_range, s2_range: range (max-min) di S0, S1, S2 (dopo baseline), diviso per il peso
      - lat_acc_ratio: rapporto Az/||A|| per ogni campione (con media e std calcolabili sul passo)
      - resampled_ratio: versione risample (100 punti) del rapporto, per il plot aggregato
      - mag_angle: angolo del magnetometro rispetto alla verticale (asse Y) per ogni campione, con media e std
          * Se si desidera centrare l’angolo a 0, decommentare la riga indicata.
      - resampled_mag_angle: versione risample (100 punti) dell’angolo magnetico, per il plot aggregato
      - temporal_distance_max e temporal_distance_min: differenza temporale fra i picchi di S0,S1,S2.
    """
    df = pd.read_csv(file_path)
    
    # Sottrazione dei valori basali per S0, S1 e S2
    df['S0'] = df['S0'] - baseline[0]
    df['S1'] = df['S1'] - baseline[1]
    df['S2'] = df['S2'] - baseline[2]
    
    # Durata del passo/emiciclo (Timestamp finale - iniziale)
    duration = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]
    
    # Range per le pressioni (diviso per il peso)
    s0_range = (df['S0'].max() - df['S0'].min()) / weight
    s1_range = (df['S1'].max() - df['S1'].min()) / weight
    s2_range = (df['S2'].max() - df['S2'].min()) / weight
    
    # Calcolo della norma dell'accelerazione e rapporto Az/||A||
    acc_norm = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
    # Se la norma è zero, imposto il rapporto a zero per evitare divisioni per zero
    ratio = np.where(acc_norm == 0, 0, df['Az'] / acc_norm)
    lat_acc_mean = np.mean(ratio)
    lat_acc_std = np.std(ratio)
    resampled_ratio = resample_series(ratio, 100)
    
    # Calcolo dell'angolo del magnetometro rispetto alla verticale (asse Y)
    mag_norm = np.sqrt(df['Mx']**2 + df['My']**2 + df['Mz']**2)
    # Limito il rapporto tra My e la norma a [-1,1] per evitare errori di dominio
    cos_val = np.clip(df['My'] / mag_norm, -1, 1)
    mag_angle = np.degrees(np.arccos(cos_val))
    # Se si desidera centrare l'angolo a 0 (cioè far sì che la verticale corrisponda a 0 gradi),
    # decommentare la seguente riga:
    # mag_angle = mag_angle - np.mean(mag_angle)
    mag_angle_mean = np.mean(mag_angle)
    mag_angle_std = np.std(mag_angle)
    resampled_mag_angle = resample_series(mag_angle, 100)
    
    # Calcolo della distanza temporale tra i picchi per S0,S1,S2:
    max_times = [df['Timestamp'].max() for col in ['S0','S1','S2']]  # I valori massimi non dipendono dalla colonna: uso .max() su df[col]
    min_times = [df['Timestamp'].min() for col in ['S0','S1','S2']]
    temporal_distance_max = max(max_times) - min(max_times)
    temporal_distance_min = max(min_times) - min(min_times)
    
    return {
        'file': file_path,
        'duration': duration,
        's0_range': s0_range,
        's1_range': s1_range,
        's2_range': s2_range,
        'lat_acc_mean': lat_acc_mean,
        'lat_acc_std': lat_acc_std,
        'resampled_ratio': resampled_ratio,
        'mag_angle_mean': mag_angle_mean,
        'mag_angle_std': mag_angle_std,
        'resampled_mag_angle': resampled_mag_angle,
        'temporal_distance_max': temporal_distance_max,
        'temporal_distance_min': temporal_distance_min
    }

def process_steps(folder, baseline, weight):
    """
    Elabora tutti i file CSV (passi o emicicli) presenti nella cartella 'folder'.
    Ritorna una lista di dizionari, uno per ogni file elaborato.
    """
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    steps = []
    for file in files:
        try:
            step_metrics = process_single_step(file, baseline, weight)
            steps.append(step_metrics)
        except Exception as e:
            print(f"Errore nell'elaborazione del file {file}: {e}")
    return steps

def aggregate_metrics(steps):
    """
    Data una lista di dizionari (uno per ogni passo o emiciclo), calcola le statistiche aggregate:
      - Media e deviazione standard per ciascuna metrica (durata, range S0,S1,S2, lat_acc, magnetometro, distanze temporali)
      - Aggrega le serie risample (100 punti) per il rapporto Az/||A|| e per l'angolo magnetico.
    Ritorna un dizionario con i valori aggregati.
    """
    if len(steps) == 0:
        return {}
    durations = [s['duration'] for s in steps]
    s0_ranges = [s['s0_range'] for s in steps]
    s1_ranges = [s['s1_range'] for s in steps]
    s2_ranges = [s['s2_range'] for s in steps]
    lat_acc_means = [s['lat_acc_mean'] for s in steps]
    lat_acc_stds = [s['lat_acc_std'] for s in steps]
    mag_angle_means = [s['mag_angle_mean'] for s in steps]
    mag_angle_stds = [s['mag_angle_std'] for s in steps]
    temporal_distance_maxs = [s['temporal_distance_max'] for s in steps]
    temporal_distance_mins = [s['temporal_distance_min'] for s in steps]
    
    # Aggrego le serie risample per il rapporto Az/||A||
    resampled_matrix = np.array([s['resampled_ratio'] for s in steps])
    resampled_mean = np.mean(resampled_matrix, axis=0)
    resampled_std = np.std(resampled_matrix, axis=0)
    
    # Aggrego le serie risample per l'angolo magnetico
    resampled_mag_matrix = np.array([s['resampled_mag_angle'] for s in steps])
    resampled_mag_mean = np.mean(resampled_mag_matrix, axis=0)
    resampled_mag_std = np.std(resampled_mag_matrix, axis=0)
    
    aggregated = {
        'duration_mean': np.mean(durations),
        'duration_std': np.std(durations),
        's0_range_mean': np.mean(s0_ranges),
        's0_range_std': np.std(s0_ranges),
        's1_range_mean': np.mean(s1_ranges),
        's1_range_std': np.std(s1_ranges),
        's2_range_mean': np.mean(s2_ranges),
        's2_range_std': np.std(s2_ranges),
        'lat_acc_mean_mean': np.mean(lat_acc_means),
        'lat_acc_mean_std': np.std(lat_acc_means),
        'lat_acc_std_mean': np.mean(lat_acc_stds),
        'lat_acc_std_std': np.std(lat_acc_stds),
        'mag_angle_mean_mean': np.mean(mag_angle_means),
        'mag_angle_mean_std': np.std(mag_angle_means),
        'mag_angle_std_mean': np.mean(mag_angle_stds),
        'mag_angle_std_std': np.std(mag_angle_stds),
        'temporal_distance_max_mean': np.mean(temporal_distance_maxs),
        'temporal_distance_max_std': np.std(temporal_distance_maxs),
        'temporal_distance_min_mean': np.mean(temporal_distance_mins),
        'temporal_distance_min_std': np.std(temporal_distance_mins),
        'resampled_mean': resampled_mean,
        'resampled_std': resampled_std,
        'resampled_mag_mean': resampled_mag_mean,
        'resampled_mag_std': resampled_mag_std
    }
    return aggregated

def generate_plot(x, mean, std, output_file, xlabel, ylabel, title):
    """
    Genera un plot con x sull'asse (tempo normalizzato) e la curva media (mean) con una fascia che rappresenta lo std.
    """
    plt.figure()
    plt.plot(x, mean, label="Media")
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, label="Deviazione Std")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def generate_concatenated_plot(x, series, n_steps, output_file):
    """
    Genera un plot in cui viene mostrata la concatenazione (in ordine) dell'angolo magnetico di ogni passo.
    Vengono tracciate anche delle linee verticali per indicare il confine tra i passi.
    """
    plt.figure()
    plt.plot(x, series, label="Angolo Magnetico")
    step_length = len(series) / n_steps
    for i in range(1, n_steps):
        plt.axvline(x=i*step_length, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("Campioni concatenati (passi consecutivi)")
    plt.ylabel("Angolo Magnetico (gradi)")
    plt.title("Angolo Magnetico su tutti i passi")
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def main():
    import sys
    # Richiede all'utente il path della cartella principale e il peso
    main_folder = input("Inserisci il path della cartella generale: ").strip()
    try:
        weight = float(input("Inserisci il peso del soggetto (in kg): ").strip())
    except ValueError:
        print("Peso non valido")
        sys.exit(1)
    
    # Richiede i valori basali per il piede DESTRO
    print("\nInserisci i valori basali per il piede DESTRO:")
    try:
        baseline_S0_right = float(input("Valore basale per S0: ").strip())
        baseline_S1_right = float(input("Valore basale per S1: ").strip())
        baseline_S2_right = float(input("Valore basale per S2: ").strip())
    except ValueError:
        print("Valore basale non valido")
        sys.exit(1)
    baseline_right = (baseline_S0_right, baseline_S1_right, baseline_S2_right)
    
    # Richiede i valori basali per il piede SINISTRO
    print("\nInserisci i valori basali per il piede SINISTRO:")
    try:
        baseline_S0_left = float(input("Valore basale per S0: ").strip())
        baseline_S1_left = float(input("Valore basale per S1: ").strip())
        baseline_S2_left = float(input("Valore basale per S2: ").strip())
    except ValueError:
        print("Valore basale non valido")
        sys.exit(1)
    baseline_left = (baseline_S0_left, baseline_S1_left, baseline_S2_left)
    
    # Mappa dei piedi: le cartelle sotto 'passi' sono "Piede_Destro" e "Piede_Sinistro"
    # per ciascuna si indica anche il nome della sottocartella di output e i rispettivi basali.
    foot_mapping = {
        "Piede_Destro": ("piede_destro", baseline_right),
        "Piede_Sinistro": ("piede_sinistro", baseline_left)
    }
    
    # Creazione della cartella "calcoli" all'interno della cartella principale
    calcoli_dir = os.path.join(main_folder, "calcoli")
    os.makedirs(calcoli_dir, exist_ok=True)
    
    for foot_folder_name, (output_folder_name, baseline) in foot_mapping.items():
        foot_folder = os.path.join(main_folder, "passi", foot_folder_name)
        if not os.path.exists(foot_folder):
            print("Cartella non trovata:", foot_folder)
            continue
        # Crea la cartella di output per questo piede
        output_dir = os.path.join(calcoli_dir, output_folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Elaborazione dei passi interi
        full_steps_folder = os.path.join(foot_folder, "Passi_Interi")
        full_steps_metrics = process_steps(full_steps_folder, baseline, weight)
        
        # Elaborazione degli emicicli
        half_steps_folder = os.path.join(foot_folder, "Mezzi_Passi")
        half_steps_all = process_steps(half_steps_folder, baseline, weight)
        # Divido in primi e secondi emicicli (supponendo che i file siano ordinati)
        first_half_metrics = [step for i, step in enumerate(half_steps_all) if i % 2 == 0]
        second_half_metrics = [step for i, step in enumerate(half_steps_all) if i % 2 == 1]
        
        # Calcolo delle statistiche aggregate
        aggregated_full = aggregate_metrics(full_steps_metrics)
        aggregated_first_half = aggregate_metrics(first_half_metrics) if first_half_metrics else {}
        aggregated_second_half = aggregate_metrics(second_half_metrics) if second_half_metrics else {}
        
        # Genera il plot per il rapporto Az/||A|| (media e std istante per istante)
        x_common = np.linspace(0, 100, 100)  # asse x in percentuale (0-100%)
        if "resampled_mean" in aggregated_full:
            plot_file_ratio = os.path.join(output_dir, f"lat_acc_ratio_plot_{output_folder_name}.png")
            generate_plot(x_common, aggregated_full["resampled_mean"], aggregated_full["resampled_std"],
                          plot_file_ratio, "Tempo normalizzato (%)", "Rapporto Az/||A||",
                          "Media istante per istante di Az/||A|| con banda std")
        
        # Genera il plot per l'angolo magnetico medio (media istante per istante) con banda std
        if "resampled_mag_mean" in aggregated_full:
            plot_file_mag_avg = os.path.join(output_dir, f"mag_angle_avg_plot_{output_folder_name}.png")
            generate_plot(x_common, aggregated_full["resampled_mag_mean"], aggregated_full["resampled_mag_std"],
                          plot_file_mag_avg, "Tempo normalizzato (%)", "Angolo Magnetico (gradi)",
                          "Media istante per istante dell'angolo magnetico con banda std")
        
        # Genera il plot concatenato per l'angolo magnetico su tutti i passi
        if len(full_steps_metrics) > 0:
            concatenated_mag = np.concatenate([s['resampled_mag_angle'] for s in full_steps_metrics])
            x_concat = np.arange(len(concatenated_mag))
            plot_file_mag_concat = os.path.join(output_dir, f"mag_angle_concat_plot_{output_folder_name}.png")
            generate_concatenated_plot(x_concat, concatenated_mag, len(full_steps_metrics), plot_file_mag_concat)
        
        # Salva i risultati in un file di testo nella cartella di output
        results_file = os.path.join(output_dir, f"risultati_{output_folder_name}.txt")
        with open(results_file, 'w') as f:
            f.write(f"Risultati per {output_folder_name}:\n\n")
            f.write("----- PASSI INTERI -----\n")
            f.write(f"Numero di passi: {len(full_steps_metrics)}\n")
            f.write(f"Durata media: {aggregated_full.get('duration_mean', 'N/A')} (std: {aggregated_full.get('duration_std', 'N/A')})\n")
            f.write(f"S0 range medio: {aggregated_full.get('s0_range_mean', 'N/A')} (std: {aggregated_full.get('s0_range_std', 'N/A')})\n")
            f.write(f"S1 range medio: {aggregated_full.get('s1_range_mean', 'N/A')} (std: {aggregated_full.get('s1_range_std', 'N/A')})\n")
            f.write(f"S2 range medio: {aggregated_full.get('s2_range_mean', 'N/A')} (std: {aggregated_full.get('s2_range_std', 'N/A')})\n")
            f.write(f"Lat Acc (Az/||A||) media: {aggregated_full.get('lat_acc_mean_mean', 'N/A')} (std: {aggregated_full.get('lat_acc_mean_std', 'N/A')})\n")
            f.write(f"Lat Acc (Az/||A||) std: {aggregated_full.get('lat_acc_std_mean', 'N/A')} (std: {aggregated_full.get('lat_acc_std_std', 'N/A')})\n")
            f.write(f"Magnetometro angle media: {aggregated_full.get('mag_angle_mean_mean', 'N/A')} (std: {aggregated_full.get('mag_angle_mean_std', 'N/A')})\n")
            f.write(f"Magnetometro angle std: {aggregated_full.get('mag_angle_std_mean', 'N/A')} (std: {aggregated_full.get('mag_angle_std_std', 'N/A')})\n")
            f.write(f"Temporal distance max (S0,S1,S2): {aggregated_full.get('temporal_distance_max_mean', 'N/A')} (std: {aggregated_full.get('temporal_distance_max_std', 'N/A')})\n")
            f.write(f"Temporal distance min (S0,S1,S2): {aggregated_full.get('temporal_distance_min_mean', 'N/A')} (std: {aggregated_full.get('temporal_distance_min_std', 'N/A')})\n")
            f.write("\n----- EMICICLI -----\n")
            f.write("Primi emicicli:\n")
            if aggregated_first_half:
                f.write(f"  Numero di emicicli: {len(first_half_metrics)}\n")
                f.write(f"  Durata media: {aggregated_first_half.get('duration_mean', 'N/A')} (std: {aggregated_first_half.get('duration_std', 'N/A')})\n")
            else:
                f.write("  Nessun dato.\n")
            f.write("Secondi emicicli:\n")
            if aggregated_second_half:
                f.write(f"  Numero di emicicli: {len(second_half_metrics)}\n")
                f.write(f"  Durata media: {aggregated_second_half.get('duration_mean', 'N/A')} (std: {aggregated_second_half.get('duration_std', 'N/A')})\n")
            else:
                f.write("  Nessun dato.\n")
            f.write("\n----- METRICHE PER OGNI PASSO -----\n")
            for i, step in enumerate(full_steps_metrics):
                f.write(f"Passo {i+1}:\n")
                f.write(f"  File: {step['file']}\n")
                f.write(f"  Durata: {step['duration']}\n")
                f.write(f"  S0 range: {step['s0_range']}\n")
                f.write(f"  S1 range: {step['s1_range']}\n")
                f.write(f"  S2 range: {step['s2_range']}\n")
                f.write(f"  Lat Acc (Az/||A||) mean: {step['lat_acc_mean']}\n")
                f.write(f"  Lat Acc (Az/||A||) std: {step['lat_acc_std']}\n")
                f.write(f"  Magnetometro angle mean: {step['mag_angle_mean']}\n")
                f.write(f"  Magnetometro angle std: {step['mag_angle_std']}\n")
                f.write(f"  Temporal distance max: {step['temporal_distance_max']}\n")
                f.write(f"  Temporal distance min: {step['temporal_distance_min']}\n")
                f.write("\n")
        print(f"Risultati salvati in: {results_file}")

if __name__ == '__main__':
    main()
