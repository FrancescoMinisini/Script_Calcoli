import os
import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def process_file(file_path, baseline, weight):
    """
    Elabora un singolo file CSV (il file n.1 di ciclo) per eseguire i calcoli:
      - Sottrazione dei valori basali da S0, S1, S2.
      - Calcolo della norma dell'accelerazione e del rapporto |Az|/norm.
      - Calcolo del range (max-min) per S0,S1,S2, diviso per il peso.
      - Calcolo della durata (Timestamp finale - Timestamp iniziale).
      - Calcolo della distanza temporale tra i picchi (massimi e minimi) per S0,S1,S2.
      - Calcolo dell'angolo del vettore magnetometrico rispetto alla verticale (Y).
      - Salvataggio di un grafico dell'angolo vs Timestamp.
    """
    # Legge il CSV
    df = pd.read_csv(file_path)
    
    # Sottrazione dei valori basali (input utente) da S0, S1 e S2
    df['S0'] = df['S0'] - baseline[0]
    df['S1'] = df['S1'] - baseline[1]
    df['S2'] = df['S2'] - baseline[2]
    
    # Calcolo della norma dell'accelerazione e rapporto |Az|/norm
    df['acc_norm'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
    df['lat_acc_ratio'] = np.abs(df['Az']) / df['acc_norm']
    lat_acc_mean = df['lat_acc_ratio'].mean()
    lat_acc_std = df['lat_acc_ratio'].std()
    
    # Calcolo del range (max-min)/peso per le pressioni S0, S1, S2
    pressure_results = {}
    for col in ['S0', 'S1', 'S2']:
        col_max = df[col].max()
        col_min = df[col].min()
        delta = (col_max - col_min) / weight
        pressure_results[col] = delta
        
    # Durata del ciclo (Timestamp finale - iniziale)
    duration = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]
    
    # Calcolo della distanza temporale tra picchi (max) e tra minimi per S0, S1, S2
    max_timestamps = []
    min_timestamps = []
    for col in ['S0', 'S1', 'S2']:
        max_idx = df[col].idxmax()
        min_idx = df[col].idxmin()
        max_timestamps.append(df.loc[max_idx, 'Timestamp'])
        min_timestamps.append(df.loc[min_idx, 'Timestamp'])
    temporal_distance_max = max(max_timestamps) - min(max_timestamps)
    temporal_distance_min = max(min_timestamps) - min(min_timestamps)
    
    # Calcolo dell'angolo del vettore magnetometrico rispetto alla verticale (asse Y)
    # Verticale = [0, 1, 0]. Calcolo: angolo = arccos(My/||M||)
    mag_norm = np.sqrt(df['Mx']**2 + df['My']**2 + df['Mz']**2)
    # Per evitare divisioni per zero
    df['mag_angle'] = np.degrees(np.arccos(df['My'] / mag_norm))
    mag_angle_mean = df['mag_angle'].mean()
    mag_angle_std = df['mag_angle'].std()
    
    # Salva il grafico dell'angolo magnetometrico vs Timestamp
    plt.figure()
    plt.plot(df['Timestamp'], df['mag_angle'])
    plt.xlabel('Timestamp')
    plt.ylabel('Angolo Magnetometro (gradi)')
    plt.title('Angolo Magnetometro vs Timestamp')
    plot_file = file_path.replace('.csv', '_mag_angle_plot.png')
    plt.savefig(plot_file)
    plt.close()
    
    return {
        'lat_acc_mean': lat_acc_mean,
        'lat_acc_std': lat_acc_std,
        'pressure_range': pressure_results,
        'duration': duration,
        'temporal_distance_max': temporal_distance_max,
        'temporal_distance_min': temporal_distance_min,
        'mag_angle_mean': mag_angle_mean,
        'mag_angle_std': mag_angle_std,
        'mag_angle_plot': plot_file
    }

def process_durations(folder):
    """
    Elabora tutti i file CSV in una cartella (Passi_Interi o Mezzi_Passi)
    e calcola la durata di ciascun file (Timestamp finale - iniziale).
    Ritorna la durata media e la deviazione standard.
    """
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    durations = []
    for f in files:
        df = pd.read_csv(f)
        duration = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]
        durations.append(duration)
    if durations:
        return np.mean(durations), np.std(durations)
    else:
        return None, None

def process_foot(foot_folder, baseline, weight):
    """
    Per una data cartella di un piede (Piede_Destro o Piede_Sinistro)
    esegue:
      - Elaborazione del file ciclo (primo file in "Passi_Interi")
      - Calcolo delle durate medie (e dev. std) per ciclo e mezzo ciclo.
    Ritorna i risultati in un dizionario strutturato.
    """
    # Percorsi delle cartelle dei passi
    passi_interni_folder = os.path.join(foot_folder, 'Passi_Interi')
    mezzi_passi_folder = os.path.join(foot_folder, 'Mezzi_Passi')
    
    cycle_files = sorted(glob.glob(os.path.join(passi_interni_folder, '*.csv')))
    if len(cycle_files) == 0:
        print("Nessun file ciclo trovato in", passi_interni_folder)
        return None
    # Utilizzo il primo file di ciclo (file n.1)
    cycle_file = cycle_files[0]
    cycle_results = process_file(cycle_file, baseline, weight)
    
    # Calcola durata media e dev. std per tutti i file in ciascuna cartella
    cycle_duration_mean, cycle_duration_std = process_durations(passi_interni_folder)
    half_cycle_duration_mean, half_cycle_duration_std = process_durations(mezzi_passi_folder)
    
    results = {
        'cycle_file': cycle_file,
        'cycle_results': cycle_results,
        'cycle_duration_mean': cycle_duration_mean,
        'cycle_duration_std': cycle_duration_std,
        'half_cycle_duration_mean': half_cycle_duration_mean,
        'half_cycle_duration_std': half_cycle_duration_std
    }
    return results

def main():
    import sys
    # Richiede all'utente il path della cartella principale e gli input richiesti
    main_folder = input("Inserisci il path della cartella generale: ").strip()
    try:
        weight = float(input("Inserisci il peso del soggetto (in kg): ").strip())
    except ValueError:
        print("Peso non valido")
        sys.exit(1)
    try:
        baseline_S0 = float(input("Inserisci il valore basale per S0: ").strip())
        baseline_S1 = float(input("Inserisci il valore basale per S1: ").strip())
        baseline_S2 = float(input("Inserisci il valore basale per S2: ").strip())
    except ValueError:
        print("Valore basale non valido")
        sys.exit(1)
    baseline = (baseline_S0, baseline_S1, baseline_S2)
    
    # Mappa per i nomi delle cartelle dei piedi
    results = {}
    foot_mapping = {
        'Piede_Destro': 'piede_destro',
        'Piede_Sinistro': 'piede_sinistro'
    }
    for foot_name in foot_mapping:
        foot_folder = os.path.join(main_folder, 'passi', foot_name)
        if not os.path.exists(foot_folder):
            print("Cartella non trovata:", foot_folder)
            continue
        res = process_foot(foot_folder, baseline, weight)
        if res is not None:
            results[foot_mapping[foot_name]] = res
    
    # Scrive i risultati in due file di testo (uno per piede) nella cartella principale
    for foot_key, res in results.items():
        output_file = os.path.join(main_folder, f"risultati_{foot_key}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Risultati per {foot_key}:\n")
            f.write(f"File ciclo usato: {res['cycle_file']}\n\n")
            
            f.write("1. Accelerazione laterale (|Az|/||A||):\n")
            f.write(f"   Media: {res['cycle_results']['lat_acc_mean']}\n")
            f.write(f"   Deviazione standard: {res['cycle_results']['lat_acc_std']}\n\n")
            
            f.write("2. Pressioni (delta = max - min, diviso per il peso):\n")
            for col, delta in res['cycle_results']['pressure_range'].items():
                f.write(f"   {col}: {delta}\n")
            f.write("\n")
            
            f.write(f"3. Durata del ciclo (file n.1): {res['cycle_results']['duration']}\n\n")
            
            f.write("4. Distanza temporale fra i picchi per S0,S1,S2:\n")
            f.write(f"   - Massimi: {res['cycle_results']['temporal_distance_max']}\n")
            f.write(f"   - Minimi: {res['cycle_results']['temporal_distance_min']}\n\n")
            
            f.write("5. Durata media (e deviazione standard):\n")
            f.write(f"   - Ciclo (Passi_Interi): media = {res['cycle_duration_mean']}, std = {res['cycle_duration_std']}\n")
            f.write(f"   - Mezzo ciclo (Mezzi_Passi): media = {res['half_cycle_duration_mean']}, std = {res['half_cycle_duration_std']}\n\n")
            
            f.write("6. Angolo magnetometro (rispetto alla verticale, asse Y):\n")
            f.write(f"   Media: {res['cycle_results']['mag_angle_mean']}\n")
            f.write(f"   Deviazione standard: {res['cycle_results']['mag_angle_std']}\n")
            f.write(f"   (Plot salvato in: {res['cycle_results']['mag_angle_plot']})\n\n")
            
            f.write("WARNING: Attenzione! L'accelerometro verticale è rappresentato dalla colonna Y (non da Z).\n")
        print(f"Risultati salvati in: {output_file}")

if __name__ == '__main__':
    main()
