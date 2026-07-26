"""
analisi_unlearning_benchmark.py
Analisi dei risultati di unlearning per i dataset del benchmark OpenML.
Funziona con la struttura:
  risultati_ridge_openml_dataset_{dataset_id}/seed_{seed}/
    - hessian_inverse.txt
    - parametri.txt
    - X_test.txt, y_test.txt
    - X_train_scaled.txt, y_train.txt (aggiunti)
    - indices_train.txt, indices_unlearned.txt
    - risultati_grado_X.txt, inversi_chiaro_grado_X.txt
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GRADI DA ANALIZZARE
# ============================================================
GRADI = [5]

# ============================================================
# PUNTI FISSI DA ANALIZZARE (per le tabelle di disagreement/accuracy)
# ============================================================
# Numero assoluto di punti rimossi per cui calcolare il disagreement
# (modello cifrato vs retrained) e le accuracy. Modificabile qui oppure da
# linea di comando con --punti_fissi (es. --punti_fissi 10 20 50 100).
PUNTI_FISSI = [10, 20, 50, 100]

# ============================================================
# IMPOSTAZIONI GRAFICI
# ============================================================
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 16
})

# ============================================================
# MAPPA DATASET ID -> TASK ID (per caricamento dataset originale)
# ============================================================
TASK_ID_MAP = {
    44091: 361057,  # wine (classificazione)
    44120: 361601,  # electricity
    44121: 361602,  # covertype
    44122: 361603,  # pol
    44123: 361604,  # house_16H
    44125: 361605,  # MagicTelescope
    44126: 361606,  # bank-marketing
    44127: 361067,  # phoneme
    44128: 361607,  # MiniBooNE
    44129: 361608,  # Higgs
    44130: 361609,  # eye_movements
    44131: 361071,  # jannis
    44089: 361600,  # credit
}

# Dataset di classificazione (per LabelEncoder)
CLASSIFICATION_DATASETS = [44120, 44121, 44122, 44123, 44125, 44126, 44127, 
                           44128, 44129, 44130, 44131, 44089, 44091]

# ============================================================
# FUNZIONI DI CARICAMENTO
# ============================================================

def trova_cartelle_seed_benchmark(dataset_id):
    """Trova tutte le cartelle seed in risultati_ridge_openml_dataset_{dataset_id}/"""
    base_dir = f"risultati_ridge_openml_dataset_{dataset_id}"
    if not os.path.exists(base_dir):
        print(f"  cartella non trovata: {base_dir}")
        return {}
    
    cartelle_seed = {}
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):
            match = re.match(r'seed_(\d+)', item)
            if match:
                seed_num = int(match.group(1))
                cartelle_seed[seed_num] = full_path
                print(f"  trovata cartella seed: {full_path} -> seed {seed_num}")
    
    return cartelle_seed

def carica_X_test(cartella):
    """Carica X_test.txt dalla cartella"""
    file_path = os.path.join(cartella, 'X_test.txt')
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, dtype=float)
    except Exception as e:
        print(f"  errore X_test: {e}")
        return None

def carica_y_test(cartella):
    """Carica y_test.txt dalla cartella"""
    file_path = os.path.join(cartella, 'y_test.txt')
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, dtype=int)
    except Exception as e:
        print(f"  errore y_test: {e}")
        return None

def carica_X_train_scalato(cartella):
    """Carica X_train_scaled.txt dalla cartella (già scalato e con colonna di 1)"""
    file_path = os.path.join(cartella, 'X_train_scaled.txt')
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, dtype=float)
    except Exception as e:
        print(f"  errore X_train_scaled: {e}")
        return None

def carica_y_train(cartella):
    """Carica y_train.txt dalla cartella"""
    file_path = os.path.join(cartella, 'y_train.txt')
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, dtype=float)
    except Exception as e:
        print(f"  errore y_train: {e}")
        return None

def carica_indici_unlearned(cartella):
    """Carica indices_unlearned.txt"""
    file_path = os.path.join(cartella, 'indices_unlearned.txt')
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, dtype=int)
    except Exception as e:
        print(f"  errore indices_unlearned: {e}")
        return None

def carica_indici_train(cartella):
    """Carica indices_train.txt"""
    file_path = os.path.join(cartella, 'indices_train.txt')
    if not os.path.exists(file_path):
        return None
    try:
        return np.loadtxt(file_path, dtype=int)
    except Exception as e:
        print(f"  errore indices_train: {e}")
        return None

def parsa_file_risultati(file_path):
    """Parsa risultati_grado_X.txt e restituisce i dati per punto"""
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r') as f:
        contenuto = f.read()
    
    # Pattern per estrarre i blocchi di ogni punto
    pattern_punto = r'--- PUNTO (\d+) ---\n(.*?)(?=--- PUNTO \d+ ---|\Z)'
    matches = re.findall(pattern_punto, contenuto, re.DOTALL)
    
    dati_per_punto = {}
    for punto_num, sezione in matches:
        punto_num = int(punto_num)
        
        # Estrai i vettori usando regex
        w_clear_match = re.search(r'w_new_clear = \[(.*?)\]', sezione, re.DOTALL)
        w_enc_match = re.search(r'w_new_encrypted = \[(.*?)\]', sezione, re.DOTALL)
        delta_clear_match = re.search(r'delta_w_clear = \[(.*?)\]', sezione, re.DOTALL)
        delta_enc_match = re.search(r'delta_w_encrypted = \[(.*?)\]', sezione, re.DOTALL)
        
        if w_clear_match and w_enc_match and delta_clear_match and delta_enc_match:
            w_clear = np.array([float(x.strip()) for x in w_clear_match.group(1).split(',')])
            w_enc = np.array([float(x.strip()) for x in w_enc_match.group(1).split(',')])
            delta_clear = np.array([float(x.strip()) for x in delta_clear_match.group(1).split(',')])
            delta_enc = np.array([float(x.strip()) for x in delta_enc_match.group(1).split(',')])
            
            dati_per_punto[punto_num] = {
                'w_clear': w_clear,
                'w_enc': w_enc,
                'delta_clear': delta_clear,
                'delta_enc': delta_enc
            }
    
    return dati_per_punto

def carica_file_inverso(file_path):
    """Carica inversi_chiaro_grado_X.txt"""
    if not os.path.exists(file_path):
        return None
    try:
        valori = np.loadtxt(file_path, dtype=float)
        if valori.ndim == 0:
            return np.array([valori])
        return valori
    except Exception:
        return None

def carica_dati_unlearning(dataset_id):
    """
    Carica tutti i dati di unlearning per i gradi 3, 5, 7.
    Restituisce dizionari organizzati per grado e punto.
    """
    cartelle_seed = trova_cartelle_seed_benchmark(dataset_id)
    
    if not cartelle_seed:
        print("  nessuna cartella seed trovata")
        return {g: {} for g in GRADI}, {g: [] for g in GRADI}, {g: {} for g in GRADI}
    
    # Strutture dati
    dati_temp = {g: {} for g in GRADI}
    tutti_inversi = {g: [] for g in GRADI}
    tempi_per_seed = {g: {} for g in GRADI}
    seed_order = {g: [] for g in GRADI}
    
    for seed_num, cartella in cartelle_seed.items():
        print(f"  elaborazione seed {seed_num}...")
        
        for grado in GRADI:
            # Carica risultati_grado_X.txt
            file_risultati = os.path.join(cartella, f'risultati_grado_{grado}.txt')
            if os.path.exists(file_risultati):
                dati_punto = parsa_file_risultati(file_risultati)
                if dati_punto:
                    dati_temp[grado][seed_num] = dati_punto
                    seed_order[grado].append(seed_num)
                    print(f"    grado {grado}: {len(dati_punto)} punti")
            
            # Carica inversi_chiaro_grado_X.txt
            file_inverso = os.path.join(cartella, f'inversi_chiaro_grado_{grado}.txt')
            valori = carica_file_inverso(file_inverso)
            if valori is not None:
                tutti_inversi[grado].extend(valori)
                print(f"    grado {grado}: {len(valori)} valori inversi")
    
    # Organizza i dati per punto
    w_clear = {g: {} for g in GRADI}
    w_enc = {g: {} for g in GRADI}
    delta_clear = {g: {} for g in GRADI}
    delta_enc = {g: {} for g in GRADI}
    
    for grado in GRADI:
        if not dati_temp[grado]:
            continue
        
        primo_seed = seed_order[grado][0]
        punti_disponibili = sorted(dati_temp[grado][primo_seed].keys())
        
        for punto in punti_disponibili:
            n_seed = len(seed_order[grado])
            n_features = len(dati_temp[grado][primo_seed][punto]['w_clear'])
            
            w_clear_mat = np.full((n_seed, n_features), np.nan)
            w_enc_mat = np.full((n_seed, n_features), np.nan)
            delta_clear_mat = np.full((n_seed, n_features), np.nan)
            delta_enc_mat = np.full((n_seed, n_features), np.nan)
            
            for idx, seed_num in enumerate(seed_order[grado]):
                if punto in dati_temp[grado][seed_num]:
                    d = dati_temp[grado][seed_num][punto]
                    w_clear_mat[idx] = d['w_clear']
                    w_enc_mat[idx] = d['w_enc']
                    delta_clear_mat[idx] = d['delta_clear']
                    delta_enc_mat[idx] = d['delta_enc']
            
            w_clear[grado][punto] = w_clear_mat
            w_enc[grado][punto] = w_enc_mat
            delta_clear[grado][punto] = delta_clear_mat
            delta_enc[grado][punto] = delta_enc_mat
    
    return w_clear, w_enc, delta_clear, delta_enc, seed_order, tutti_inversi, tempi_per_seed

def carica_tutti_dati_test(dataset_id):
    """Carica X_test, y_test, X_train_scaled, y_train, indices_unlearned, indices_train per tutti i seed"""
    cartelle_seed = trova_cartelle_seed_benchmark(dataset_id)
    
    X_test_dict = {}
    y_test_dict = {}
    X_train_dict = {}
    y_train_dict = {}
    unlearned_dict = {}
    train_indices_dict = {}
    
    for seed_num, cartella in cartelle_seed.items():
        X_data = carica_X_test(cartella)
        if X_data is not None:
            X_test_dict[seed_num] = X_data
        
        y_data = carica_y_test(cartella)
        if y_data is not None:
            y_test_dict[seed_num] = y_data
        
        # Carica i dati di training scalati
        X_train_data = carica_X_train_scalato(cartella)
        if X_train_data is not None:
            X_train_dict[seed_num] = X_train_data
        
        y_train_data = carica_y_train(cartella)
        if y_train_data is not None:
            y_train_dict[seed_num] = y_train_data
        
        unlearned = carica_indici_unlearned(cartella)
        if unlearned is not None:
            unlearned_dict[seed_num] = unlearned
        
        train_idx = carica_indici_train(cartella)
        if train_idx is not None:
            train_indices_dict[seed_num] = train_idx
    
    return X_test_dict, y_test_dict, X_train_dict, y_train_dict, unlearned_dict, train_indices_dict

def carica_parametri(cartella):
    """
    Carica parametri.txt e restituisce il lambda usato nel training/unlearning
    originale (quello con cui sono stati generati w_clear/w_enc).

    Formato reale del file (una riga 'chiave = valore' per riga), es:
        dataset_id = 44120
        n_punti = 100
        seed = 15
        lambda_reg = 1e-08
        ...

    Il valore rilevante è 'lambda_reg' ed è unico per seed (vale per tutti i
    gradi 3, 5, 7, dato che il file non distingue per grado).

    Restituisce un dizionario {grado: lambda} (stesso valore per ogni grado)
    oppure None se non trova la chiave 'lambda_reg'.
    """
    file_path = os.path.join(cartella, 'parametri.txt')
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        contenuto = f.read()

    match = re.search(r'^\s*lambda_reg\s*=\s*([\d.eE+\-]+)\s*$', contenuto, re.MULTILINE)
    if not match:
        print(f"  ATTENZIONE: chiave 'lambda_reg' non trovata in {file_path}")
        return None

    lambda_reg = float(match.group(1))
    return {g: lambda_reg for g in GRADI}

def carica_tutti_lambda(dataset_id):
    """Carica il lambda originale (per grado) per ogni seed da parametri.txt"""
    cartelle_seed = trova_cartelle_seed_benchmark(dataset_id)

    lambda_dict = {}
    for seed_num, cartella in cartelle_seed.items():
        lambda_per_grado = carica_parametri(cartella)
        if lambda_per_grado is not None:
            lambda_dict[seed_num] = lambda_per_grado
            print(f"  seed {seed_num}: lambda originali = {lambda_per_grado}")
        else:
            print(f"  seed {seed_num}: lambda originale non trovato "
                  f"(verrà usato un fallback via grid-search)")

    return lambda_dict

# ============================================================
# FUNZIONI DI ANALISI (da test_vari.py)
# ============================================================

def calcola_media_varianza_differenze_assolute(w_clear, w_enc, delta_clear, delta_enc):
    """Calcola norma L1 media e varianza delle differenze assolute"""
    medie_varianze = {}
    
    for grado in GRADI:
        if grado not in w_clear or not w_clear[grado]:
            continue
        
        punti = sorted(w_clear[grado].keys())
        
        medie_ass_w = []
        std_ass_w = []
        medie_ass_delta = []
        std_ass_delta = []
        medie_perc_w = []
        std_perc_w = []
        medie_perc_delta = []
        std_perc_delta = []
        medie_norma_w_clear = []
        std_norma_w_clear = []
        medie_norma_w_enc = []
        std_norma_w_enc = []
        
        for punto in punti:
            w_clear_mat = w_clear[grado][punto]
            w_enc_mat = w_enc[grado][punto]
            
            # Norma L1 per w
            diff_ass_w = np.abs(w_enc_mat - w_clear_mat)
            norma_l1_w_per_seed = np.sum(diff_ass_w, axis=1)
            medie_ass_w.append(np.nanmean(norma_l1_w_per_seed))
            std_ass_w.append(np.nanstd(norma_l1_w_per_seed))
            
            # Errore percentuale per w
            norma_l1_clear_w = np.sum(np.abs(w_clear_mat), axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                errore_perc_w_per_seed = (norma_l1_w_per_seed / norma_l1_clear_w) * 100.0
            medie_perc_w.append(np.nanmean(errore_perc_w_per_seed))
            std_perc_w.append(np.nanstd(errore_perc_w_per_seed))
            
            # Norma L1 per delta
            delta_clear_mat = delta_clear[grado][punto]
            delta_enc_mat = delta_enc[grado][punto]
            
            diff_ass_delta = np.abs(delta_enc_mat - delta_clear_mat)
            norma_l1_delta_per_seed = np.sum(diff_ass_delta, axis=1)
            medie_ass_delta.append(np.nanmean(norma_l1_delta_per_seed))
            std_ass_delta.append(np.nanstd(norma_l1_delta_per_seed))
            
            # Errore percentuale per delta
            norma_l1_clear_delta = np.sum(np.abs(delta_clear_mat), axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                errore_perc_delta_per_seed = (norma_l1_delta_per_seed / norma_l1_clear_delta) * 100.0
            medie_perc_delta.append(np.nanmean(errore_perc_delta_per_seed))
            std_perc_delta.append(np.nanstd(errore_perc_delta_per_seed))
            
            # Norma L1 di w_clear e w_enc
            norma_l1_w_clear_per_seed = np.sum(np.abs(w_clear_mat), axis=1)
            norma_l1_w_enc_per_seed = np.sum(np.abs(w_enc_mat), axis=1)
            medie_norma_w_clear.append(np.nanmean(norma_l1_w_clear_per_seed))
            std_norma_w_clear.append(np.nanstd(norma_l1_w_clear_per_seed))
            medie_norma_w_enc.append(np.nanmean(norma_l1_w_enc_per_seed))
            std_norma_w_enc.append(np.nanstd(norma_l1_w_enc_per_seed))
        
        medie_varianze[grado] = {
            'punti': punti,
            'medie_ass_w': np.array(medie_ass_w),
            'std_ass_w': np.array(std_ass_w),
            'medie_ass_delta': np.array(medie_ass_delta),
            'std_ass_delta': np.array(std_ass_delta),
            'medie_perc_w': np.array(medie_perc_w),
            'std_perc_w': np.array(std_perc_w),
            'medie_perc_delta': np.array(medie_perc_delta),
            'std_perc_delta': np.array(std_perc_delta),
            'medie_norma_w_clear': np.array(medie_norma_w_clear),
            'std_norma_w_clear': np.array(std_norma_w_clear),
            'medie_norma_w_enc': np.array(medie_norma_w_enc),
            'std_norma_w_enc': np.array(std_norma_w_enc)
        }
    
    return medie_varianze

def _trova_indice_punto_fisso(punti_array, p_fisso, grado):
    """
    Converte un numero assoluto di punti rimossi (es. 10, 20, 50, 100)
    nell'indice del PUNTO corrispondente all'interno dei dati caricati da
    risultati_grado_X.txt.

    IMPORTANTE: i PUNTO nei file sono indicizzati da 0, coerentemente con
    la convenzione già usata altrove nello script (vedi analizza_punti_fissi,
    dove `indice_punto = n_punti - 1`). Quindi "100 punti rimossi"
    corrisponde al PUNTO 99, non al PUNTO 100.

    Se il punto non viene trovato, stampa un messaggio diagnostico con il
    range di punti effettivamente disponibili, così è chiaro se manca per
    un problema di indicizzazione o perché l'esperimento non ha
    semplicemente raccolto abbastanza punti rimossi.
    """
    punto_atteso = p_fisso - 1
    idx = np.where(punti_array == punto_atteso)[0]
    if len(idx) == 0:
        if len(punti_array) > 0:
            print(f"  attenzione: punto fisso {p_fisso} (PUNTO {punto_atteso}) "
                  f"non trovato per grado {grado}. Punti disponibili: "
                  f"{punti_array.min()}-{punti_array.max()} "
                  f"({len(punti_array)} punti totali)")
        else:
            print(f"  attenzione: nessun punto disponibile per grado {grado}")
        return None
    return idx[0]

def estrai_errore_relativo_punti_fissi(medie_varianze, punti_fissi):
    """
    Estrae, per ciascun grado, l'errore relativo (NON percentuale, quindi
    nell'intervallo [0, 1] circa) su w e delta in corrispondenza dei
    punti fissi richiesti (es. 10, 20, 50, 100 punti rimossi).

    L'errore relativo è semplicemente l'errore percentuale già calcolato
    diviso per 100 (medie_perc_w / 100, medie_perc_delta / 100).

    Restituisce due liste di dizionari (righe_w, righe_delta), una riga
    per ogni combinazione (grado, punto_fisso) trovata nei dati.
    """
    righe_w = []
    righe_delta = []

    for grado in GRADI:
        if grado not in medie_varianze:
            continue

        dati = medie_varianze[grado]
        punti_array = np.array(dati['punti'])

        for p_fisso in punti_fissi:
            i = _trova_indice_punto_fisso(punti_array, p_fisso, grado)
            if i is None:
                continue

            errore_relativo_w = dati['medie_perc_w'][i] / 100.0
            std_relativo_w = dati['std_perc_w'][i] / 100.0
            righe_w.append({
                'grado': grado,
                'punti_rimossi': p_fisso,
                'errore_relativo_medio_w': errore_relativo_w,
                'errore_relativo_std_w': std_relativo_w
            })

            errore_relativo_delta = dati['medie_perc_delta'][i] / 100.0
            std_relativo_delta = dati['std_perc_delta'][i] / 100.0
            righe_delta.append({
                'grado': grado,
                'punti_rimossi': p_fisso,
                'errore_relativo_medio_delta': errore_relativo_delta,
                'errore_relativo_std_delta': std_relativo_delta
            })

    return righe_w, righe_delta

def estrai_errore_assoluto_punti_fissi(medie_varianze, punti_fissi):
    """
    Estrae, per ciascun grado, l'errore assoluto (norma L1 media della
    differenza, NON normalizzato rispetto alla norma di w/delta) su w e
    delta in corrispondenza dei punti fissi richiesti (es. 10, 20, 50, 100
    punti rimossi).

    Restituisce due liste di dizionari (righe_w, righe_delta), una riga
    per ogni combinazione (grado, punto_fisso) trovata nei dati.
    """
    righe_w = []
    righe_delta = []

    for grado in GRADI:
        if grado not in medie_varianze:
            continue

        dati = medie_varianze[grado]
        punti_array = np.array(dati['punti'])

        for p_fisso in punti_fissi:
            i = _trova_indice_punto_fisso(punti_array, p_fisso, grado)
            if i is None:
                continue

            righe_w.append({
                'grado': grado,
                'punti_rimossi': p_fisso,
                'errore_assoluto_medio_w': dati['medie_ass_w'][i],
                'errore_assoluto_std_w': dati['std_ass_w'][i]
            })

            righe_delta.append({
                'grado': grado,
                'punti_rimossi': p_fisso,
                'errore_assoluto_medio_delta': dati['medie_ass_delta'][i],
                'errore_assoluto_std_delta': dati['std_ass_delta'][i]
            })

    return righe_w, righe_delta

def salva_csv_errore_relativo(medie_varianze, dataset_id, punti_fissi):
    """
    Salva due tabelle CSV con l'errore relativo (non percentuale) su w e
    su delta, ai punti fissi indicati (default 10, 20, 50, 100 punti
    rimossi), una riga per ogni combinazione grado/punto.
    """
    righe_w, righe_delta = estrai_errore_relativo_punti_fissi(medie_varianze, punti_fissi)

    if righe_w:
        df_w = pd.DataFrame(righe_w)
        filename_w = f"errore_relativo_w_dataset_{dataset_id}.csv"
        df_w.to_csv(filename_w, index=False)
        print(f"  tabella errore relativo w salvata in {filename_w}")
    else:
        print("  nessun dato disponibile per la tabella errore relativo w")

    if righe_delta:
        df_delta = pd.DataFrame(righe_delta)
        filename_delta = f"errore_relativo_delta_dataset_{dataset_id}.csv"
        df_delta.to_csv(filename_delta, index=False)
        print(f"  tabella errore relativo delta salvata in {filename_delta}")
    else:
        print("  nessun dato disponibile per la tabella errore relativo delta")

def salva_csv_errore_assoluto(medie_varianze, dataset_id, punti_fissi):
    """
    Salva due tabelle CSV con l'errore assoluto (norma L1 media, non
    normalizzato) su w e su delta, ai punti fissi indicati (default 10,
    20, 50, 100 punti rimossi), una riga per ogni combinazione grado/punto.
    """
    righe_w, righe_delta = estrai_errore_assoluto_punti_fissi(medie_varianze, punti_fissi)

    if righe_w:
        df_w = pd.DataFrame(righe_w)
        filename_w = f"errore_assoluto_w_dataset_{dataset_id}.csv"
        df_w.to_csv(filename_w, index=False)
        print(f"  tabella errore assoluto w salvata in {filename_w}")
    else:
        print("  nessun dato disponibile per la tabella errore assoluto w")

    if righe_delta:
        df_delta = pd.DataFrame(righe_delta)
        filename_delta = f"errore_assoluto_delta_dataset_{dataset_id}.csv"
        df_delta.to_csv(filename_delta, index=False)
        print(f"  tabella errore assoluto delta salvata in {filename_delta}")
    else:
        print("  nessun dato disponibile per la tabella errore assoluto delta")

# ============================================================
# FUNZIONI PER GRAFICI
# ============================================================

def genera_grafici_analisi(medie_varianze, dataset_id):
    """Genera tutti i grafici di analisi"""
    
    for grado in GRADI:
        if grado not in medie_varianze:
            continue
        
        dati = medie_varianze[grado]
        punti = dati['punti']
        
        # Grafico errori assoluti
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # w
        media_w = dati['medie_ass_w']
        std_w = dati['std_ass_w']
        ax1.fill_between(punti, media_w - std_w, media_w + std_w,
                         alpha=0.3, color='blue', label='1σ banda')
        ax1.plot(punti, media_w, 'b-', linewidth=2, label='media')
        ax1.set_xlabel('punto i')
        ax1.set_ylabel('errore assoluto medio su w')
        ax1.set_title(f'grado {grado}: errore assoluto medio su w')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.legend(loc='best')
        
        # delta
        media_delta = dati['medie_ass_delta']
        std_delta = dati['std_ass_delta']
        ax2.fill_between(punti, media_delta - std_delta, media_delta + std_delta,
                         alpha=0.3, color='green', label='1σ banda')
        ax2.plot(punti, media_delta, 'g-', linewidth=2, label='media')
        ax2.set_xlabel('punto i')
        ax2.set_ylabel('errore assoluto medio su delta')
        ax2.set_title(f'grado {grado}: errore assoluto medio su delta')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'errori_assoluti_grado_{grado}_dataset_{dataset_id}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Grafico errori percentuali
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        media_w = dati['medie_perc_w']
        std_w = dati['std_perc_w']
        ax1.fill_between(punti, media_w - std_w, media_w + std_w,
                         alpha=0.3, color='blue', label='1σ banda')
        ax1.plot(punti, media_w, 'b-', linewidth=2, label='media')
        ax1.set_xlabel('punto i')
        ax1.set_ylabel('errore percentuale medio su w (%)')
        ax1.set_title(f'grado {grado}: errore percentuale medio su w')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.legend(loc='best')
        
        media_delta = dati['medie_perc_delta']
        std_delta = dati['std_perc_delta']
        ax2.fill_between(punti, media_delta - std_delta, media_delta + std_delta,
                         alpha=0.3, color='green', label='1σ banda')
        ax2.plot(punti, media_delta, 'g-', linewidth=2, label='media')
        ax2.set_xlabel('punto i')
        ax2.set_ylabel('errore percentuale medio su delta (%)')
        ax2.set_title(f'grado {grado}: errore percentuale medio su delta')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'errori_percentuali_grado_{grado}_dataset_{dataset_id}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Grafico norma w
        fig, ax = plt.subplots(figsize=(14, 6))
        
        media_clear = dati['medie_norma_w_clear']
        std_clear = dati['std_norma_w_clear']
        ax.fill_between(punti, media_clear - std_clear, media_clear + std_clear,
                        alpha=0.3, color='blue', label='1σ banda w_clear')
        ax.plot(punti, media_clear, 'b-', linewidth=2, label='media w_clear')
        
        media_enc = dati['medie_norma_w_enc']
        std_enc = dati['std_norma_w_enc']
        ax.fill_between(punti, media_enc - std_enc, media_enc + std_enc,
                        alpha=0.3, color='orange', label='1σ banda w_enc')
        ax.plot(punti, media_enc, color='orange', linestyle='-', linewidth=2, label='media w_enc')
        
        ax.set_xlabel('punto i')
        ax.set_ylabel('norma l1 di w')
        ax.set_title(f'grado {grado}: norma l1 di w_clear e w_enc')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f'norma_w_grado_{grado}_dataset_{dataset_id}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"grafici salvati per grado {grado}")

def genera_istogramma_inversi(tutti_inversi, dataset_id):
    """Genera istogramma dei valori inversi"""
    tutti_valori = []
    for grado in GRADI:
        if tutti_inversi[grado]:
            tutti_valori.extend(tutti_inversi[grado])
    
    if not tutti_valori:
        print("nessun valore inverso da visualizzare")
        return
    
    tutti_valori = np.array(tutti_valori)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(tutti_valori, bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('valore inverso (chiaro)')
    ax.set_ylabel('frequenza')
    ax.set_title(f'distribuzione valori inversi - dataset {dataset_id}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'istogramma_inversi_dataset_{dataset_id}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"istogramma inversi salvato per dataset {dataset_id}")

# ============================================================
# FUNZIONI PER CONFRONTO CON RETRAINING
# ============================================================

LAMBDA_VALUES = np.logspace(-8, 5, 27).tolist()  # 27 valori da 1e-8 a 1e5

def evaluate_lambda_holdout(X_train, y_train, X_val, y_val, lambda_val):
    """Valuta un lambda su holdout (validation set)"""
    n_samples = X_train.shape[0]
    ridge = Ridge(alpha=lambda_val * n_samples, fit_intercept=False)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return accuracy_score(y_val, y_pred_binary)

def find_best_lambda_holdout(X_train, y_train, X_val, y_val, lambda_values):
    """Trova il miglior lambda usando holdout (validation set)"""
    results = []
    for lambda_val in lambda_values:
        score = evaluate_lambda_holdout(X_train, y_train, X_val, y_val, lambda_val)
        results.append((lambda_val, score))
    best_idx = np.argmax([r[1] for r in results])
    best_lambda = results[best_idx][0]
    best_score = results[best_idx][1]
    print(f"    miglior lambda holdout: {best_lambda:.4e} (validation accuracy: {best_score:.6f})")
    return best_lambda, results

def retrain_model(X_train, y_train, lambda_reg=0.05):
    n_samples = X_train.shape[0]
    ridge = Ridge(alpha=lambda_reg * n_samples, fit_intercept=False)
    ridge.fit(X_train, y_train)
    return ridge.coef_

def predici_con_w(w, X_test, soglia=0.5):
    scores = X_test @ w
    predizioni = (scores >= soglia).astype(int)
    return scores, predizioni

def confronta_con_retraining(w_clear, w_enc, seed_order, 
                             X_test_dict, y_test_dict,
                             X_train_dict, y_train_dict,
                             unlearned_dict, train_indices_dict,
                             lambda_dict,
                             percentuali=[ 0.10, 0.20, 0.50, 1]):
    """
    Confronta unlearning (chiaro e cifrato) con retraining per ogni seed.
    USA I DATI DI TRAINING SCALATI DAI FILE, NON RICARICA DA OPENML!
    """
    risultati = {}
    
    # Determina il numero totale di punti unlearned
    totale_punti_unlearned = 0
    for seed in unlearned_dict.values():
        if seed is not None:
            totale_punti_unlearned = max(totale_punti_unlearned, len(seed))
    
    if totale_punti_unlearned == 0:
        print("  nessun punto unlearned trovato")
        return risultati
    
    mappa_percentuale_punto = {}
    for perc in percentuali:
        n_punti = int(totale_punti_unlearned * perc)
        mappa_percentuale_punto[perc] = max(0, n_punti - 1)
        print(f"  {perc*100:.0f}% rimozione -> {n_punti} punti -> punto {mappa_percentuale_punto[perc]}")
    
    # Inizializza struttura risultati
    for perc in percentuali:
        risultati[perc] = {g: {} for g in GRADI}
        for grado in GRADI:
            risultati[perc][grado] = {
                'mae_enc_vs_retrained': [],
                'accuracy_clear': [],
                'accuracy_enc': [],
                'accuracy_retrained': [],
                'discrepanze_enc': [],
                'seed_list': [],
                'norma_l1_w_enc_vs_retrained': [],
                'norma_l1_w_enc_vs_retrained_perc': []
            }
    
    # Seed disponibili: ora includiamo anche X_train_dict e y_train_dict
    seed_disponibili = set(X_test_dict.keys()) & set(y_test_dict.keys()) & \
                       set(X_train_dict.keys()) & set(y_train_dict.keys()) & \
                       set(unlearned_dict.keys()) & set(train_indices_dict.keys())
    print(f"\n  Seed disponibili per confronto: {sorted(seed_disponibili)}")
    
    for seed_num in sorted(seed_disponibili):
        print(f"\n  elaborazione seed {seed_num}...")
        
        X_test = X_test_dict[seed_num]
        y_test = y_test_dict[seed_num]
        X_train_full = X_train_dict[seed_num]  # DATI GIÀ SCALATI E CON COLONNA DI 1!
        y_train_full = y_train_dict[seed_num]
        unlearned = unlearned_dict[seed_num]
        train_idx = train_indices_dict[seed_num]
        
        # Verifica che X_train_full abbia la colonna di 1 (ultima colonna = 1)
        if X_train_full.shape[1] > 0 and not np.allclose(X_train_full[:, -1], 1.0):
            print(f"    ATTENZIONE: X_train_full senza colonna di 1 per seed {seed_num}, la aggiungo...")
            X_train_full = np.column_stack([X_train_full, np.ones(X_train_full.shape[0])])
        
        lambda_per_grado_seed = lambda_dict.get(seed_num, {})
        lambda_fallback_cache = {}  # grado -> lambda, calcolato via grid-search solo se serve
        
        for perc in percentuali:
            indice_punto = mappa_percentuale_punto[perc]
            n_da_rimuovere = int(len(unlearned) * perc)
            
            if n_da_rimuovere == 0:
                continue
            
            indici_rimossi = unlearned[:n_da_rimuovere]
            mask_da_rimuovere = np.isin(train_idx, indici_rimossi)
            
            # USA I DATI SCALATI DAI FILE!
            X_train_retrain = X_train_full[~mask_da_rimuovere]
            y_train_retrain = y_train_full[~mask_da_rimuovere]
            
            for grado in GRADI:
                if grado not in w_clear or grado not in seed_order:
                    continue
                
                try:
                    seed_idx = seed_order[grado].index(seed_num)
                except ValueError:
                    continue
                
                if indice_punto not in w_clear[grado]:
                    continue
                
                # Usa il lambda originale (da parametri.txt) se disponibile;
                # altrimenti fallback a grid-search holdout (con avviso), calcolato
                # una sola volta per seed/grado e riusato per tutte le percentuali.
                if grado in lambda_per_grado_seed:
                    lambda_da_usare = lambda_per_grado_seed[grado]
                else:
                    if grado not in lambda_fallback_cache:
                        print(f"    ATTENZIONE: lambda originale mancante per seed {seed_num}, "
                              f"grado {grado} -> fallback a grid-search holdout")
                        # Usiamo i dati scalati per il grid-search
                        X_tr_full = X_train_full
                        y_tr_full = y_train_full
                        from sklearn.model_selection import train_test_split as _tts
                        X_tr_gs, X_val_gs, y_tr_gs, y_val_gs = _tts(
                            X_tr_full, y_tr_full,
                            test_size=0.2, random_state=seed_num,
                            stratify=y_tr_full if len(np.unique(y_tr_full)) > 1 else None
                        )
                        lambda_fb, _ = find_best_lambda_holdout(
                            X_tr_gs, y_tr_gs, X_val_gs, y_val_gs, LAMBDA_VALUES
                        )
                        lambda_fallback_cache[grado] = lambda_fb
                    lambda_da_usare = lambda_fallback_cache[grado]
                
                w_retrained = retrain_model(X_train_retrain, y_train_retrain, lambda_reg=lambda_da_usare)
                
                w_unlearned_clear = w_clear[grado][indice_punto][seed_idx]
                w_unlearned_enc = w_enc[grado][indice_punto][seed_idx]
                
                _, pred_clear = predici_con_w(w_unlearned_clear, X_test)
                _, pred_enc = predici_con_w(w_unlearned_enc, X_test)
                scores_retrained, pred_retrained = predici_con_w(w_retrained, X_test)
                
                mae_enc = mean_absolute_error(scores_retrained, w_unlearned_enc @ X_test.T)
                
                acc_clear = accuracy_score(y_test, pred_clear)
                acc_enc = accuracy_score(y_test, pred_enc)
                acc_ret = accuracy_score(y_test, pred_retrained)
                
                discrepanze_enc = np.sum(pred_enc != pred_retrained)
                
                # Norma L1 tra pesi
                norma_l1_w = np.sum(np.abs(w_unlearned_enc - w_retrained))
                norma_w_ret = np.sum(np.abs(w_retrained))
                perc_diff = (norma_l1_w / norma_w_ret * 100) if norma_w_ret > 1e-10 else np.nan
                
                risultati[perc][grado]['mae_enc_vs_retrained'].append(mae_enc)
                risultati[perc][grado]['accuracy_clear'].append(acc_clear)
                risultati[perc][grado]['accuracy_enc'].append(acc_enc)
                risultati[perc][grado]['accuracy_retrained'].append(acc_ret)
                risultati[perc][grado]['discrepanze_enc'].append(discrepanze_enc)
                risultati[perc][grado]['seed_list'].append(seed_num)
                risultati[perc][grado]['norma_l1_w_enc_vs_retrained'].append(norma_l1_w)
                risultati[perc][grado]['norma_l1_w_enc_vs_retrained_perc'].append(perc_diff)
                
                print(f"    grado {grado}, {perc*100:.0f}%: lambda={lambda_da_usare:.4e}, "
                      f"acc_enc={acc_enc:.3f}, acc_ret={acc_ret:.3f}")
    
    return risultati


# ============================================================
# ANALISI A PUNTI FISSI (10, 20, 50, 100 punti rimossi)
# ============================================================
# A differenza di confronta_con_retraining (che lavora per PERCENTUALE del
# totale di punti unlearned), qui lavoriamo per NUMERO ASSOLUTO di punti
# rimossi, così come richiesto per le tabelle di disagreement e di accuracy.

def analizza_punti_fissi(w_enc, seed_order,
                          X_test_dict, y_test_dict,
                          X_train_dict, y_train_dict,
                          unlearned_dict, train_indices_dict,
                          lambda_dict,
                          punti_fissi=None):
    """
    Per ogni seed/grado e per ogni numero assoluto di punti rimossi in
    punti_fissi:
      - riaddestra il modello (retrained) rimuovendo i primi n_punti punti
        unlearned;
      - calcola il disagreement (numero e percentuale di predizioni diverse
        sul test set) tra il modello cifrato (w_enc) e il retrained;
      - calcola accuracy_enc e accuracy_retrained a quel numero di punti.
    Calcola inoltre, una sola volta per seed (indipendente da n_punti),
    l'accuracy iniziale: quella del modello riaddestrato su TUTTO il
    training set, prima di qualunque rimozione.

    Restituisce:
      disagreement_risultati[n_punti][grado] = {
          'discrepanze': [...], 'discrepanze_perc': [...], 'seed_list': [...]
      }
      accuracy_risultati[grado] = {
          'accuracy_iniziale': [...],
          'accuracy_retrained': {n_punti: [...]},
          'accuracy_enc': {n_punti: [...]},
          'seed_list': [...]
      }
    """
    if punti_fissi is None:
        punti_fissi = PUNTI_FISSI

    disagreement_risultati = {
        n: {g: {'discrepanze': [], 'discrepanze_perc': [], 'seed_list': []} for g in GRADI}
        for n in punti_fissi
    }
    accuracy_risultati = {
        g: {
            'accuracy_iniziale': [],
            'accuracy_retrained': {n: [] for n in punti_fissi},
            'accuracy_enc': {n: [] for n in punti_fissi},
            'seed_list': []
        } for g in GRADI
    }

    seed_disponibili = set(X_test_dict.keys()) & set(y_test_dict.keys()) & \
                        set(X_train_dict.keys()) & set(y_train_dict.keys()) & \
                        set(unlearned_dict.keys()) & set(train_indices_dict.keys())

    if not seed_disponibili:
        print("  nessun seed disponibile per l'analisi a punti fissi")
        return disagreement_risultati, accuracy_risultati

    print(f"\n  Seed disponibili per analisi a punti fissi: {sorted(seed_disponibili)}")

    cache_iniziale = {}       # (seed_num, lambda arrotondato) -> accuracy iniziale
    lambda_fallback_cache = {}  # (seed_num, grado) -> lambda

    for seed_num in sorted(seed_disponibili):
        print(f"\n  [punti fissi] elaborazione seed {seed_num}...")

        X_test = X_test_dict[seed_num]
        y_test = y_test_dict[seed_num]
        X_train_full = X_train_dict[seed_num]
        y_train_full = y_train_dict[seed_num]
        unlearned = unlearned_dict[seed_num]
        train_idx = train_indices_dict[seed_num]

        if X_train_full.shape[1] > 0 and not np.allclose(X_train_full[:, -1], 1.0):
            X_train_full = np.column_stack([X_train_full, np.ones(X_train_full.shape[0])])

        lambda_per_grado_seed = lambda_dict.get(seed_num, {})

        for grado in GRADI:
            if grado not in w_enc or grado not in seed_order:
                continue
            try:
                seed_idx = seed_order[grado].index(seed_num)
            except ValueError:
                continue

            # Lambda: usa quello originale se disponibile, altrimenti fallback
            if grado in lambda_per_grado_seed:
                lambda_da_usare = lambda_per_grado_seed[grado]
            else:
                key = (seed_num, grado)
                if key not in lambda_fallback_cache:
                    print(f"    ATTENZIONE: lambda originale mancante per seed {seed_num}, "
                          f"grado {grado} -> fallback a grid-search holdout")
                    from sklearn.model_selection import train_test_split as _tts
                    X_tr_gs, X_val_gs, y_tr_gs, y_val_gs = _tts(
                        X_train_full, y_train_full,
                        test_size=0.2, random_state=seed_num,
                        stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
                    )
                    lambda_fb, _ = find_best_lambda_holdout(
                        X_tr_gs, y_tr_gs, X_val_gs, y_val_gs, LAMBDA_VALUES
                    )
                    lambda_fallback_cache[key] = lambda_fb
                lambda_da_usare = lambda_fallback_cache[key]

            # ---- accuracy iniziale (nessun punto rimosso) ----
            cache_key = (seed_num, round(float(lambda_da_usare), 12))
            if cache_key not in cache_iniziale:
                w_iniziale = retrain_model(X_train_full, y_train_full, lambda_reg=lambda_da_usare)
                _, pred_iniziale = predici_con_w(w_iniziale, X_test)
                cache_iniziale[cache_key] = accuracy_score(y_test, pred_iniziale)
            acc_iniziale = cache_iniziale[cache_key]

            accuracy_risultati[grado]['accuracy_iniziale'].append(acc_iniziale)
            accuracy_risultati[grado]['seed_list'].append(seed_num)

            # ---- per ogni numero fisso di punti rimossi ----
            for n_punti in punti_fissi:
                if n_punti > len(unlearned):
                    print(f"    seed {seed_num}, grado {grado}: solo {len(unlearned)} punti "
                          f"unlearned disponibili, salto n_punti={n_punti}")
                    continue

                indice_punto = n_punti - 1
                if indice_punto not in w_enc[grado]:
                    print(f"    seed {seed_num}, grado {grado}: punto {indice_punto} "
                          f"non disponibile in w_enc, salto n_punti={n_punti}")
                    continue

                indici_rimossi = unlearned[:n_punti]
                mask_da_rimuovere = np.isin(train_idx, indici_rimossi)
                X_train_retrain = X_train_full[~mask_da_rimuovere]
                y_train_retrain = y_train_full[~mask_da_rimuovere]

                w_retrained = retrain_model(X_train_retrain, y_train_retrain, lambda_reg=lambda_da_usare)
                w_unlearned_enc = w_enc[grado][indice_punto][seed_idx]

                _, pred_enc = predici_con_w(w_unlearned_enc, X_test)
                _, pred_retrained = predici_con_w(w_retrained, X_test)

                discrepanze = int(np.sum(pred_enc != pred_retrained))
                discrepanze_perc = discrepanze / len(y_test) * 100.0

                acc_enc = accuracy_score(y_test, pred_enc)
                acc_ret = accuracy_score(y_test, pred_retrained)

                disagreement_risultati[n_punti][grado]['discrepanze'].append(discrepanze)
                disagreement_risultati[n_punti][grado]['discrepanze_perc'].append(discrepanze_perc)
                disagreement_risultati[n_punti][grado]['seed_list'].append(seed_num)

                accuracy_risultati[grado]['accuracy_retrained'][n_punti].append(acc_ret)
                accuracy_risultati[grado]['accuracy_enc'][n_punti].append(acc_enc)

                print(f"    grado {grado}, n_punti={n_punti}: disagreement={discrepanze} "
                      f"({discrepanze_perc:.2f}%), acc_enc={acc_enc:.4f}, acc_ret={acc_ret:.4f}")

    return disagreement_risultati, accuracy_risultati


def salva_csv_disagreement(disagreement_risultati, dataset_id, punti_fissi=None):
    """
    Salva una tabella CSV (una riga per grado x n_punti_rimossi) con il
    disagreement (totale e percentuale sul test set) tra modello cifrato e
    retrained, alla rimozione di 10/20/50/100 punti.
    """
    if punti_fissi is None:
        punti_fissi = PUNTI_FISSI

    righe = []
    for n_punti in punti_fissi:
        for grado in GRADI:
            dati = disagreement_risultati.get(n_punti, {}).get(grado, {})
            discrepanze = dati.get('discrepanze', [])
            if not discrepanze:
                continue
            discrepanze_perc = dati.get('discrepanze_perc', [])
            righe.append({
                'dataset_id': dataset_id,
                'grado': grado,
                'n_punti_rimossi': n_punti,
                'n_seed': len(discrepanze),
                'disagreement_medio': np.mean(discrepanze),
                'disagreement_std': np.std(discrepanze),
                'disagreement_min': np.min(discrepanze),
                'disagreement_max': np.max(discrepanze),
                'disagreement_perc_medio': np.mean(discrepanze_perc),
                'disagreement_perc_std': np.std(discrepanze_perc),
            })

    filename = f'disagreement_dataset_{dataset_id}.csv'
    if righe:
        df = pd.DataFrame(righe)
        df.to_csv(filename, index=False)
        print(f"tabella disagreement salvata in {filename}")
    else:
        print(f"  nessun dato di disagreement da salvare per dataset {dataset_id}")
        df = pd.DataFrame(columns=['dataset_id', 'grado', 'n_punti_rimossi', 'n_seed',
                                    'disagreement_medio', 'disagreement_std',
                                    'disagreement_min', 'disagreement_max',
                                    'disagreement_perc_medio', 'disagreement_perc_std'])
        df.to_csv(filename, index=False)

    return df


def estrai_righe_riassuntive(accuracy_risultati, dataset_id, n_finale=100):
    """
    Estrae, per ogni grado, una riga con:
      - accuracy iniziale (media/std, nessun punto rimosso)
      - accuracy di retraining dopo la rimozione di n_finale punti (media/std)
      - accuracy del modello cifrato dopo la rimozione di n_finale punti (media/std)
    Da usare per costruire la tabella riassuntiva su tutti i dataset.
    """
    righe = []
    for grado in GRADI:
        dati = accuracy_risultati.get(grado, {})
        acc_iniziale = dati.get('accuracy_iniziale', [])
        acc_ret = dati.get('accuracy_retrained', {}).get(n_finale, [])
        acc_enc = dati.get('accuracy_enc', {}).get(n_finale, [])

        if not acc_iniziale and not acc_ret and not acc_enc:
            continue

        righe.append({
            'dataset_id': dataset_id,
            'grado': grado,
            'n_seed_iniziale': len(acc_iniziale),
            'accuracy_iniziale_media': np.mean(acc_iniziale) if acc_iniziale else np.nan,
            'accuracy_iniziale_std': np.std(acc_iniziale) if acc_iniziale else np.nan,
            f'n_seed_retraining_{n_finale}': len(acc_ret),
            f'accuracy_retraining_{n_finale}_media': np.mean(acc_ret) if acc_ret else np.nan,
            f'accuracy_retraining_{n_finale}_std': np.std(acc_ret) if acc_ret else np.nan,
            f'n_seed_cifrato_{n_finale}': len(acc_enc),
            f'accuracy_cifrato_{n_finale}_media': np.mean(acc_enc) if acc_enc else np.nan,
            f'accuracy_cifrato_{n_finale}_std': np.std(acc_enc) if acc_enc else np.nan,
        })

    return righe


# ============================================================
# GRAFICI CONFRONTO RETRAINING
# ============================================================

def genera_grafici_confronto(risultati_confronto, dataset_id):
    """Genera grafici di confronto con retraining"""
    if not risultati_confronto:
        print("  nessun risultato di confronto disponibile")
        return
    
    percentuali = sorted(risultati_confronto.keys())
    gradi = GRADI
    
    # MAE
    fig, ax = plt.subplots(figsize=(10, 7))
    for grado in gradi:
        mae_enc_means = []
        mae_enc_stds = []
        for p in percentuali:
            if grado in risultati_confronto[p]:
                vals = risultati_confronto[p][grado]['mae_enc_vs_retrained']
                if vals:
                    mae_enc_means.append(np.mean(vals))
                    mae_enc_stds.append(np.std(vals))
                else:
                    mae_enc_means.append(np.nan)
                    mae_enc_stds.append(np.nan)
            else:
                mae_enc_means.append(np.nan)
                mae_enc_stds.append(np.nan)
        
        x = [p*100 for p in percentuali]
        ax.errorbar(x, mae_enc_means, yerr=mae_enc_stds, marker='s',
                   label=f'grado {grado}', linestyle='-', linewidth=2, capsize=5)
    
    ax.set_xlabel('percentuale di punti rimossi (%)')
    ax.set_ylabel('MAE tra score (encrypted vs retrained)')
    ax.set_title(f'Errore Medio Assoluto - dataset {dataset_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'confronto_retraining_mae_dataset_{dataset_id}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"grafico MAE salvato per dataset {dataset_id}")
    
    # Accuracy
    fig, axes = plt.subplots(1, len(gradi), figsize=(5 * len(gradi), 5))
    if len(gradi) == 1:
        axes = [axes]
    
    for idx, grado in enumerate(gradi):
        ax = axes[idx]
        
        acc_enc_means = []
        acc_enc_stds = []
        acc_ret_means = []
        acc_ret_stds = []
        
        for p in percentuali:
            if grado in risultati_confronto[p]:
                vals_enc = risultati_confronto[p][grado]['accuracy_enc']
                vals_ret = risultati_confronto[p][grado]['accuracy_retrained']
                if vals_enc:
                    acc_enc_means.append(np.mean(vals_enc))
                    acc_enc_stds.append(np.std(vals_enc))
                else:
                    acc_enc_means.append(np.nan)
                    acc_enc_stds.append(np.nan)
                if vals_ret:
                    acc_ret_means.append(np.mean(vals_ret))
                    acc_ret_stds.append(np.std(vals_ret))
                else:
                    acc_ret_means.append(np.nan)
                    acc_ret_stds.append(np.nan)
            else:
                acc_enc_means.append(np.nan)
                acc_enc_stds.append(np.nan)
                acc_ret_means.append(np.nan)
                acc_ret_stds.append(np.nan)
        
        x = [p*100 for p in percentuali]
        ax.errorbar(x, acc_enc_means, yerr=acc_enc_stds, marker='s',
                   label='unlearning cifrato', linestyle='-', linewidth=2, capsize=5)
        ax.errorbar(x, acc_ret_means, yerr=acc_ret_stds, marker='^',
                   label='retraining', linestyle='-', linewidth=2, capsize=5)
        
        ax.set_xlabel('percentuale di punti rimossi (%)')
        ax.set_ylabel('accuracy')
        ax.set_title(f'grado {grado}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'confronto_retraining_accuracy_dataset_{dataset_id}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"grafico accuracy salvato per dataset {dataset_id}")

# ============================================================
# MAIN
# ============================================================

def genera_grafico_errori_tutti_gradi(medie_varianze, dataset_id, gradi=None):
    """
    Genera un unico grafico (invece di uno per grado) con l'errore assoluto
    medio su w e su delta per TUTTI i gradi analizzati, ciascuno con un
    colore diverso, così da poterli confrontare direttamente.
    """
    if gradi is None:
        gradi = GRADI

    gradi_disponibili = [g for g in gradi if g in medie_varianze]
    if not gradi_disponibili:
        print("  nessun dato disponibile per il grafico comparativo tra gradi")
        return

    cmap = plt.get_cmap('tab10')
    colori = {g: cmap(i % cmap.N) for i, g in enumerate(gradi_disponibili)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for grado in gradi_disponibili:
        dati = medie_varianze[grado]
        punti = dati['punti']
        colore = colori[grado]

        media_w = dati['medie_ass_w']
        std_w = dati['std_ass_w']
        ax1.plot(punti, media_w, color=colore, linewidth=2, label=f'grado {grado}')
        ax1.fill_between(punti, media_w - std_w, media_w + std_w, color=colore, alpha=0.15)

        media_delta = dati['medie_ass_delta']
        std_delta = dati['std_ass_delta']
        ax2.plot(punti, media_delta, color=colore, linewidth=2, label=f'grado {grado}')
        ax2.fill_between(punti, media_delta - std_delta, media_delta + std_delta, color=colore, alpha=0.15)

    ax1.set_xlabel('punto i')
    ax1.set_ylabel('errore assoluto medio su w')
    ax1.set_title(f'errore assoluto medio su w - dataset {dataset_id} (confronto tra gradi)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.legend(loc='best')

    ax2.set_xlabel('punto i')
    ax2.set_ylabel('errore assoluto medio su delta')
    ax2.set_title(f'errore assoluto medio su delta - dataset {dataset_id} (confronto tra gradi)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.legend(loc='best')

    plt.tight_layout()
    filename = f'errori_assoluti_tutti_gradi_dataset_{dataset_id}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"grafico comparativo tra gradi salvato in {filename}")


def elabora_dataset(dataset_id, punti_fissi=None, n_finale_riassunto=100):
    """
    Esegue l'intera pipeline di analisi per un singolo dataset.
    Restituisce la lista di righe (una per grado) da usare per la tabella
    riassuntiva globale (accuracy iniziale / retraining / cifrato).
    """
    if punti_fissi is None:
        punti_fissi = PUNTI_FISSI

    print("="*60)
    print(f"ANALISI UNLEARNING - DATASET OPENML ID: {dataset_id}")
    print("="*60)
    
    # 1. Carica dati di unlearning
    print("\n1. Caricamento dati di unlearning...")
    w_clear, w_enc, delta_clear, delta_enc, seed_order, tutti_inversi, tempi_per_seed = carica_dati_unlearning(dataset_id)
    
    # 2. Carica tutti i dati (test e training scalati)
    print("\n2. Caricamento dati di test e training scalati...")
    X_test_dict, y_test_dict, X_train_dict, y_train_dict, unlearned_dict, train_indices_dict = carica_tutti_dati_test(dataset_id)
    
    # 3. Carica lambda originali
    print("\n3. Caricamento lambda originali da parametri.txt...")
    lambda_dict = carica_tutti_lambda(dataset_id)
    
    # 4. Analisi errori
    print("\n4. Calcolo errori assoluti e percentuali...")
    medie_varianze = calcola_media_varianza_differenze_assolute(w_clear, w_enc, delta_clear, delta_enc)

    print(f"\n4b. Salvataggio tabelle CSV errore relativo (w, delta) ai punti fissi {punti_fissi}...")
    salva_csv_errore_relativo(medie_varianze, dataset_id, punti_fissi)

    print(f"\n4c. Salvataggio tabelle CSV errore assoluto (w, delta) ai punti fissi {punti_fissi}...")
    salva_csv_errore_assoluto(medie_varianze, dataset_id, punti_fissi)

    # 5. Genera grafici di analisi
    print("\n5. Generazione grafici di analisi...")
    genera_grafici_analisi(medie_varianze, dataset_id)
    genera_istogramma_inversi(tutti_inversi, dataset_id)
    genera_grafico_errori_tutti_gradi(medie_varianze, dataset_id)

    righe_riassuntive = []

    # 6. Confronto con retraining (USA I DATI SCALATI DAI FILE!)
    if X_train_dict and y_train_dict:
        print("\n6. Confronto con retraining usando dati scalati dai file...")
        percentuali = [0.10, 0.20, 0.50, 1]
        risultati_confronto = confronta_con_retraining(
            w_clear, w_enc, seed_order,
            X_test_dict, y_test_dict,
            X_train_dict, y_train_dict,
            unlearned_dict, train_indices_dict,
            lambda_dict,
            percentuali
        )
        
        print("\n7. Generazione grafici di confronto...")
        genera_grafici_confronto(risultati_confronto, dataset_id)
        
        # Salva risultati
        salva_risultati_txt(medie_varianze, tutti_inversi, risultati_confronto, dataset_id)

        # 8. Disagreement e accuracy a punti fissi (10, 20, 50, 100 punti rimossi)
        print(f"\n8. Analisi a punti fissi {punti_fissi} (disagreement cifrato vs retrained)...")
        disagreement_risultati, accuracy_risultati = analizza_punti_fissi(
            w_enc, seed_order,
            X_test_dict, y_test_dict,
            X_train_dict, y_train_dict,
            unlearned_dict, train_indices_dict,
            lambda_dict,
            punti_fissi
        )

        print("\n9. Salvataggio tabella CSV di disagreement...")
        salva_csv_disagreement(disagreement_risultati, dataset_id, punti_fissi)

        righe_riassuntive = estrai_righe_riassuntive(
            accuracy_risultati, dataset_id, n_finale=n_finale_riassunto
        )
    else:
        print("\n6. ATTENZIONE: Dati di training scalati non trovati nei file!")
        print("   Esegui prima ridge_intervallo_diretto.py per generare X_train_scaled.txt e y_train.txt")
        # Salva solo risultati parziali
        salva_risultati_parziali(medie_varianze, tutti_inversi, dataset_id)
    
    print(f"\nFile generati per dataset {dataset_id}:")
    print("  - errori_assoluti_grado_X_dataset_Y.png (uno per grado)")
    print("  - errori_percentuali_grado_X_dataset_Y.png (uno per grado)")
    print("  - norma_w_grado_X_dataset_Y.png (uno per grado)")
    print(f"  - errori_assoluti_tutti_gradi_dataset_{dataset_id}.png (tutti i gradi, colori diversi)")
    print(f"  - istogramma_inversi_dataset_{dataset_id}.png")
    print(f"  - errore_relativo_w_dataset_{dataset_id}.csv")
    print(f"  - errore_relativo_delta_dataset_{dataset_id}.csv")
    print(f"  - errore_assoluto_w_dataset_{dataset_id}.csv")
    print(f"  - errore_assoluto_delta_dataset_{dataset_id}.csv")
    if X_train_dict:
        print(f"  - confronto_retraining_mae_dataset_{dataset_id}.png")
        print(f"  - confronto_retraining_accuracy_dataset_{dataset_id}.png")
        print(f"  - risultati_confronto_retraining_dataset_{dataset_id}.txt")
        print(f"  - disagreement_dataset_{dataset_id}.csv")

    return righe_riassuntive


def main():
    global GRADI, PUNTI_FISSI

    parser = argparse.ArgumentParser(description='Analisi unlearning per benchmark OpenML (uno o più dataset)')
    parser.add_argument('--dataset_ids', type=int, required=True, nargs='+',
                        help='Uno o più ID di dataset OpenML separati da spazio '
                             '(es. --dataset_ids 44120 44121 44122)')
    parser.add_argument('--gradi', type=int, nargs='+', default=GRADI,
                         help=f'Gradi del polinomio da analizzare (default: {GRADI}). '
                              'Es. --gradi 5, oppure --gradi 3 5 7 9')
    parser.add_argument('--punti_fissi', type=int, nargs='+', default=PUNTI_FISSI,
                         help=f'Numero assoluto di punti rimossi per cui calcolare '
                              f'disagreement e accuracy (default: {PUNTI_FISSI}). '
                              'Es. --punti_fissi 10 20 50 100')
    parser.add_argument('--n_finale_riassunto', type=int, default=100,
                         help='Numero di punti rimossi da usare nella tabella riassuntiva '
                              'finale per accuracy_retraining e accuracy_cifrato (default: 100). '
                              'Deve essere uno dei valori presenti in --punti_fissi.')
    args = parser.parse_args()

    GRADI = args.gradi
    PUNTI_FISSI = args.punti_fissi
    dataset_ids = args.dataset_ids
    n_finale_riassunto = args.n_finale_riassunto

    print(f"Gradi selezionati per l'analisi: {GRADI}")
    print(f"Punti fissi selezionati per disagreement/accuracy: {PUNTI_FISSI}")
    print(f"Dataset da elaborare: {dataset_ids}")

    tutte_le_righe_riassuntive = []

    for dataset_id in dataset_ids:
        righe = elabora_dataset(dataset_id, punti_fissi=PUNTI_FISSI,
                                 n_finale_riassunto=n_finale_riassunto)
        tutte_le_righe_riassuntive.extend(righe)
        print("\n" + "="*60)
        print(f"COMPLETATA ELABORAZIONE DATASET {dataset_id}")
        print("="*60 + "\n")

    # Tabella riassuntiva su tutti i dataset: accuracy iniziale, di retraining
    # e del modello cifrato dopo la rimozione di n_finale_riassunto punti.
    if tutte_le_righe_riassuntive:
        df_riassunto = pd.DataFrame(tutte_le_righe_riassuntive)
        filename_riassunto = 'tabella_accuracy_riassuntiva.csv'
        df_riassunto.to_csv(filename_riassunto, index=False)
        print(f"\nTabella riassuntiva (tutti i dataset) salvata in {filename_riassunto}")
    else:
        print("\nNessuna riga riassuntiva da salvare (nessun dataset con dati di retraining).")

    print("\n" + "="*60)
    print("ELABORAZIONE COMPLETATA PER TUTTI I DATASET!")
    print("="*60)
    print(f"Dataset elaborati: {dataset_ids}")
    print(f"Gradi analizzati: {GRADI}")
    print(f"Punti fissi analizzati: {PUNTI_FISSI}")



def salva_risultati_txt(medie_varianze, tutti_inversi, risultati_confronto, dataset_id):
    """Salva risultati in formato testo"""
    filename = f"risultati_analisi_dataset_{dataset_id}.txt"
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RISULTATI ANALISI UNLEARNING - DATASET {dataset_id}\n")
        f.write("="*80 + "\n\n")
        
        # Errori per grado (come prima)
        for grado in GRADI:
            if grado not in medie_varianze:
                continue
            dati = medie_varianze[grado]
            f.write(f"\nGRADO {grado}\n")
            f.write("-"*40 + "\n")
            f.write("Punto\tMAE_w\tMAE_delta\tErr%_w\tErr%_delta\n")
            for i, p in enumerate(dati['punti']):
                f.write(f"{p}\t{dati['medie_ass_w'][i]:.6f}\t{dati['medie_ass_delta'][i]:.6f}\t"
                       f"{dati['medie_perc_w'][i]:.2f}\t{dati['medie_perc_delta'][i]:.2f}\n")
        
        # Confronto retraining CON DISCREPANZE
        if risultati_confronto:
            f.write("\n\nCONFRONTO RETRAINING\n")
            f.write("="*60 + "\n")
            for perc in sorted(risultati_confronto.keys()):
                f.write(f"\n{perc*100:.0f}% rimozione:\n")
                for grado in GRADI:
                    if grado in risultati_confronto[perc]:
                        d = risultati_confronto[perc][grado]
                        if d['accuracy_enc']:
                            # Calcola statistiche
                            acc_enc_mean = np.mean(d['accuracy_enc'])
                            acc_enc_std = np.std(d['accuracy_enc'])
                            acc_ret_mean = np.mean(d['accuracy_retrained'])
                            acc_ret_std = np.std(d['accuracy_retrained'])
                            
                            # ===== AGGIUNGI DISCREPANZE =====
                            discrepanze = np.array(d['discrepanze_enc'])
                            disc_mean = np.mean(discrepanze)
                            disc_std = np.std(discrepanze)
                            disc_min = np.min(discrepanze)
                            disc_max = np.max(discrepanze)
                            
                            f.write(f"  Grado {grado}:\n")
                            f.write(f"    acc_enc = {acc_enc_mean:.4f} +/- {acc_enc_std:.4f}\n")
                            f.write(f"    acc_ret = {acc_ret_mean:.4f} +/- {acc_ret_std:.4f}\n")
                            f.write(f"    discrepanze_enc_vs_ret = {disc_mean:.2f} +/- {disc_std:.2f} "
                                   f"(min={disc_min:.0f}, max={disc_max:.0f})\n")
    
    print(f"Risultati salvati in {filename}")

def salva_risultati_parziali(medie_varianze, tutti_inversi, dataset_id):
    """Salva risultati parziali (senza retraining)"""
    filename = f"risultati_analisi_dataset_{dataset_id}_parziali.txt"
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RISULTATI ANALISI UNLEARNING (PARZIALI) - DATASET {dataset_id}\n")
        f.write("="*80 + "\n\n")
        
        for grado in GRADI:
            if grado not in medie_varianze:
                continue
            dati = medie_varianze[grado]
            f.write(f"\nGRADO {grado}\n")
            f.write("-"*40 + "\n")
            f.write("Punto\tMAE_w\tMAE_delta\n")
            for i, p in enumerate(dati['punti']):
                f.write(f"{p}\t{dati['medie_ass_w'][i]:.6f}\t{dati['medie_ass_delta'][i]:.6f}\n")
    
    print(f"Risultati parziali salvati in {filename}")

if __name__ == "__main__":
    main()