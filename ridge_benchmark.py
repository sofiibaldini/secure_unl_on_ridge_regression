import openml
import numpy as np
import pandas as pd
from scipy.linalg import inv
import os
import argparse
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
import random
import warnings
import subprocess
import re
import tempfile


# Dopo gli import, aggiungi:
SOLLYA_DISPONIBILE = False
# Sollya è installato in /usr/bin/sollya
SOLLYA_PATH = "/usr/bin/sollya"
SOLLYA_DISPONIBILE = os.path.exists(SOLLYA_PATH)

if SOLLYA_DISPONIBILE:
    print(f"Sollya trovato in: {SOLLYA_PATH}")
else:
    print("Sollya non trovato. Installa con: sudo apt-get install sollya")

def calcola_polinomio_remez(intervallo_dict, grado=5):
    """
    Calcola il polinomio di Remez usando Sollya da riga di comando.
    """
    if not SOLLYA_DISPONIBILE:
        print("Sollya non disponibile, impossibile calcolare il polinomio di Remez.")
        return None
    
    a = float(intervallo_dict['estremo1'])
    b = float(intervallo_dict['estremo2']) # QUA per cambiare intervallo
    #a=0.5
    #b=1
    
    print(f"  Calcolo polinomio di Remez su [{a:.6f}, {b:.6f}] con Sollya...")
    
    # Usiamo marker espliciti per separare i coefficienti dall'errore finale,
    # così il parsing non dipende dal contare le righe.
    comando = (
        f'f = 1/x;\n'
        f'p = remez(f, {grado}, [{a}, {b}]);\n'
        f'for i from 0 to {grado} do print("COEFF:", coeff(p, i));\n'
        f'errore = dirtyinfnorm(p - f, [{a}, {b}]);\n'
        f'print("ERR:", errore);\n'
    )
    
    tmp_path = None
    try:
        # Scriviamo il comando in un file .sollya su disco invece di passarlo
        # via 'echo | sollya': il pipe con echo come processo esterno è fragile
        # (espansioni di shell, indentazione che finisce nel testo, encoding),
        # mentre un file letto con 'sollya script.sollya' è deterministico.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sollya',
                                          delete=False) as tmp:
            tmp.write(comando)
            tmp_path = tmp.name
        
        sollya_process = subprocess.run(
            [SOLLYA_PATH, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        
        output = sollya_process.stdout.decode('utf-8')
        error_output = sollya_process.stderr.decode('utf-8')
        
        # NOTA: non usiamo il return code come criterio di successo. Sollya in
        # alcune build/versioni esce con codice diverso da 0 anche quando il
        # calcolo è completato correttamente (es. warning interni in batch
        # mode). Il criterio affidabile è: siamo riusciti a estrarre tutti i
        # coefficienti attesi dall'output? Se sì, è andata bene a prescindere
        # dal returncode.
        
        # Parsing robusto basato sui marker COEFF:/ERR: invece che sulla posizione di riga
        coeffs = []
        errore = None
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("COEFF:"):
                valore = line.replace("COEFF:", "").strip()
                try:
                    coeffs.append(float(valore))
                except ValueError:
                    print(f"  ERRORE: non riesco a convertire '{valore}' in float")
            elif line.startswith("ERR:"):
                valore = line.replace("ERR:", "").strip()
                try:
                    errore = float(valore)
                except ValueError:
                    pass
        
        if len(coeffs) != grado + 1:
            print(f"  ATTENZIONE: trovati {len(coeffs)} coefficienti, attesi {grado+1}")
            if sollya_process.returncode != 0:
                print(f"  Return code di Sollya: {sollya_process.returncode}")
            print(f"  Comando inviato:\n{comando}")
            print(f"  Output completo di Sollya:")
            print(output)
            if error_output:
                print(f"  Stderr di Sollya:\n{error_output}")
            return None
        
        if sollya_process.returncode != 0:
            print(f"  (Nota: Sollya ha terminato con return code {sollya_process.returncode}, "
                  f"ma tutti i {grado+1} coefficienti sono stati estratti correttamente, "
                  f"quindi procediamo comunque.)")
        
        print(f"  Polinomio calcolato con successo!")
        if errore is not None:
            print(f"  Errore di approssimazione: {errore:.6e}")
        
        return {
            'coefficienti': coeffs,
            'errore': errore
        }
        
    except subprocess.TimeoutExpired:
        print("Timeout nella chiamata a Sollya")
        return None
    except Exception as e:
        print(f"Errore durante l'esecuzione di Sollya: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)




def salva_coefficienti_remez(output_dir, coeffs, errore, intervallo, grado=5):
    """Salva i coefficienti del polinomio di Remez in un file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "coefficienti_remez.txt")
    
    with open(file_path, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# COEFFICIENTI DEL POLINOMIO DI REMEZ\n")
        f.write("# ============================================================\n")
        f.write(f"# grado = {grado}\n")
        f.write(f"# intervallo = [{intervallo['estremo1']:.10f}, {intervallo['estremo2']:.10f}]\n")
        f.write(f"# errore_approssimazione = {errore:.10e}\n")
        f.write("\n")
        f.write("# Coefficienti dal grado più basso al più alto (c0 + c1*x + c2*x^2 + ...)\n")
        f.write("# formato: c0, c1, c2, ..., cn\n")
        f.write("\n")
        
        for i, c in enumerate(coeffs):
            f.write(f"c{i} = {c:.15f}\n")
        
        f.write("\n")
        f.write("# ============================================================\n")
        f.write("# FORMATO PER C++ (array di coefficienti)\n")
        f.write("# ============================================================\n")
        f.write("const vector<double> COEFFS = {\n    ")
        f.write(", ".join([f"{c:.15f}" for c in coeffs]))
        f.write("\n};\n")
    
    print(f"Coefficienti salvati in: {file_path}")


# Mappa completa (da aggiornare con i risultati dello script)
TASK_ID_MAP = {
    44091: 361057,  # wine (regressione)
    44120: 361601,  # electricity
    44121: 361602,  # covertype
    44122: 361603,  # pol
    44123: 361604,  # house_16H
    44124: 361064,  # kdd_ipsums_la_97-small
    44125: 361605,  # MagicTelescope
    44126: 361606,  # bank-marketing
    44127: 361067,  # phoneme
    44128: 361607,  # MiniBooNE
    44129: 361608,  # Higgs
    44130: 361609,  # eye_movements
    44131: 361071,  # jannis
    44089: 361600,  # credit
    44090: 361056,  # california (regressione)
}

def get_task_id(dataset_id):
    if dataset_id in TASK_ID_MAP:
        return TASK_ID_MAP[dataset_id]
    else:
        raise ValueError(f"Task ID non trovato per dataset_id {dataset_id}")

# ============================================================
# PARSER ARGOMENTI DA RIGA DI COMANDO
# ============================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Ridge Regression con split predefiniti OpenML')
    
    parser.add_argument('--dataset_id', type=int, required=True,
                        help='ID del dataset OpenML trasformato (es. 44120 per electricity)')
    parser.add_argument('--n_punti', type=int, required=True,
                        help='Numero di punti da selezionare per l\'unlearning')
    parser.add_argument('--is_classification', type=lambda x: x.lower() == 'true', required=True,
                        help='True per classificazione, False per regressione')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per la generazione casuale (default: 42)')
    parser.add_argument('--lambda_reg', type=float, default=None,
                        help='Lambda di regolarizzazione (se None, viene ottimizzato con grid search su validation)')
    parser.add_argument('--max_train_size', type=int, default=50000,
                        help='Dimensione massima del training set (default: 50000)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory di output')
    parser.add_argument('--metodo_sigma_min', type=int, default=1, choices=[1, 2],
                        help='Metodo per sigma_min: 1=min_i{sigma_i>sigma_max*eps}, 2=max{min_i{sigma_i}, sigma_max*eps}')
    
    return parser.parse_args()
# ============================================================
# CONFIGURAZIONE LAMBDA PER GRID SEARCH
# ============================================================
LAMBDA_VALUES = np.logspace(-8, 5, 27).tolist()  # 27 valori da 1e-8 a 1e5

def evaluate_lambda_holdout(X_train, y_train, X_val, y_val, lambda_val, is_classification):
    """Valuta un lambda su holdout (validation set)"""
    n_samples = X_train.shape[0]
    ridge = Ridge(alpha=lambda_val * n_samples, fit_intercept=True)
    ridge.fit(X_train, y_train)
    
    y_pred = ridge.predict(X_val)
    
    if is_classification:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return accuracy_score(y_val, y_pred_binary)
    else:
        return r2_score(y_val, y_pred)

def find_best_lambda_holdout(X_train, y_train, X_val, y_val, lambda_values, is_classification):
    """Trova il miglior lambda usando holdout (validation set)"""
    print(f"\n{'='*60}")
    print(f"GRID SEARCH CON HOLDOUT")
    print(f"{'='*60}")
    print(f"  Lambda testati: {len(lambda_values)} valori")
    print(f"  Da: {lambda_values[0]:.1e} a {lambda_values[-1]:.1e}")
    print(f"  Validation set size: {X_val.shape[0]} campioni")
    
    results = []
    
    for lambda_val in lambda_values:
        score = evaluate_lambda_holdout(X_train, y_train, X_val, y_val, lambda_val, is_classification)
        results.append((lambda_val, score))
    
    # Trova il miglior lambda
    best_idx = np.argmax([r[1] for r in results])
    best_lambda = results[best_idx][0]
    best_score = results[best_idx][1]
    
    print(f"\n{'='*60}")
    print(f"MIGLIOR LAMBDA TROVATO: {best_lambda:.4e}")
    print(f"  Validation score associato: {best_score:.6f}")
    print(f"{'='*60}")
    
    return best_lambda, results


# ============================================================
# FUNZIONI PER CALCOLO NORME E VALORI SINGOLARI
# ============================================================

def calcola_norme_punti_training(X_train):
    """
    Calcola la norma L2 (euclidea) di ogni punto di training.
    Restituisce il minimo, il massimo e tutti i valori.
    """
    # Calcola la norma L2 di ogni riga (campione)
    norme = np.linalg.norm(X_train, axis=1)
    return {
        'min': np.min(norme),
        'max': np.max(norme),
        'mean': np.mean(norme),
        'std': np.std(norme),
        'tutti': norme
    }

def calcola_valori_singolari_H(H):
    """
    Calcola i valori singolari della matrice H.
    Restituisce il minimo, il massimo e tutti i valori.
    """
    # Calcola i valori singolari
    valori_singolari = np.linalg.svdvals(H)
    return {
        'min': np.min(valori_singolari),
        'max': np.max(valori_singolari),
        'mean': np.mean(valori_singolari),
        'std': np.std(valori_singolari),
        'tutti': valori_singolari
    }

def calcola_condizionamento_H(H):
    """
    Calcola il numero di condizionamento di H (rapporto max/min valori singolari).
    """
    valori_singolari = np.linalg.svdvals(H)
    if np.min(valori_singolari) < 1e-15:
        return np.inf  # Matrice singolare o mal condizionata
    return np.max(valori_singolari) / np.min(valori_singolari)


def calcola_sigma_min_metodo1(s, sigma_max, eps=1e-8):
    """
    Metodo 1: sigma_min = min_i {sigma_i : sigma_i > sigma_max * eps}
    Prende il più piccolo valore singolare che supera il cutoff
    """
    cutoff = eps * sigma_max
    s_valid = s[s > cutoff]
    
    if len(s_valid) > 0:
        return np.min(s_valid), cutoff
    else:
        return cutoff, cutoff

def calcola_sigma_min_metodo2(s, sigma_max, eps=1e-8):
    """
    Metodo 2: sigma_min = max { min_i {sigma_i}, sigma_max * eps}
    Prende il massimo tra il minimo valore singolare e il cutoff
    """
    cutoff = eps * sigma_max
    sigma_min_tradizionale = np.min(s)
    
    return max(sigma_min_tradizionale, cutoff), cutoff


def calcola_intervallo_tradizionale_con_cutoff(H, X_train, eps=1e-8, metodo=1):
    """
    Calcola l'intervallo usando i valori singolari di H con cutoff.
    H è la matrice Hessiana.
    """
    valori_singolari = np.linalg.svdvals(H)
    
    # sv_max: SENZA cutoff
    sv_max = np.max(valori_singolari)
    
    # Calcola sv_min con il metodo scelto
    if metodo == 1:
        sv_min, cutoff = calcola_sigma_min_metodo1(valori_singolari, sv_max, eps)
        metodo_str = "min_i {sv_i : sv_i > sv_max * eps}"
    else:
        sv_min, cutoff = calcola_sigma_min_metodo2(valori_singolari, sv_max, eps)
        metodo_str = "max { min_i {sv_i}, sv_max * eps }"
    
    # Norma al quadrato dei punti di training
    norme_quadre = np.sum(X_train**2, axis=1)
    norma_quadra_min = np.min(norme_quadre)
    norma_quadra_max = np.max(norme_quadre)
    
    # Intervallo per beta = ||x||² / sv
    beta_min = norma_quadra_min / sv_max
    beta_max = norma_quadra_max / sv_min
    
    # Intervallo per (1 - beta)
    estremo1 = 1.0 - beta_max
    estremo2 = 1.0 - beta_min
    
    # Correggi se l'intervallo è invertito
    if estremo1 > estremo2:
        print(f"!!!!!Cambio gli estremi!!!!!!!")
        estremo1, estremo2 = estremo2, estremo1
    
    # Assicura che l'intervallo sia valido
    # if estremo1 < 1e-6:
      #  estremo1 = 1e-6
    #if estremo2 > 1.0:
      #  estremo2 = 1.0
    
    return {
        'estremo1': estremo1,
        'estremo2': estremo2,
        'sv_max': sv_max,
        'sv_min': sv_min,
        'cutoff': cutoff,
        'eps': eps,
        'metodo': metodo_str,
        'metodo_id': metodo,
        'norma_quadra_min': norma_quadra_min,
        'norma_quadra_max': norma_quadra_max,
        'cond_H': sv_max / sv_min if sv_min > 0 else np.inf
    }

# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================
def compute_hessian_ridge(X, lambda_reg):
    """Calcola l'Hessiana della Ridge (come in calcolo_hessiana_v2.py)"""
    n_samples, n_features = X.shape
    XTX = X.T @ X
    H = 2 * XTX + 2 * lambda_reg * n_samples * np.eye(n_features)
    return H

def matrice_a_diagonali(M, n_real):
    """Converte una matrice n×n in un array di n diagonali"""
    n = M.shape[0]
    diagonali = []
    
    for d in range(n):
        diag = np.zeros(n)
        for i in range(n):
            j = (i + d) % n
            if i < n_real and j < n_real:
                diag[i] = M[i, j]
        diagonali.append(diag)
    
    return diagonali

def prossima_potenza_due(n):
    return 1 << (n - 1).bit_length()



# ============================================================
# CARICAMENTO DATASET CON SPLIT PREDEFINITI
# ============================================================
def load_dataset_with_openml_split(dataset_id, max_train_size=50000, random_state=42, is_classification=True):
    """
    Carica il dataset usando gli split predefiniti del task OpenML
    Restituisce X_train, X_test, y_train, y_test come numpy array
    """
    print(f"\n{'='*60}")
    print(f"CARICAMENTO DATASET OPENML ID: {dataset_id}")
    print(f"{'='*60}")
    
    print(f"  Tipo: {'Classificazione' if is_classification else 'Regressione'}")
    
    # Trova il task ID usando la mappa
    task_id = get_task_id(dataset_id)
    if task_id is None:
        raise ValueError(f"Nessun task trovato per dataset_id={dataset_id}")
    
    print(f"  Task ID trovato: {task_id}")
    
    # Carica il task
    task = openml.tasks.get_task(task_id)
    
    # Ottieni gli split predefiniti (holdout fisso)
    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)
    
    # Carica il dataset
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    print(f"  Dataset originale: {X.shape[0]} campioni, {X.shape[1]} feature totali")
    
    # ============================================================
    # RIMUOVI COLONNE NON NUMERICHE PRIMA DEGLI SPLIT
    # ============================================================
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    
    if non_numeric_cols:
        print(f"  Rimosse colonne non numeriche: {non_numeric_cols}")
        X = X[numeric_cols]
    
    print(f"  Feature numeriche rimaste: {X.shape[1]}")
    
    # ============================================================
    # APPLICA SPLIT (ORA X HA SOLO COLONNE NUMERICHE)
    # ============================================================
    X_train = X.iloc[train_indices]
    X_test_raw = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test_raw = y[test_indices]
    
    # Limita train set a max_train_size 
    original_train_size = len(X_train)
    if len(X_train) > max_train_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_train), max_train_size, replace=False)
        X_train = X_train.iloc[indices]
        y_train = y_train.iloc[indices]
        print(f"  Train set limitato da {original_train_size} a {max_train_size} campioni")

    # Limita test set a max_train_size 
    original_test_size = len(X_test_raw)
    if len(X_test_raw) > max_train_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_test_raw), max_train_size, replace=False)
        X_test_raw = X_test_raw.iloc[indices]
        y_test_raw = y_test_raw.iloc[indices]
        print(f"  Train set limitato da {original_test_size} a {max_train_size} campioni")

    print(f"\n[DEBUG] Calcolo percentuali reali:")
    print(f"  Train size: {len(X_train)}")
    print(f"  Validation + test size: {len(X_test_raw)}")
    totale = len(X_train) +  len(X_test_raw)
    print(f"  Train %: {len(X_train)/totale*100:.1f}%")
    print(f"  Test + val%: {len(X_test_raw)/totale*100:.1f}%")
    
    # ============================================================
    # SPLIT DI X_test_raw IN VALIDATION (30%) E TEST FINALE (70%)
    # ============================================================
    print(f"\n  Split del test set originale ({len(X_test_raw)} campioni) in validation e test finale:")
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_raw, y_test_raw, 
        test_size=0.7,
        random_state=random_state,
        stratify=y_test_raw if is_classification else None
    )
    
    print(f"    Validation set: {len(X_val)} campioni ({len(X_val)/len(X_test_raw)*100:.1f}% del test originale)")
    print(f"    Test finale:    {len(X_test)} campioni ({len(X_test)/len(X_test_raw)*100:.1f}% del test originale)")
    
    # Converti in numpy (ora sono tutte numeriche)
    X_train = X_train.values.astype(float)
    X_val = X_val.values.astype(float)
    X_test = X_test.values.astype(float)
    
    # Prepara target
    if is_classification:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
        print(f"  Classi: {le.classes_}")
    else: # converte in array
        y_train_enc = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
        y_val_enc = y_val.values.ravel() if hasattr(y_val, 'values') else y_val
        y_test_enc = y_test.values.ravel() if hasattr(y_test, 'values') else y_test
    
    print(f"\n  Train set:      {X_train.shape[0]} campioni, {X_train.shape[1]} feature")
    print(f"  Validation set: {X_val.shape[0]} campioni")
    print(f"  Test finale:    {X_test.shape[0]} campioni")
    
    return X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc
# ============================================================
# SELEZIONE PUNTI CASUALI
# ============================================================
def seleziona_punti_casuali(X, y, n_punti, random_state):
    """Seleziona casualmente n_punti dal dataset"""
    
    print(f"\n{'='*60}")
    print(f"SELEZIONE PUNTI CASUALI")
    print(f"{'='*60}")
    print(f"  Seed: {random_state}")
    print(f"  Numero punti da selezionare: {n_punti}")
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    indici_disponibili = list(range(X.shape[0]))
    indici_scelti = random.sample(indici_disponibili, min(n_punti, len(indici_disponibili)))
    
    punti_x = []
    punti_y_originali = []
    
    #print(f"\n  Punti selezionati:")
    for idx in indici_scelti:
        x_random = X[idx]
        y_random = y[idx]
        punti_x.append(x_random)
        punti_y_originali.append(y_random)
        # print(f"    Indice {idx}: y = {y_random}")
    
    return indici_scelti, punti_x, punti_y_originali


## SAlVA FILE CON HESSIANA
def salva_info_H(output_dir, norme_punti, valori_singolari_H, condizionamento_H, intervallo, lambda_reg, n_train_samples, n_features):
    """
    Salva le informazioni sulla matrice Hessiana in un file separato.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "info_H.txt")
    
    with open(file_path, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# INFORMAZIONI SULLA MATRICE HESSIANA H\n")
        f.write("# ============================================================\n")
        f.write(f"# lambda_reg = {lambda_reg:.10e}\n")
        f.write(f"# n_train_samples = {n_train_samples}\n")
        f.write(f"# n_features = {n_features}\n")
        f.write("\n")
        
        # ============================================================
        # NORME DEI PUNTI DI TRAINING
        # ============================================================
        f.write("# ============================================================\n")
        f.write("# STATISTICHE NORME PUNTI DI TRAINING (norma L2)\n")
        f.write("# ============================================================\n")
        f.write(f"# norma_min = {norme_punti['min']:.10e}\n")
        f.write(f"# norma_max = {norme_punti['max']:.10e}\n")
        f.write(f"# norma_mean = {norme_punti['mean']:.10e}\n")
        f.write(f"# norma_std = {norme_punti['std']:.10e}\n")
        f.write("\n")
        
        # ============================================================
        # VALORI SINGOLARI DI H
        # ============================================================
        f.write("# ============================================================\n")
        f.write("# STATISTICHE VALORI SINGOLARI DELLA MATRICE HESSIANA\n")
        f.write("# ============================================================\n")
        f.write(f"# sv_min = {valori_singolari_H['min']:.10e}\n")
        f.write(f"# sv_max = {valori_singolari_H['max']:.10e}\n")
        f.write(f"# sv_mean = {valori_singolari_H['mean']:.10e}\n")
        f.write(f"# sv_std = {valori_singolari_H['std']:.10e}\n")
        f.write(f"# condizionamento = {condizionamento_H:.2f}\n")
        f.write("\n")
        
        # ============================================================
        # INTERVALLO [1 - norma_min/sv_max, 1 - norma_max/sv_min]
        # ============================================================
        f.write("# ============================================================\n")
        f.write("# INTERVALLO [1 - norma_min_x / sv_max_H, 1 - norma_max_x / sv_min_H]\n")
        f.write("# ============================================================\n")
        f.write(f"# intervallo_estremo1 = {intervallo['estremo1']:.10e}\n")
        f.write(f"# intervallo_estremo2 = {intervallo['estremo2']:.10e}\n")
        f.write(f"# intervallo_larghezza = {intervallo['estremo2'] - intervallo['estremo1']:.10e}\n")
        f.write("\n")
        
        # ============================================================
        # DETTAGLI (opzionale: tutti i valori per analisi più approfondita)
        # ============================================================
        f.write("# ============================================================\n")
        f.write("# DETTAGLIO VALORI SINGOLARI (tutti i valori)\n")
        f.write("# ============================================================\n")
        for i, v in enumerate(valori_singolari_H['tutti']):
            f.write(f"# sv_{i:04d} = {v:.10e}\n")
        f.write("\n")
        
        # Norme di tutti i punti (se non sono troppi)
        if len(norme_punti['tutti']) <= 100:
            f.write("# ============================================================\n")
            f.write("# NORME DI TUTTI I PUNTI DI TRAINING\n")
            f.write("# ============================================================\n")
            for i, n in enumerate(norme_punti['tutti']):
                f.write(f"# norma_{i:04d} = {n:.10e}\n")
        else:
            f.write(f"# norme_punti_totali = {len(norme_punti['tutti'])}\n")
            f.write("# (elenco non riportato per motivi di spazio)\n")
        
        f.write("\n")
        f.write("# ============================================================\n")
        f.write("# FINE FILE\n")
        f.write("# ============================================================\n")
    
    print(f"Informazioni su H salvate in: {file_path}")



# ============================================================
# SALVATAGGIO FILE
# ============================================================

# SALVATAGGIO FILE
# ============================================================
def salva_file_output(output_dir, diagonali, punti_x, punti_y, w_star, lambda_reg, 
                      grid_results, args, n_train_samples, n_features, is_classification,
                      test_accuracy, test_mse):
    """Salva i risultati in file di testo (formato compatibile)"""
    
    os.makedirs(output_dir, exist_ok=True)
    txt_file = os.path.join(output_dir, "hessian_inverse.txt")
    
    metric_name = "accuracy" if is_classification else "r2"
    
    with open(txt_file, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# PARAMETRI DI CONFIGURAZIONE\n")
        f.write("# ============================================================\n")
        f.write(f"# dataset_id = {args.dataset_id}\n")
        f.write(f"# n_punti = {args.n_punti}\n")
        f.write(f"# seed = {args.seed}\n")
        f.write(f"# is_classification = {args.is_classification}\n")
        f.write(f"# lambda_ottimale = {lambda_reg}\n")
        f.write(f"# max_train_size = {args.max_train_size}\n")
        f.write(f"# test_{metric_name} = {test_accuracy:.6f}\n")
        f.write(f"# test_mse = {test_mse:.6f}\n")
        f.write(f"# n_train_samples = {n_train_samples}\n")
        f.write(f"# n_features = {n_features}\n")
        f.write("\n")
        
        f.write("# ============================================================\n")
        f.write("# RISULTATI GRID SEARCH (validation set)\n")
        f.write("# ============================================================\n")
        for lambda_val, score in grid_results:
            f.write(f"# lambda={lambda_val:.4e}: validation_score={score:.6f}\n")
        f.write("\n")
        
        f.write("# ============================================================\n")
        f.write("# MATRICE HESSIANA INVERSA (formato diagonali)\n")
        f.write("# ============================================================\n")
        for d, diag in enumerate(diagonali):
            for val in diag:
                f.write(f"{val:15.10f}")
            f.write("\n")
        f.write("\n")
        
        f.write("# ============================================================\n")
        f.write("# PUNTI DA DIMENTICARE\n")
        f.write("# ============================================================\n")
        for i, (x, y_val) in enumerate(zip(punti_x, punti_y)):
            f.write(f"x_random = [{', '.join([f'{val:.10f}' for val in x])}]\n")
            f.write(f"y_random = {y_val}\n")
        f.write("\n")
        
        f.write("# ============================================================\n")
        f.write("# PESI INIZIALI (w*)\n")
        f.write("# ============================================================\n")
        f.write(f"w_star = [{', '.join([f'{val:.10f}' for val in w_star])}]\n")
    
    return txt_file

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_arguments()
    
    print("="*60)
    print("RIDGE REGRESSION CON SPLIT PREDEFINITI OPENML")
    print("="*60)
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Numero punti da rimuovere: {args.n_punti}")
    print(f"Is Classification: {args.is_classification}")
    print(f"Seed: {args.seed}")
    print(f"Max train size: {args.max_train_size}")
    print("="*60)
    
    # imposta seed casuali a argomento
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Carica dataset con split predefiniti (train + validation + test finale)
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_with_openml_split(
        args.dataset_id, args.max_train_size, args.seed, args.is_classification
    )
    
    n_features = X_train.shape[1]
    n_train_samples = X_train.shape[0]

    scaler = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=min(1000, X_train.shape[0]),
    random_state=args.seed,
    subsample=int(1e9),
    )
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_val = np.column_stack([X_val, np.ones(X_val.shape[0])])
    X_test = np.column_stack([X_test, np.ones(X_test.shape[0])])
    
    # 2. Se non specificato, trova il miglior lambda usando IL VALIDATION SET
    if args.lambda_reg is None:
        lambda_reg, grid_results = find_best_lambda_holdout(
            X_train, y_train, X_val, y_val, LAMBDA_VALUES, args.is_classification
        )
    else:
        lambda_reg = args.lambda_reg
        val_score = evaluate_lambda_holdout(X_train, y_train, X_val, y_val, lambda_reg, args.is_classification)
        grid_results = [(lambda_reg, val_score)]
        print(f"\nLambda specificato: {lambda_reg}, validation score: {val_score:.6f}")
    
    # 3. Calcola Hessiana e modello finale sul TRAIN set
    print(f"\n{'='*60}")
    print(f"CALCOLO MODELLO FINALE SUL TRAIN SET")
    print(f"{'='*60}")
    print(f"  Lambda usato: {lambda_reg:.4e}")
    
    H = compute_hessian_ridge(X_train, lambda_reg)
    H_inv = np.linalg.inv(H)


    print(f"\n{'='*60}")
    print(f"CALCOLO STATISTICHE SUI DATI")
    print(f"{'='*60}")
    
    norme_punti = calcola_norme_punti_training(X_train)
    print(f"  Norma L2 dei punti di training:")
    print(f"    Min: {norme_punti['min']:.6f}")
    print(f"    Max: {norme_punti['max']:.6f}")
    print(f"    Media: {norme_punti['mean']:.6f}")
    print(f"    Std: {norme_punti['std']:.6f}")
    
    # ============================================================
    # NUOVO: CALCOLO VALORI SINGOLARI DI H
    # ============================================================
    valori_singolari_H = calcola_valori_singolari_H(H)
    condizionamento_H = calcola_condizionamento_H(H)
    print(f"\n  Valori singolari della matrice Hessiana:")
    print(f"    Min: {valori_singolari_H['min']:.6e}")
    print(f"    Max: {valori_singolari_H['max']:.6e}")
    print(f"    Media: {valori_singolari_H['mean']:.6e}")
    print(f"    Std: {valori_singolari_H['std']:.6e}")
    print(f"    Numero di condizionamento: {condizionamento_H:.2f}")
    if condizionamento_H > 1e6:
        print(f"    ATTENZIONE: Matrice mal condizionata!")

    ## GENERO L'INTERVALLO DI REMEZ

    # ============================================================
    # CALCOLO INTERVALLO
    # ============================================================
    print(f"\n{'='*60}")
    print(f"CALCOLO INTERVALLO PER REMEZ")
    print(f"{'='*60}")

    EPS = 1e-8
    METODO_SIGMA_MIN = args.metodo_sigma_min

    print(f"  Metodo sigma_min scelto: {METODO_SIGMA_MIN}")
    if METODO_SIGMA_MIN == 1:
        print(f"    sigma_min = min_i {{sigma_i : sigma_i > sigma_max * eps}}")
    else:
        print(f"    sigma_min = max {{ min_i {{sigma_i}}, sigma_max * eps }}")
    
    # Calcola il condizionamento di H
    cond_H = condizionamento_H
    

    intervallo_trad = calcola_intervallo_tradizionale_con_cutoff(
        H, X_train, eps=EPS, metodo=METODO_SIGMA_MIN
    )
    
    print(f"  Intervallo calcolato:")
    print(f"    [{intervallo_trad['estremo1']:.6f}, {intervallo_trad['estremo2']:.6f}]")
    print(f"    sv_max = {intervallo_trad['sv_max']:.6e}")
    print(f"    sv_min = {intervallo_trad['sv_min']:.6e}")
    print(f"    cutoff = {intervallo_trad['cutoff']:.6e}")
    print(f"    ||x||²_min = {intervallo_trad['norma_quadra_min']:.6e}")
    print(f"    ||x||²_max = {intervallo_trad['norma_quadra_max']:.6e}")
    
    intervallo = {
        'estremo1': intervallo_trad['estremo1'],
        'estremo2': intervallo_trad['estremo2'],
        'norma_quadra_min': intervallo_trad['norma_quadra_min'],
        'norma_quadra_max': intervallo_trad['norma_quadra_max'],
        'sv_min': intervallo_trad['sv_min'],
        'sv_max': intervallo_trad['sv_max'],
        'cutoff': intervallo_trad['cutoff'],
        'eps': EPS,
        'metodo': f'tradizionale_{intervallo_trad["metodo"]}',
        'metodo_id': METODO_SIGMA_MIN,
        'cond_H': intervallo_trad['cond_H']
    }

    print(f"\n  Intervallo finale per Remez:")
    print(f"    [{intervallo['estremo1']:.6f}, {intervallo['estremo2']:.6f}]")
    print(f"    Larghezza: {intervallo['estremo2'] - intervallo['estremo1']:.6f}")
    
    
    
    ## ADDESTRAMENTO RIDGE
    
    ridge_model = Ridge(alpha=lambda_reg * n_train_samples, fit_intercept=True)
    ridge_model.fit(X_train, y_train)
    w_star = ridge_model.coef_
    
    print(f"  w_star shape: {w_star.shape}")
    print(f"  Hessiana inversa shape: {H_inv.shape}")
    
    ridge_model = Ridge(alpha=lambda_reg * n_train_samples, fit_intercept=True)
    ridge_model.fit(X_train, y_train)
    w_star = ridge_model.coef_
    
    print(f"  w_star shape: {w_star.shape}")
    print(f"  Hessiana inversa shape: {H_inv.shape}")
    
    # 4. VALUTAZIONE SUL TEST FINALE (NON USATO PER LA GRID SEARCH!)
    print(f"\n{'='*60}")
    print(f"VALUTAZIONE MODELLO SUL TEST SET")
    print(f"{'='*60}")
    
    y_pred = ridge_model.predict(X_test)
    
    if args.is_classification:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        test_accuracy = accuracy_score(y_test, y_pred_binary)
        test_mse = mean_squared_error(y_test, y_pred) 
        print(f"  Test Accuracy: {test_accuracy:.6f}")
        print(f"  (Confronto: miglior validation score = {max([s for _, s in grid_results]):.6f})")
    else:
        test_accuracy = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred) 
        print(f"  Test R²: {test_accuracy:.6f}")
        print(f"  (Confronto: miglior validation R² = {max([s for _, s in grid_results]):.6f})")
    
    # 5. Seleziona punti casuali DAL TRAIN SET
    indici_scelti, punti_x, punti_y = seleziona_punti_casuali(
        X_train, y_train, args.n_punti, args.seed
    )
    
    

    base_dir = args.output_dir if args.output_dir else f"risultati_ridge_openml_dataset_{args.dataset_id}"
    output_dir = os.path.join(base_dir, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    salva_info_H(output_dir, norme_punti, valori_singolari_H, condizionamento_H, intervallo, lambda_reg, n_train_samples, n_features)

    # SALVA POLINOMIO REMEZ
    if SOLLYA_DISPONIBILE:
        print(f"\n{'='*60}")
        print("CALCOLO POLINOMIO DI REMEZ CON SOLLYA")
        
        
        risultato_remez = calcola_polinomio_remez(intervallo, grado=5)
        
        if risultato_remez and risultato_remez['coefficienti']:
            coeffs = risultato_remez['coefficienti']
            errore = risultato_remez['errore']
            
            # Salva i coefficienti
            salva_coefficienti_remez(output_dir, coeffs, errore, intervallo, grado=5)
            
            print("\n  Coefficienti del polinomio di Remez (c0 + c1*x + c2*x^2 + ...):")
            for i, c in enumerate(coeffs):
                print(f"    c{i} = {c:.15f}")
            
            if errore is not None:
                print(f"  Errore di approssimazione: {errore:.6e}")
        else:
            print("  ERRORE: Impossibile calcolare il polinomio di Remez.")
    else:
        print("\n  SKIP: Sollya non disponibile.")
    
    # Salva gli indici
    np.savetxt(os.path.join(output_dir, "indices_train.txt"), np.arange(n_train_samples), fmt='%d')
    np.savetxt(os.path.join(output_dir, "indices_val.txt"), np.arange(len(y_val)), fmt='%d')
    np.savetxt(os.path.join(output_dir, "indices_test.txt"), np.arange(len(y_test)), fmt='%d')
    np.savetxt(os.path.join(output_dir, "indices_unlearned.txt"), indici_scelti, fmt='%d')
    
    # Salva validation e test set
    np.savetxt(os.path.join(output_dir, "X_val.txt"), X_val, fmt='%.10f')
    np.savetxt(os.path.join(output_dir, "y_val.txt"), y_val, fmt='%d' if args.is_classification else '%.10f')
    np.savetxt(os.path.join(output_dir, "X_test.txt"), X_test, fmt='%.10f')
    np.savetxt(os.path.join(output_dir, "y_test.txt"), y_test, fmt='%d' if args.is_classification else '%.10f')
    
    # 7. Padding e diagonali
    n_originale = H_inv.shape[0]
    n_padded = prossima_potenza_due(n_originale)
    
    H_inv_padded = np.zeros((n_padded, n_padded))
    H_inv_padded[:n_originale, :n_originale] = H_inv
    
    diagonali = matrice_a_diagonali(H_inv_padded, n_originale)
    
    # 8. Salva file principale
    txt_file = salva_file_output(
        output_dir, diagonali, punti_x, punti_y, w_star, 
        lambda_reg, grid_results, args, n_train_samples, n_features, args.is_classification,
        test_accuracy, test_mse
    )
    
    # 9. Salva parametri.txt
    params_file = os.path.join(output_dir, "parametri.txt")
    metric_name = "accuracy" if args.is_classification else "r2"
    with open(params_file, 'w') as f:
        f.write(f"dataset_id = {args.dataset_id}\n")
        f.write(f"n_punti = {args.n_punti}\n")
        f.write(f"seed = {args.seed}\n")
        f.write(f"is_classification = {args.is_classification}\n")
        f.write(f"lambda_reg = {lambda_reg}\n")
        f.write(f"max_train_size = {args.max_train_size}\n")
        f.write(f"n_train_samples = {n_train_samples}\n")
        f.write(f"n_validation_samples = {len(y_val)}\n")
        f.write(f"n_test_samples = {len(y_test)}\n")
        f.write(f"n_features = {n_features}\n")
        f.write(f"n_padded = {n_padded}\n")
        f.write(f"indici_scelti = {indici_scelti}\n")
        f.write(f"best_validation_score = {max([s for _, s in grid_results]):.6f}\n")
        f.write(f"test_{metric_name} = {test_accuracy:.6f}\n")
        f.write(f"test_mse = {test_mse:.6f}\n") 
        
    
    # 10. Report finale
    print(f"\n{'='*60}")
    print("ELABORAZIONE COMPLETATA")
    print(f"{'='*60}")
    print(f"File salvato: {txt_file}")
    print(f"  - Dimensioni matrice: {n_originale}x{n_originale}")
    print(f"  - Padding a: {n_padded}x{n_padded}")
    print(f"  - Numero diagonali: {len(diagonali)}")
    print(f"  - Punti selezionati: {len(punti_x)}/{n_train_samples}")
    print(f"  - Feature: {n_features}")
    print(f"  - Lambda ottimale: {lambda_reg:.4e}")
    print(f"  - Test {metric_name}: {test_accuracy:.6f}")
    print(f"Parametri salvati in: {params_file}")

if __name__ == "__main__":
    main()