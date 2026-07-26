// E' QUEllo giusto

#include <iostream>
#include <seal/seal.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace seal;


// dato un intero, trova la potenza di due maggiore più vicina
// usato nelle rotazioni dei vettori packed per riempire precisamente gli slot
int prossima_potenza_due_log(int n) {
    if (n <= 0) return 1;
    return int(pow(2, ceil(log2(n))));
}

// legge la matrice creata dal programma python e ci inizializza la matrice Ainv
vector<vector<double>> leggiMatriceDaFile(const std::string& filename) {
    vector<vector<double>> H_inv;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Errore: impossibile aprire il file " << filename << endl;
        return H_inv;
    }
    
    string line;
    // bool in_matrix_section = false; // dice se sto ancpra leggendo la matrice
    int righe_lette = 0;
    
    cout << "Lettura del file: " << filename << endl;
    
    while (getline(file, line)) {
        
            // Se la linea è vuota o contiene solo spazi, termina la lettura
            if (line.empty() || line.find_first_not_of(" \t") == string::npos) {
                break;
            }
            
            vector<double> riga;
            istringstream iss(line); // converte riga in uno stream
            double valore;
            
            while (iss >> valore) { // legge i valori separati da spazi
                riga.push_back(valore); // aggiunge elemento a fine vettore
            }
            
            if (!riga.empty()) {
                H_inv.push_back(riga); // aggiunge vettore a matrice
                righe_lette++;
            }
    }
    
    file.close();
    
    cout << "Lette " << righe_lette << " righe" << std::endl;
    if (!H_inv.empty()) {
        cout << "Dimensioni matrice: " << H_inv.size() << " x " << H_inv[0].size() << std::endl;
    }
    
    return H_inv;
}

// legge i coefficienti del polinomio di Remez dal file generato da ridge_benchmark.py
// si aspetta righe nel formato "ci = valore" (es. "c0 = 0.991107960988289")
vector<double> leggiCoeffDaFile(const std::string& filename) {
    vector<double> coeffs;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Errore: impossibile aprire il file " << filename << endl;
        return coeffs;
    }

    string line;
    cout << "Lettura del file: " << filename << endl;

    while (getline(file, line)) {
        // Salta righe vuote o commenti (iniziano con #)
        size_t prima_non_spazio = line.find_first_not_of(" \t");
        if (prima_non_spazio == string::npos || line[prima_non_spazio] == '#') {
            continue;
        }

        // Cerca il pattern "ci = valore"
        size_t pos_uguale = line.find('=');
        if (pos_uguale == string::npos) {
            continue;
        }

        string parte_sinistra = line.substr(0, pos_uguale);
        size_t pos_c = parte_sinistra.find('c');
        if (pos_c == string::npos) {
            continue;
        }

        try {
            int indice = stoi(parte_sinistra.substr(pos_c + 1));
            double valore = stod(line.substr(pos_uguale + 1));

            // Assicura che il vettore sia grande abbastanza
            if ((int)coeffs.size() <= indice) {
                coeffs.resize(indice + 1);
            }
            coeffs[indice] = valore;
        } catch (...) {
            // riga non nel formato atteso, la ignoro
            continue;
        }
    }

    file.close();

    cout << "Letti " << coeffs.size() << " coefficienti" << std::endl;

    return coeffs;
}

// decifra anche la matrice

vector<vector<double>> scrivi_matrice_su_file(
    const vector<Ciphertext>& Anew,
    SEALContext &context,
    Decryptor &decryptor,
    CKKSEncoder &encoder,
    size_t n,
    const string& filename
) {
    cout << "Scrittura nuova matrice su file: " << filename << endl;
    
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Errore: impossibile aprire il file " << filename << " per la scrittura" << endl;
        return vector<vector<double>>();
    }
    
    vector<vector<double>> Anew_vals (n, vector<double>(n));

    // Per ogni diagonale (che rappresenta una riga della matrice)
    for (int i = 0; i < n; i++) {
        Plaintext diag_pt;
        decryptor.decrypt(Anew[i], diag_pt);
        encoder.decode(diag_pt, Anew_vals[i]);
        
        // Per ogni elemento della riga
        for (int j = 0; j < n; j++) {
            double valore = Anew_vals[i][j];
            
            // Se il valore è molto vicino a zero, scrivi 0 - no
            /*if (fabs(valore) < soglia_zero) {
                file << "0";
            } else {
                file << valore;
            }*/
           file << valore; 
            
            // Aggiungi spazio tra gli elementi, ma non alla fine della riga
            if (j < n - 1) {
                file << " ";
            }
        }
        
        // nuovo a capo dopo ogni riga della matrice
        file << endl;
    }

    file<<endl<<endl;
    
    file.close();
    cout << "Matrice scritta con successo in " << filename << endl;

    return Anew_vals;
}

// trasforma matrice in formato diagonale in formato pieno

vector<vector<double>> diagonali_a_matrice(const vector<vector<double>>& diagonali, int n, int n_real) {
    vector<vector<double>> M(n_real, vector<double>(n_real, 0.0));
    
    for (int d = 0; d < n; d++) {
        for (int i = 0; i < n; i++) {
            int j = (i + d) % n;
            if (i < n_real && j < n_real) {
                M[i][j] = diagonali[d][i];
            }
        }
    }
    
    return M;
}

// Modifica della struttura DatiCompleti per includere multipli punti
struct DatiCompleti {
    vector<vector<double>> H_inv;
    vector<vector<double>> x_vals_list;  // Lista di vettori x
    vector<double> y_vals_list;          // Lista di y corrispondenti
    vector<double> w_star;
    vector<vector<double>> delta_vals_list; // Lista di delta (opzionale)
};



int leggiNtrainDaFileParametri(const std::string& filepath) {
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire il file " << filepath << std::endl;
        return -1;
    }
    
    std::string line;
    int n_train = -1;
    
    while (std::getline(file, line)) {
        if (line.find("n_train_samples") != std::string::npos) {
            size_t equal_pos = line.find('=');
            if (equal_pos != std::string::npos) {
                std::string value_str = line.substr(equal_pos + 1);
                value_str.erase(0, value_str.find_first_not_of(" \t"));
                value_str.erase(value_str.find_last_not_of(" \t") + 1);
                n_train = std::stoi(value_str);
                std::cout << "  Letto n_train_samples = " << n_train << std::endl;
                break;
            }
        }
    }
    
    file.close();
    return n_train;
}

// funzione per leggere multipli punti dal file (INCLUDENDO LAMBDA)
auto leggiTuttiIDatiMultipli(const std::string& filename) {
    vector<vector<double>> H_inv;
    vector<vector<double>> x_vals_list;
    vector<double> y_vals_list;
    vector<double> w_star;
    double lambda_reg = 0.01;
    
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Errore: impossibile aprire il file " << filename << endl;
        return make_tuple(H_inv, x_vals_list, y_vals_list, w_star, lambda_reg);
    }
    
    string line;
    bool in_matrix_section = true;
    int righe_lette = 0;
    
    cout << "Lettura del file: " << filename << endl;
    
    // ============================================================
    // PRIMA PARTE: LEGGI LA MATRICE FINCHE' NON TROVI x_random
    // ============================================================
    while (getline(file, line) && in_matrix_section) {
        // Leggi lambda dai commenti
        if (line.find("# lambda_ottimale") != string::npos) { // se trovo la stringa nella riga
            size_t equal_pos = line.find('='); // cerco l'=
            if (equal_pos != string::npos) {
                lambda_reg = stod(line.substr(equal_pos + 1)); // estrae sottostringa dopo = e converte in double
                cout << "  Lambda letto dal file: " << lambda_reg << endl;
            }
            continue;
        }
        
        // Salta commenti e righe vuote
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Se trova x_random, usciamo dalla sezione matrice
        if (line.find("x_random") != string::npos) {
            in_matrix_section = false;
            break;  // Esce dal while, la riga corrente sarà processata nel prossimo loop
        }
        
        // Se la riga contiene solo spazi, salta
        if (line.find_first_not_of(" \t") == string::npos) {
            continue;
        }
        
        // Leggi una riga di numeri (diagonale della matrice)
        vector<double> riga;
        istringstream iss(line);
        double valore;
        while (iss >> valore) {
            riga.push_back(valore);
        }
        
        if (!riga.empty()) {
            H_inv.push_back(riga);
            righe_lette++;
        }
    }
    
    cout << "Lette " << righe_lette << " righe per la matrice" << endl;
    
    // ============================================================
    // SECONDA PARTE: LEGGI TUTTI I PUNTI (x_random, y_random)
    // ============================================================
    vector<double> current_x;
    int punti_letti = 0;
    
    // Processa la riga che ha causato l'uscita (se contiene x_random)
    if (!line.empty() && line.find("x_random") != string::npos) {
        // Estrai x_random da questa riga
        size_t start = line.find('[');
        size_t end = line.find(']');
        if (start != string::npos && end != string::npos) {
            string values_str = line.substr(start + 1, end - start - 1);
            istringstream values_ss(values_str);
            string token;
            current_x.clear();
            while (getline(values_ss, token, ',')) {
                token.erase(remove(token.begin(), token.end(), ' '), token.end()); // rimuove gli spazi: remove sposta alla fine e erase cancella
                if (!token.empty()) {
                    current_x.push_back(stod(token));
                }
            }
        }
    }
    
    // Continua a leggere il resto del file
    while (getline(file, line)) {
        // Salta righe vuote
        if (line.empty() || line.find_first_not_of(" \t") == string::npos) {
            continue;
        }
        
        // Commento: salta
        if (line[0] == '#') {
            continue;
        }
        
        // Se troviamo w_star, abbiamo finito i punti
        if (line.find("w_star") != string::npos) {
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != string::npos && end != string::npos) {
                string values_str = line.substr(start + 1, end - start - 1);
                istringstream values_ss(values_str);
                string token;
                while (getline(values_ss, token, ',')) {
                    token.erase(remove(token.begin(), token.end(), ' '), token.end());
                    if (!token.empty()) {
                        w_star.push_back(stod(token));
                    }
                }
                cout << "Letto w_star con " << w_star.size() << " elementi" << endl;
            }
            break;  // Fine file
        }
        
        // Leggi x_random
        if (line.find("x_random") != string::npos) {
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != string::npos && end != string::npos) {
                string values_str = line.substr(start + 1, end - start - 1);
                istringstream values_ss(values_str);
                string token;
                current_x.clear();
                while (getline(values_ss, token, ',')) {
                    token.erase(remove(token.begin(), token.end(), ' '), token.end());
                    if (!token.empty()) {
                        current_x.push_back(stod(token));
                    }
                }
            }
        }
        // Leggi y_random
        else if (line.find("y_random") != string::npos) {
            size_t equal_pos = line.find('=');
            if (equal_pos != string::npos) {
                string val_str = line.substr(equal_pos + 1);
                val_str.erase(remove(val_str.begin(), val_str.end(), ' '), val_str.end());
                double y_val = stod(val_str);
                
                if (!current_x.empty()) {
                    x_vals_list.push_back(current_x);
                    y_vals_list.push_back(y_val);
                    punti_letti++;
                    //cout << "Letto punto " << punti_letti << ": y=" << y_val << endl;
                }
            }
            current_x.clear();
        }
    }
    
    file.close();
    
    cout << "\nLette " << x_vals_list.size() << " coppie (x, y) dal file" << endl;
    cout << "Lambda regolarizzazione: " << lambda_reg << endl;
    
    return make_tuple(H_inv, x_vals_list, y_vals_list, w_star, lambda_reg);
}

// allinea i livelli di due cifrati al minore
void align_ciphertexts(Ciphertext &a, Ciphertext &b, Evaluator &evaluator, SEALContext &context) {
    auto a_level = context.get_context_data(a.parms_id())->chain_index();
    auto b_level = context.get_context_data(b.parms_id())->chain_index();
    size_t min_level = min(a_level, b_level);

    while (context.get_context_data(a.parms_id())->chain_index() > min_level)
        evaluator.mod_switch_to_next_inplace(a);
    while (context.get_context_data(b.parms_id())->chain_index() > min_level)
        evaluator.mod_switch_to_next_inplace(b);

    
}



// Calcola la norma L1 dell'errore empirico tra due vettori
double compute_l1_error(const vector<double>& a, const vector<double>& b, size_t n_real) {
    double err = 0.0;
    for (size_t i = 0; i < n_real; i++) {
        err += fabs(a[i] - b[i]);
    }
    return err;
}

// Scrive i bound teorici e pratici in un file unico (append, indipendente dal seed)
void write_bounds_to_file(
    const string& output_dir,    // directory del seed (per ricavare dataset_id)
    int seed,
    int degree,
    double eps_remez,
    double bound_sample_dependent,
    double bound_uniforme,
    double error_empirico,       // L1 tra w_enc e w_clear per il punto rimosso
    double error_empirico_uniforme, // massimo errore su tutti i punti (opzionale)
    double error_chiaro_vs_retrain
) {
    // Costruisci il path del file unico nella directory base (non nella sottocartella seed)
    string base_dir = output_dir;
    // Risali di un livello per ottenere la directory base (rimuovi "/seed_XX")
    size_t pos = base_dir.rfind("/seed_");
    if (pos != string::npos) {
        base_dir = base_dir.substr(0, pos);
    }
    
    string filename = base_dir + "/bounds_teorici_pratici.txt";
    
    // Apri in modalità append
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "ERRORE: Impossibile aprire il file " << filename << endl;
        return;
    }
    
    // Se il file è vuoto, scrivi l'intestazione
    if (file.tellp() == 0) {
        file << "# seed, degree, eps_remez, bound_sample_dependent, bound_uniforme, "
             << "error_empirico_L1, error_empirico_uniforme, error_sample_bound_ratio, "
             << "error_uniform_bound_ratio" << endl;
    }
    
    // Calcola i rapporti
    double ratio_sample = (bound_sample_dependent > 0) ? error_empirico / bound_sample_dependent : 0.0;
    double ratio_uniform = (bound_uniforme > 0) ? error_empirico_uniforme / bound_uniforme : 0.0;
    
    // Scrivi i dati
    cout<<"scrivo i dati su file"<<endl;
    file << seed << ", "
         << degree << ", "
         << scientific << setprecision(10) << eps_remez << ", "
         << bound_sample_dependent << ", "
         << bound_uniforme << ", "
         << error_empirico << ", "
         << error_empirico_uniforme << ", "
         << ratio_sample << ", "
         << ratio_uniform << ", "
         << error_chiaro_vs_retrain << endl;
    
    file.close();
    cout << "Bounds scritti in: " << filename << endl;
}


// allinea i livelli di due cifrati al minore e allinea anche le scale
// da utilizzare quando le scale sono molto simili ma non uguali
void align_ciphertexts_scale(Ciphertext &a, Ciphertext &b, Evaluator &evaluator, SEALContext &context) {
    auto a_level = context.get_context_data(a.parms_id())->chain_index();
    auto b_level = context.get_context_data(b.parms_id())->chain_index();
    if (a_level>b_level) evaluator.mod_switch_to_inplace(a,b.parms_id());
    else if (a_level<b_level) evaluator.mod_switch_to_inplace(b,a.parms_id());

    if (a.scale() < b.scale())
        a.scale() = b.scale();
    if (a.scale() > b.scale())
        b.scale() = a.scale();
    
}

// esegue il prodotto scalare tra i due vettori:
// li allinea, moltiplica tramite multiply, rilinearizzando e scalando al prossimo livello
// replica il vettore risultato in ogni slot utilizzando rotazioni e somme

Ciphertext moltiplica_vettori(
    const Ciphertext &x,                
    const Ciphertext &y,
    SEALContext &context,            
    Evaluator &evaluator,
    CKKSEncoder &encoder,
    Decryptor &decryptor,
    RelinKeys &relin_keys,
    GaloisKeys &galois_keys,
    size_t n,
    double scale
) 
{

    Ciphertext ct_x=x;
    Ciphertext ct_y=y; // salvo in copie per non modificare gli originali
    int slot_count = encoder.slot_count();

    // stampa vettori prima per debugging
    // cout<<"stampo i due vettori PRIMA"<<endl;
    /*
    Plaintext pt_x;
        decryptor.decrypt(ct_x, pt_x);
        vector<double> xxx;
        encoder.decode(pt_x, xxx);
        cout << "(X):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "x[" << i << "] = " << xxx[i] << endl;
        }
     Plaintext pt_y;
        decryptor.decrypt(ct_y, pt_y);

        vector<double> yyy;
        encoder.decode(pt_y, yyy);

        cout << "(Y):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "y[" << i << "] = " << yyy[i] << endl;
        }

    */

    //cout << "\nEseguo align.." << std::endl;
    align_ciphertexts(ct_x, ct_y, evaluator, context);
    Ciphertext result;

    //cout << "\nEseguo moltiplicazione..." << std::endl;
    evaluator.multiply(ct_x, ct_y, result);
    evaluator.relinearize_inplace(result, relin_keys);
    evaluator.rescale_to_next_inplace(result);


    // stampa risultato moltiplicazione per debug

    /*
    vector<double> yy;
    Plaintext pt_res3;
        decryptor.decrypt(result, pt_res3);

        
        encoder.decode(pt_res3, yy);

        cout << "(primi 2n slots QUA):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "yy[" << i << "] = " << yy[i] << endl;
        }
    */

    // RIPETO IL VETTORE RISULTATO PER CALCOLARE LE SOMME IN TUTTI I PRIMI N SLOT
    //NOTA: la moltiplicazione per la maschera non è necessaria (non aumenta l'errore e aggiunge una moltiplicazione)
/*
    vector<double> mask(slot_count, 0.0);
    for (size_t i = 0; i < n; i++) mask[i] = 1.0;
    
    Plaintext plain_mask;
    encoder.encode(mask, scale, plain_mask);
    
    // Applica la maschera per azzerare gli slot oltre n
    evaluator.multiply_plain_inplace(result, plain_mask);
    // RIGUARDA: qua potresti dover riscalare
*/

    // Rotazioni per replicare il vettore (di n, 2n,...)
    Ciphertext rotated;

    size_t block = n;
    while (block < slot_count) {
        evaluator.rotate_vector(result, block, galois_keys, rotated);
        evaluator.add_inplace(result, rotated);
        block *= 2;
    }

    // stampa per debug
    /*
    Plaintext pt_res3;
        decryptor.decrypt(result, pt_res3);

        //vector<double> yy;
        encoder.decode(pt_res3, yy);

        cout << "(first 2n slots):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "yy[" << i << "] = " << yy[i] << endl;
        }
            /*
    
    /*for (size_t offset = 1; offset < slot_count; offset *= 2) {
        Ciphertext rotated;
        evaluator.rotate_vector(result, offset, galois_keys, rotated);
        evaluator.add_inplace(result, rotated);
    }*/

    // ROTAZIONI PER SOMMARE (logaritmiche da 1 a n-1)

    Ciphertext rotated2;

    for (size_t step = 1; step < n; step*=2) {
        evaluator.rotate_vector(result, step, galois_keys, rotated2);
        
        // stampa debug
        /*
        Plaintext pt_r;
        decryptor.decrypt(result, pt_r);

        vector<double> yr;
        encoder.decode(pt_r, yr);

        cout << "rotato:" << endl;
        for (size_t i = 0; i <= n; i++)
        {
            cout << "y[" << i << "] = " << yr[i] << endl;
        }*/

        evaluator.add_inplace(result, rotated2);

        // stampa debug
        /*
        Plaintext pt_res;
        decryptor.decrypt(result, pt_res);

        vector<double> y;
        encoder.decode(pt_res, y);

        cout << "Output y (first n slots):" << endl;
        for (size_t i = 0; i < n; i++)
        {
            cout << "y[" << i << "] = " << y[i] << endl;
        }*/
    }
    return result;
}


// come moltiplica_vettori, ma uno è plain
// serve nel calcolo di delta
Ciphertext moltiplica_vettori_plain(
    vector<double> x,                
    const Ciphertext &y,
    SEALContext &context,            
    Evaluator &evaluator,
    CKKSEncoder &encoder,
    Decryptor &decryptor,
    RelinKeys &relin_keys,
    GaloisKeys &galois_keys,
    size_t n,
    double scale
) 
{   
    Ciphertext ct_y=y;
    int slot_count = encoder.slot_count();
    
    // stampa debug
    /*
    cout<<"stampo i due vettori PRIMA"<<endl;
    Plaintext pt_x;
        decryptor.decrypt(ct_x, pt_x);

        vector<double> xxx;
        encoder.decode(pt_x, xxx);

        cout << "(X):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "x[" << i << "] = " << xxx[i] << endl;
        }

     Plaintext pt_y;
        decryptor.decrypt(ct_y, pt_y);

        vector<double> yyy;
        encoder.decode(pt_y, yyy);

        cout << "(Y):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "y[" << i << "] = " << yyy[i] << endl;
        }

    */

    Plaintext pt_x;
    encoder.encode(x, y.scale(), pt_x); // encoding con scale di y per compatibilità

    //cout << "\nEseguo moltiplicazione..." << endl;
    // Moltiplica x_n per w*^T
    Ciphertext result;
    evaluator.multiply_plain(ct_y, pt_x, result);
    evaluator.relinearize_inplace(result, relin_keys);
    evaluator.rescale_to_next_inplace(result);

    // stampa debug
    /*
    vector<double> yy;
    Plaintext pt_res3;
        decryptor.decrypt(result, pt_res3);

        
        encoder.decode(pt_res3, yy);

        cout << "(primi 2n slots QUA):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "yy[" << i << "] = " << yy[i] << endl;
        }
    */

// moltiplica per mask - non necessaria
/*
    vector<double> mask(slot_count, 0.0);
    for (size_t i = 0; i < n; i++) mask[i] = 1.0;
    
    Plaintext plain_mask;
    encoder.encode(mask, scale, plain_mask);
    
    // Applica la maschera per azzerare gli slot oltre n
    evaluator.multiply_plain_inplace(result, plain_mask);
    // RIGUARDA: qua potresti dover riscalare
    evaluator.relinearize_inplace(result, relin_keys);  // Riduce dimensione da 3 a 2
    evaluator.rescale_to_next_inplace(result);          // Per CKKS, riduce la scala
*/

    // 2. Replica usando rotazioni e somme
    // Rotazione di n posizioni per copiare i primi n slot nei secondi n slot
    Ciphertext rotated;
    size_t block = n;
    while (block < slot_count) {
        evaluator.rotate_vector(result, block, galois_keys, rotated);
        evaluator.add_inplace(result, rotated);
        block *= 2;
    }

    //stampa debug
    /*
    Plaintext pt_res3;
        decryptor.decrypt(result, pt_res3);

        //vector<double> yy;
        encoder.decode(pt_res3, yy);

        cout << "(first 2n slots):" << endl;
        for (size_t i = 0; i < 2*n; i++)
        {
            cout << "yy[" << i << "] = " << yy[i] << endl;
        }
    */

    // ROTAZIONI PER SOMMARE

    Ciphertext rotated2;

    for (size_t step = 1; step < n; step*=2) {
        evaluator.rotate_vector(result, step, galois_keys, rotated2);
        // stampa debug
        /*
        Plaintext pt_r;
        decryptor.decrypt(result, pt_r);

        vector<double> yr;
        encoder.decode(pt_r, yr);

        cout << "rotato:" << endl;
        for (size_t i = 0; i <= n; i++)
        {
            cout << "y[" << i << "] = " << yr[i] << endl;
        }*/

        evaluator.add_inplace(result, rotated2);

        // stampa debug
        /*
        Plaintext pt_res;
        decryptor.decrypt(result, pt_res);

        vector<double> y;
        encoder.decode(pt_res, y);

        cout << "Output y (first n slots):" << endl;
        for (size_t i = 0; i < n; i++)
        {
            cout << "y[" << i << "] = " << y[i] << endl;
        }*/
    }
    return result;
}


// eseguo prodotto esterno del vettore per se stesso

vector<Ciphertext> prodotto_esterno_vettore(
    const Ciphertext &ct_x,                // Vettore cifrato di input
    SEALContext &context,            // Contesto SEAL
    Evaluator &evaluator,            // Evaluator
    Decryptor &decryptor,            // Decryptor (per debug opzionale)
    CKKSEncoder &encoder,            // Encoder CKKS
    const GaloisKeys &galois_keys,   // Chiavi per rotazioni
    RelinKeys &relin_keys,
    size_t n,                        // Dimensione matrice/vettore
    size_t n_real,
    double scale                     // Scala CKKS
) { 
    size_t slot_count = encoder.slot_count();

    // replica del vettore

     Plaintext pt_debug;
        decryptor.decrypt(ct_x, pt_debug);
        vector<double> debug_vals;
        encoder.decode(pt_debug, debug_vals);
        
        // stampa debug
        /*cout << "Vettore iniziale (primi " << 2*n << " slot):" << endl;
        for (size_t i = 0; i < 2*n; i++) {
            cout << "  [" << i << "] = " << debug_vals[i];
            if (i >= n) cout << " (replica)";
            cout << endl;
        }*/


    Ciphertext ct_x_rep = ct_x;
    // moltiplico per maschera 1xn 0... per isolare i primi n slot e avere 0 sugli altri
    vector<double> mask_first_n(slot_count, 0.0);
    for (size_t i = 0; i < n; i++) mask_first_n[i] = 1.0;

    Plaintext plain_mask_first_n;
    // creo maschera allineata con vettore
    encoder.encode(mask_first_n, ct_x.scale(), plain_mask_first_n);
    evaluator.mod_switch_to_inplace(plain_mask_first_n, ct_x.parms_id());
    evaluator.multiply_plain_inplace(ct_x_rep, plain_mask_first_n);
    evaluator.rescale_to_next_inplace(ct_x_rep); // FIX: mancava, raddoppiava la scala (~40 -> ~80 bit)
    
    // replica
    for (size_t offset = n; offset < slot_count; offset *=2) {
        Ciphertext rotated;
        evaluator.rotate_vector(ct_x_rep, offset, galois_keys, rotated);
        evaluator.add_inplace(ct_x_rep, rotated);
    }
    

    // stampa debug opzionale, mostra vettore replicato
    bool debug_mode = false;
    if (debug_mode) {
        Plaintext pt_debug;
        decryptor.decrypt(ct_x_rep, pt_debug);
        vector<double> debug_vals;
        encoder.decode(pt_debug, debug_vals);
        
        cout << "Vettore replicato (primi " << 2*n << " slot):" << endl;
        for (size_t i = 0; i < 2*n; i++) {
            cout << "  [" << i << "] = " << debug_vals[i];
            if (i >= n) cout << " (replica)";
            cout << endl;
        }
    }
    

    vector<Ciphertext> A(n);
    for(int i=0;i<n;i++) 
    {
        // parto dal vettore v
        A[i]=ct_x;
        // ruoto v di i posizioni
        Ciphertext rotated;
        if (i == 0) {
            // diagonale principale: non ruotare
            rotated = ct_x_rep;
        } else {
            // Altre diagonali: ruota di d posizioni
            evaluator.rotate_vector(ct_x_rep, i, galois_keys, rotated);
        }

        
        // stampa debug
        // stampo rotated e diagonals di d
        /*
        Plaintext pt_dbg;
        decryptor.decrypt(rotated, pt_dbg);

        vector<double> vals_dbg;
        encoder.decode(pt_dbg, vals_dbg);

        cout << "\n[DEBUG] d = " << d << " | rotated (first " << n << " slots):\n";
        for (size_t i = 0; i < n; i++) {
            cout << "  slot[" << i << "] = " << vals_dbg[i] << endl;
        }*/

        /*cout << "  scale = " << log2(rotated.scale()) << " bits"
            << " | level = "
            << context.get_context_data(rotated.parms_id())->chain_index()
            << endl;*/


        // moltiplica per la diagonale
        evaluator.mod_switch_to_inplace(A[i], rotated.parms_id());
        evaluator.multiply_inplace(A[i], rotated);
        evaluator.relinearize_inplace(A[i], relin_keys);
        //cout << "scala: " << rotated.scale() << endl;
        evaluator.rescale_to_next_inplace(A[i]);

    }
   
    return A;
}

// divide vecchio che decifrava
// decifra, calcola l'inverso (1/x) per ogni slot e ricifra
/*Ciphertext divide(
    const Ciphertext& ct,           // Ciphertext da invertire
    SEALContext& context,
    Encryptor& encryptor,
    Decryptor& decryptor,
    Evaluator& evaluator,
    CKKSEncoder& encoder,
    double scale) {
    cout << "\n=== DIVIDE NAIVE ===" << endl;
    
    // 1. Decifra il ciphertext
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    
    // 2. Decodifica in un vettore di double
    vector<double> values;
    encoder.decode(pt, values);
    
    
    
    // 3. Calcola l'inverso per ogni slot
    vector<double> inverses(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        if (abs(values[i]) < 1e-10) {  // Evita divisione per zero
            inverses[i] = 0.0;
            //cout << "ATTENZIONE: slot " << i << " quasi zero (" << values[i] 
             //    << "), imposto inverso a 0" << endl;
        } else {
            inverses[i] = 1.0 / values[i];
        }
    }
    
    cout << "Valori inversi (primi 5 slot):" << endl;
    for (size_t i = 0; i < min((size_t)5, inverses.size()); i++) {
        cout << "  inv[" << i << "] = " << inverses[i] 
             << " (1/" << values[i] << ")" << endl;
    }
    
    // 4. Ricodifica in plaintext
    Plaintext pt_inverse;
    encoder.encode(inverses, scale, pt_inverse);
    
    // 5. Ricifra
    Ciphertext ct_inverse;
    encryptor.encrypt(pt_inverse, ct_inverse);
    
    
    
    return ct_inverse;
}
*/



// Funzione per matrice × vettore con replica e packing diagonali
// codifico la matrice per diagonali
// replico il vettore in ogni slot
// moltiplico diagonale i x vettore ruotato di i posizioni

// NOTA: sia la matrice che i vettori sono padded con 0 fino alla potenza di due successiva, per rendere le rotazioni precise

Ciphertext moltiplica_matrice_vettore(
    const Ciphertext &ct_x,                // Vettore cifrato di input
    const vector<vector<double>> &A, // Matrice n×n in chiaro
    SEALContext &context,            // Contesto SEAL
    Evaluator &evaluator,            // Evaluator
    Decryptor &decryptor,            // Decryptor (per debug opzionale)
    CKKSEncoder &encoder,            // Encoder CKKS
    const GaloisKeys &galois_keys,   // Chiavi per rotazioni
    size_t n,                        // Dimensione matrice/vettore
    size_t n_real,
    double scale                     // Scala CKKS
) {
    size_t slot_count = encoder.slot_count();

    // replica del vettore

     Plaintext pt_debug;
        decryptor.decrypt(ct_x, pt_debug);
        vector<double> debug_vals;
        encoder.decode(pt_debug, debug_vals);
        
        // stampa debug
        /*cout << "Vettore iniziale (primi " << 2*n << " slot):" << endl;
        for (size_t i = 0; i < 2*n; i++) {
            cout << "  [" << i << "] = " << debug_vals[i];
            if (i >= n) cout << " (replica)";
            cout << endl;
        }*/


    Ciphertext ct_x_rep = ct_x;
    // moltiplico per maschera 1xn 0... per isolare i primi n slot e avere 0 sugli altri
    vector<double> mask_first_n(slot_count, 0.0);
    for (size_t i = 0; i < n; i++) mask_first_n[i] = 1.0;

    Plaintext plain_mask_first_n;
    // creo maschera allineata con vettore
    encoder.encode(mask_first_n, ct_x.scale(), plain_mask_first_n);
    evaluator.mod_switch_to_inplace(plain_mask_first_n, ct_x.parms_id());
    evaluator.multiply_plain_inplace(ct_x_rep, plain_mask_first_n);
    evaluator.rescale_to_next_inplace(ct_x_rep); // FIX: mancava, raddoppiava la scala (~40 -> ~80 bit),
                                                  // causa di "scale out of bounds" a valle (beta/alfa/divide)
    
    // replica
    for (size_t offset = n; offset < slot_count; offset *=2) {
        Ciphertext rotated;
        evaluator.rotate_vector(ct_x_rep, offset, galois_keys, rotated);
        evaluator.add_inplace(ct_x_rep, rotated);
    }
    

    // stampa debug opzionale, mostra vettore replicato
    bool debug_mode = false;
    if (debug_mode) {
        Plaintext pt_debug;
        decryptor.decrypt(ct_x_rep, pt_debug);
        vector<double> debug_vals;
        encoder.decode(pt_debug, debug_vals);
        
        cout << "Vettore replicato (primi " << 2*n << " slot):" << endl;
        for (size_t i = 0; i < 2*n; i++) {
            cout << "  [" << i << "] = " << debug_vals[i];
            if (i >= n) cout << " (replica)";
            cout << endl;
        }
    }
    

    // encoding diagonali (la matrice è già organizzata in diagonali e padded dal programma python)
    vector<Plaintext> diagonals(n);
    for (size_t d = 0; d < n; d++) {
        encoder.encode(A[d], ct_x.scale(), diagonals[d]);
    }
    
    // moltiplicazione matrice-vettore

    //cout << "calcolo matrix × vector..." << endl;
    Ciphertext result;
    bool initialized = false;
    
    for (size_t d = 0; d < n; d++) {
        //cout<<"1"<<endl;
        Ciphertext rotated;
        //cout<<"2"<<endl;
        if (d == 0) {
            // diagonale principale: non ruotare
            rotated = ct_x_rep;
        } else {
            // Altre diagonali: ruota di d posizioni
            evaluator.rotate_vector(ct_x_rep, d, galois_keys, rotated);
            //cout<<"3"<<endl;
        }

        
        // stampa debug
        // stampo rotated e diagonals di d
        
        Plaintext pt_dbg;
        decryptor.decrypt(rotated, pt_dbg);

        vector<double> vals_dbg;
        encoder.decode(pt_dbg, vals_dbg);

        /*cout << "\n[DEBUG] d = " << d << " | rotated (first " << n << " slots):\n";
        for (size_t i = 0; i < n; i++) {
            cout << "  slot[" << i << "] = " << vals_dbg[i] << endl;
        }*/

        /*cout << "  scale = " << log2(rotated.scale()) << " bits"
            << " | level = "
            << context.get_context_data(rotated.parms_id())->chain_index()
            << endl;*/


        // FIX: controllo spostato QUI, PRIMA della multiply_plain_inplace.
        // Prima il controllo veniva fatto DOPO aver già moltiplicato per diagonals[d]:
        // se la diagonale è (quasi) tutta zero (es. diagonali di padding oltre n_real,
        // o per coincidenza nei dati reali), multiply_plain_inplace produce un
        // ciphertext "trasparente" e SEAL lancia "result ciphertext is transparent"
        // PRIMA che il codice arrivi mai a leggere max_diag.
        //double max_diag = *max_element(A[d].begin(), A[d].end(), [](double a, double b){ return fabs(a) < fabs(b); });
        /*if (diagonals[d].is_transparent()) {
            // salta completamente questa diagonale: niente moltiplicazione, niente add
            cout << "diagonale saltata perchè piccola (d=" << d << ")" << endl;
            continue;
        }*/


        try {
            // moltiplica per la diagonale
            evaluator.mod_switch_to_inplace(diagonals[d], rotated.parms_id());
            evaluator.multiply_plain_inplace(rotated, diagonals[d]);
            //cout << "scala: " << rotated.scale() << endl;
            evaluator.rescale_to_next_inplace(rotated);
        } catch (const std::logic_error &e) {
            cout << "diagonale d=" << d << " -> ciphertext trasparente, salto (" << e.what() << ")" << endl;
            continue;
        }

        vector<double> diag_dbg;
        encoder.decode(diagonals[d], diag_dbg);

        /*cout << "\n[DEBUG] d = " << d << " | diagonale (first " << n << " slots):\n";
        for (size_t i = 0; i < n; i++) {
            cout << "  slot[" << i << "] = " << diag_dbg[i] << endl;
        }*/

        
        
        if (!initialized) {
            // prima iterazione (o prima diagonale non nulla trovata): assegna direttamente
            result = rotated;
            initialized = true;
        } else {
            // Allinea livello e somma
            evaluator.mod_switch_to_inplace(rotated, result.parms_id());
            //cout<< "aggiungo"<<endl;
            evaluator.add_inplace(result, rotated);
        }

    }
    //evaluator.rescale_to_next_inplace(result);


    // stampa debug
    /*
     cout << " scale = " << log2(result.scale()) << " bits"
            << " | level = "
            << context.get_context_data(result.parms_id())->chain_index()
            << endl;
    */
   
    return result;
}

// divide vecchio che decifrava
// decifra, calcola l'inverso (1/x) per ogni slot e ricifra
/*Ciphertext divide(
    const Ciphertext& ct,           // Ciphertext da invertire
    SEALContext& context,
    Encryptor& encryptor,
    Decryptor& decryptor,
    Evaluator& evaluator,
    CKKSEncoder& encoder,
    double scale) {
    cout << "\n=== DIVIDE NAIVE ===" << endl;
    
    // 1. Decifra il ciphertext
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    
    // 2. Decodifica in un vettore di double
    vector<double> values;
    encoder.decode(pt, values);
    
    
    
    // 3. Calcola l'inverso per ogni slot
    vector<double> inverses(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        if (abs(values[i]) < 1e-10) {  // Evita divisione per zero
            inverses[i] = 0.0;
            //cout << "ATTENZIONE: slot " << i << " quasi zero (" << values[i] 
             //    << "), imposto inverso a 0" << endl;
        } else {
            inverses[i] = 1.0 / values[i];
        }
    }
    
    cout << "Valori inversi (primi 5 slot):" << endl;
    for (size_t i = 0; i < min((size_t)5, inverses.size()); i++) {
        cout << "  inv[" << i << "] = " << inverses[i] 
             << " (1/" << values[i] << ")" << endl;
    }
    
    // 4. Ricodifica in plaintext
    Plaintext pt_inverse;
    encoder.encode(inverses, scale, pt_inverse);
    
    // 5. Ricifra
    Ciphertext ct_inverse;
    encryptor.encrypt(pt_inverse, ct_inverse);
    
    
    
    return ct_inverse;
}
*/


// calcola l'inverso di un numero (1+x)
// usa approssimazione polinomiale con coeffcienti precalcolati con algoritmo remez
// il grado del polinomio è 5
// l'intervallo è 1-2, perchè il modulo di x è con alta probabilità tra -1 e 1 e 
// l'hessiana è definita positiva, quindi xAx è > 0 -> 1+xAx sta tra 1 e 2



/*
// vecchio divide, solo per grado 5
Ciphertext divide(
    Ciphertext &ct_x,                 // x già cifrato
    SEALContext &context,
    Evaluator &evaluator,
    CKKSEncoder &encoder,
    Encryptor &encryptor,
    RelinKeys &relin_keys,
    GaloisKeys &galois_keys,
    size_t n,                         // dimensione del vettore da usare
    double scale
) {

    // Calcolo potenze cifrate x^0 ... x^5
    vector<Ciphertext> powers(6);
    
    // x^0 = 1 (plaintext costante)
    vector<double> ones(n, 1.0);
    Plaintext plain_one;
    encoder.encode(ones, scale, plain_one);
    encryptor.encrypt(plain_one, powers[0]);

    // x^1 =x
    powers[1] = ct_x;

    // x^2
    Ciphertext ct_x2;
    evaluator.multiply(powers[1], powers[1], ct_x2);
    evaluator.relinearize_inplace(ct_x2, relin_keys);
    evaluator.rescale_to_next_inplace(ct_x2);
    powers[2] = ct_x2;

    // x^3
    align_ciphertexts(powers[1], powers[2], evaluator, context);
    Ciphertext ct_x3;
    evaluator.multiply(powers[1], powers[2], ct_x3);
    evaluator.relinearize_inplace(ct_x3, relin_keys);
    evaluator.rescale_to_next_inplace(ct_x3);
    powers[3] = ct_x3;

    // x^4
    Ciphertext ct_x4;
    evaluator.multiply(powers[2], powers[2], ct_x4);
    evaluator.relinearize_inplace(ct_x4, relin_keys);
    evaluator.rescale_to_next_inplace(ct_x4);
    powers[4] = ct_x4;

    // x^5
    align_ciphertexts(powers[3], powers[2], evaluator, context);
    Ciphertext ct_x5;
    evaluator.multiply(powers[3], powers[2], ct_x5);
    evaluator.relinearize_inplace(ct_x5, relin_keys);
    evaluator.rescale_to_next_inplace(ct_x5);
    powers[5] = ct_x5;

    // ============================================================
    // 2. Moltiplicazione per coefficienti polinomiale
    // ============================================================
    const vector<double> COEFFS = {
         8.5710678742407527,     // x^0
        -30.2408012841011456,    // x^1
        56.2203518755371966,    // x^2
        -58.0919760279726909,    // x^3
        31.6400756261164759,    // x^4
        -7.0987924021784943     // x^5
    };

    // encoding dei coefficienti
    vector<Plaintext> plain_coeffs(6);
    for (int i = 0; i <= 5; i++) {
        vector<double> coeff_vector(n, COEFFS[i]);
        encoder.encode(coeff_vector, scale, plain_coeffs[i]);
    }

    // moltiplico coefficienti per potenze
    vector<Ciphertext> terms(6);
    for (int i = 0; i <= 5; i++) {
        evaluator.mod_switch_to_inplace(plain_coeffs[i], powers[i].parms_id());
        evaluator.multiply_plain(powers[i], plain_coeffs[i], terms[i]);
        evaluator.relinearize_inplace(terms[i], relin_keys);
        evaluator.rescale_to_next_inplace(terms[i]);
    }


    // Somma dei termini
    // NOTA: la somma si può trasformare in un for normale, è fatta così per ridurre la quantità di cambi di scala
    Ciphertext somma01, somma23, somma45;
    align_ciphertexts_scale(terms[1], terms[0], evaluator, context);
    evaluator.add(terms[0], terms[1], somma01);

    align_ciphertexts_scale(terms[3], terms[2], evaluator, context);
    evaluator.add(terms[2], terms[3], somma23);

    align_ciphertexts_scale(terms[5], terms[4], evaluator, context);
    evaluator.add(terms[4], terms[5], somma45);

    Ciphertext somma03;
    align_ciphertexts_scale(somma01, somma23, evaluator, context);
    evaluator.add(somma01, somma23, somma03);

    Ciphertext result;
    align_ciphertexts_scale(somma03, somma45, evaluator, context);
    evaluator.add(somma03, somma45, result);

    return result;
}

*/



Ciphertext divide(
    Ciphertext &ct_x,                 // x già cifrato
    SEALContext &context,
    Evaluator &evaluator,
    CKKSEncoder &encoder,
    Encryptor &encryptor,
    RelinKeys &relin_keys,
    GaloisKeys &galois_keys,
    size_t n,                         // dimensione del vettore da usare
    double scale,
    const vector<double> &COEFFS,     // coefficienti del polinomio di Remez, letti una volta nel main
    int degree = 5                    // grado del polinomio (2-8)
) {
    // ============================================================
    // 1. Coefficienti dei polinomi di Remez (grado 2-8)
    // Ogni riga contiene i coefficienti dal grado più basso al più alto
    // ============================================================
    // Seleziona i coefficienti in base al grado richiesto
    if (degree < 2 || degree > 8) {
        throw std::invalid_argument("Seleziona grado tra 2 e 8");
    }

    // ============================================================
    // 1. Calcolo delle potenze cifrate x^0 ... x^degree
    // ============================================================
    vector<Ciphertext> powers(degree + 1);
    
    // x^0 = 1 (plaintext costante cifrato)
    vector<double> ones(n, 1.0);
    Plaintext plain_one;
    encoder.encode(ones, scale, plain_one);
    encryptor.encrypt(plain_one, powers[0]);

    // x^1 = x
    powers[1] = ct_x;

    // Calcolo potenze successive: x^i = x^(i-1) * x
    for (int i = 2; i <= degree; i++) {
        Ciphertext ct_power;
        //cout<<"DEBUG: inizio moltiplicazione x^"<<i<<endl;
        align_ciphertexts(powers[i-1], powers[1], evaluator, context);
        evaluator.multiply(powers[i-1], powers[1], ct_power);
        evaluator.relinearize_inplace(ct_power, relin_keys);
        evaluator.rescale_to_next_inplace(ct_power);
        powers[i] = ct_power;
    }

    // ============================================================
    // 2. Moltiplicazione coefficienti per potenze
    // ============================================================
    vector<Plaintext> plain_coeffs(degree + 1);
    for (int i = 0; i <= degree; i++) {
        vector<double> coeff_vector(n, COEFFS[i]);
        encoder.encode(coeff_vector, scale, plain_coeffs[i]);
    }

    //cout<<"DEBUG: moltiplico per i coefficienti"<<endl;
    vector<Ciphertext> terms(degree + 1);
    for (int i = 0; i <= degree; i++) {
        //cout<<"DEBUG:switch n."<<i<<endl;
        evaluator.mod_switch_to_inplace(plain_coeffs[i], powers[i].parms_id());
        //cout<<"DEBUG:mult n."<<i<<endl;
        evaluator.multiply_plain(powers[i], plain_coeffs[i], terms[i]);
        //cout<<"DEBUG:relin n."<<i<<endl;
        evaluator.relinearize_inplace(terms[i], relin_keys);
        evaluator.rescale_to_next_inplace(terms[i]);
    }

    //cout<<"DEBUG: moltiplicato per i coefficienti"<<endl;

    // ============================================================
    // 3. Somma dei termini con un normale ciclo for
    // ============================================================
    Ciphertext result = terms[0];
    for (size_t i = 1; i < terms.size(); i++) {
        align_ciphertexts_scale(result, terms[i], evaluator, context);
        evaluator.add_inplace(result, terms[i]);
    }
    //cout<<"DEBUG: ho sommato i termini"<<endl;
    
    return result;
}


auto sherman_morrison_apply_delta(
    Ciphertext &ct_x,                    // x cifrato
    Ciphertext &ct_y,                    // y cifrato    
    vector <double> &w_star,                // w*, in chiaro
    double lambda,
    const vector<vector<double>> &Ainv,  // A^{-1} in chiaro
    SEALContext &context,
    Evaluator &evaluator,
    Encryptor &encryptor,
    Decryptor &decryptor,
    CKKSEncoder &encoder,
    RelinKeys &relin_keys,
    GaloisKeys &galois_keys,
    size_t n,
    size_t n_real,
    double scale,
    const vector<double> &COEFFS,        // coefficienti del polinomio di Remez, letti una volta nel main
    int degree,
    int n_train
)
{

    bool debug_mode = true;
    // --- CALCOLO DEL DELTA ---
    // delta = w*^T x_n - y_n

    // 1. Moltiplicazione scalare w*^T x_n
    Ciphertext wx = moltiplica_vettori_plain
        (
        w_star,       // x_n cifrato
        ct_x,     // w* in chiaro
        context,
        evaluator,
        encoder,
        decryptor,
        relin_keys,
        galois_keys,
        n,
        scale
        );

    //cout<<"moltiplicati plain"<<endl;

    // Sottrai y_n
    Ciphertext ct_delta;
    align_ciphertexts_scale(wx, ct_y, evaluator, context); // allinea livelli
    evaluator.sub_inplace(wx, ct_y);                 // delta = w*^T x_n - y_n
    //cout <<"sottratti wx-y"<<endl;

    // stampa debug

    /*
    Plaintext pt_e;
    decryptor.decrypt(wx, pt_e);
    vector<double> e_vals;
    encoder.decode(pt_e, e_vals);
    cout << "Errore scalare e: " << e_vals[0] << endl;
    */

    //cout<<"molt di nuovo per x"<<endl;
    // moltiplico il risultato per x

    align_ciphertexts(wx, ct_x, evaluator, context);
    evaluator.multiply_inplace(wx,ct_x); // ri moltiplico per x
    evaluator.relinearize_inplace(wx, relin_keys);
    evaluator.rescale_to_next_inplace(wx);

    vector<double> vec_2(n, 2.0);
    Plaintext plain_2;
    encoder.encode(vec_2, wx.scale(), plain_2);
    evaluator.mod_switch_to_inplace(plain_2, wx.parms_id());
    evaluator.multiply_plain_inplace(wx, plain_2);
    evaluator.rescale_to_next_inplace(wx);




    // moltiplico w per lambda per la regolarizzazione del modello
    vector<double> w_lambda_vec(n, 0.0);
    for(int i=0;i<n_real; i++)
    w_lambda_vec[i] = 2.0 * n_train *lambda * w_star[i];

    
    // sommo i due fattori per ottenere delta
    //cout<<"aggiunta finale"<<endl;
    Plaintext w_lambda;
    //cout<<"w per lambda: "<<endl;
    //for (int i=0; i<n; i++) cout << " , " <<w_lambda_vec[i];

    encoder.encode(w_lambda_vec, wx.scale(), w_lambda);
    evaluator.mod_switch_to_inplace(w_lambda, wx.parms_id());
    evaluator.add_plain(wx, w_lambda, ct_delta);


    // stampa debug
    
    Plaintext pt_delta;
    decryptor.decrypt(ct_delta, pt_delta);
    vector<double> delta_vals;
    encoder.decode(pt_delta, delta_vals);
    //cout << "\nDelta (primi " << n << " slot): ";
    //for (size_t i = 0; i < n; i++) cout << delta_vals[i] << " ";
    //cout << endl;
    
    

    // moltiplico per 100 per avere valori meno vicini a 0
    // QUA metti o togli 100
    vector<double> vec_1000(n, 100.0); // ##
    // codifico il vettore plaintext con scala del ciphertext delta
    Plaintext plain_vec_1000;
    encoder.encode(vec_1000, ct_delta.scale(), plain_vec_1000);
    
    // allineo e moltiplico per delta
    evaluator.mod_switch_to_inplace(plain_vec_1000, ct_delta.parms_id());
    evaluator.multiply_plain_inplace(ct_delta, plain_vec_1000);
    evaluator.relinearize_inplace(ct_delta, relin_keys);
    evaluator.rescale_to_next_inplace(ct_delta);
    
    
    
    // ho calcolato delta


    // u = A^{-1} x -----------------------------------------------------------------------
    
    Ciphertext u = moltiplica_matrice_vettore(
        ct_x, Ainv,
        context, evaluator, decryptor,
        encoder, galois_keys,
        n, n_real, scale);
    
    //stampa debug

    if(debug_mode)
    {
        Plaintext pt_u;
        decryptor.decrypt(u, pt_u);
        vector<double> u_vals;
        encoder.decode(pt_u, u_vals);
        //cout << "DEBUG u (primi " << n << " slot): ";
        //for(size_t i=0; i<n; i++) cout << u_vals[i] << " ";
        //cout << endl;
    }
    
        
    
    

    // v = A^{-1} delta ------------------------------------------------------------------

    Ciphertext v = moltiplica_matrice_vettore(
        ct_delta, Ainv,
        context, evaluator, decryptor,
        encoder, galois_keys,
        n, n_real, scale
    );

    // stampa debug

    if (debug_mode)
    {
        Plaintext pt_v;
        decryptor.decrypt(v, pt_v);
        vector<double> v_vals;
        encoder.decode(pt_v, v_vals);
        //cout << "DEBUG v (primi " << n << " slot): ";
        //for(size_t i=0; i<n; i++) cout << v_vals[i] << " ";
        //cout << endl;
    }
        
    
   //cout<<"calcolo nuova mat"<<endl;
    
    Ciphertext u_copia =u;
    vector<Ciphertext> Anew(n);
    Anew = prodotto_esterno_vettore(
        u_copia,            // Vettore cifrato di input
        context,            // Contesto SEAL
        evaluator,            // Evaluator
        decryptor,            // Decryptor (per debug opzionale)
        encoder,            // Encoder CKKS
        galois_keys,   // Chiavi per rotazioni
        relin_keys,
        n,                        // Dimensione matrice/vettore
        n_real,
        scale                     // Scala CKKS
    );

    //cout<<"calcolo beta"<<endl;
    // beta = x^T u ------------------------------------------------------------------
    Ciphertext beta = moltiplica_vettori(
        ct_x, u,
        context, evaluator,
        encoder, decryptor,relin_keys, galois_keys, 
        n, scale
    );
    // beta è uno scalare replicato

    // stampa debug
    
    Plaintext beta_pt;
    decryptor.decrypt(beta, beta_pt);
    vector<double> beta_vals;
    encoder.decode(beta_pt, beta_vals);
    //cout << "DEBUG beta (primo slot): ";
    //for(size_t i=0; i<1; i++) cout << beta_vals[i] << " ";
    //cout << endl;
    
        
    
    
    //cout<<"calcolo alfa"<<endl;

    //alpha = x^T v ------------------------------------------------------------------
    Ciphertext alpha = moltiplica_vettori(
        ct_x, v,
        context, evaluator,
        encoder, decryptor, relin_keys, galois_keys,
        n, scale
    );

    
    // stampa debug
    if (debug_mode)
    {
        Plaintext pt_alpha;
        decryptor.decrypt(alpha, pt_alpha);
        vector<double> alpha_vals;
        encoder.decode(pt_alpha, alpha_vals);
        //cout << "DEBUG alpha: " << alpha_vals[0] << endl;

        //cout << " alfa | Level: " << context.get_context_data(alpha.parms_id())->chain_index()  << " | Scale: " << log2(alpha.scale()) << " bits" << endl;
            
    }
        
    
    //cout << "calcolo p"<< endl;

    // p = 1 / (1 - beta) ------------------------------------------------------------------
    // 1 - beta
    Plaintext one;
    encoder.encode(1.0, beta.scale(), one);

    Ciphertext ct_one;
    encryptor.encrypt(one, ct_one);  // cifra il 1

    // --- Sottrazione: p = 1 - beta ---
    align_ciphertexts(ct_one, beta, evaluator, context); // allinea i livelli e parms_id
    evaluator.sub(ct_one, beta, beta);                       // beta = 1 - beta


    //cout<<"DEBUG: calcolo inverso"<<endl;


    // calcolo inverso
    Ciphertext p = divide(
        beta,context, evaluator, encoder, encryptor, relin_keys, galois_keys ,n, scale, COEFFS, degree);



    // moltiplico anew per p

    //cout<<"DEBUG: moltiplico anew per p"<<endl;

    for(int i=0;i<n;i++)
    {
        align_ciphertexts(Anew[i], p, evaluator, context);
        evaluator.multiply_inplace(Anew[i], p);
        evaluator.relinearize_inplace(Anew[i], relin_keys);
        evaluator.rescale_to_next_inplace(Anew[i]);
    }

    // decifro e scrivo la nuova matrice su file

    // questo è l'update di Ainv
    string nome_file_hessiana = "nuova_hessiana_grado_"+to_string(degree)+".txt";
    vector<vector <double>> Anew_vals = scrivi_matrice_su_file(Anew, context, decryptor, encoder,n, nome_file_hessiana);
    // in Anew_vals salvo la matrice iniziale - l'update
    for (int i=0;i<n; i++)
    {
        for (int j=0;j<n;j++)
        {
            Anew_vals[i][j] = Ainv[i][j] + Anew_vals[i][j];
        }
    }


    // 6. coeff = alpha * p ------------------------------------------------------------------
    
    
    align_ciphertexts(alpha, p, evaluator, context);
    
    /*
    cout<< "ho allineato"<<endl;
    cout << " alfa | Level: " << context.get_context_data(alpha.parms_id())->chain_index() 
     << " | Scale: " << log2(alpha.scale()) << " bits" << endl;
    cout << " p | Level: " << context.get_context_data(p.parms_id())->chain_index() 
    << " | Scale: " << log2(p.scale()) << " bits" << endl;
    */
    
    Ciphertext coeff;
    evaluator.multiply(alpha, p, coeff);

    /*
    cout<< "ho moltiplicato"<<endl;
    cout << " coeff | Level: " << context.get_context_data(coeff.parms_id())->chain_index() 
     << " | Scale: " << log2(coeff.scale()) << " bits" << endl;
    */
    
    evaluator.relinearize_inplace(coeff, relin_keys);
    //cout<<"ho relin"<<endl;
    evaluator.rescale_to_next_inplace(coeff);
    //cout<<"ho riscalato"<<endl;
    // coeff è scalare replicato

    /*
        Plaintext pt_coeff;
        decryptor.decrypt(coeff, pt_coeff);
        vector<double> coeff_vals;
        encoder.decode(pt_coeff, coeff_vals);
        cout << "DEBUG coeff (primo slot): " << coeff_vals[0] << endl;

        cout<<"ho riscalato"<<endl;
    */
    

    //term = coeff * u ----------------------------------------------------------------------

    /*
        cout << " coeff | Level: " << context.get_context_data(coeff.parms_id())->chain_index() 
        << " | Scale: " << log2(coeff.scale()) << " bits" << endl;
        cout << " u | Level: " << context.get_context_data(u.parms_id())->chain_index() 
        << " | Scale: " << log2(u.scale()) << " bits" << endl;
    */
    
    align_ciphertexts(coeff, u, evaluator, context);
    //cout<< "ho allineato di nuovo, ora:"<<endl;

    /*
        cout << " coeff | Level: " << context.get_context_data(coeff.parms_id())->chain_index() 
        << " | Scale: " << log2(coeff.scale()) << " bits" << endl;
        cout << " u | Level: " << context.get_context_data(u.parms_id())->chain_index() 
        << " | Scale: " << log2(u.scale()) << " bits" << endl;
    */
    
    Ciphertext term;
    evaluator.multiply(coeff, u, term);
    evaluator.relinearize_inplace(term, relin_keys);
    evaluator.rescale_to_next_inplace(term);
    //cout<<"terza molt"<<endl;

    

    /*
        Plaintext pt_term;
        decryptor.decrypt(term, pt_term);
        vector<double> term_vals;
        encoder.decode(pt_term, term_vals);
        for(int i=0;i<n;i++)
        {
            cout << "DEBUG term ("<<i<<" slot): " << term_vals[i] << endl;
        }
    */

    return make_pair(term, Anew_vals);  // = A_new^{-1} * delta
}



// Modifica nel main
int main(int argc, char* argv[])
{

   int degree=5;
    string input_base_dir;
    int seed;
    
    if (argc >= 3) {
        input_base_dir = argv[1];
        seed = atoi(argv[2]);
    } else {
        cout << "Uso: " << argv[0] << "<directory_base> <seed>" << endl;
        cout << "Esempio: " << argv[0] << " risultati_ridge_openml_dataset_44120 42" << endl;
        return 1;
    }
    
    // Costruisci il percorso della sottocartella per il seed
    string input_dir = input_base_dir + "/seed_" + to_string(seed);

    string nomefileinput = input_dir + "/hessian_inverse.txt";
    string nomefileparametri = input_dir + "/parametri.txt";
    string nomefilecoeff = input_dir + "/coefficienti_remez.txt";

    // legge i coefficienti del polinomio di Remez una sola volta
    const vector<double> COEFFS = leggiCoeffDaFile(nomefilecoeff);

    /***********************
     * PARAMETRI
     ***********************/                
    size_t poly_modulus_degree = 32768;
    double scale = pow(2.0, 40);

    /***********************
     * CONTEXT CKKS
     ***********************/
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50}));

    SEALContext context(parms);
    cout << "Contesto CKKS creato" << endl;

    /***********************
     * CHIAVI
     ***********************/
    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    cout << "Chiavi generate" << endl;

    PublicKey public_key;
    keygen.create_public_key(public_key);

    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Slot count: " << slot_count << endl;

    /***********************
     * DATI DI TEST: leggo dal file
     ***********************/
    cout << "\n=== PREPARAZIONE DATI ===" << endl;

    // Leggi tutti i dati (matrice inversa + lista di punti)
    
    auto [Ainv_diagonals, x_vals_list, y_vals_list, w_star_initial, lambda_reg] = leggiTuttiIDatiMultipli(nomefileinput);
    
    int n_train = leggiNtrainDaFileParametri(nomefileparametri);

    if (x_vals_list.empty()) {
        cerr << "Errore: nessun punto letto dal file" << endl;
        return 1;
    }

    int n_padded = Ainv_diagonals.size();  // Dimensione con padding (potenza di 2)
    int n_real = x_vals_list[0].size();    // Dimensione reale dei vettori
    
    if (n_real > slot_count)
        throw runtime_error("n > slot_count");

    cout << "\n=== DIMENSIONI ===\n";
    cout << "  Dimensione reale: " << n_real << endl;
    cout << "  Dimensione con padding: " << n_padded << endl;
    cout << "  Numero punti: " << x_vals_list.size() << endl;

    double lambda = lambda_reg;

    // Vettori per accumulare risultati
    vector<vector<double>> all_results_encrypted;
    vector<vector<double>> all_results_clear;

    cout << "\n=== UNLEARNING SEQUENZIALE ===\n";



    string nome_error_file = input_dir + "/erroripar_grado_" + to_string(degree) + ".txt";
    ofstream error_file(nome_error_file, ios::app);
    string inversi_filename = input_dir + "/inversi_chiaro_grado_" + to_string(degree) + ".txt";
    ofstream inv_file(inversi_filename, ios::app);

    if (error_file.tellp() == 0) {
        error_file << "# punto_index, errore_medio%, errore_massimo%, errore_medio_abs, errore_massimo_abs" << endl;
    }

    // ============================================================
    // INIZIALIZZAZIONE: matrice corrente e pesi correnti
    // ============================================================
    
    // Matrice corrente in formato diagonale (per le funzioni cifrate)
    vector<vector<double>> current_Ainv_diagonals = Ainv_diagonals;
    
    // Matrice corrente in formato matrice piena (per i calcoli in chiaro)
    vector<vector<double>> current_Ainv_matrix = diagonali_a_matrice(current_Ainv_diagonals, n_padded, n_real);
    
    // Pesi correnti (inizialmente w_star)
    vector<double> current_w_clear = w_star_initial;      // per calcoli in chiaro
    vector<double> current_w_encrypted = w_star_initial;  // per calcoli cifrati


    // Loop su tutti i punti

    auto start_total = chrono::high_resolution_clock::now();// faccio partire timer
    for (size_t idx = 0; idx < x_vals_list.size(); idx++) {
        cout << "\n" << string(60, '=') << endl;
        cout << "ITERAZIONE " << idx << " - PUNTO " << idx << endl;
        cout << string(60, '=') << endl;
        
        vector<double> x_vals = x_vals_list[idx];
        double y = y_vals_list[idx];

        // Cifra x
        Plaintext pt_x;
        encoder.encode(x_vals, scale, pt_x);
        Ciphertext ct_x;
        encryptor.encrypt(pt_x, ct_x);
        
        // Cifra y (come vettore costante)
        vector<double> y_vals(n_real, y);
        Plaintext pt_y;
        encoder.encode(y_vals, scale, pt_y);
        Ciphertext ct_y;
        encryptor.encrypt(pt_y, ct_y);

        cout << "\n===== CALCOLO DELTA IN CHIARO =====\n";

        // Calcolo delta in chiaro (usa current_w)
        double w_dot_x = 0.0;
        for(int i = 0; i < n_real; i++)
            w_dot_x += current_w_clear[i] * x_vals[i];

        double residual = w_dot_x - y;

        vector<double> gradient(n_real);
        for(int i = 0; i < n_real; i++)
            gradient[i] = 2.0 * residual * x_vals[i];

        vector<double> delta_plain(n_real);
        for(int i = 0; i < n_real; i++)
            delta_plain[i] = (2.0 * lambda * n_train * current_w_clear[i] + gradient[i]) * 100;  // ##Scala opzionale QUA SCALO PER 100 plain
            
        //cout << "Delta plain (primi 5): ";
        for(int i = 0; i < min(5, n_real); i++) cout << delta_plain[i] << " ";
        //cout << endl;

        cout << "\n=== SHERMAN-MORRISON CIFRATO (con matrice corrente) ===" << endl;

        // Applica Sherman-Morrison in cifrato usando current_Ainv_diagonals

        auto [ct_result, Anew_enc_vals]  = sherman_morrison_apply_delta(
            ct_x, ct_y, current_w_encrypted, lambda, current_Ainv_diagonals,
            context, evaluator, encryptor, decryptor,
            encoder, relin_keys, galois_keys,
            n_padded, n_real, scale, COEFFS, degree, n_train
        );

        //cout<<"il livello finale è: "<<context.get_context_data(ct_result.parms_id())->chain_index()<<endl;

        // aggiorno la matrice cifrata
        current_Ainv_diagonals = Anew_enc_vals; 

        // Decifra il risultato
        Plaintext pt_result;
        decryptor.decrypt(ct_result, pt_result);
        vector<double> result_vals;
        encoder.decode(pt_result, result_vals);
        
        all_results_encrypted.push_back(result_vals);

        // ============================================================
        // CALCOLO IN CHIARO PER CONFRONTO (usa current_Ainv_matrix)
        // ============================================================
        
        vector<double> u(n_real, 0.0), v(n_real, 0.0);
        
        for (size_t i = 0; i < n_real; i++) {
            for (size_t j = 0; j < n_real; j++) {
                u[i] += current_Ainv_matrix[i][j] * x_vals[j];
                v[i] += current_Ainv_matrix[i][j] * delta_plain[j];
            }
        }

        double beta = 0.0;
        for (size_t i = 0; i < n_real; i++) beta += x_vals[i] * u[i];

        //cout<<"DEBUG PLAIN BETA: "<<beta<<endl;

        double alpha = 0.0;
        for (size_t i = 0; i < n_real; i++) alpha += x_vals[i] * v[i];

        inv_file << fixed << setprecision(6) << 1.0-beta << endl;

        double coeff = alpha / (1.0 - beta);

        vector<double> result_clear(n_real, 0.0);
        for (size_t i = 0; i < n_real; i++) {
            result_clear[i] = coeff * u[i];
        }
        all_results_clear.push_back(result_clear);

        // ============================================================
        // AGGIORNAMENTO DELLA MATRICE PER LA PROSSIMA ITERAZIONE
        // ============================================================
        
        // Calcola l'aggiornamento di Sherman-Morrison in chiaro
        // A_new = A_old - (A_old * x * x^T * A_old) / (1 + x^T * A_old * x)
        // Ma in realtà abbiamo già u = A_old * x e v = A_old * delta
        
        vector<vector<double>> A_update(n_real, vector<double>(n_real, 0.0));
        double denom = 1.0 - beta;
        
        for (size_t i = 0; i < n_real; i++) {
            for (size_t j = 0; j < n_real; j++) {
                A_update[i][j] = (u[i] * u[j]) / denom;
            }
        }

        /*cout<< "//////////////////////////  STAMPO MATRICE ///////////////////////////"<<endl;
        for(int i=0;i<5;i++)
        {
            for (int j=0;j<5; j++)
            {
                cout << " "<<A_update[i][j]<<" ";
            }
            cout<<endl;
        }*/
        
        // Aggiorna la matrice corrente in chiaro (formato matrice piena)
        vector<vector<double>> new_Ainv_matrix(n_real, vector<double>(n_real, 0.0));
        for (size_t i = 0; i < n_real; i++) {
            for (size_t j = 0; j < n_real; j++) {
                new_Ainv_matrix[i][j] = current_Ainv_matrix[i][j] + A_update[i][j];
            }
        }

        
        
        // Sostituisci current_Ainv_matrix con la nuova matrice
        current_Ainv_matrix = new_Ainv_matrix;
        
        
        // Aggiorna anche current_w per il prossimo punto
        for (size_t i = 0; i < n_real; i++) {
            current_w_clear[i] = current_w_clear[i] + result_clear[i]/100; //QUA SCALO I W PER 1000 ##
        }
        for (size_t i = 0; i < n_real; i++) {
            current_w_encrypted[i] = current_w_encrypted[i] + result_vals[i]/100; //##
        }


        // Calcola errore percentuale per questo punto
        double max_percent_error = 0.0, mean_percent_error = 0.0;
        double max_abs_error = 0.0, mean_abs_error = 0.0;
        double soglia = 1e-12;

        for (int i = 0; i < n_real; i++) {
            double err_abs = fabs(result_vals[i] - result_clear[i]);
            mean_abs_error += err_abs;
            if (err_abs > max_abs_error) max_abs_error = err_abs;
            
            double err_percent;
            /*if (fabs(result_clear[i]) > soglia) {
                err_percent = (err_abs / fabs(result_clear[i])) * 100.0;
            } else {
                err_percent = err_abs * 100.0;
            }*/
           err_percent = (err_abs / fabs(result_clear[i])) * 100.0;
            
            mean_percent_error += err_percent;
            if (err_percent > max_percent_error) max_percent_error = err_percent;
        }
        mean_abs_error /= n_real;
        mean_percent_error /= n_real;

        // Scrivi su file
        error_file << idx << ", " << setprecision(12) << mean_percent_error << ", " 
           << max_percent_error << ", "
           << mean_abs_error << ", " 
           << max_abs_error << endl;

        cout << "Punto " << idx << " - Errore % medio: " << mean_percent_error 
            << "%, Errore % max: " << max_percent_error << "%" << endl;
        cout << "           Errore assoluto medio: " << mean_abs_error 
            << ", Errore assoluto max: " << max_abs_error << endl;

    } // Fine loop punti

    // blocco timer
    auto end_total = chrono::high_resolution_clock::now();
    chrono::duration<double> durata = end_total - start_total;
    int hours = (int)durata.count() / 3600;
    int minutes = ((int)durata.count() % 3600) / 60;
    int seconds = (int)durata.count() % 60;

    cout << "\n" << string(60, '=') << endl;
    cout << "TEMPO TOTALE UNLEARNING" << endl;
    cout << string(60, '=') << endl;
    cout << "Tempo totale per " << x_vals_list.size() << " punti: " 
        << fixed << setprecision(2) << durata.count() << " secondi" << endl;
    cout << "Tempo medio per punto: " << (durata.count() / x_vals_list.size()) << " secondi" << endl;



    // ============================================================
    // LETTURA BOUND TEORICI DAL FILE GENERATO DA PYTHON
    // ============================================================
    string bounds_file = input_dir + "/bounds_teorici.txt";
    double eps_remez = 0.0;
    double bound_sample_dependent = 0.0;
    double bound_uniforme = 0.0;

    ifstream bf(bounds_file);
    if (bf.is_open()) {

        cout<<"Ho aperto il file dei bound"<<endl;
        string line;
        while (getline(bf, line)) {
            if (line.find("eps_remez") != string::npos) {
                eps_remez = stod(line.substr(line.find("=") + 1));
                cout<<"Trovato epsilon"<<endl;
            } else if (line.find("bound_sample_dependent") != string::npos) {
                bound_sample_dependent = stod(line.substr(line.find("=") + 1));
                cout<<"trovato sample"<<endl;
            } else if (line.find("bound_uniforme") != string::npos) {
                bound_uniforme = stod(line.substr(line.find("=") + 1));
                cout<<"trovato unif"<<endl;
            }
        }
        bf.close();
    } else {
        cerr << "ATTENZIONE: Impossibile aprire " << bounds_file << endl;
    }


    // ============================================================
    // LETTURA W_RETRAIN DAL FILE GENERATO DA PYTHON
    // ============================================================
    vector<double> w_retrain;
    string retrain_file = input_dir + "/w_retrain.txt";
    ifstream rf(retrain_file);
    if (rf.is_open()) {
        string line;
        while (getline(rf, line)) {
            if (line.find("w_retrain") != string::npos) {
                size_t start = line.find('[');
                size_t end = line.find(']');
                if (start != string::npos && end != string::npos) {
                    string values_str = line.substr(start + 1, end - start - 1);
                    istringstream values_ss(values_str);
                    string token;
                    while (getline(values_ss, token, ',')) {
                        token.erase(remove(token.begin(), token.end(), ' '), token.end());
                        if (!token.empty()) w_retrain.push_back(stod(token));
                    }
                }
            }
        }
        rf.close();
        cout << "Letto w_retrain con " << w_retrain.size() << " elementi" << endl;
    } else {
        cerr << "ATTENZIONE: Impossibile aprire " << retrain_file << endl;
    }


    // ============================================================
    // ERRORE CHIARO VS RETRAIN (solo il primo punto rimosso)
    // ============================================================
    double error_chiaro_vs_retrain = 0.0;
    if (!w_retrain.empty() && !all_results_clear.empty()) {
        vector<double> w_clear_punto0(n_real);
        for (int i = 0; i < n_real; i++) {
            w_clear_punto0[i] = w_star_initial[i] + all_results_clear[0][i] / 100.0;  // stesso scaling di w_new_clear
        }
        error_chiaro_vs_retrain = compute_l1_error(w_clear_punto0, w_retrain, n_real);
    }

    cout << "\nErrore L1 chiaro (Sherman-Morrison esatto) vs retrain (primo punto): "
        << scientific << setprecision(10) << error_chiaro_vs_retrain << endl;

    // ============================================================
    // CALCOLO ERRORI EMPIRICI (un solo punto rimosso)
    // ============================================================

    // Errore empirico L1 per l'unico punto rimosso
    double error_empirico = 0.0;
    if (!all_results_encrypted.empty() && !all_results_clear.empty()) {
        // Prendi il primo (e unico) risultato
        error_empirico = compute_l1_error(all_results_encrypted[0], all_results_clear[0], n_real);
    }

    // Per il bound uniforme, l'errore empirico uniforme è lo stesso (un solo punto)
    double error_empirico_uniforme = error_empirico;

    // ============================================================
    // SCRITTURA BOUND NEL FILE UNICO (INDIPENDENTE DAL SEED)
    // ============================================================
    write_bounds_to_file(
        input_dir,
        seed,
        degree,
        eps_remez,
        bound_sample_dependent,
        bound_uniforme,
        error_empirico,
        error_empirico_uniforme,
        error_chiaro_vs_retrain
    );


    // Statistiche finali su tutti i punti
    // rimetto la precisione più alta
    cout << defaultfloat << setprecision(8);
    cout << "\n" << string(60, '=') << endl;
    cout << "STATISTICHE FINALI SU " << x_vals_list.size() << " PUNTI" << endl;
    cout << string(60, '=') << endl;

    double global_mean_error = 0.0;
    double global_max_error = 0.0;

    for (size_t p = 0; p < x_vals_list.size(); p++) {
        double punto_mean = 0.0;
        double punto_max = 0.0;
        
        for (int i = 0; i < n_real; i++) {
            double err = fabs(all_results_encrypted[p][i] - all_results_clear[p][i]);
            punto_mean += err;
            if (err > punto_max) punto_max = err;
        }
        punto_mean /= n_real;
        
        global_mean_error += punto_mean;
        if (punto_max > global_max_error) global_max_error = punto_max;
        
        cout << "Punto " << p << " - Errore medio: " << punto_mean 
             << ", Errore max: " << punto_max << endl;
    }

    global_mean_error /= x_vals_list.size();

    cout << "\n" << string(40, '-') << endl;
    cout << "ERRORE MEDIO GLOBALE: " << global_mean_error << endl;
    cout << "ERRORE MASSIMO GLOBALE: " << global_max_error << endl;

    // Salva risultati su file

    string output_filename = input_dir + "/risultati_grado_" + to_string(degree) + ".txt";
    ofstream file(output_filename);


    
    
    
    
    if (file.is_open()) {
        file << "w_star_iniziale = [";
        for (size_t i = 0; i < n_real; i++) {
            file << w_star_initial[i];
            if (i < n_real - 1) file << ", ";
        }
        file << "]\n\n";

        // QUA: vettori che tengono traccia del modello corrente dopo ogni aggiornamento.
        // inizializzati a w_star_initial; ad ogni punto p vengono aggiornati sommando
        // il delta del punto 
        vector<double> w_current_clear(n_real), w_current_encrypted(n_real);
        for (size_t i = 0; i < n_real; i++) {
            w_current_clear[i]     = w_star_initial[i];
            w_current_encrypted[i] = w_star_initial[i];
        }

        for (size_t p = 0; p < x_vals_list.size(); p++) {
            file << "\n--- PUNTO " << p << " ---\n";
            
            file << "x_random = [";
            for (size_t i = 0; i < n_real; i++) {
                file << x_vals_list[p][i];
                if (i < n_real - 1) file << ", ";
            }
            file << "]\n";
            
            file << "y_random = " << y_vals_list[p] << "\n";

            // QUA: w_new_clear e' il modello in chiaro corrente (punto precedente)
            // piu' il delta in chiaro del punto p 
            // w_current_clear viene aggiornato per essere usato come base al punto p+1
            file << "w_new_clear = [";
            for (size_t i = 0; i < n_real; i++) {
                w_current_clear[i] += all_results_clear[p][i]/100;// ##
                file << w_current_clear[i];
                if (i < n_real - 1) file << ", ";
            }
            file << "]\n";
            
            // QUA: stesso aggiornamento cumulativo per il modello cifrato
            file << "w_new_encrypted = [";
            for (size_t i = 0; i < n_real; i++) {
                w_current_encrypted[i] += all_results_encrypted[p][i]/100; // ##
                file << w_current_encrypted[i];
                if (i < n_real - 1) file << ", ";
            }
            file << "]\n";

            file << "delta_w_clear = [";
            for (size_t i = 0; i < n_real; i++) {
                file << all_results_clear[p][i]/100; // ##
                if (i < n_real - 1) file << ", ";
            }
            file << "]\n";
            
            file << "delta_w_encrypted = [";
            for (size_t i = 0; i < n_real; i++) {
                file << all_results_encrypted[p][i]/100; //##
                if (i < n_real - 1) file << ", ";
            }
            file << "]\n";
        }

        file << "# Tempo totale: " << hours << "h " << minutes << "m " << seconds << "s\n";
        
        file.close();
        cout << "\nRisultati salvati in: " << output_filename << endl;
    }

    error_file.close();

    return 0;
}