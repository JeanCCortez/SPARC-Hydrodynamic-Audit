import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import random
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SPARC MONTE CARLO CROSS-VALIDATION (1000 ITERATIONS)
# Data 100% Real. Zero Hardcoding.
# ==============================================================================
A0_RRT = 1.2001e-10  
KPC_TO_M = 3.086e19
KM_TO_M = 1000.0

SPARC_DATA_DIR = r"E:\Documentos\rrt\art2\new\Rotmod_LTG"

class GalaxyData:
    """Estrutura leve para processamento ultrarrápido em vetores Numpy."""
    def __init__(self, rad, vobs, vgas, vdisk, vbul):
        self.rad = rad
        self.vobs = vobs
        self.vgas = vgas
        self.vdisk = vdisk
        self.vbul = vbul

def parse_sparc_file(filepath):
    """Leitor blindado do SPARC."""
    with open(filepath, 'r') as f: lines = f.readlines()
    header, data = [], []
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            if 'RAD' in line.upper() or 'VOBS' in line.upper(): 
                header = line.replace('#', '').strip().split()
        else: data.append(line.split())
        
    if not header or not data: return None
    df = pd.DataFrame(data)
    
    if len(header) >= len(df.columns): df.columns = header[:len(df.columns)]
    else: df.columns = header + list(df.columns[len(header):])
    
    rename_map = {}
    for col in df.columns:
        c = str(col).upper()
        if 'RAD' in c or c == 'R': rename_map[col] = 'Rad'
        elif 'VOBS' in c: rename_map[col] = 'Vobs'
        elif 'VGAS' in c: rename_map[col] = 'Vgas'
        elif 'VDISK' in c: rename_map[col] = 'Vdisk'
        elif 'VBUL' in c: rename_map[col] = 'Vbul'
        
    df = df.rename(columns=rename_map).loc[:, ~df.columns.duplicated()]
    if 'Vbul' not in df.columns: df['Vbul'] = 0.0
    for col in ['Rad', 'Vobs', 'Vgas', 'Vdisk', 'Vbul']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=['Rad', 'Vobs', 'Vgas', 'Vdisk'])

def evaluate_mae(upsilon_d, galaxy_list):
    """Calcula o Erro Médio Absoluto (%) para um conjunto de galáxias e um dado Upsilon."""
    all_vobs = []
    all_vpred = []
    
    for g in galaxy_list:
        # Newton: Preservando os sinais corretos das velocidades
        v2_bar = (g.vgas * np.abs(g.vgas)) + (upsilon_d * g.vdisk * np.abs(g.vdisk)) + (0.7 * g.vbul * np.abs(g.vbul))
        valid = v2_bar > 0
        if not np.any(valid): continue
        
        g_bar = (v2_bar[valid] * (KM_TO_M**2)) / (g.rad[valid] * KPC_TO_M)
        nu = 0.5 + np.sqrt(0.25 + (A0_RRT / g_bar))
        v_pred = np.sqrt((v2_bar[valid] * (KM_TO_M**2)) * nu) / KM_TO_M
        
        all_vobs.extend(g.vobs[valid])
        all_vpred.extend(v_pred)
        
    all_vobs = np.array(all_vobs)
    all_vpred = np.array(all_vpred)
    
    if len(all_vobs) == 0: return np.inf
    
    mae = np.nanmean(np.abs((all_vobs - all_vpred) / all_vobs)) * 100
    return mae

def run_montecarlo_cross_validation(n_iterations=1000):
    print("="*80)
    print(f" 🔄 RRT MONTE CARLO SPLIT-HALF CROSS-VALIDATION ({n_iterations} ITERAÇÕES) 🔄")
    print("="*80)
    
    all_files = [f for f in os.listdir(SPARC_DATA_DIR) if f.endswith('.dat')]
    galaxies = []
    
    for file in all_files:
        filepath = os.path.join(SPARC_DATA_DIR, file)
        df = parse_sparc_file(filepath)
        if df is not None and len(df) > 0:
            galaxies.append(GalaxyData(df['Rad'].values, df['Vobs'].values, df['Vgas'].values, df['Vdisk'].values, df['Vbul'].values))
            
    print(f"[INFO] {len(galaxies)} Galáxias carregadas na memória vetorial.")
    print("[ALGORITHM] Executando o motor de otimização cega. Aguarde...\n")
    
    opt_upsilons = []
    out_of_sample_errors = []
    
    for i in range(n_iterations):
        # Embaralha e divide ao meio
        random.shuffle(galaxies)
        mid = len(galaxies) // 2
        train_set = galaxies[:mid]
        test_set = galaxies[mid:]
        
        # Otimizador Cego: Procura o Upsilon que minimiza o Erro no Treino (limites razoáveis 0.1 a 1.0)
        res = minimize_scalar(evaluate_mae, bounds=(0.1, 1.0), args=(train_set,), method='bounded')
        best_upsilon = res.x
        
        # Testa esse Upsilon na outra metade de galáxias (dados nunca vistos)
        test_error = evaluate_mae(best_upsilon, test_set)
        
        opt_upsilons.append(best_upsilon)
        out_of_sample_errors.append(test_error)
        
        if (i + 1) % 200 == 0:
            print(f" -> {i + 1}/{n_iterations} iterações concluídas...")

    final_upsilon_mean = np.mean(opt_upsilons)
    final_upsilon_std = np.std(opt_upsilons)
    final_error_mean = np.mean(out_of_sample_errors)
    final_error_std = np.std(out_of_sample_errors)
    
    print("\n" + "="*80)
    print(" RESULTADO DA VALIDAÇÃO CRUZADA (SPLIT-HALF)")
    print("="*80)
    print(f" -> Upsilon_d Otimizado (Média)      : {final_upsilon_mean:.4f} M_sun/L_sun")
    print(f" -> Upsilon_d Dispersão (Desvio)     : ± {final_upsilon_std:.4f} M_sun/L_sun")
    print("-" * 80)
    print(f" -> Erro Out-of-Sample Bruto (Teste) : {final_error_mean:.2f} %")
    print(f" -> Estabilidade do Erro (Desvio)    : ± {final_error_std:.2f} %")
    print("="*80)

if __name__ == "__main__":
    run_montecarlo_cross_validation(n_iterations=1000)