import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SPARC KINEMATIC AUDIT: THE CHI-SQUARED REVELATION
# ==============================================================================
A0_RRT = 1.2001e-10  
KPC_TO_M = 3.086e19
KM_TO_M = 1000.0

SPARC_DATA_DIR = r"E:\Documentos\rrt\art2\new\Rotmod_LTG"

def calculate_hydrodynamic_velocity(row, upsilon_d=0.5, upsilon_b=0.7):
    # Newton: Preservando os sinais corretos das velocidades
    v2_bar = (row['Vgas'] * np.abs(row['Vgas'])) + (upsilon_d * row['Vdisk'] * np.abs(row['Vdisk'])) + (upsilon_b * row['Vbul'] * np.abs(row['Vbul']))
    if v2_bar <= 0: return np.nan 
    
    g_bar = (v2_bar * (KM_TO_M**2)) / (row['Rad'] * KPC_TO_M)
    nu = 0.5 + np.sqrt(0.25 + (A0_RRT / g_bar))
    return np.sqrt((v2_bar * (KM_TO_M**2)) * nu) / KM_TO_M

def parse_sparc_file(filepath):
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
        elif 'ERR' in c: rename_map[col] = 'errV'
        elif 'VGAS' in c: rename_map[col] = 'Vgas'
        elif 'VDISK' in c: rename_map[col] = 'Vdisk'
        elif 'VBUL' in c: rename_map[col] = 'Vbul'
        
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'Vbul' not in df.columns: df['Vbul'] = 0.0
    if 'errV' not in df.columns: return None
    
    for col in ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul']:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])

def run_chi_square_audit():
    print("="*80)
    print(" 🔎 RRT: A REVELAÇÃO DO QUI-QUADRADO (A ORIGEM DO 1.33 E 5.81) 🔎")
    print("="*80)
    
    all_files = [f for f in os.listdir(SPARC_DATA_DIR) if f.endswith('.dat')]
    
    global_chi2 = 0.0
    total_points = 0
    absolute_errors_kms = []
    
    for file in all_files:
        filepath = os.path.join(SPARC_DATA_DIR, file)
        df = parse_sparc_file(filepath)
        if df is None or len(df) == 0: continue
        
        # Filtro vital: Remover erros instrumentais zerados para não quebrar o Chi-Quadrado
        df = df[df['errV'] > 0]
        
        df['V_pred'] = df.apply(lambda row: calculate_hydrodynamic_velocity(row, upsilon_d=0.5), axis=1)
        df = df.dropna(subset=['V_pred'])
        if len(df) == 0: continue
            
        # 1. Equação 6 do seu LaTeX: Chi-Quadrado
        chi2_galaxy = np.sum( ((df['Vobs'] - df['V_pred']) / df['errV'])**2 )
        global_chi2 += chi2_galaxy
        total_points += len(df)
        
        # 2. Erro Absoluto em km/s (Para achar o seu 5.81 km/s)
        abs_err_kms = np.abs(df['Vobs'] - df['V_pred'])
        absolute_errors_kms.extend(abs_err_kms.values)

    # Cálculo Final
    reduced_chi2 = global_chi2 / total_points
    mean_abs_err_kms = np.nanmean(absolute_errors_kms)
    median_abs_err_kms = np.nanmedian(absolute_errors_kms)
    
    print(f" -> Pontos Analisados (N)        : {total_points}")
    print(f" -> Qui-Quadrado Global (X²)     : {global_chi2:.2f}")
    print("-" * 80)
    print(f" -> QUI-QUADRADO REDUZIDO (X²_v) : {reduced_chi2:.3f}")
    print(f"    (O Santo Graal da Astrofísica. Perto de 1.33 = Perfeição)")
    print("-" * 80)
    print(f" -> Erro Absoluto Médio (MAE)    : {mean_abs_err_kms:.2f} km/s")
    print(f" -> Erro Absoluto Mediano        : {median_abs_err_kms:.2f} km/s")
    print(f"    (O seu texto citava ~5.81 km/s)")
    print("="*80)

if __name__ == "__main__":
    run_chi_square_audit()