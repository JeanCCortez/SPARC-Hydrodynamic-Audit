import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SPARC INSTRUMENTAL NOISE FLOOR AUDIT
# Objetivo: Provar matematicamente o limite de precisão do telescópio.
# ==============================================================================

SPARC_DATA_DIR = r"E:\Documentos\rrt\art2\new\Rotmod_LTG"

def parse_sparc_file(filepath):
    """Leitor blindado."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    header, data = [], []
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            upper_line = line.upper()
            if 'RAD' in upper_line or 'VOBS' in upper_line:
                header = line.replace('#', '').strip().split()
        else:
            data.append(line.split())
            
    if not header or not data: return None
    df = pd.DataFrame(data)
    
    if len(header) >= len(df.columns): df.columns = header[:len(df.columns)]
    else: df.columns = header + list(df.columns[len(header):])
        
    rename_map = {}
    for col in df.columns:
        c = str(col).upper()
        if c == 'R' or 'RAD' in c: rename_map[col] = 'Rad'
        elif 'OBS' in c: rename_map[col] = 'Vobs'
        elif 'ERR' in c: rename_map[col] = 'errV'
        
    df.rename(columns=rename_map, inplace=True)
    if 'Vobs' not in df.columns or 'errV' not in df.columns: return None
    
    df['Vobs'] = pd.to_numeric(df['Vobs'], errors='coerce')
    df['errV'] = pd.to_numeric(df['errV'], errors='coerce')
    return df.dropna(subset=['Vobs', 'errV'])

def run_noise_floor_audit():
    print("="*80)
    print(" 🔎 SPARC INSTRUMENTAL NOISE FLOOR AUDIT 🔎")
    print("="*80)
    
    all_files = [f for f in os.listdir(SPARC_DATA_DIR) if f.endswith('.dat')]
    
    instrumental_pct_errors = []
    pontos_medidos = 0
    
    for file in all_files:
        filepath = os.path.join(SPARC_DATA_DIR, file)
        df = parse_sparc_file(filepath)
        
        if df is None or len(df) == 0: continue
        
        # Filtro básico de segurança contra divisão por zero
        df = df[df['Vobs'] > 0]
        
        # O Erro Instrumental Percentual: (errV / Vobs) * 100
        instr_err_pct = (df['errV'] / df['Vobs']) * 100
        
        instrumental_pct_errors.extend(instr_err_pct.values)
        pontos_medidos += len(df)

    mean_noise_floor = np.nanmean(instrumental_pct_errors)
    median_noise_floor = np.nanmedian(instrumental_pct_errors)
    
    print(f"Total de pontos radiais analisados : {pontos_medidos}")
    print("\n" + "-"*80)
    print(" O CHÃO DE RUÍDO ABSOLUTO DO TELESCÓPIO (O LIMITE DA CIÊNCIA)")
    print("-"*80)
    print(f" -> Erro Instrumental Médio   : {mean_noise_floor:.2f} %")
    print(f" -> Erro Instrumental Mediano : {median_noise_floor:.2f} %")
    print("="*80)

if __name__ == "__main__":
    run_noise_floor_audit()