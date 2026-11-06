import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================
A0_THRESHOLD = 1.2001e-10  
KPC_TO_M = 3.086e19
KM_TO_M = 1000.0

SPARC_DATA_DIR = r"E:\Documentos\rrt\art2\new\Rotmod_LTG"
OUTPUT_DIR = "sparc_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# PHYSICAL MODELS
# ==============================================================================
def v_baryonic_squared(v_gas, v_disk, v_bulge, upsilon_disk=0.5):
    """Calculates the squared velocity generated strictly by baryons."""
    return (v_gas * np.abs(v_gas) + 
            upsilon_disk * v_disk * np.abs(v_disk) + 
            0.7 * v_bulge * np.abs(v_bulge))

def hydrodynamic_velocity(r, v_gas, v_disk, v_bulge, upsilon_d=0.5):
    """Predicts velocity using the parameter-free Hydrodynamic Ansatz."""
    v2_bar = v_baryonic_squared(v_gas, v_disk, v_bulge, upsilon_disk=upsilon_d)
    v2_bar_ms = v2_bar * (KM_TO_M**2)
    r_m = r * KPC_TO_M
    
    # Avoid division by zero warnings dynamically
    with np.errstate(divide='ignore', invalid='ignore'):
        g_bar = v2_bar_ms / r_m
        nu = 0.5 + np.sqrt(0.25 + (A0_THRESHOLD / g_bar))
        v_hydro = np.sqrt(v2_bar_ms * nu) / KM_TO_M
        
    return v_hydro

def nfw_velocity(r, v200, c):
    """Calculates the dark matter halo velocity contribution (NFW profile)."""
    r200 = v200 / (10 * 0.7) 
    rs = r200 / c
    x = r / rs
    with np.errstate(divide='ignore', invalid='ignore'):
        v_halo_sq = v200**2 * (1 / x) * (np.log(1 + x) - x / (1 + x)) / (np.log(1 + c) - c / (1 + c))
    return np.sqrt(np.maximum(0, v_halo_sq))

def combined_nfw_velocity(r, v_gas, v_disk, v_bulge, v200, c):
    """Combines baryonic components with the NFW halo."""
    v2_bar = v_baryonic_squared(v_gas, v_disk, v_bulge, upsilon_disk=0.5)
    v2_halo = nfw_velocity(r, v200, c)**2
    return np.sqrt(np.maximum(0, v2_bar + v2_halo))

def fit_nfw(r, v_obs, v_err, v_gas, v_disk, v_bulge):
    """Fits the NFW profile to the rotation curve, optimizing V200 and c."""
    def objective(r_val, v200, c):
        return combined_nfw_velocity(r_val, v_gas, v_disk, v_bulge, v200, c)
    
    try:
        # Bounds: V200 [10, 500] km/s, c [1, 50]
        popt, _ = curve_fit(objective, r, v_obs, sigma=v_err, absolute_sigma=True, bounds=([10, 1], [500, 50]))
        v_fit = objective(r, *popt)
        return v_fit, 2 # 2 free parameters (k=2)
    except:
        return np.full_like(r, np.nan), 2

def calculate_bic(v_obs, v_model, v_err, k):
    """Calculates the Bayesian Information Criterion (BIC)."""
    valid = ~np.isnan(v_model) & (v_err > 0)
    if not np.any(valid): return np.nan
    chi2 = np.sum(((v_obs[valid] - v_model[valid]) / v_err[valid])**2)
    n = np.sum(valid)
    if n <= k: return np.nan
    return chi2 + k * np.log(n)

def parse_sparc_file(filepath):
    """Robust parser for SPARC catalog .dat files."""
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
        
    df = df.rename(columns=rename_map).loc[:, ~df.columns.duplicated()]
    if 'Vbul' not in df.columns: df['Vbul'] = 0.0
    for col in ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk'])

def generate_comparative_panel():
    print("="*80)
    print(" GENERATING COMPARATIVE ROTATION CURVE PANEL (HYDRODYNAMIC VS NFW)")
    print("="*80)
    
    # Representative sample of galaxies using robust base names
    target_galaxies_base = ['NGC2841', 'UGC04325', 'NGC5055', 'DDO154']    
    try:
        all_files = os.listdir(SPARC_DATA_DIR)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Directory {SPARC_DATA_DIR} not found.")
        return
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, base_name in enumerate(target_galaxies_base):
        # Dynamically matches the exact filename in the user's directory
        matching_files = [f for f in all_files if base_name in f.upper() or base_name in f]
        matching_files = [f for f in matching_files if f.endswith('.dat')]
        
        if not matching_files:
            print(f" -> Warning: Could not find data file for {base_name}. Panel {i} will remain blank.")
            continue
            
        filepath = os.path.join(SPARC_DATA_DIR, matching_files[0])
        
        df = parse_sparc_file(filepath)
        if df is None or len(df) == 0: 
            print(f" -> Warning: Data file for {base_name} is empty or unreadable.")
            continue
            
        r = df['Rad'].values
        v_obs = df['Vobs'].values
        v_err = df['errV'].values
        v_gas = df['Vgas'].values
        v_disk = df['Vdisk'].values
        v_bulge = df['Vbul'].values
        
        # Generates the kinematics based STRICTLY on real data
        v_hydro = hydrodynamic_velocity(r, v_gas, v_disk, v_bulge, upsilon_d=0.5)
        v_nfw, k_nfw = fit_nfw(r, v_obs, v_err, v_gas, v_disk, v_bulge)
        
        # Calculates BIC scoring
        bic_hydro = calculate_bic(v_obs, v_hydro, v_err, k=0)
        bic_nfw = calculate_bic(v_obs, v_nfw, v_err, k=2)
        
        # Plotting routines
        axs[i].errorbar(r, v_obs, yerr=v_err, fmt='o', color='black', label='SPARC Obs', markersize=4)
        axs[i].plot(r, v_hydro, '-', color='red', linewidth=2, label=f'Hydrodynamic Ansatz (BIC: {bic_hydro:.1f})')
        axs[i].plot(r, v_nfw, '--', color='blue', linewidth=2, label=f'NFW Fit (BIC: {bic_nfw:.1f})')
        
        axs[i].set_title(f'Galaxy: {base_name}', fontweight='bold')
        axs[i].set_xlabel('Radius [kpc]')
        axs[i].set_ylabel('Velocity [km/s]')
        axs[i].legend(loc='lower right', fontsize=9)
        axs[i].grid(alpha=0.3)
        
        print(f" -> Successfully plotted {base_name} from {matching_files[0]}")
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "sparc_comparative_panel.jpg")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\n -> Comparative panel successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_comparative_panel()