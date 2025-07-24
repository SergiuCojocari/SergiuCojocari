import numpy as np
from pyscf import gto, dft, tdscf
import os
import csv
from scipy.optimize import fsolve
import warnings

# Constants
hartree_to_ev = 27.2114
nstates_per_multiplicity = 5  # Numărul de stări excitate pentru FIECARE multiplicitate (singlet/triplet)
step_size = 0.1
start_distance = 2.1
end_distance = 2.4

# --- Alegeți solventul aici ---
SOLVENT_TO_USE = None
FUNCTIONAL_USED = 'wB97M-V'

# Distance values
d_values = np.arange(start_distance, end_distance + step_size, step_size)
if not np.isclose(d_values[-1], end_distance):
    if end_distance > d_values[-1]:
         d_values = np.append(d_values, end_distance)


# --- Definirea etichetelor pentru stări ---
state_labels = ['S0']
for i in range(1, nstates_per_multiplicity + 1):
    state_labels.append(f'S{i}')
for i in range(1, nstates_per_multiplicity + 1):
    state_labels.append(f'T{i}')

total_electronic_states_count = len(state_labels)

# --- Stocarea energiilor pentru toate stările ---
energies_tz = {label: [] for label in state_labels}
energies_qz = {label: [] for label in state_labels}
energies_5z = {label: [] for label in state_labels}
energies_cbs = {label: [] for label in state_labels}

# Basis set mapping for extrapolation
basis_sets_cardinal = {
    'aug-cc-pVTZ': 3,
    'aug-cc-pVQZ': 4,
    'aug-cc-pV5Z': 5
}

# Set threads
os.environ['OMP_NUM_THREADS'] = '32'

# Cache pentru energiile atomului de Ar izolat
e_isolated_ar_cache = {}

def get_isolated_ar_energy(basis_name, xc_functional=FUNCTIONAL_USED, conv_tol_val=1e-9, grid_level_val=9, max_cycle_val=200, solvent_name=None):
    cache_key = (basis_name, xc_functional, solvent_name)
    if cache_key in e_isolated_ar_cache:
        return e_isolated_ar_cache[cache_key]
    print(f"Calculating isolated Ar energy for basis: {basis_name}, Functional: {xc_functional}, Solvent: {solvent_name if solvent_name else 'Vacuum'}")
    mol_isolated_Ar = gto.M(atom=[['Ar', (0.0, 0.0, 0.0)]],
                              basis=basis_name, unit='Angstrom', charge=0, spin=0)
    # MODIFICAT: Folosim RKS pentru sisteme closed-shell (spin=0)
    if mol_isolated_Ar.spin == 0:
        mf = dft.RKS(mol_isolated_Ar).x2c()
    else: # Fallback la UKS pentru sisteme open-shell (nu e cazul aici pentru Ar)
        mf = dft.UKS(mol_isolated_Ar).x2c()
    mf.xc = xc_functional
    mf.conv_tol = conv_tol_val
    mf.grids.level = grid_level_val
    mf.max_cycle = max_cycle_val
    if solvent_name:
        mf = mf.ddPCM()
        mf.solvent_name = solvent_name
    mf.kernel()
    e_isolated_ar_cache[cache_key] = mf.e_tot
    return mf.e_tot

def run_dft_calculation(atom_spec, basis, xc_functional=FUNCTIONAL_USED, charge=0, spin=0, conv_tol_val=1e-9, grid_level_val=9, max_cycle_val=200, solvent_name=None):
    mol = gto.M(atom=atom_spec, basis=basis, unit='Angstrom', charge=charge, spin=spin)
    # MODIFICAT: Folosim RKS pentru sisteme closed-shell (spin=0)
    if spin == 0:
        mf = dft.RKS(mol).x2c()
    else: # Fallback la UKS pentru sisteme open-shell
        mf = dft.UKS(mol).x2c()
    mf.xc = xc_functional
    mf.conv_tol = conv_tol_val
    mf.grids.level = grid_level_val
    mf.max_cycle = max_cycle_val
    if solvent_name:
        mf = mf.ddPCM()
        mf.solvent_name = solvent_name
    mf.kernel()
    return mf

def calculate_energies_cp(distance, basis_name, xc_functional=FUNCTIONAL_USED, solvent_name=None):
    """Calculează S0, stările excitate de SINGLET și TRIPLET cu corecție CP."""
    atom_dimer = [['Ar', (0.0, 0.0, 0.0)], ['Ar', (0.0, 0.0, distance)]]
    # mf_dimer va fi un obiect RKS datorită modificării în run_dft_calculation
    mf_dimer = run_dft_calculation(atom_dimer, basis_name, xc_functional=xc_functional, charge=0, spin=0, solvent_name=solvent_name)
    e_dimer_hartree_raw = mf_dimer.e_tot

    atom_Ar1_ghost_Ar2 = [['Ar', (0.0, 0.0, 0.0)], ['ghost:Ar', (0.0, 0.0, distance)]]
    mf_Ar1_ghost_Ar2 = run_dft_calculation(atom_Ar1_ghost_Ar2, basis_name, xc_functional=xc_functional, charge=0, spin=0, solvent_name=solvent_name)
    e_Ar1_ghost_Ar2_hartree = mf_Ar1_ghost_Ar2.e_tot

    atom_Ar2_ghost_Ar1 = [['ghost:Ar', (0.0, 0.0, 0.0)], ['Ar', (0.0, 0.0, distance)]]
    mf_Ar2_ghost_Ar1 = run_dft_calculation(atom_Ar2_ghost_Ar1, basis_name, xc_functional=xc_functional, charge=0, spin=0, solvent_name=solvent_name)
    e_Ar2_ghost_Ar1_hartree = mf_Ar2_ghost_Ar1.e_tot

    e_isolated_Ar_hartree = get_isolated_ar_energy(basis_name, xc_functional=xc_functional, solvent_name=solvent_name)

    bsse_hartree = (2 * e_isolated_Ar_hartree) - (e_Ar1_ghost_Ar2_hartree + e_Ar2_ghost_Ar1_hartree)
    
    energies_hartree_list_cp = [] 

    e_s0_cp_corrected_hartree = e_dimer_hartree_raw + bsse_hartree
    energies_hartree_list_cp.append(e_s0_cp_corrected_hartree)

    # --- Calculul Stărilor Excitate de SINGLET ---
    # Pentru referință RKS, tdscf.TDDFT este echivalent cu tdscf.TDHFTD(mf_dimer). TDRKS
    td_singlet = tdscf.TDDFT(mf_dimer) 
    td_singlet.nstates = nstates_per_multiplicity 
    td_singlet.singlet = True   
    print(f"  Calculating {nstates_per_multiplicity} Singlet excited states for basis {basis_name} at {distance:.2f} Å...")
    try:
        td_singlet.kernel() 
        for j in range(nstates_per_multiplicity):
            if j < len(td_singlet.e):
                excited_state_energy_hartree_raw = mf_dimer.e_tot + td_singlet.e[j]
                excited_state_cp_corrected_hartree = excited_state_energy_hartree_raw + bsse_hartree
                energies_hartree_list_cp.append(excited_state_cp_corrected_hartree)
            else:
                print(f"    Singlet state {j+1} for basis {basis_name} at {distance:.2f} Å not found by TDDFT.")
                energies_hartree_list_cp.append(np.nan)
    except Exception as e:
        print(f"⚠️ Singlet TDDFT failed at {distance:.2f} Å, basis {basis_name}, Functional: {xc_functional}, Solvent: {solvent_name if solvent_name else 'Vacuum'}: {e}")
        for _ in range(nstates_per_multiplicity):
            energies_hartree_list_cp.append(np.nan)

    # --- Calculul Stărilor Excitate de TRIPLET ---
    td_triplet = tdscf.TDDFT(mf_dimer) 
    td_triplet.nstates = nstates_per_multiplicity
    td_triplet.singlet = False  
    print(f"  Calculating {nstates_per_multiplicity} Triplet excited states for basis {basis_name} at {distance:.2f} Å...")
    try:
        td_triplet.kernel() 
        for j in range(nstates_per_multiplicity):
            if j < len(td_triplet.e):
                excited_state_energy_hartree_raw = mf_dimer.e_tot + td_triplet.e[j]
                excited_state_cp_corrected_hartree = excited_state_energy_hartree_raw + bsse_hartree
                energies_hartree_list_cp.append(excited_state_cp_corrected_hartree)
            else:
                print(f"    Triplet state {j+1} for basis {basis_name} at {distance:.2f} Å not found by TDDFT.")
                energies_hartree_list_cp.append(np.nan)
    except Exception as e:
        print(f"⚠️ Triplet TDDFT failed at {distance:.2f} Å, basis {basis_name}, Functional: {xc_functional}, Solvent: {solvent_name if solvent_name else 'Vacuum'}: {e}")
        for _ in range(nstates_per_multiplicity):
            energies_hartree_list_cp.append(np.nan)

    energies_ev_list_cp = [e * hartree_to_ev if not np.isnan(e) else np.nan for e in energies_hartree_list_cp]
    return energies_ev_list_cp

# Inițializarea cache-ului și pre-calcularea energiilor atomilor izolați
print(f"Pre-calculating isolated Argon energies with Functional: {FUNCTIONAL_USED}, Solvent: {SOLVENT_TO_USE if SOLVENT_TO_USE else 'Vacuum'}...")
get_isolated_ar_energy('aug-cc-pVTZ', xc_functional=FUNCTIONAL_USED, solvent_name=SOLVENT_TO_USE)
get_isolated_ar_energy('aug-cc-pVQZ', xc_functional=FUNCTIONAL_USED, solvent_name=SOLVENT_TO_USE)
get_isolated_ar_energy('aug-cc-pV5Z', xc_functional=FUNCTIONAL_USED, solvent_name=SOLVENT_TO_USE)
print("--- Finished pre-calculation ---")


def cbs_extrapolate_2point_B3(E_X_large, E_Y_small, X_large, Y_small):
    if np.isnan(E_X_large) or np.isnan(E_Y_small):
        return np.nan
    if X_large < Y_small: 
        E_X_large, E_Y_small = E_Y_small, E_X_large
        X_large, Y_small = Y_small, X_large
    X_L_cubed = X_large**3
    Y_S_cubed = Y_small**3
    if abs(X_L_cubed - Y_S_cubed) < 1e-9:
        return np.nan 
    return (X_L_cubed * E_X_large - Y_S_cubed * E_Y_small) / (X_L_cubed - Y_S_cubed)

def cbs_extrapolate_variable_exponent(E_tz, E_qz, E_5z, X_tz, X_qz, X_5z, 
                                     fallback_func_2point, E_fallback_qz, E_fallback_5z, X_fb_qz, X_fb_5z):
    energies = np.array([E_tz, E_qz, E_5z])
    if np.any(np.isnan(energies)):
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz) 
    
    epsilon = 1e-7 
    cond_desc = (E_tz >= E_qz - epsilon and E_qz >= E_5z - epsilon)
    cond_cresc = (E_tz <= E_qz + epsilon and E_qz <= E_5z + epsilon)

    if not (cond_desc or cond_cresc) or abs(E_qz - E_5z) < 1e-9 :
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)
        
    if abs(E_qz - E_5z) < 1e-12 : 
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)

    ratio_E = (E_tz - E_qz) / (E_qz - E_5z)

    if ratio_E <= 1e-6 : 
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)

    def equation_for_B(B_val_arr):
        B_val = B_val_arr[0]
        if B_val < 0.1 or B_val > 10.0: 
            return 1e6 
        term_Xtz_B = X_tz**(-B_val)
        term_Xqz_B = X_qz**(-B_val)
        term_X5z_B = X_5z**(-B_val)
        numerator_X = term_Xtz_B - term_Xqz_B
        denominator_X = term_Xqz_B - term_X5z_B
        if abs(denominator_X) < 1e-12: 
            return 1e6 
        ratio_X = numerator_X / denominator_X
        return ratio_E - ratio_X

    B_solution = np.nan
    solved_B_successfully = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning) 
            initial_guesses_B = [3.0, 4.0, 2.5, 3.5, 2.0, 5.0] 
            for guess_B in initial_guesses_B:
                sol, infodict, ier, mesg = fsolve(equation_for_B, x0=[guess_B], full_output=True, xtol=1e-7, maxfev=200)
                if ier == 1 and 0.5 < sol[0] < 8.0: 
                    B_solution = sol[0]
                    solved_B_successfully = True
                    break
            if not solved_B_successfully:
                return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)
    except Exception:
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)

    B = B_solution
    denominator_A = X_tz**(-B) - X_qz**(-B)
    if abs(denominator_A) < 1e-12:
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)

    A = (E_tz - E_qz) / denominator_A
    E_CBS = E_tz - A * (X_tz**(-B))
    
    tolerance_check = 1e-5 
    if (E_tz > E_qz and E_qz > E_5z and E_CBS > E_5z + tolerance_check) or \
       (E_tz < E_qz and E_qz < E_5z and E_CBS < E_5z - tolerance_check):
        return fallback_func_2point(E_fallback_5z, E_fallback_qz, X_fb_5z, X_fb_qz)
    return E_CBS

# --- Main loop ---
for d in d_values:
    print(f"\n--- Distance: {d:.2f} Å --- Functional: {FUNCTIONAL_USED}, Solvent: {SOLVENT_TO_USE if SOLVENT_TO_USE else 'Vacuum'} ---")

    e_tz_all_states_cp_ev = calculate_energies_cp(d, 'aug-cc-pVTZ', xc_functional=FUNCTIONAL_USED, solvent_name=SOLVENT_TO_USE)
    e_qz_all_states_cp_ev = calculate_energies_cp(d, 'aug-cc-pVQZ', xc_functional=FUNCTIONAL_USED, solvent_name=SOLVENT_TO_USE)
    e_5z_all_states_cp_ev = calculate_energies_cp(d, 'aug-cc-pV5Z', xc_functional=FUNCTIONAL_USED, solvent_name=SOLVENT_TO_USE)

    for list_idx, current_state_label in enumerate(state_labels):
        val_tz = e_tz_all_states_cp_ev[list_idx] if len(e_tz_all_states_cp_ev) > list_idx else np.nan
        val_qz = e_qz_all_states_cp_ev[list_idx] if len(e_qz_all_states_cp_ev) > list_idx else np.nan
        val_5z = e_5z_all_states_cp_ev[list_idx] if len(e_5z_all_states_cp_ev) > list_idx else np.nan

        energies_tz[current_state_label].append(val_tz)
        energies_qz[current_state_label].append(val_qz)
        energies_5z[current_state_label].append(val_5z)

        e_cbs_val = cbs_extrapolate_variable_exponent(
            val_tz, val_qz, val_5z, 
            basis_sets_cardinal['aug-cc-pVTZ'], 
            basis_sets_cardinal['aug-cc-pVQZ'], 
            basis_sets_cardinal['aug-cc-pV5Z'],
            fallback_func_2point=cbs_extrapolate_2point_B3,
            E_fallback_qz=val_qz, 
            E_fallback_5z=val_5z, 
            X_fb_qz=basis_sets_cardinal['aug-cc-pVQZ'], 
            X_fb_5z=basis_sets_cardinal['aug-cc-pV5Z']
        )
        energies_cbs[current_state_label].append(e_cbs_val)

# --- Save to CSV ---
solvent_tag_filename = f"_solvent_{SOLVENT_TO_USE.lower().replace('-', '_')}" if SOLVENT_TO_USE else "_vacuum"
filename = f"argon_dimer_ST_energies_cbs_varB_TQ5_cp_{FUNCTIONAL_USED.replace('-', '_').lower()}{solvent_tag_filename}.csv" 

header_solvent_tag_display = f" {SOLVENT_TO_USE.title()}" if SOLVENT_TO_USE else " Vacuum"
header_functional_display = f" {FUNCTIONAL_USED}"
cbs_method_tag_display = " CBS(varB,TQ5)" 

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Distance (Å)'] + [f'{label}{cbs_method_tag_display}_CP{header_functional_display}{header_solvent_tag_display} (eV)' for label in state_labels]
    writer.writerow(header)

    num_distances_processed = len(d_values) 

    for i in range(num_distances_processed):
        d_val = d_values[i] 
        row = [d_val]
        for label in state_labels:
            if i < len(energies_cbs[label]):
                row.append(energies_cbs[label][i])
            else:
                row.append(np.nan) 
        writer.writerow(row)

print(f"\n✅ Calculul și extrapolarea (Singlet & Triplet) CBS(varB,TQ5) cu corecție CP, Functional: {FUNCTIONAL_USED}, Solvent ({SOLVENT_TO_USE if SOLVENT_TO_USE else 'Vacuum'}) s-au încheiat.")
print(f"Rezultatele au fost salvate în '{filename}'")
