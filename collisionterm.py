import numpy as np
import vegas

# --- Input Parameters ---
T1 = 10.0      # Temperature of species 1 (MeV)
T2 = 10.0      # Temperature of species 2 (MeV)
T3 = 8.0       # Temperature of species 3 (MeV)
T4 = 8.0       # Temperature of species 4 (MeV)
m = 100.0      # Mediator mass (MeV)
g = 0.1        # Coupling constant

def fermi_dirac(E, T):
    if E / T > 500: return 0.0
    return 1.0 / (np.exp(E / T) + 1.0)

def phase_space_integrand(x):
    # 1. Map [0,1] variables to physical ranges
    E1 = -T1 * np.log(1.0 - x[0]) 
    E2 = -T2 * np.log(1.0 - x[1])
    
    c_12 = 2.0 * x[2] - 1.0       # cos(theta_12)
    c_star = 2.0 * x[3] - 1.0     # cos(theta*) wrt boost axis
    phi_star = 2.0 * np.pi * x[4] # Azimuthal angle
    
    # 2. Kinematics
    s = 2.0 * E1 * E2 * (1.0 - c_12)
    # Return a dictionary of zeros if kinematics are invalid
    if s <= 0: return {'approx': 0.0, 'full': 0.0}
    
    # Total momentum P in the lab frame
    P2 = E1**2 + E2**2 + 2.0 * E1 * E2 * c_12
    P = np.sqrt(P2) if P2 > 0 else 0.0
    
    # Exact Lab frame energies for E3 and E4
    E3 = 0.5 * (E1 + E2 + P * c_star)
    E4 = E1 + E2 - E3
    
    # Momentum transfer t requires the scattering angle wrt p1*
    cos_alpha = (E1 + E2 * c_12) / P if P > 0 else 1.0
    sin_alpha = np.sqrt(max(0.0, 1.0 - cos_alpha**2))
    sin_star = np.sqrt(max(0.0, 1.0 - c_star**2))
    
    cos_theta_scatter = cos_alpha * c_star + sin_alpha * sin_star * np.cos(phi_star)
    t = -0.5 * s * (1.0 - cos_theta_scatter)
    
    # 3. Matrix Elements Squared
    M2_approx = ((g**2) / (-m**2))**2 * (t**2 / 4.0)
    M2_full = ((g**2) / (t - m**2))**2 * (t**2 / 4.0)
    
    # 4. Statistical Factor (Forward - Reverse)
    F = (fermi_dirac(E1, T1) * fermi_dirac(E2, T2) * (1.0 - fermi_dirac(E3, T3)) * (1.0 - fermi_dirac(E4, T4)) - 
         fermi_dirac(E3, T3) * fermi_dirac(E4, T4) * (1.0 - fermi_dirac(E1, T1)) * (1.0 - fermi_dirac(E2, T2)))
    
    # 5. Jacobians & Phase Space Volume
    jac = np.exp(E1/T1) * np.exp(E2/T2) * (T1 * T2) * (8.0 * np.pi)
    phase_space_prefactor = 1.0 / (256.0 * (np.pi**5))
    
    common_term = phase_space_prefactor * E1 * E2 * F * jac * E1
    
    # Return a dictionary so modern Vegas tracks both integrals cleanly
    return {'approx': common_term * M2_approx, 'full': common_term * M2_full}

# --- Integration ---
integ = vegas.Integrator([[0.0, 1.0]] * 5)

integ(phase_space_integrand, nitn=5, neval=20000) # Training phase
result = integ(phase_space_integrand, nitn=10, neval=100000) # Final phase

# Extract results using dictionary keys
approx_val = result['approx'].mean
approx_err = result['approx'].sdev
full_val = result['full'].mean
full_err = result['full'].sdev

diff = abs(full_val - approx_val) / abs(approx_val) * 100 if approx_val != 0 else 0.0

print(f"Analytical Approx (C33 style) : {approx_val:.4e} ± {approx_err:.4e}")
print(f"Full Propagator (with q)      : {full_val:.4e} ± {full_err:.4e}")
print(f"Percentage Difference         : {diff:.2f}%")
