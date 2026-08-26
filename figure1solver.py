import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

M_PL = 1.22e22          # Planck mass in MeV
G_F = 1.1663787e-11     # Fermi constant in MeV^-2

# 2. Standard Model Degrees of Freedom
TRANSITIONS = [
    (35.0, 3.5, 0.4),      # Muons
    (45.0, 3.0, 0.4),      # Pions
    (150.0, 44.5, 0.4),    # QCD Phase Transition
    (400.0, 10.5, 0.4),    # Charm 
    (600.0, 3.5, 0.4),     # Tau 
    (1400.0, 10.5, 0.4),   # Bottom 
    (30000.0, 20.5, 0.5)   # W, Z, Higgs, Top
]

def g_star(T):
    g = 10.75  
    for Tc, dg, w in TRANSITIONS:
        g += dg * 0.5 * (1.0 + np.tanh((np.log(T) - np.log(Tc)) / w))
    return g

def dg_star_dlnT(T):
    dg = 0.0
    for Tc, dg_val, w in TRANSITIONS:
        sech2 = np.cosh((np.log(T) - np.log(Tc)) / w)**-2
        dg += dg_val * 0.5 * sech2 / w
    return dg


# 3. ODE Solver (No Interpolation Artifacts)
def solve_evolution(G_S_factor):
    G_S_val = G_S_factor * G_F
    
    def dr_dx(x, y):
        r = y[0]
        T = np.exp(x)
        r_safe = max(1e-10, r)
        
        g_SM = g_star(T)
        dg_dlnT = dg_star_dlnT(T)
        delta = dg_dlnT / (3.0 * g_SM)
        
        rho_SM = (np.pi**2 / 30) * g_SM * T**4
        rho_nuR = (np.pi**2 / 30) * 5.25 * (r_safe * T)**4
        H = np.sqrt((8 * np.pi / (3 * M_PL**2)) * (rho_SM + rho_nuR))
        
        s_SM = (2 * np.pi**2 / 45) * g_SM * T**3
        B = 3 * H * T * s_SM
        S_r = g_SM / (5.25 * r_safe**3)
        
        C0_1 = (72 * 0.8840 / np.pi**5) * G_S_val**2
        C0_5 = (36 * 0.8118 / np.pi**5) * G_S_val**2
        C = (C0_1 * (1.0 - r_safe**9) + C0_5 * r_safe**4 * (1.0 - r_safe)) * T**9
        
        if C < -0.9 * B:
            C = -0.9 * B
            
        numerator = r_safe * delta - (C / B) * (S_r * (1.0 + delta) + r_safe)
        denominator = 1.0 + C / B
        
        return [numerator / denominator]

    x_start = np.log(1e6)
    x_end = np.log(10.0)
    
    sol = solve_ivp(dr_dx, [x_start, x_end], [1.0], 
                    method='Radau', 
                    rtol=1e-10, atol=1e-13,
                    max_step=0.01)
    
    T_SM_vals = np.exp(sol.t)
    r_vals = sol.y[0]
    
    return T_SM_vals, r_vals

# 4. Plotting
Gs_factors = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

plt.figure(figsize=(9, 6))

for Gs in Gs_factors:
    print(f"Solving for G_S = 10^{int(np.log10(Gs))} G_F...")
    T_SM_vals, r_vals = solve_evolution(Gs)
    
    ratio_4th = r_vals**4
    exponent = int(np.log10(Gs))
    label_str = rf'$G_S = 10^{{{exponent}}} G_F$'
    
    plt.plot(T_SM_vals, ratio_4th, lw=2.5, label=label_str)

plt.xscale('log')
plt.yscale('linear')
plt.ylim(0.0, 1.1)

plt.xlim(10.0, 1e6) 

tick_locations = [10**i for i in range(1, 7)]
tick_labels = [rf'$10^{{{i}}}$' for i in range(1, 7)]
plt.xticks(tick_locations, tick_labels)

plt.xlabel(r'$T_{SM}$ [MeV]', fontsize=14)
plt.ylabel(r'$(T_{\nu_R} / T_{SM})^4$', fontsize=14)
plt.title('Evolution of Right-Handed Neutrino Temperature', fontsize=15)
plt.legend(loc='lower right', fontsize=12) 
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()