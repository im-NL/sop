"""
Run the resonance QMC 5 times with m_phi=10 MeV, average and plot.
"""
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.special import gammaincinv
from scipy.stats import qmc as scipy_qmc
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(r'C:\Users\sannidhya\Downloads\stuff\5d\10mev')
OUT.mkdir(parents=True, exist_ok=True)

GF = 1.1663787e-11
MPL = 1.2209e22
NNU = 3
NNU_R = 6
S_ANN = 0.5

T_GEV = np.array([.010,.020,.050,.100,.130,.140,.150,.160,.170,.180,
                   .190,.200,.214,.500,1,2,5,10,20,50,100,200,500,1000])
G_RHO = np.array([10.76,11.33,14.63,18.00,21.76,23.77,26.31,29.51,33.47,38.27,
                   44.01,50.75,62.49,69.26,76.34,82.50,85.60,86.22,88.45,97.40,
                   103.53,105.90,106.61,106.72])
G_P   = np.array([10.75,10.99,13.40,16.21,18.16,19.05,20.13,21.42,22.98,24.84,
                   27.04,29.62,61.52,66.43,72.97,79.69,84.68,85.85,87.22,93.94,
                   100.80,104.75,106.38,106.65])
LR = np.log(T_GEV)
GR = PchipInterpolator(LR, G_RHO)
GP = PchipInterpolator(LR, G_P)
DGR = GR.derivative()


def gstar(T):
    x = np.log(np.clip(T/1000, T_GEV[0], T_GEV[-1]))
    return float(GR(x)), float(GP(x)), float(DGR(x))/T


def rho_components(T, TR):
    gr, gp, dg = gstar(T)
    rho_sm = np.pi**2/30*gr*T**4
    p_sm   = np.pi**2/90*gp*T**4
    rho_r  = NNU_R*7/8*np.pi**2/30*TR**4
    return rho_sm, p_sm, rho_r, rho_r/3, dg


def C_scatt_contact_paper(T, TR, GS):
    return 6/np.pi**5*NNU_R*GS**2*T**4*TR**4*(T-TR)*0.8118


def decoupling_temperature(GS):
    return 2*(np.sqrt(.28)*GS/GF)**(-2/3)


class QMC:
    def __init__(self, n_pow=9, seed=1, K=10):
        self.n = 2**n_pow
        self.K = K
        self.U = np.clip(
            scipy_qmc.Sobol(5, scramble=True, seed=seed).random_base2(n_pow),
            1e-12, 1-1e-12)

    def collision(self, T, TR, GS, mphi, gamma_frac, res=True):
        U = self.U; n = self.n
        p1 = TR*gammaincinv(2, U[:,0])
        p2 = TR*gammaincinv(2, U[:,1])
        Smax = 4*p1*p2
        G = gamma_frac*mphi; a = mphi*G; d = self.K*a
        lo = np.maximum(0, mphi*mphi-d)
        hi = np.minimum(Smax, mphi*mphi+d)
        hasR = Smax > lo; hasAbove = Smax > hi
        nint = np.where(hasAbove, 3, np.where(hasR, 2, 1))
        w = U[:,2]*nint; k = np.minimum(w.astype(int), nint-1); frac = w-k
        A = np.choose(k, [np.zeros(n), np.full(n, lo), hi])
        B = np.choose(k, [np.minimum(Smax, lo), hi, Smax])
        width = B-A; s = A+frac*width
        q_mu = (2*p1*p2)/(nint*width)
        mu = 1-s/(2*p1*p2)
        sinmu = np.sqrt(np.maximum(0, 1-mu*mu))
        P0 = p1+p2; Px = p2*sinmu; Pz = p1+p2*mu
        Pmag = np.sqrt(Px*Px+Pz*Pz)
        bx = Px/P0; bz = Pz/P0
        b2 = np.minimum((Pmag/P0)**2, 1-1e-15); gam = 1/np.sqrt(1-b2)
        z = 2*U[:,3]-1; ph = 2*np.pi*U[:,4]
        sth = np.sqrt(np.maximum(0, 1-z*z))
        nx = -bz*sth*np.sin(ph)+bx*z
        nz =  bx*sth*np.sin(ph)+bz*z
        est = .5*np.sqrt(s); pstar = est
        bp = bx*(pstar*nx)+bz*(pstar*nz)
        E3 = gam*(est+bp); E4 = P0-E3; E1 = p1; E2 = p2
        f1 = 1/(np.exp(np.clip(E1/TR, 0, 700))+1)
        f2 = 1/(np.exp(np.clip(E2/TR, 0, 700))+1)
        f3 = 1/(np.exp(np.clip(E3/T,  0, 700))+1)
        f4 = 1/(np.exp(np.clip(E4/T,  0, 700))+1)
        F = f1*f2*(1-f3)*(1-f4) - f3*f4*(1-f1)*(1-f2)
        if res:
            M2 = 4*(GS*mphi**2)**2*s*s/((s-mphi**2)**2+a*a)
        else:
            M2 = 4*GS*GS*s*s
        q1 = p1*np.exp(-p1/TR)/(TR*TR)
        q2 = p2*np.exp(-p2/TR)/(TR*TR)
        q = q1*q2*q_mu*.5/(2*np.pi)
        coeff = 1/(1024*np.pi**6)
        val = -NNU_R*S_ANN*coeff*p1*p2*E1*F*M2/q
        val = np.where(np.isfinite(val), val, 0)
        return float(np.mean(val)), float(np.std(val, ddof=1)/np.sqrt(n))


def collision_avg(mcs, T, TR, GS, mphi, gamma, res):
    vals = []; ses = []
    for mc in mcs:
        v, se = mc.collision(T, TR, GS, mphi, gamma, res)
        vals.append(v); ses.append(se)
    return float(np.mean(vals)), float(np.sqrt(np.sum(np.array(ses)**2))/len(mcs))


def rhs(x, y, mcs, GS, mphi, gamma, res):
    T = float(np.exp(x))
    r = float(np.clip(y[0], 1e-7, 1)); TR = r*T
    rho_sm, p_sm, rho_r, p_r, dg = rho_components(T, TR)
    H = np.sqrt(8*np.pi*(rho_sm+rho_r)/(3*MPL**2))
    C = collision_avg(mcs, T, TR, GS, mphi, gamma, res)[0] \
        + C_scatt_contact_paper(T, TR, GS)
    gr, _, _ = gstar(T)
    drsm = np.pi**2/30*(4*gr*T**3+T**4*dg)
    drr  = 4*rho_r/TR
    dTR_dT = ((3*H*(rho_r+p_r)-C)/(3*H*(rho_sm+p_sm)+C))*drsm/drr
    return [dTR_dT - r]


def solve(GS, mcs, mphi, gamma, n_eval=100):
    Ts = min(1e6, max(10, 20*decoupling_temperature(GS)))
    xs = np.linspace(np.log(Ts), np.log(10), n_eval)
    fun = lambda x, y: rhs(x, y, mcs, GS, mphi, gamma, True)
    sol = solve_ivp(fun, (xs[0], xs[-1]), [1.0],
                    t_eval=xs, rtol=5e-3, atol=3e-5,
                    max_step=.18, method='RK45')
    return np.exp(xs), np.clip(sol.y[0], 0, 1)**4, sol


# ── Configuration ──────────────────────────────────────────
MPHI    = 10     # MeV
GAMMA   = 0.01
N_RUNS  = 5
N_POW   = 9      # 512 Sobol samples -- enough noise to see run-to-run scatter
COUPLINGS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# Each run uses a single QMC object so individual runs show real MC noise.
RUN_SEEDS = [1001, 2002, 3003, 4004, 5005]

# ── Run ────────────────────────────────────────────────────
all_runs = {}   # coupling -> {'T': array, 'ys': [array, ...]}

for run_idx in range(N_RUNS):
    seed = RUN_SEEDS[run_idx]
    mcs = [QMC(n_pow=N_POW, seed=seed, K=10)]
    print(f"\n=== Run {run_idx+1}/{N_RUNS}  (seed {seed}) ===", flush=True)
    for a in COUPLINGS:
        GS = a * GF
        print(f"  G_S = {a:.0e} G_F ... ", end="", flush=True)
        T, y, sol = solve(GS, mcs, MPHI, GAMMA, 100)
        print(f"ok={sol.success}  y(10 MeV)={y[-1]:.6f}")
        if a not in all_runs:
            all_runs[a] = {'T': T, 'ys': []}
        all_runs[a]['ys'].append(y)

# ── Plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
cmap = plt.cm.plasma
colors = [cmap(v) for v in np.linspace(0.10, 0.85, len(COUPLINGS))]

for i, a in enumerate(COUPLINGS):
    T   = all_runs[a]['T']
    ys  = np.array(all_runs[a]['ys'])          # (5, n_eval)
    y_m = np.mean(ys, axis=0)
    y_s = np.std(ys, axis=0)
    ax.semilogx(T, y_m, color=colors[i], lw=2.2,
                label=rf'$G_S = {a:.0e}\,G_F$')
    ax.fill_between(T, y_m-y_s, y_m+y_s, color=colors[i], alpha=0.15)

ax.axvline(MPHI/2, ls='--', lw=1.3, alpha=0.65, color='grey',
           label=rf'$m_\phi/2 = {MPHI/2:.0f}$ MeV')
ax.set_xlim(10, 1e6)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r'$T_{\rm SM}$ [MeV]', fontsize=13)
ax.set_ylabel(r'$(T_{\nu_R}/T_{\rm SM})^4$', fontsize=13)
ax.set_title(
    rf'QMC average of {N_RUNS} runs — '
    rf'$m_\phi = {MPHI}$ MeV, $\Gamma/m = {GAMMA}$',
    fontsize=13)
ax.legend(fontsize=9, frameon=False, loc='lower right')
ax.grid(alpha=0.2, which='both')
fig.tight_layout()
outpath = OUT / 'qmc_5run_average.png'
fig.savefig(outpath, dpi=220)
plt.close(fig)
print(f"\nPlot saved -> {outpath}")

# ── CSV ────────────────────────────────────────────────────
csv_path = OUT / 'qmc_5run_endpoints.csv'
with open(csv_path, 'w') as f:
    f.write('GS_over_GF,mean_y10,std_y10\n')
    for a in COUPLINGS:
        ys = np.array(all_runs[a]['ys'])
        f.write(f'{a:.0e},{np.mean(ys[:,-1]):.8f},{np.std(ys[:,-1]):.8f}\n')
print(f"CSV saved  -> {csv_path}")
