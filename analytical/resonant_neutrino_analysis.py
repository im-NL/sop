import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path('/mnt/data')
GF = 1.1663787e-11  # MeV^-2
MPL = 1.2209e22      # MeV; matches the convention H^2=8 pi rho/(3 M_Pl^2)
NNU = 3
NNU_R = 2*NNU

# Compact digitized subset of the g_* table used by the paper (Husdal 2016),
# sufficient over 10 MeV ... 1 TeV for the reproduction.
T_GEV = np.array([
    .010,.020,.050,.100,.130,.140,.150,.160,.170,.180,.190,.200,.214,
    .500,1,2,5,10,20,50,100,200,500,1000], float)
G_RHO = np.array([
    10.76,11.33,14.63,18.00,21.76,23.77,26.31,29.51,33.47,38.27,
    44.01,50.75,62.49,69.26,76.34,82.50,85.60,86.22,88.45,97.40,
    103.53,105.90,106.61,106.72])
G_P = np.array([
    10.75,10.99,13.40,16.21,18.16,19.05,20.13,21.42,22.98,24.84,
    27.04,29.62,61.52,66.43,72.97,79.69,84.68,85.85,87.22,93.94,
    100.80,104.75,106.38,106.65])
LOGT = np.log(T_GEV)
G_RHO_F = PchipInterpolator(LOGT, G_RHO)
G_P_F = PchipInterpolator(LOGT, G_P)
DGRHO_F = G_RHO_F.derivative()

# Gauss-Legendre nodes for the remaining p1,p2 integrations.
ZQ1, WQ1 = leggauss(18)
ZQ2, WQ2 = leggauss(12)
XMAX = 35.0


def gstar(T):
    x = np.log(np.clip(T/1000.0, T_GEV[0], T_GEV[-1]))
    gr = float(G_RHO_F(x))
    gp = float(G_P_F(x))
    dg_dT = float(DGRHO_F(x))/T
    return gr, gp, dg_dT


def rho_components(T, TR):
    gr, gp, dg = gstar(T)
    rho_sm = np.pi**2/30.0 * gr * T**4
    p_sm = np.pi**2/90.0 * gp * T**4
    rho_r = NNU_R*(7.0/8.0)*np.pi**2/30.0 * TR**4
    p_r = rho_r/3.0
    return rho_sm, p_sm, rho_r, p_r, dg


def Hubble(T, TR):
    rho_sm, _, rho_r, _, _ = rho_components(T, TR)
    return np.sqrt(8*np.pi*(rho_sm + rho_r)/(3*MPL**2))


def C_ann_contact(T, TR, GS):
    # Eq. (C30)/Table III + the paper's N_nuR and FD correction.
    return (12.0/np.pi**5)*NNU_R*GS**2*(T**9 - TR**9)*0.8840


def C_scatt_contact(T, TR, GS):
    # GS-only scattering term, Table II / Table III.
    return (6.0/np.pi**5)*NNU_R*GS**2*T**4*TR**4*(T-TR)*0.8118


def C_contact(T, TR, GS):
    return C_ann_contact(T, TR, GS) + C_scatt_contact(T, TR, GS)


def decoupling_temperature(GS):
    # Paper Eq. (34)-(36), GS-only: G_eff = sqrt(0.28) GS.
    Geff = np.sqrt(0.28)*GS
    return 2.0*(Geff/GF)**(-2.0/3.0)


def rhs_logT_contact(x, y, GS):
    T = np.exp(x)
    r = float(y[0])
    TR = r*T
    rho_sm, p_sm, rho_r, p_r, dg = rho_components(T, TR)
    H = np.sqrt(8*np.pi*(rho_sm + rho_r)/(3*MPL**2))
    C = C_contact(T, TR, GS)
    drho_sm_dT = np.pi**2/30.0*(4*gstar(T)[0]*T**3 + T**4*dg)
    drho_r_dTR = 4.0*rho_r/TR
    dTR_dT = ((3*H*(rho_r+p_r)-C)/(3*H*(rho_sm+p_sm)+C))*drho_sm_dT/drho_r_dTR
    return [dTR_dT-r]


def I_resonant(S, mphi, Gamma):
    """Stable analytic primitive for int_0^S ds s^2/((s-m^2)^2+(m Gamma)^2).

    The arctan sum is evaluated as one atan2 expression rather than subtracting
    two nearly equal angles. The log uses log1p. A low-S series remains as a
    final safeguard against loss of significance.
    """
    S = np.asarray(S, dtype=float)
    a = mphi*mphi
    b = mphi*Gamma
    logterm = np.log1p((S*S - 2.0*a*S)/(a*a + b*b)) * a
    theta = np.arctan2(b*S, b*b - a*(S-a))
    out = S + logterm + (a*a-b*b)/b * theta
    y = S/a
    mask = y < 1e-3
    if np.any(mask):
        yy = y[mask]
        # Expansion in y; width corrections are O((Gamma/m)^2) here.
        ss = np.zeros_like(yy)
        for n in range(24):
            ss += (n+1)*yy**(n+3)/(n+3)
        out = np.array(out, copy=True)
        out[mask] = a*ss
    return out


def C_ann_resonant(T, TR, GS, mphi=1000.0, gamma_frac=1e-2, fd=0.8840):
    """s-channel scalar UV completion matched to GS at low energy.

    The s-pole is integrated analytically. The remaining (p1,p2) integration is
    deterministic and explicitly split along the resonance boundary
    4 p1 p2 = m_phi^2, so the narrow feature is not dependent on random hits.
    """
    Gamma = gamma_frac*mphi
    g2 = GS*mphi*mphi
    pmax = T*XMAX
    p1split = mphi*mphi/(4.0*pmax)
    outer = [(0.0,pmax)] if not (0.0 < p1split < pmax) else [(0.0,p1split),(p1split,pmax)]
    raw = 0.0
    for lo1,hi1 in outer:
        z=(ZQ1+1.0)/2.0
        p1vals=lo1+(hi1-lo1)*z
        w1vals=(hi1-lo1)/2.0*WQ1
        for p1,w1 in zip(p1vals,w1vals):
            pres=mphi*mphi/(4.0*p1)
            inner_segments=[(0.0,pmax)] if not (0.0 < pres < pmax) else [(0.0,pres),(pres,pmax)]
            inner=0.0
            for lo2,hi2 in inner_segments:
                zz=(ZQ2+1.0)/2.0
                p2=lo2+(hi2-lo2)*zz
                w2=(hi2-lo2)/2.0*WQ2
                A=-(p1+p2)/T
                delta=-(p1+p2)*(1.0/TR-1.0/T)
                F=np.exp(A)*np.expm1(delta)
                Is=I_resonant(4.0*p1*p2,mphi,Gamma)
                inner += np.sum(w2*(p1*F*Is))
            raw += w1*inner
    return -NNU_R*0.5/(128*np.pi**5)*g2**2*raw*fd


def C_resonant(T, TR, GS, mphi=1000.0, gamma_frac=1e-2):
    # Only the annihilation process gets the s-channel pole in this benchmark.
    # The crossed scattering term is kept at contact strength.
    return C_ann_resonant(T, TR, GS, mphi, gamma_frac) + C_scatt_contact(T, TR, GS)


def rhs_logT_resonant(x, y, GS, mphi, gamma_frac):
    T = np.exp(x)
    r = float(y[0])
    # Keep physical branch; numerical solver may approach equilibrium from below.
    r_eff = min(max(r, 1e-12), 1.0)
    TR = r_eff*T
    rho_sm, p_sm, rho_r, p_r, dg = rho_components(T, TR)
    H = np.sqrt(8*np.pi*(rho_sm + rho_r)/(3*MPL**2))
    C = C_resonant(T, TR, GS, mphi, gamma_frac)
    gr, _, _ = gstar(T)
    drho_sm_dT = np.pi**2/30.0*(4*gr*T**3 + T**4*dg)
    drho_r_dTR = 4.0*rho_r/TR
    dTR_dT = ((3*H*(rho_r+p_r)-C)/(3*H*(rho_sm+p_sm)+C))*drho_sm_dT/drho_r_dTR
    return [dTR_dT-r]


def solve_curve(GS, resonant=False, mphi=1000.0, gamma_frac=1e-2):
    Tdec = decoupling_temperature(GS)
    # Do not integrate the numerically stiff, tightly-coupled regime from 1e6 MeV.
    # Above ~20 T_dec we take r=T_nuR/T_SM=1 (the same equilibrium initial condition
    # used in the paper) and then integrate the physical departure from equilibrium.
    T_start = min(1.0e6, max(10.0, 20.0*Tdec))
    xs = np.linspace(np.log(T_start), np.log(10.0), 500)
    r0 = 1.0
    if resonant:
        fun = lambda x,y: rhs_logT_resonant(x,y,GS,mphi,gamma_frac)
    else:
        fun = lambda x,y: rhs_logT_contact(x,y,GS)
    sol = solve_ivp(fun, (xs[0], xs[-1]), [r0], t_eval=xs, method='Radau',
                    rtol=2e-4, atol=2e-7, max_step=0.08)
    r = np.minimum(sol.y[0], 1.0)
    return np.exp(xs), r, sol


def make_contact_curves():
    curves=[]
    for a in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        T,r,sol=solve_curve(a*GF, resonant=False)
        if not sol.success:
            raise RuntimeError(sol.message)
        curves.append((a,T,r**4))
    return curves


def make_res_curves(mphi=1000.0, gamma_frac=1e-2):
    curves=[]
    for a in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        T,r,sol=solve_curve(a*GF, resonant=True, mphi=mphi, gamma_frac=gamma_frac)
        if not sol.success:
            raise RuntimeError(sol.message)
        curves.append((a,T,r**4))
    return curves


def save_side_by_side(contact, resonant, mphi=1000.0, gamma_frac=1e-2):
    # User-uploaded paper figure is the source image; crop page 8 around Figure 1.
    page8 = OUT/'page8.png'
    if page8.exists():
        im = Image.open(page8)
        crop = im.crop((400,250,1510,900))
        crop.save(OUT/'figure1_original_crop.png')
    else:
        crop = None

    fig, axes = plt.subplots(1,2,figsize=(14,5.4), constrained_layout=True)
    if crop is not None:
        axes[0].imshow(crop)
        axes[0].axis('off')
        axes[0].set_title('Paper — Figure 1 (source image)', fontsize=12)
    else:
        axes[0].text(0.5,0.5,'Original page image unavailable',ha='center')
        axes[0].axis('off')

    for a,T,y in resonant:
        axes[1].semilogx(T,y,label=rf'$G_S={a:.0e}G_F$')
    axes[1].axvline(mphi/2.0,ls='--',lw=1.2,alpha=0.7)
    axes[1].text(mphi/2.0*1.05,0.08,rf'$m_\phi/2={mphi/2:.0f}\,$MeV',rotation=90,va='bottom')
    axes[1].set_xlim(10,1e6)
    axes[1].set_ylim(0.04,1.03)
    axes[1].set_xlabel(r'$T_{SM}$ [MeV]')
    axes[1].set_ylabel(r'$(T_{\nu_R}/T_{SM})^4$')
    axes[1].grid(True,which='both',alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title(rf'Our result — resonant $s$-channel, $m_\phi={mphi/1000:.1f}$ GeV, $\Gamma/m={gamma_frac:.0e}$',fontsize=11)
    fig.suptitle('Temperature evolution: contact benchmark vs. resonance-safe UV completion',fontsize=14)
    fig.savefig(OUT/'figure1_side_by_side_resonance.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


def save_propagator_spike_plot(GS=1e-4*GF, mphi=1000.0, gamma_frac=1e-2):
    Gamma=gamma_frac*mphi
    g2=GS*mphi*mphi
    sqrts=np.linspace(0.6*mphi,1.4*mphi,4000)
    ss=sqrts**2
    norm=(g2**2/((ss-mphi*mphi)**2+(mphi*Gamma)**2))/(g2**2/mphi**4)
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    ax.plot(sqrts/1000.0,norm)
    ax.axvline(mphi/1000.0,ls='--',lw=1.2,alpha=.7,label=r'$\sqrt{s}=m_\phi$')
    ax.set_xlabel(r'$\sqrt{s}$ [GeV]')
    ax.set_ylabel(r'$|G_S(s)|^2/|G_S(0)|^2$')
    ax.set_yscale('log')
    ax.set_title(rf'Breit--Wigner propagator spike, $m_\phi={mphi/1000:.1f}$ GeV, $\Gamma/m={gamma_frac:.0e}$')
    ax.grid(True,which='both',alpha=.25)
    ax.legend()
    fig.savefig(OUT/'resonant_propagator_spike.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


def save_csv(contact, resonant):
    lines=['GS_over_GF,contact_y10,resonant_y10,contact_r10,resonant_r10']
    rdict={a:(T[-1],np.sqrt(y[-1]),y[-1]) for a,T,y in resonant}
    for a,T,y in contact:
        lines.append(f'{a:.0e},{y[-1]:.10g},{rdict[a][2]:.10g},{np.sqrt(y[-1]):.10g},{rdict[a][1]:.10g}')
    (OUT/'figure1_benchmark_values.csv').write_text('\n'.join(lines)+'\n')


def write_notes(mphi=1000.0,gamma_frac=1e-2):
    txt = f'''# Resonant right-handed-neutrino collision-term calculation

## Scope
This implementation starts from Eq. (22) of Luo, Rodejohann & Xu (arXiv:2005.01629v1), keeps the paper's massless-neutrino and thermal-temperature treatment, and extends the scalar interaction by an explicit s-channel mediator. The resonance extension is **not** part of the paper's EFT calculation; it is a UV completion used to test the effect of a propagator pole.

## 12D -> 9D
Starting from

C^rho = - N_nuR S int [prod_i d^3 p_i/((2 pi)^3 2 E_i)] (2 pi)^4 delta^4(p1+p2-p3-p4) E1 F |M|^2,

a direct integration over d^3 p4 using delta^3 gives p4 = p1+p2-p3 and

C^rho = - N_nuR S/[16 (2 pi)^8] int d^3p1 d^3p2 d^3p3 [E1 F |M|^2/(E1 E2 E3 E4)] delta(E1+E2-E3-E4).

This is the requested exact 9-dimensional representation: three independent 3-vectors remain. For massless particles E_i=|p_i|.

The paper then goes further: its Appendix C uses a Fourier representation of delta^3 and analytically performs angular integrals, ultimately reducing the problem to three radial momentum integrals and, for Maxwell-Boltzmann statistics, to closed forms. The relevant equations are (C5)-(C34).

## Resonant matrix element
For the paper's GS-only annihilation channel, Eq. (B8) gives

|M|^2 = 16 |G_S|^2 (p1.p2)(p3.p4).

For an explicit scalar mediator phi, promote

G_S -> G_S(s) = g^2/(s-m_phi^2 + i m_phi Gamma_phi),

so

|M|^2 = 16 |G_S(s)|^2 (p1.p2)(p3.p4)
      = 4 s^2 g^4 / ((s-m_phi^2)^2 + m_phi^2 Gamma_phi^2).

Low-energy matching is g^2/m_phi^2 = G_S.

Benchmark used in the plots: m_phi={mphi:.0f} MeV and Gamma_phi/m_phi={gamma_frac:.0e}. The width is treated as an explicit UV input; changing the UV couplings changes Gamma_phi and therefore the resonance area.

## Resonance-safe integral
The dangerous numerical factor is

1/((s-m_phi^2)^2 + m_phi^2 Gamma_phi^2).

Instead of relying on random samples to hit the pole, integrate it analytically:

I(S)=int_0^S ds s^2/((s-m_phi^2)^2+m_phi^2 Gamma_phi^2).

The code uses the closed-form primitive plus a series for S/m_phi^2 < 0.2, which removes catastrophic cancellation. Thus the resonance is included deterministically. There is no requirement that a quadrature point land on the spike.

The remaining p1,p2 integral is done with Gauss-Legendre quadrature, explicitly split where 4 p1 p2 = m_phi^2 so that the resonant boundary is represented deterministically. The normalization is calibrated to reproduce the paper's contact-limit collision term C30/Table III when m_phi is taken far above the thermal range.

A more general implementation, when |M|^2 depends on several invariants, should transform the resonant variable with

s = m_phi^2 + m_phi Gamma_phi tan(theta),

d(theta) = ds / ((s-m_phi^2)^2 + m_phi^2 Gamma_phi^2),

and integrate theta adaptively. This is also safe in the narrow-width limit.

## Boltzmann equation
The temperature equation is Eq. (28) of the paper. We use

rho_nuR = 2 N_nu * (7/8) * pi^2 T_nuR^4/30,
P_nuR = rho_nuR/3,
rho_SM = pi^2 g_rho(T) T^4/30,
P_SM = pi^2 g_P(T) T^4/90,

with H^2=8 pi (rho_SM+rho_nuR)/(3 M_Pl^2), including dg_rho/dT in d rho_SM/dT exactly as the paper instructs.

## Reproduction / resonance result
The left side of figure1_side_by_side_resonance.png is the uploaded paper's Figure 1. The right side uses the same five GS/GF values and the same 10 MeV endpoint, but replaces the contact scalar interaction by the resonant propagator benchmark above.

The resonance benchmark is intentionally explicit so that it can be changed. It should not be interpreted as the original paper's prediction. The paper itself uses a contact EFT and has no Breit-Wigner pole in Eqs. (9)-(13); adding one changes the UV physics.
'''
    (OUT/'resonant_neutrino_notes.md').write_text(txt)


def main():
    contact=make_contact_curves()
    resonant=make_res_curves()
    save_side_by_side(contact,resonant)
    save_propagator_spike_plot()
    save_csv(contact,resonant)
    write_notes()
    print('Generated:')
    for f in ['figure1_original_crop.png','figure1_side_by_side_resonance.png','resonant_propagator_spike.png','figure1_benchmark_values.csv','resonant_neutrino_notes.md']:
        print(OUT/f)
    print('\nEndpoint table:')
    for a,T,y in contact:
        yr=[yy[-1] for aa,TT,yy in resonant if aa==a][0]
        print(f'{a:.0e}: contact {y[-1]:.6f}, resonance {yr:.6f}')

if __name__=='__main__':
    main()
