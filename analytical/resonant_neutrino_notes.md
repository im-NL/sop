# Resonant right-handed-neutrino collision-term calculation

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

Benchmark used in the plots: m_phi=1000 MeV and Gamma_phi/m_phi=1e-02. The width is treated as an explicit UV input; changing the UV couplings changes Gamma_phi and therefore the resonance area.

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
