# Resonance-resolved phase-space QMC

- $m_phi$ = 1000.0 MeV
- Gamma/m = 0.01
- samples per collision evaluation = 2048
- resonance stratum = |s-m_phi^2| < 10 m_phi Gamma
- physical 2->2 final state generated exactly in CM and boosted to plasma frame
- full Fermi-Dirac factors evaluated at each sample
- incoming radial proposal: q(p)=p exp(-p/T)/T^2 (Gamma shape k=2)
- s uses an equal-probability mixture of kinematic strata, so the resonance window is sampled even when narrow
- contact scattering term added from the paper Table II; the resonant modification is applied to the annihilation channel
