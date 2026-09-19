import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.special import gammaincinv
from scipy.stats import qmc
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from PIL import Image

OUT=Path(r'C:\Users\aryan\code\sop\latest\5d\10mev')
GF=1.1663787e-11; MPL=1.2209e22; NNU=3; NNU_R=6; S_ANN=0.5
T_GEV=np.array([.010,.020,.050,.100,.130,.140,.150,.160,.170,.180,.190,.200,.214,.500,1,2,5,10,20,50,100,200,500,1000])
G_RHO=np.array([10.76,11.33,14.63,18.00,21.76,23.77,26.31,29.51,33.47,38.27,44.01,50.75,62.49,69.26,76.34,82.50,85.60,86.22,88.45,97.40,103.53,105.90,106.61,106.72])
G_P=np.array([10.75,10.99,13.40,16.21,18.16,19.05,20.13,21.42,22.98,24.84,27.04,29.62,61.52,66.43,72.97,79.69,84.68,85.85,87.22,93.94,100.80,104.75,106.38,106.65])
LR=np.log(T_GEV); GR=PchipInterpolator(LR,G_RHO); GP=PchipInterpolator(LR,G_P); DGR=GR.derivative()

def gstar(T):
    x=np.log(np.clip(T/1000,T_GEV[0],T_GEV[-1])); gr=float(GR(x)); gp=float(GP(x)); dg=float(DGR(x))/T; return gr,gp,dg

def rho_components(T,TR):
    gr,gp,dg=gstar(T); rho_sm=np.pi**2/30*gr*T**4; p_sm=np.pi**2/90*gp*T**4; rho_r=NNU_R*7/8*np.pi**2/30*TR**4; return rho_sm,p_sm,rho_r,rho_r/3,dg

def C_ann_contact_paper(T,TR,GS): return 12/np.pi**5*NNU_R*GS**2*(T**9-TR**9)*0.8840

def C_scatt_contact_paper(T,TR,GS): return 6/np.pi**5*NNU_R*GS**2*T**4*TR**4*(T-TR)*0.8118

def decoupling_temperature(GS): return 2*(np.sqrt(.28)*GS/GF)**(-2/3)

class QMC:
    def __init__(self,n_pow=9,seed=1,K=10):
        self.n=2**n_pow; self.K=K; self.seed=seed; self.U=np.clip(qmc.Sobol(5,scramble=True,seed=seed).random_base2(n_pow),1e-12,1-1e-12)
    def collision(self,T,TR,GS,mphi=1000.,gamma_frac=.01,res=True):
        U=self.U; n=self.n
        # radial invariant phase-space proposal q(p)=p exp(-p/T)/T^2
        p1=T*0 + TR*gammaincinv(2,U[:,0]); p2=TR*gammaincinv(2,U[:,1])
        Smax=4*p1*p2; G=gamma_frac*mphi; a=mphi*G; d=self.K*a; lo=np.maximum(0,mphi*mphi-d); hi=np.minimum(Smax,mphi*mphi+d)
        hasR=Smax>lo; hasAbove=Smax>hi
        nint=np.where(hasAbove,3,np.where(hasR,2,1))
        w=U[:,2]*nint; k=np.minimum(w.astype(int),nint-1); frac=w-k
        # interval endpoints for k=0,1,2
        A=np.choose(k,[np.zeros(n),np.full(n,lo),hi]); B=np.choose(k,[np.minimum(Smax,lo),hi,Smax])
        width=B-A; s=A+frac*width
        q_mu=(2*p1*p2)/(nint*width)
        mu=1-s/(2*p1*p2)
        # incoming geometry
        sinmu=np.sqrt(np.maximum(0,1-mu*mu)); p1v_z=p1; p2x=p2*sinmu; p2z=p2*mu
        P0=p1+p2; Px=p2x; Pz=p1v_z+p2z; Pmag=np.sqrt(Px*Px+Pz*Pz)
        bx=Px/P0; bz=Pz/P0; b2=np.minimum((Pmag/P0)**2,1-1e-15); gam=1/np.sqrt(1-b2)
        # final CM direction relative to boost axis ez=(bx,0,bz), with ex chosen y-axis and ey=e_z x e_x
        z=2*U[:,3]-1; ph=2*np.pi*U[:,4]; sth=np.sqrt(np.maximum(0,1-z*z))
        # ex=(0,1,0), ey=(-bz,0,bx)
        nx=-bz*sth*np.sin(ph)+bx*z; ny=sth*np.cos(ph); nz=bx*sth*np.sin(ph)+bz*z
        est=.5*np.sqrt(s); pstar=.5*np.sqrt(s)
        bp=bx*(pstar*nx)+bz*(pstar*nz)
        E3=gam*(est+bp)
        # E4=P0-E3
        E4=P0-E3; E1=p1; E2=p2
        f1=1/(np.exp(np.clip(E1/TR,0,700))+1); f2=1/(np.exp(np.clip(E2/TR,0,700))+1)
        f3=1/(np.exp(np.clip(E3/T,0,700))+1); f4=1/(np.exp(np.clip(E4/T,0,700))+1)
        F=f1*f2*(1-f3)*(1-f4)-f3*f4*(1-f1)*(1-f2)
        if res:
            M2=4*(GS*mphi*mphi)**2*s*s/((s-mphi*mphi)**2+a*a)
        else:
            M2=4*GS*GS*s*s
        q1=p1*np.exp(-p1/TR)/(TR*TR); q2=p2*np.exp(-p2/TR)/(TR*TR); q=q1*q2*q_mu*.5/(2*np.pi)
        coeff=1/(1024*np.pi**6)
        val=-NNU_R*S_ANN*coeff*p1*p2*E1*F*M2/q
        val=np.where(np.isfinite(val),val,0)
        return float(np.mean(val)),float(np.std(val,ddof=1)/np.sqrt(n)),{'s':s,'M2':M2,'val':val,'qmu':q_mu,'nint':nint,'E3':E3,'E4':E4}

def collision_avg(mcs,T,TR,GS,mphi,gamma,res):
    vals=[]; ses=[]
    for mc in mcs:
        v,se,_=mc.collision(T,TR,GS,mphi,gamma,res)
        vals.append(v); ses.append(se)
    return float(np.mean(vals)), float(np.sqrt(np.sum(np.array(ses)**2))/len(mcs))

def rhs(x,y,mc,GS,mphi,gamma,res):
    T=float(np.exp(x)); r=float(np.clip(y[0],1e-7,1)); TR=r*T; rho_sm,p_sm,rho_r,p_r,dg=rho_components(T,TR); H=np.sqrt(8*np.pi*(rho_sm+rho_r)/(3*MPL**2)); C=collision_avg(mc,T,TR,GS,mphi,gamma,res)[0]+C_scatt_contact_paper(T,TR,GS)
    gr,_,_=gstar(T); drsm=np.pi**2/30*(4*gr*T**3+T**4*dg); drr=4*rho_r/TR; dTRdT=((3*H*(rho_r+p_r)-C)/(3*H*(rho_sm+p_sm)+C))*drsm/drr
    return [dTRdT-r]

def solve(GS,mcs,res,mphi,gamma,n_eval=100):
    Ts=min(1e6,max(10,20*decoupling_temperature(GS))); xs=np.linspace(np.log(Ts),np.log(10),n_eval)
    fun=lambda x,y:rhs(x,y,mcs,GS,mphi,gamma,res)
    sol=solve_ivp(fun,(xs[0],xs[-1]),[1.0],t_eval=xs,rtol=5e-3,atol=3e-5,max_step=.18,method='RK45')
    return np.exp(xs),np.clip(sol.y[0],0,1)**4,sol

if __name__=='__main__':
    mphi=10; gamma=.01; mcs=[QMC(n_pow=11,seed=s,K=10) for s in [20260909,20260910,20260911,20260912]]; mc=mcs[0]
    # Contact MC validation at one representative state, multiple scrambles.
    rows=[]
    for seed in [1,2,3,4]:
        q=QMC(n_pow=12,seed=seed,K=10); mcval,se,_=q.collision(300,240,1e-4*GF,mphi,gamma,False); pp=C_ann_contact_paper(300,240,1e-4*GF); rows.append((seed,mcval,se,pp,(mcval-pp)/pp))
        print('validation',rows[-1])
    print('doing convergence')
    conv=[]
    for pw in [8,9,10,11,12,13]:
        vals=[]
        for seed in [100+pw,200+pw]:
            q=QMC(n_pow=pw,seed=seed,K=10); est,se,_=q.collision(300,240,1e-4*GF,mphi,gamma,True); vals.append(est); conv.append((2**pw,seed,est,se))
        print(pw,vals)
    # Curves
    couplings=[1e-3,1e-4,1e-5,1e-6,1e-7]
    rescur=[]
    for a in couplings:
        print('curve',a)
        T,y,sol=solve(a*GF,mcs,True,mphi,gamma,100); print('success',sol.success,'end',y[-1]); rescur.append((a,T,y))
    # Original fig crop
    crop=Image.open(OUT/'page8.png').crop((400,250,1510,900)); crop.save(OUT/'figure1_original_crop_qmc.png')
    fig,ax=plt.subplots(figsize=(7,5.2))
    for a,T,y in rescur: ax.semilogx(T,y,label=rf'$G_S={a:.0e}G_F$')
    ax.axvline(mphi/2,ls='--',lw=1.0,alpha=.7,label=r'$m_\phi/2$')
    ax.set_xlim(10,1e6); ax.set_ylim(0,1.03); ax.set_xlabel(r'$T_{SM}$ [MeV]'); ax.set_ylabel(r'$(T_{\nu_R}/T_{SM})^4$'); ax.set_title(rf'Resonance-resolved QMC: $m_\phi={mphi:g}$ MeV, $\Gamma/m={gamma:g}$'); ax.legend(fontsize=8,frameon=False); ax.grid(alpha=.2,which='both'); fig.tight_layout(); fig.savefig(OUT/'figure1_qmc_resonant_result.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    # Sample distribution demonstration at T=100 MeV, TR=80 MeV
    q=QMC(n_pow=15,seed=20260910,K=10); _,_,dat=q.collision(100,80,1e-4*GF,mphi,gamma,True,)
    ax.hist(dat['s']/mphi**2,bins=180,density=True,log=True); ax.axvline(1,ls='--',lw=1.2,label=r'$s=m_\phi^2$'); ax.set_xlim(0,3); ax.set_xlabel(r'$s/m_\phi^2$'); ax.set_ylabel('QMC sample density'); ax.set_title('Dedicated resonance stratum'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'qmc_resonance_sample_histogram.png',dpi=220); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2),constrained_layout=True); axes[0].imshow(crop); axes[0].axis('off'); axes[0].set_title('Paper Figure 1 (source)');
    for a,T,y in rescur: axes[1].semilogx(T,y,label=rf'$G_S={a:.0e}G_F$')
    axes[1].axvline(mphi/2,ls='--',lw=1,alpha=.7); axes[1].set_xlim(10,1e6); axes[1].set_ylim(0,1.03); axes[1].set_xlabel(r'$T_{SM}$ [MeV]'); axes[1].set_ylabel(r'$(T_{\nu_R}/T_{SM})^4$'); axes[1].set_title(rf'Our QMC + resonant propagator ($m_\phi={mphi:g}$ MeV, $\Gamma/m={gamma:g}$)'); axes[1].legend(fontsize=8,frameon=False); axes[1].grid(alpha=.2,which='both'); fig.savefig(OUT/'figure1_qmc_side_by_side.png',dpi=220); plt.close(fig)
    with open(OUT/'qmc_validation.csv','w') as f:
        f.write('seed,MC_ann,se,paper_ann,relative_error\n'); [f.write(','.join(map(str,r))+'\n') for r in rows]
    with open(OUT/'qmc_convergence.csv','w') as f:
        f.write('N,seed,estimate,se\n'); [f.write(','.join(map(str,r))+'\n') for r in conv]
    with open(OUT/'qmc_run_summary.md','w') as f:
        f.write(f'# Resonance-resolved phase-space QMC\n\n- $m_phi$ = {mphi} MeV\n- Gamma/m = {gamma}\n- samples per collision evaluation = {mc.n}\n- resonance stratum = |s-m_phi^2| < 10 m_phi Gamma\n- physical 2->2 final state generated exactly in CM and boosted to plasma frame\n- full Fermi-Dirac factors evaluated at each sample\n- incoming radial proposal: q(p)=p exp(-p/T)/T^2 (Gamma shape k=2)\n- s uses an equal-probability mixture of kinematic strata, so the resonance window is sampled even when narrow\n- contact scattering term added from the paper Table II; the resonant modification is applied to the annihilation channel\n')
    print('ENDPOINTS')
    for a,T,y in rescur: print(a,y[-1])
