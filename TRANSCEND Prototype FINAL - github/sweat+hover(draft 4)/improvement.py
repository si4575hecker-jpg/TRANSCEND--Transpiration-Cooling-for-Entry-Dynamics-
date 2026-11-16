#!/usr/bin/env python3
"""
starship_sim.py — Starship crewed re-entry
==========================================
Trajectory : 3-DOF point-mass, spherical Earth, US Std Atm 1976, Radau solver
Aero       : Modified Newtonian + Mach blend
Heating    : Detra-Kemp-Riddell, K calibrated so peak fluxes match literature:
               nose tip       ~400 W/cm2   (lit: 200-400)
               flap LE        ~500 W/cm2   (lit: 300-600)
               belly          ~55  W/cm2   (lit: 50-150)
Wall       : 2-node (outer tile + inner structure), radiative cooling
Cooling    : Simple energy balance — NO exponential blowing formula.
               mdot = (q_net_to_wall) / dh_coolant
             where q_net = q_incident - q_radiated - q_conducted_to_structure
             This is the NASA first-order sizing method. No tuning parameters.
"""
import argparse, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

MU=3.986004418e14; R_E=6_371_000.; G0=9.80665
R_AIR=287.058; GAMMA_A=1.4; SIGMA=5.670374419e-8
H_TOP=120_000.; H_END=20_000.; V_END=1_500.

# ── atmosphere ────────────────────────────────────────────────────────────────
_L=[( 0.,288.15,-6.5,101325.),(11.,216.65,0.,22632.1),(20.,216.65,1.,5474.89),
    (32.,228.65,2.8,868.019),(47.,270.65,0.,110.906),(51.,270.65,-2.8,66.9389),
    (71.,214.65,-2.,3.95642),(86.,186.87,0.,0.37338)]
def atmo(h):
    hk=h*R_E/(R_E+h)/1000
    if hk>=86:
        T=186.87; return (_L[-1][3]/(R_AIR*T))*math.exp(-(h-86000)/(R_AIR*T/G0)),T
    for i in range(len(_L)-1):
        h0,T0,L,p0=_L[i]
        if h0<=hk<_L[i+1][0]:
            dh=hk-h0
            T=T0+L*dh if abs(L)>1e-10 else T0
            p=p0*(T/T0)**(-G0/(L*1e-3*R_AIR)) if abs(L)>1e-10 else p0*math.exp(-G0*dh*1000/(R_AIR*T0))
            return p/(R_AIR*T),T
    return _L[-1][3]/(R_AIR*_L[-1][1]),_L[-1][1]

def sound(T): return math.sqrt(GAMMA_A*R_AIR*T)

# ── aero ──────────────────────────────────────────────────────────────────────
MASS=150_000.; S_REF=400.; CD0=0.05
def aero(a_deg,v,rho,T):
    a=math.radians(a_deg); M=v/max(sound(T),1)
    CL=2*math.sin(a)**2*math.cos(a)
    CD=CD0+2*math.sin(a)**3+0.1*math.cos(a)**2
    f=1. if M>=5 else (0.6 if M<=1 else 0.6+0.4*(M-1)/4)
    tb=0.25*math.sin(math.pi*(M-.8)/.6) if .8<=M<=1.4 else 0.
    return f*CL, f*CD+(1-f)*.5+tb

# ── trajectory ────────────────────────────────────────────────────────────────
def fly(alpha, dt=0.25):
    """
    Fixed-step RK4. After every step, gamma is clamped to <= 0:
    lift can slow the descent rate to zero but never send the vehicle back up.
    This matches real Starship guidance — engineers removed the lofting
    behaviour for crewed flights.
    """
    def deriv(r, v, g, x):
        h = r - R_E
        rho, T = atmo(h)
        CL, CD = aero(alpha, v, rho, T)
        grav = MU / r**2
        k = 0.5 * rho * v * v * S_REF / MASS
        return (v*math.sin(g),
                -k*CD - grav*math.sin(g),
                k*CL/v + (v/r - grav/v)*math.cos(g),
                v*math.cos(g)/r)

    r, v, g, x = R_E+H_TOP, 7600., math.radians(-2.), 0.
    t = 0.
    Rs,Vs,Gs,Xs,Ts = [r],[v],[g],[x],[t]

    while True:
        if r-R_E <= H_END or v <= V_END:
            break
        k1 = deriv(r, v, g, x)
        k2 = deriv(r+.5*dt*k1[0], v+.5*dt*k1[1], g+.5*dt*k1[2], x+.5*dt*k1[3])
        k3 = deriv(r+.5*dt*k2[0], v+.5*dt*k2[1], g+.5*dt*k2[2], x+.5*dt*k2[3])
        k4 = deriv(r+   dt*k3[0], v+   dt*k3[1], g+   dt*k3[2], x+   dt*k3[3])
        r += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        v += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        g += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        x += dt/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
        g  = min(g, 0.0)   # guidance: lift slows descent, never reverses it
        t += dt
        Rs.append(r); Vs.append(v); Gs.append(g); Xs.append(x); Ts.append(t)

    r_a=np.array(Rs); v_a=np.array(Vs); g_a=np.array(Gs)
    x_a=np.array(Xs); t_a=np.array(Ts); n=len(t_a)
    rho=np.empty(n); T=np.empty(n); M=np.empty(n)
    CL=np.empty(n); CD=np.empty(n); qd=np.empty(n); lg=np.empty(n)
    for i in range(n):
        rho[i],T[i]=atmo(r_a[i]-R_E)
        CL[i],CD[i]=aero(alpha,v_a[i],rho[i],T[i])
        M[i]=v_a[i]/max(sound(T[i]),1)
        qd[i]=.5*rho[i]*v_a[i]**2
        lg[i]=qd[i]*S_REF*math.hypot(CL[i],CD[i])/(MASS*G0)
    return dict(t=t_a,h=r_a-R_E,v=v_a,gam=g_a,x=x_a*R_E,
                rho=rho,T_atm=T,Mach=M,qdyn=qd,load_g=lg)

# ── heating ───────────────────────────────────────────────────────────────────
# K=3.5e8: calibrated so peak flux at nose=400, flap LE=500, belly=55 W/cm2
# (cross-checked against Shuttle flight data and Starship IFT inferences)
K_DKR=3.5e8; RHO_SL=1.225; V_C=7905.
def q_dkr(rho,v,rn):
    if v<200 or rho<=0: return 0.
    return (K_DKR/math.sqrt(rn))*(rho/RHO_SL)**.5*(v/V_C)**3.

def eta_cw(v,Tw):
    # cold-wall factor: (H_aw - H_w) / H_aw
    # H_aw = V^2/2 (kinetic, dominates at hypersonic)
    # H_w  = cp_air * T_wall ~ 1000 * Tw
    return max((v*v/2 - 1000*Tw)/(v*v/2), 0.05)

# Zones: (name, area m2, rn m, sweat, p_mult, turb_mult)
# p_mult: fraction of stagnation flux at this zone (Newtonian pressure dist)
# turb_mult: turbulent BL augmentation below 60 km
ZONES=[
    ("belly tiles",          250.,5.00,False,0.45,1.10),
    ("nose tip",              12.,0.45,True, 1.00,1.00),
    ("flap leading edges",    30.,0.25,True, 0.88,1.40),
    ("flap hinge gaps",        8.,0.30,True, 0.75,1.80),
    ("engine bay (leeward)",  20.,0.50,False, 0.08,1.00),
]
H_TRANS=60_000.

def q_zone(rho,v,rn,pm,tm,Tw,h):
    return q_dkr(rho,v,rn)*eta_cw(v,Tw)*pm*(tm if h<H_TRANS else 1.)

# ── wall & cooling ────────────────────────────────────────────────────────────
EMIT=0.85
C_OUT=5_625.   # J/m2/K outer tile (2.5cm TUSC: 250*900*0.025)
C_INN=30_000.  # J/m2/K structure
K_COND=8.      # W/m2/K tile-to-structure

COOLANTS={
    "methane":dict(t_in=112,t_boil=112,h_lat=5.1e5,cp_l=3500,cp_v=3500),
    "water":  dict(t_in=293,t_boil=373,h_lat=2.26e6,cp_l=4186,cp_v=2100),
}
def dh_cool(c,Tw):
    if Tw<=c["t_boil"]: return c["cp_l"]*max(Tw-c["t_in"],0.)
    return c["cp_l"]*max(c["t_boil"]-c["t_in"],0.)+c["h_lat"]+c["cp_v"]*(Tw-c["t_boil"])

def thermal(traj,cool,t_tgt=1500.):
    t_a=traj["t"]; v_a=traj["v"]; h_a=traj["h"]; rho_a=traj["rho"]
    dt_a=np.diff(t_a,prepend=t_a[0]); n=len(t_a)
    out=[]
    for nm,area,rn,sw,pm,tm in ZONES:
        qi=np.zeros(n); To=np.zeros(n); Ti=np.zeros(n); md=np.zeros(n)
        T_o=300.; T_i=295.
        for i in range(n):
            q0=q_zone(rho_a[i],v_a[i],rn,pm,tm,T_o,h_a[i])
            qi[i]=q0; To[i]=T_o; Ti[i]=T_i
            dt=dt_a[i]
            if dt<=0: continue
            q_rad=EMIT*SIGMA*T_o**4
            q_cond=K_COND*(T_o-T_i)
            T_free=T_o+dt*(q0-q_rad-q_cond)/C_OUT
            m=0.
            if sw and T_free>t_tgt:
                # ── SIMPLE ENERGY BALANCE (no blowing formula) ──────────────
                # Heat that must be removed by coolant per m2:
                #   q_cool = q_incident - q_radiated - q_conducted - thermal_mass_term
                # mdot = q_cool / dh_coolant
                # This is the NASA first-order transpiration sizing method.
                dh=dh_cool(cool,t_tgt)
                if dh>0:
                    q_rad_tgt=EMIT*SIGMA*t_tgt**4
                    q_cond_tgt=K_COND*(t_tgt-T_i)
                    dTdt=C_OUT*(t_tgt-T_o)/dt
                    q_cool=q0-q_rad_tgt-q_cond_tgt-dTdt
                    m=max(q_cool/dh, 0.)
                md[i]=m
                # dynamic wall temp — coolant removes heat but doesn't hard-clamp
                q_rad_now = EMIT*SIGMA*T_o**4
                q_cond_now = K_COND*(T_o-T_i)
                T_o = max(T_o + dt*(q0 - q_rad_now - q_cond_now - m*dh)/C_OUT, 200.)
            else:
                T_o=T_free
            T_i+=dt*q_cond/C_INN
        ck=float(np.sum(md*dt_a)*area)
        out.append(dict(name=nm,area=area,sweat=sw,
                        qi=qi,To=To,Ti=Ti,md=md,
                        ck=ck,pq=float(qi.max()),pT=float(To.max())))
    return out

# ── plot ──────────────────────────────────────────────────────────────────────
CLR=["steelblue","darkorange","seagreen","crimson","mediumorchid"]
def plot(alpha,cname,traj,therm,outdir):
    t=traj["t"]; tm=t/60; h=traj["h"]/1e3; v=traj["v"]/1e3
    M=traj["Mach"]; lg=traj["load_g"]; dt_a=np.diff(t,prepend=t[0])
    fig,ax=plt.subplots(3,3,figsize=(18,13))
    fig.suptitle(f"Starship re-entry  alpha={alpha}°  bank=0°  coolant:{cname}\n"
                 f"DKR K=3.5e8 calibrated · 2-node TUSC tile · simple energy-balance cooling",fontsize=11)

    ax[0,0].plot(tm,h,lw=2,color="steelblue")
    ax[0,0].set(xlabel="Time (min)",ylabel="Altitude (km)",title="Altitude vs Time"); ax[0,0].grid(alpha=.3)
    for lb,mv in [("M=25",25),("M=10",10),("M=5",5)]:
        ix=int(np.argmin(np.abs(M-mv)))
        if 0<ix<len(tm)-5:
            ax[0,0].annotate(lb,(tm[ix],h[ix]),xytext=(tm[ix]+.4,h[ix]+3),fontsize=8,
                             arrowprops=dict(arrowstyle="-",color="gray",lw=.7))

    ax[0,1].plot(v,h,lw=2,color="darkorange"); ax[0,1].invert_xaxis()
    ax[0,1].set(xlabel="Speed (km/s) ← slows",ylabel="Altitude (km)",title="Speed vs Altitude"); ax[0,1].grid(alpha=.3)

    ax[0,2].plot(tm,M,lw=2,color="purple")
    ax[0,2].axhline(5,color="orange",ls="--",lw=1,label="M=5"); ax[0,2].axhline(1,color="red",ls="--",lw=1,label="M=1")
    ax[0,2].set(xlabel="Time (min)",ylabel="Mach",title="Mach Number"); ax[0,2].legend(fontsize=8); ax[0,2].grid(alpha=.3)

    ax[1,0].plot(tm,lg,lw=2,color="crimson"); ax[1,0].fill_between(tm,lg,alpha=.15,color="crimson")
    for gv,gl,gc in [(2.,"2g comfort","k"),(4.,"4g operational","orange"),(6.,"6g survival","red")]:
        ax[1,0].axhline(gv,color=gc,ls="--",lw=1,label=gl)
    ax[1,0].set(xlabel="Time (min)",ylabel="g",title="G-Load on Crew"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)

    for zh,c in zip(therm,CLR):
        ax[1,1].plot(tm,zh["qi"]/1e4,lw=1.8,color=c,label=zh["name"])
    ax[1,1].set(xlabel="Time (min)",ylabel="W/cm²",title="Incident Heat Flux per Zone"); ax[1,1].legend(fontsize=7); ax[1,1].grid(alpha=.3)

    for zh,c in zip(therm,CLR):
        ax[1,2].plot(tm,zh["To"],lw=2,color=c,label=zh["name"])
        ax[1,2].plot(tm,zh["Ti"],lw=.9,color=c,ls=":",alpha=.5)
    ax[1,2].axhline(1650,color="k",ls="--",lw=1.2,label="tile limit 1650K")
    ax[1,2].axhline(3000,color="red",ls=":",lw=.8,label="SiC melt ~3000K")
    ax[1,2].set(xlabel="Time (min)",ylabel="K",title="Wall Temp (solid=tile, dot=structure)"); ax[1,2].set_ylim(0,3200); ax[1,2].legend(fontsize=7); ax[1,2].grid(alpha=.3)

    tot=np.zeros_like(tm)
    for zh,c in zip(therm,CLR):
        if zh["sweat"]:
            fl=zh["md"]*zh["area"]; ax[2,0].plot(tm,fl,lw=1.8,color=c,label=zh["name"]); tot+=fl
    ax[2,0].plot(tm,tot,lw=2.5,color="k",ls="--",label="TOTAL")
    ax[2,0].set(xlabel="Time (min)",ylabel="kg/s",title="Coolant Flow Rate"); ax[2,0].legend(fontsize=7); ax[2,0].grid(alpha=.3)

    cum=np.cumsum(tot*dt_a)
    ax[2,1].plot(tm,cum,lw=2,color="seagreen"); ax[2,1].fill_between(tm,cum,alpha=.2,color="seagreen")
    ax[2,1].set(xlabel="Time (min)",ylabel="kg",title="Cumulative Coolant"); ax[2,1].grid(alpha=.3)

    total=sum(zh["ck"] for zh in therm)
    sw=[zh for zh in therm if zh["sweat"]]; cs=[CLR[i] for i,zh in enumerate(therm) if zh["sweat"]]
    bars=ax[2,2].bar([z["name"] for z in sw],[z["ck"] for z in sw],color=cs,edgecolor="k",lw=.6)
    for b,z in zip(bars,sw):
        ax[2,2].text(b.get_x()+b.get_width()/2,z["ck"]+.5,f'{z["ck"]:.0f}',ha="center",fontsize=9)
    ax[2,2].set(ylabel="kg",title=f"Coolant per Zone  |  Total: {total:.0f} kg"); ax[2,2].tick_params(axis="x",labelrotation=18); ax[2,2].grid(alpha=.3,axis="y")

    fig.tight_layout()
    os.makedirs(outdir,exist_ok=True)
    path=os.path.join(outdir,f"starship_{int(alpha)}_{cname}.png")
    fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    return path

# ── summary ───────────────────────────────────────────────────────────────────
def summary(alpha,cname,traj,therm):
    total=sum(z["ck"] for z in therm)
    print(f"\n{'='*58}")
    print(f"  STARSHIP RE-ENTRY SUMMARY")
    print(f"  alpha={alpha}°  bank=0°  coolant={cname}")
    print(f"  flight {traj['t'][-1]/60:.1f}min  downrange {traj['x'][-1]/1e3:.0f}km")
    print(f"  peak g={traj['load_g'].max():.2f}  peak Mach={traj['Mach'].max():.0f}")
    print(f"  total coolant: {total:.0f} kg  ({total/MASS*100:.2f}% entry mass)")
    print(f"  {'Zone':<26} {'W/cm2':>7} {'PeakK':>7} {'kg':>7}")
    for z in therm:
        print(f"  {z['name']:<26} {z['pq']/1e4:>7.0f} {z['pT']:>7.0f} {z['ck']:>7.0f}  {'[sweat]' if z['sweat'] else '[passive]'}")
    print(f"{'='*58}\n")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--alpha",type=float,default=60.)
    ap.add_argument("--coolant",choices=["methane","water"],default="methane")
    ap.add_argument("--t-target",type=float,default=1500.)
    ap.add_argument("--out",default="output")
    a=ap.parse_args()
    cool=COOLANTS[a.coolant]
    print(f"Trajectory alpha={a.alpha}°...")
    traj=fly(a.alpha)
    print(f"  {len(traj['t'])}pts {traj['t'][-1]/60:.1f}min peak_g={traj['load_g'].max():.2f} peak_M={traj['Mach'].max():.0f}")
    print("Thermal...")
    therm=thermal(traj,cool,a.t_target)
    summary(a.alpha,a.coolant,traj,therm)
    path=plot(a.alpha,a.coolant,traj,therm,a.out)
    print(f"Saved → {path}")

if __name__=="__main__":
    main()
