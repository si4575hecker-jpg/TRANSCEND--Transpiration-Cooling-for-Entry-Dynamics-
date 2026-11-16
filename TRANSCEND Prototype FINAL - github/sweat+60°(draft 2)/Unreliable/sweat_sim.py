#!/usr/bin/env python3
"""
sweat_sim.py — Starship crewed re-entry: single smooth dive at alpha=60 deg,
bank=0 (wings level), transpiration cooling on hot zones.
No skipping, no bank maneuvers.

Run:
    pip install numpy scipy matplotlib
    python sweat_sim.py
    python sweat_sim.py --coolant water
    python sweat_sim.py --alpha 55      # try a different pitch angle
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ────────────────────────────────────────────────────────────────
MU, R_E, G0 = 3.986004418e14, 6_371_000.0, 9.80665
SB = 5.670374419e-8
K_SG = 1.7415e-4        # Sutton-Graves
CP_GAS = 1300.0
H_TOP, H_END, V_END = 120_000.0, 20_000.0, 1_500.0

# ── atmosphere ───────────────────────────────────────────────────────────────
_AK = np.arange(0, 151, 10.0)
_R  = np.array([1.225,0.4135,0.08891,0.01841,0.003996,0.001027,3.097e-4,
                8.283e-5,1.846e-5,3.416e-6,5.604e-7,9.708e-8,2.222e-8,
                8.152e-9,3.831e-9,2.076e-9])
_LR = np.log(_R)
def rho(h):
    if h<=0: return float(_R[0])
    hk=h/1000
    if hk>=150: return float(_R[-1]*math.exp(-(h-150000)/25000))
    return float(math.exp(np.interp(hk,_AK,_LR)))

# ── vehicle ───────────────────────────────────────────────────────────────────
MASS   = 150_000   # kg
S_REF  = 400.0     # m^2
CD0    = 0.05
EMIT   = 0.85
C_SKIN = 7200.0    # J/m^2/K thermal mass of tile skin

def aero(alpha_deg):
    a = math.radians(alpha_deg)
    cl = 2*math.sin(a)**2*math.cos(a)
    cd = CD0 + 2*math.sin(a)**3
    return cl, cd

# ── integrate trajectory (RK4, bank=0) ──────────────────────────────────────
def fly(alpha_deg, dt=0.1):
    cl, cd = aero(alpha_deg)
    k = 0.5*S_REF/MASS
    r, v, gam, x = R_E+H_TOP, 7600.0, math.radians(-3.0), 0.0
    T,H,V,GAM,X = [],[],[],[],[]
    while True:
        h = r-R_E
        rr = rho(h)
        g  = MU/r**2
        ld = k*rr*v*v
        dv = -ld*cd - g*math.sin(gam)
        dg = ld*cl/v + (v/r - g/v)*math.cos(gam)
        dr = v*math.sin(gam)
        dx = v*math.cos(gam)/r
        T.append(r-R_E); H.append(h); V.append(v); GAM.append(gam); X.append(x*R_E)
        if h<=H_END or v<=V_END: break
        r+=dt*dr; v+=dt*dv; gam+=dt*dg; x+=dt*dx
    return (np.array(T)/1000, np.array(H), np.array(V),
            np.degrees(np.array(GAM)), np.array(X)/1000,
            np.arange(len(T))*dt)

# ── heating & sweating ────────────────────────────────────────────────────────
ZONES = [
    dict(name="belly tiles",       area=250, rn=5.00, sweat=False, qm=1.0),
    dict(name="nose tip",          area=12,  rn=0.60, sweat=True,  qm=1.0),
    dict(name="flap leading edges",area=30,  rn=0.25, sweat=True,  qm=1.0),
    dict(name="flap hinge gaps",   area=8,   rn=0.35, sweat=True,  qm=1.5),
    dict(name="engine bay lip",    area=20,  rn=0.40, sweat=True,  qm=1.8),
]

COOLANTS = {
    "methane": dict(t_in=112, t_boil=112, h_lat=5.1e5, cp_l=3500, cp_v=3500),
    "water":   dict(t_in=293, t_boil=373, h_lat=2.26e6,cp_l=4186, cp_v=2100),
}

def absorb(c, tw):
    if tw<=c["t_boil"]: return c["cp_l"]*max(tw-c["t_in"],0)
    return c["cp_l"]*max(c["t_boil"]-c["t_in"],0)+c["h_lat"]+c["cp_v"]*(tw-c["t_boil"])

def q_cold(rho_arr, v_arr, rn, qm):
    return qm*K_SG*np.sqrt(rho_arr/rn)*v_arr**3

def run_thermal(h_arr, v_arr, t_arr, cool, t_tgt=1500, eta=0.5):
    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    rho_arr = np.array([rho(h) for h in h_arr])
    results = []
    for z in ZONES:
        q0 = q_cold(rho_arr, v_arr, z["rn"], z["qm"])
        tw, mdot = 300.0, np.zeros(len(t_arr))
        temps = np.empty(len(t_arr))
        for i,dt in enumerate(dt_arr):
            if dt>0:
                rad = EMIT*SB*tw**4
                t_free = tw + dt*(q0[i]-rad)/C_SKIN
                if z["sweat"] and t_free>t_tgt:
                    # bisect for mdot
                    dh = absorb(cool, t_tgt)
                    stored = C_SKIN*(t_tgt-tw)/dt
                    rad2 = EMIT*SB*t_tgt**4
                    def f(m):
                        if m<=0: return q0[i]-rad2-stored
                        b = eta*m*(cool["cp_v"]/CP_GAS)*0.5*v_arr[i]**2/max(q0[i],1)
                        qb = q0[i]*b/max(math.expm1(min(b,60)),1e-12)
                        return qb-rad2-stored-m*dh
                    if f(0)<=0:
                        m=0
                    else:
                        lo,hi=0,q0[i]/max(dh,1)
                        for _ in range(40):
                            mid=0.5*(lo+hi)
                            if f(mid)>0: lo=mid
                            else: hi=mid
                        m=0.5*(lo+hi)
                    mdot[i]=m
                    tw = t_tgt if m>0 else t_free
                else:
                    tw = t_free
            temps[i]=tw
        results.append(dict(zone=z, q0=q0, temps=temps, mdot=mdot,
                            coolant_kg=float(np.sum(mdot*dt_arr)*z["area"]),
                            peak_q=float(q0.max()), peak_t=float(temps.max())))
    return results

# ── load factor ───────────────────────────────────────────────────────────────
def load_g(h_arr, v_arr, alpha_deg):
    cl,cd = aero(alpha_deg)
    lg = []
    for h,v in zip(h_arr,v_arr):
        qd = 0.5*rho(h)*v*v
        lg.append(qd*S_REF*math.hypot(cl,cd)/(MASS*G0))
    return np.array(lg)

# ── plot ───────────────────────────────────────────────────────────────────────
def plot(alpha_deg, cool_name, outdir):
    h_km, h_m, v, gam, x_km, t = fly(alpha_deg)
    rr = np.array([rho(h) for h in h_m])
    cool = COOLANTS[cool_name]
    res = run_thermal(h_m, v, t, cool)
    lg = load_g(h_m, v, alpha_deg)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Starship re-entry  |  alpha={alpha_deg}°, bank=0° (wings level)  |  coolant: {cool_name}", fontsize=13)

    # 1 altitude vs time
    ax[0,0].plot(t/60, h_m/1000, lw=2)
    ax[0,0].set(xlabel="Time (min)", ylabel="Altitude (km)", title="Altitude vs time")
    ax[0,0].grid(alpha=0.3)

    # 2 alt-velocity map
    ax[0,1].plot(v/1000, h_m/1000, lw=2, color="tab:orange")
    ax[0,1].set(xlabel="Speed (km/s)", ylabel="Altitude (km)", title="Altitude–velocity map")
    ax[0,1].grid(alpha=0.3)

    # 3 load factor
    ax[0,2].plot(t/60, lg, lw=2, color="tab:red")
    ax[0,2].axhline(2.0, color="k", ls="--", lw=1, label="2g comfort limit")
    ax[0,2].axhline(4.0, color="r", ls="--", lw=1, label="4g hard limit")
    ax[0,2].set(xlabel="Time (min)", ylabel="Load factor (g)", title="G-load on crew")
    ax[0,2].legend(fontsize=8); ax[0,2].grid(alpha=0.3)

    # 4 heat flux per zone
    for r in res:
        ax[1,0].plot(t/60, r["q0"]/1e4, lw=1.5, label=r["zone"]["name"])
    ax[1,0].set(xlabel="Time (min)", ylabel="Cold-wall flux (W/cm²)", title="Heating per zone")
    ax[1,0].legend(fontsize=7); ax[1,0].grid(alpha=0.3)

    # 5 wall temp: dry vs wet
    for r in res:
        if r["zone"]["sweat"]:
            ax[1,1].plot(t/60, r["temps"], lw=1.8, label=r["zone"]["name"]+" (sweating)")
    ax[1,1].axhline(1650, color="k", ls=":", lw=1, label="tile limit 1650 K")
    ax[1,1].set(xlabel="Time (min)", ylabel="Wall temp (K)", title="Wall temp with sweating ON")
    ax[1,1].legend(fontsize=7); ax[1,1].grid(alpha=0.3)

    # 6 coolant use
    total = sum(r["coolant_kg"] for r in res)
    names = [r["zone"]["name"] for r in res if r["zone"]["sweat"]]
    kgs   = [r["coolant_kg"]   for r in res if r["zone"]["sweat"]]
    bars = ax[1,2].bar(names, kgs, color="tab:green")
    ax[1,2].set(ylabel="Coolant mass (kg)", title=f"Coolant per zone  |  total: {total:.0f} kg")
    ax[1,2].tick_params(axis="x", labelrotation=15)
    ax[1,2].grid(alpha=0.3, axis="y")
    for b,k in zip(bars,kgs):
        ax[1,2].text(b.get_x()+b.get_width()/2, k+5, f"{k:.0f}", ha="center", fontsize=8)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"sweat_alpha{alpha_deg}_{cool_name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved → {path}")

    # print summary
    print(f"\n── Summary ─────────────────────────────────────")
    print(f"  Alpha:          {alpha_deg}°  |  Bank: 0° (wings level)")
    print(f"  Entry speed:    7600 m/s  |  Entry angle: -3°")
    print(f"  Flight time:    {t[-1]/60:.1f} min")
    print(f"  Downrange:      {x_km[-1]:.0f} km")
    print(f"  Peak g-load:    {lg.max():.2f} g")
    print(f"  Peak heat flux: {max(r['peak_q'] for r in res)/1e4:.1f} W/cm²  (flap hinge gaps)")
    print(f"  Total coolant:  {total:.0f} kg  ({cool_name})")
    print(f"  Zone breakdown:")
    for r in res:
        tag = " ← sweating" if r["zone"]["sweat"] else " ← passive tiles"
        print(f"    {r['zone']['name']:<22} peak {r['peak_q']/1e4:6.1f} W/cm²  "
              f"Tmax {r['peak_t']:5.0f} K  coolant {r['coolant_kg']:6.1f} kg{tag}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha",   type=float, default=60.0)
    ap.add_argument("--coolant", choices=["methane","water"], default="methane")
    ap.add_argument("--out",     default="output")
    a = ap.parse_args()
    plot(a.alpha, a.coolant, a.out)

if __name__=="__main__":
    main()
