#!/usr/bin/env python3
"""
skip_sweat_sim.py
=================
Reduced-order simulator for a Starship-class vehicle doing a SKIP re-entry
with TRANSPIRATION ("sweating") COOLING on the hottest surfaces.

What it does
------------
1. Flies a planar 3-DOF point-mass re-entry (spherical, non-rotating Earth,
   tabulated US-1976-style atmosphere, Newtonian flat-plate aerodynamics).
   Control = bank angle (scales the *vertical* component of lift) and angle
   of attack, both set per atmospheric pass, so you can design skips.
2. Post-processes the trajectory into stagnation-type convective heating
   (Sutton-Graves) on several surface zones (belly tiles, nose tip, flap
   leading edges, hinge gaps).
3. Runs a lumped thermal model of each zone (thermal mass + radiative
   re-emission) with and without transpiration cooling.  When sweating is
   on, the sim finds the coolant flow needed to hold the wall at a target
   temperature (film-theory blocking + coolant heat pick-up), and integrates
   coolant mass.
4. Compares presets (ballistic, lift-up, equilibrium glide, skips), prints a
   table, saves plots + CSV, and can random-search skip parameters that
   minimise coolant mass.

Run
---
    pip install numpy scipy matplotlib
    python skip_sweat_sim.py                       # all presets, saves to ./output
    python skip_sweat_sim.py --coolant water        # use water instead of methane
    python skip_sweat_sim.py --detail skip2          # detailed plots for one preset
    python skip_sweat_sim.py --optimize 150          # search skip bank/alpha angles
    python skip_sweat_sim.py --show                 # also open plot windows

IMPORTANT LIMITS (read before trusting any number)
--------------------------------------------------
* Convective heating only (Sutton-Graves).  Radiative heating is small at
  orbital entry speeds (~7.6 km/s) but matters above ~9-10 km/s.
* Planar model: bank only scales vertical lift; crossrange is ignored.
* Heating is *stagnation-point-like* with per-zone radius and multipliers.
  Real Starship heating is 3D, has boundary-layer transition, shocks, gap
  flow, etc. that this does not capture.
* Film-theory blocking, constant coolant properties, no coolant cracking,
  no pore clogging, no plenum dynamics.  Treat coolant masses as trend
  indicators, not predictions.  Calibrate against CFD or arc-jet data.
* Vehicle numbers below are assumptions, not SpaceX data - edit them with
  --mass / --area or in the Vehicle dataclass.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MU = 3.986004418e14          # Earth GM, m^3/s^2
R_E = 6_371_000.0            # mean Earth radius, m
G0 = 9.80665                 # standard gravity, m/s^2
SIGMA_SB = 5.670374419e-8    # Stefan-Boltzmann, W/m^2/K^4
K_SG = 1.7415e-4             # Sutton-Graves constant for Earth air (SI)
CP_GAS = 1300.0              # effective boundary-layer gas cp, J/kg/K (tuning knob)

H_INTERFACE = 120_000.0      # entry interface / "top of atmosphere", m
H_END = 20_000.0             # stop the sim here (start of terminal descent), m
V_END = 1_500.0              # ... or when slower than this, m/s

# --------------------------------------------------------------------------
# Atmosphere: log-linear interpolation of a US-1976-like density table
# --------------------------------------------------------------------------
_ALT_KM = np.arange(0, 151, 10.0)
_RHO = np.array([
    1.225, 0.4135, 0.08891, 0.01841, 0.003996, 0.001027, 3.097e-4,
    8.283e-5, 1.846e-5, 3.416e-6, 5.604e-7, 9.708e-8, 2.222e-8,
    8.152e-9, 3.831e-9, 2.076e-9,
])
_LN_RHO = np.log(_RHO)
_H_UPPER = 25_000.0          # scale height used above 150 km, m


def rho_air(h: float) -> float:
    """Air density (kg/m^3) at geometric altitude h (m)."""
    if h <= 0.0:
        return float(_RHO[0])
    hk = h / 1000.0
    if hk >= 150.0:
        return float(_RHO[-1] * math.exp(-(h - 150_000.0) / _H_UPPER))
    return float(math.exp(np.interp(hk, _ALT_KM, _LN_RHO)))


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Vehicle:
    mass: float = 150_000.0        # kg at entry (dry + payload + residuals)  [assumption]
    s_ref: float = 400.0           # m^2 planform reference area              [assumption]
    cd0: float = 0.05              # zero-lift drag add-on
    emissivity: float = 0.85       # surface emissivity of hot tile/coating
    skin_heat_cap: float = 7200.0  # J/m^2/K lumped surface thermal mass (tile skin)

    def coeffs(self, alpha_deg: float):
        """Newtonian flat-plate CL, CD at angle of attack (deg)."""
        a = math.radians(alpha_deg)
        cl = 2.0 * math.sin(a) ** 2 * math.cos(a)
        cd = self.cd0 + 2.0 * math.sin(a) ** 3
        return cl, cd

    def ballistic_coeff(self, alpha_deg: float) -> float:
        _, cd = self.coeffs(alpha_deg)
        return self.mass / (cd * self.s_ref)


@dataclass
class EntryState:
    h0: float = H_INTERFACE
    v0: float = 7_600.0          # inertial-ish speed, m/s (planet rotation ignored)
    gamma0_deg: float = -3.0     # flight path angle, deg (negative = descending)
    # NOTE: -3 deg is deliberately steeper than a shallow ballistic-entry-
    # coasting deorbit (~-1 deg). A shallower entry (try --gamma0 -1.2) never
    # builds enough aero load for this vehicle/alpha to trigger a pull-out at
    # all -- it just glides straight down in one dive. Skipping needs a dive
    # aggressive enough to load up, then lift can arrest it. Sweep --gamma0
    # yourself; the required steepness depends on beta and alpha.


@dataclass
class Profile:
    """Guidance profile, built from LOBES.

    At Starship's re-entry speed (~7.6 km/s) the vehicle is well below local
    circular velocity, so it physically CANNOT loft itself back out to vacuum
    (see the note in the module docstring / README). A "skip" for a vehicle
    like this means diving into the thicker air, pulling out before it gets
    too hot or too high-g, climbing to a shallower (but still sub-orbital)
    lob, then diving again. That is what "lobe" describes here.

    Lobe k = one dive-and-optional-pullout:
      * DIVE phase: fly bank_dive[k] at alpha[k].
      * PULL phase: once the instantaneous aero load factor reaches
        n_pull[k] (g), switch to bank_pull (default 0 = full lift-up) at the
        same alpha[k], which arrests the dive. The vehicle climbs to a local,
        in-atmosphere apogee and starts descending again.
      * When it starts descending again (gamma crosses back through zero),
        lobe k+1 begins in its own DIVE phase.

    The LAST lobe has no pull-out: once n_pull[-1] is reached it just keeps
    flying bank_dive[-1]/alpha[-1] down to the end of the sim (or, if
    final_mode="eqglide", switches to an equilibrium glide instead).

    len(bank_dive) sets the number of lobes. n_pull / alpha shorter than
    that repeat their last entry. n_lobes=1 with a high n_pull is just a
    single dive (no skip) -- that is what "ballistic"/"liftup" below use.
    """
    name: str
    bank_dive: List[float]
    n_pull: List[float]
    alpha: List[float]
    bank_pull: float = 0.0
    final_mode: str = "const"    # "const" or "eqglide" for the last lobe
    label: str = ""

    @property
    def n_lobes(self) -> int:
        return len(self.bank_dive)

    def alpha_at(self, k: int) -> float:
        return self.alpha[min(k, len(self.alpha) - 1)]

    def npull_at(self, k: int) -> float:
        return self.n_pull[min(k, len(self.n_pull) - 1)]

    def bank_at(self, k: int) -> float:
        return self.bank_dive[min(k, len(self.bank_dive) - 1)]


@dataclass
class Zone:
    name: str
    area: float                  # m^2
    r_nose: float                # effective radius for Sutton-Graves, m
    t_limit: float                # material limit, K
    sweat: bool = False           # is this zone transpiration-cooled?
    q_mult: float = 1.0           # hot-spot / interference multiplier


@dataclass(frozen=True)
class Coolant:
    name: str
    t_in: float                  # storage temperature, K
    t_boil: float                # boiling point at low pressure, K
    h_lat: float                 # latent heat, J/kg
    cp_liq: float                # J/kg/K
    cp_vap: float                # mean vapour cp from t_boil to wall temp, J/kg/K

    def absorb(self, t_wall: float) -> float:
        """Heat the coolant can absorb per kg going from storage to t_wall (J/kg)."""
        if t_wall <= self.t_boil:
            return self.cp_liq * max(t_wall - self.t_in, 0.0)
        return (self.cp_liq * max(self.t_boil - self.t_in, 0.0)
                + self.h_lat + self.cp_vap * (t_wall - self.t_boil))


COOLANTS: Dict[str, Coolant] = {
    # methane is already on board; real CH4 cracks/cokes above ~1000 K, ignored here
    "methane": Coolant("methane", 112.0, 112.0, 5.1e5, 3500.0, 3500.0),
    "water": Coolant("water", 293.0, 373.0, 2.26e6, 4186.0, 2100.0),
}


@dataclass
class SweatConfig:
    coolant: Coolant
    t_target: float = 1500.0     # hold sweated walls at this temperature, K
    blocking_eff: float = 0.5    # 0..1 knob scaling film-theory blowing parameter
    # (1.0 = ideal film theory; lower = more conservative; calibrate with tests)


def default_zones() -> List[Zone]:
    return [
        Zone("belly tiles",        250.0, 5.00, 1650.0, sweat=False, q_mult=1.0),
        Zone("nose tip",            12.0, 0.60, 1650.0, sweat=True,  q_mult=1.0),
        Zone("flap leading edges",  30.0, 0.25, 1650.0, sweat=True,  q_mult=1.0),
        Zone("flap hinge gaps",      8.0, 0.35, 1650.0, sweat=True,  q_mult=1.5),
    ]


# --------------------------------------------------------------------------
# Trajectory simulation
# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    t: np.ndarray
    h: np.ndarray
    v: np.ndarray
    gamma: np.ndarray
    x: np.ndarray                # downrange, m
    pass_idx: np.ndarray         # lobe number (int), same convention as Profile lobes
    phase_pull: np.ndarray       # bool: True while in the "pull-up" part of a lobe
    n_passes: int                # number of lobes actually flown (skips = n_passes - 1)
    end_reason: str
    ok: bool


def _accel(veh: Vehicle, r: float, v: float, gam: float, cl: float, cd: float,
           bank_deg: float, mode: str):
    h = r - R_E
    rho = rho_air(h)
    g = MU / (r * r)
    k = 0.5 * veh.s_ref * rho * v * v / veh.mass
    lm = k * cl
    dm = k * cd
    if mode == "eqglide":
        c = (g - v * v / r) * math.cos(gam) / max(lm, 1e-9)
        c = max(-1.0, min(1.0, c))
    else:
        c = math.cos(math.radians(bank_deg))
    dr = v * math.sin(gam)
    dv = -dm - g * math.sin(gam)
    dgam = lm * c / v + (v / r - g / v) * math.cos(gam)
    dx = v * math.cos(gam) / r
    return dr, dv, dgam, dx


def simulate(veh: Vehicle, prof: Profile, entry: EntryState,
             t_max: float = 6000.0, dt: float = 0.05) -> Trajectory:
    """Fixed-step RK4 integration with a dive/pull-up guidance state machine
    (see Profile docstring). Bank/alpha are re-evaluated every step from the
    current lobe + phase, so lobes and pull-outs are detected as the flight
    unfolds rather than pre-scheduled by altitude events."""
    r = R_E + entry.h0
    v = entry.v0
    gam = math.radians(entry.gamma0_deg)
    x = 0.0
    t = 0.0

    lobe = 0
    pulling = False
    seen_climb_this_lobe = False   # have we gone gamma>0 since the last dive started?
    last_lobe = prof.n_lobes - 1

    T, H, V, GAM, X, PIDX, PPULL = [t], [r - R_E], [v], [gam], [x], [lobe], [pulling]
    end_reason, ok = "timeout", False

    def load_factor(rr, vv, cl, cd):
        rho = rho_air(rr - R_E)
        qd = 0.5 * rho * vv * vv
        return qd * veh.s_ref * math.hypot(cl, cd) / (veh.mass * G0)

    n_steps = int(t_max / dt)
    for _ in range(n_steps):
        alpha = prof.alpha_at(lobe)
        cl, cd = veh.coeffs(alpha)
        is_last = lobe >= last_lobe

        if pulling:
            bank, mode = prof.bank_pull, "const"
        elif is_last and prof.final_mode == "eqglide":
            bank, mode = 0.0, "eqglide"
        else:
            bank, mode = prof.bank_at(lobe), "const"

        ng = load_factor(r, v, cl, cd)

        # guidance state machine, evaluated on the current state
        if not is_last:
            if not pulling and ng >= prof.npull_at(lobe):
                pulling = True
                seen_climb_this_lobe = False
            elif pulling:
                if gam > 0:
                    seen_climb_this_lobe = True
                if seen_climb_this_lobe and gam <= 0:
                    lobe += 1
                    pulling = False
                    seen_climb_this_lobe = False

        # RK4 step
        def f(rr, vv, gg, xx, bank_, mode_):
            return _accel(veh, rr, vv, gg, cl, cd, bank_, mode_)

        k1 = f(r, v, gam, x, bank, mode)
        k2 = f(r + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], gam + 0.5 * dt * k1[2],
               x + 0.5 * dt * k1[3], bank, mode)
        k3 = f(r + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], gam + 0.5 * dt * k2[2],
               x + 0.5 * dt * k2[3], bank, mode)
        k4 = f(r + dt * k3[0], v + dt * k3[1], gam + dt * k3[2], x + dt * k3[3],
               bank, mode)
        r += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        gam += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        x += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        t += dt

        h = r - R_E
        T.append(t); H.append(h); V.append(v); GAM.append(gam); X.append(x)
        PIDX.append(lobe); PPULL.append(pulling)

        if h <= H_END:
            end_reason, ok = "reached terminal-descent altitude", True
            break
        if v <= V_END:
            end_reason, ok = "slowed below terminal speed", True
            break
        if h > H_INTERFACE + 20_000.0:
            end_reason, ok = "climbed well past 120 km (unrealistic for this entry speed)", False
            break
        if v * v * 0.5 - MU / r > 0:
            end_reason = "escaped (bank/alpha gave too much lift for this speed)"
            break
    else:
        end_reason = "timeout (did not reach terminal descent in t_max)"

    return Trajectory(
        t=np.array(T), h=np.array(H), v=np.array(V), gamma=np.array(GAM),
        x=np.array(X) * R_E,   # X was integrated as a central angle (rad); convert to arc length (m)
        pass_idx=np.array(PIDX, dtype=int), phase_pull=np.array(PPULL, dtype=bool),
        n_passes=int(max(PIDX)) + 1, end_reason=end_reason, ok=ok,
    )


# --------------------------------------------------------------------------
# Resample onto a fine grid (fine inside atmosphere, coarse in space)
# --------------------------------------------------------------------------
@dataclass
class Grid:
    t: np.ndarray
    dt: np.ndarray
    h: np.ndarray
    v: np.ndarray
    gamma: np.ndarray
    x: np.ndarray
    rho: np.ndarray
    qdyn: np.ndarray
    load_g: np.ndarray           # total aero load factor
    pass_idx: np.ndarray


def build_grid(veh: Vehicle, prof: Profile, traj: Trajectory,
               dt_atm: float = 0.5, dt_space: float = 2.0) -> Grid:
    ts = [traj.t[0]]
    while ts[-1] < traj.t[-1]:
        hcur = np.interp(ts[-1], traj.t, traj.h)
        ts.append(ts[-1] + (dt_atm if hcur < 135_000.0 else dt_space))
    t = np.array(ts)
    t[-1] = min(t[-1], traj.t[-1])
    h = np.interp(t, traj.t, traj.h)
    v = np.interp(t, traj.t, traj.v)
    gam = np.interp(t, traj.t, traj.gamma)
    x = np.interp(t, traj.t, traj.x)
    pidx = np.interp(t, traj.t, traj.pass_idx.astype(float)).round().astype(int)
    rho = np.array([rho_air(hh) for hh in h])
    qd = 0.5 * rho * v * v
    load_g = np.zeros_like(t)
    for i in range(t.size):
        alpha = prof.alpha_at(max(pidx[i], 0))
        cl, cd = veh.coeffs(alpha)
        load_g[i] = qd[i] * veh.s_ref * math.hypot(cl, cd) / (veh.mass * G0)
    dt = np.diff(t, prepend=t[0])
    return Grid(t, dt, h, v, gam, x, rho, qd, load_g, pidx)


# --------------------------------------------------------------------------
# Heating + thermal + transpiration cooling
# --------------------------------------------------------------------------
def sutton_graves(rho: np.ndarray, v: np.ndarray, r_nose: float, mult: float = 1.0):
    """Cold-wall convective stagnation heat flux, W/m^2."""
    return mult * K_SG * np.sqrt(rho / r_nose) * v ** 3


def blocked_flux(q0: float, mdot: float, v: float, cool: Coolant, eta: float) -> float:
    """Convective flux reaching the wall with blowing (film theory):
       q = q0 * B / (exp(B) - 1),  B = eta * mdot * (cp_c/cp_gas) * (v^2/2) / q0."""
    if mdot <= 0.0 or q0 <= 0.0:
        return q0
    b = eta * mdot * (cool.cp_vap / CP_GAS) * 0.5 * v * v / q0
    if b < 1e-9:
        return q0
    return q0 * b / math.expm1(min(b, 60.0))


def solve_mdot(q0: float, v: float, t_prev: float, t_tgt: float, dt: float,
               cool: Coolant, eta: float, c_area: float, eps: float) -> float:
    """Coolant mass flux (kg/m^2/s) that holds the wall at t_tgt over this step,
    found by bisection on the wall energy balance:
        blocked_flux(mdot) = radiated + stored-in-skin + carried-off-by-coolant
    """
    rad = eps * SIGMA_SB * t_tgt ** 4
    stored = c_area * (t_tgt - t_prev) / dt
    dh = cool.absorb(t_tgt)
    if dh <= 0.0:
        return 0.0

    def f(m):
        return blocked_flux(q0, m, v, cool, eta) - rad - stored - m * dh

    if f(0.0) <= 0.0:
        return 0.0
    lo, hi = 0.0, q0 / dh            # f(hi) <= 0 always, since blocked flux <= q0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class ZoneResult:
    zone: Zone
    q_cold: np.ndarray           # W/m^2, unblown cold-wall
    temp: np.ndarray             # wall temperature, K
    mdot: np.ndarray             # kg/m^2/s coolant (zeros if unsweated / this run has no sweat)
    coolant_kg: float
    heat_load: float             # J/m^2 (integral of q_cold)
    peak_q: float
    peak_t: float


def analyse_heating(veh: Vehicle, grid: Grid, zones: List[Zone],
                    sweat: Optional[SweatConfig]) -> List[ZoneResult]:
    out = []
    for z in zones:
        q0 = sutton_graves(grid.rho, grid.v, z.r_nose, z.q_mult)
        n = grid.t.size
        temp = np.empty(n)
        mdot = np.zeros(n)
        tcur = 300.0
        use = sweat is not None and z.sweat
        for i in range(n):
            dt = grid.dt[i] if i > 0 else 0.0
            if dt > 0.0:
                rad = veh.emissivity * SIGMA_SB * tcur ** 4
                t_free = tcur + dt * (q0[i] - rad) / veh.skin_heat_cap
                if use and t_free > sweat.t_target:
                    m = solve_mdot(q0[i], grid.v[i], tcur, sweat.t_target, dt,
                                   sweat.coolant, sweat.blocking_eff,
                                   veh.skin_heat_cap, veh.emissivity)
                    mdot[i] = m
                    tcur = sweat.t_target if m > 0.0 else t_free
                else:
                    tcur = t_free
            temp[i] = tcur
        coolant_kg = float(np.sum(mdot * grid.dt) * z.area)
        out.append(ZoneResult(
            zone=z, q_cold=q0, temp=temp, mdot=mdot, coolant_kg=coolant_kg,
            heat_load=float(np.sum(q0 * grid.dt)), peak_q=float(q0.max()),
            peak_t=float(temp.max())))
    return out


# --------------------------------------------------------------------------
# One full run of a profile
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    prof: Profile
    traj: Trajectory
    grid: Grid
    dry: List[ZoneResult]        # no sweating anywhere (baseline)
    wet: List[ZoneResult]        # sweating on flagged zones

    @property
    def peak_g(self): return float(self.grid.load_g.max())
    @property
    def peak_qdyn(self): return float(self.grid.qdyn.max())
    @property
    def coolant_total(self): return float(sum(r.coolant_kg for r in self.wet))
    @property
    def hottest_dry_q(self): return max(r.peak_q for r in self.dry)
    @property
    def hottest_dry_t(self): return max(r.peak_t for r in self.dry)
    @property
    def hottest_wet_t(self): return max(r.peak_t for r in self.wet)
    @property
    def apogee_km(self):
        """Highest altitude reached AFTER the first pull-up (i.e. the top of
        the first skip lobe), not counting the initial entry-interface
        altitude itself. NaN if the profile never pulls up (no skip)."""
        after_dive = self.traj.h < H_INTERFACE - 500.0
        if not after_dive.any():
            return float("nan")
        first_dive_i = int(np.argmax(after_dive))
        rest = self.traj.h[first_dive_i:]
        return float(rest.max() / 1000.0) if rest.size else float("nan")


def run_profile(veh: Vehicle, prof: Profile, entry: EntryState, zones: List[Zone],
                sweat: SweatConfig, dt_atm: float = 0.5, dt_space: float = 2.0) -> RunResult:
    traj = simulate(veh, prof, entry)
    grid = build_grid(veh, prof, traj, dt_atm, dt_space)
    dry = analyse_heating(veh, grid, zones, None)
    wet = analyse_heating(veh, grid, zones, sweat)
    return RunResult(prof, traj, grid, dry, wet)


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
def preset_profiles() -> Dict[str, Profile]:
    return {
        "ballistic": Profile("ballistic", [90.0], [99.0], [55.0],
                             label="Ballistic-ish (lift sideways, bank 90, no pull-out)"),
        "eqglide": Profile("eqglide", [0.0], [99.0], [55.0], final_mode="eqglide",
                           label="Equilibrium glide (gamma-dot = 0, single dive)"),
        "dive1": Profile("dive1", [0.0], [99.0], [55.0],
                         label="Single lift-up dive, no pull-out (baseline)"),
        "skip2": Profile("skip2", [70.0, 0.0], [1.8], [55.0], bank_pull=0.0,
                         label="1 skip: bank 70 deg dive, pull to bank 0 at 1.8 g, then dive out"),
        "skip3": Profile("skip3", [70.0, 60.0, 0.0], [2.0, 1.3], [55.0], bank_pull=0.0,
                         label="2 skips: bank 70/60 deg dives, pull at 2.0/1.3 g, then dive out"),
        "skip4": Profile("skip4", [75.0, 65.0, 55.0, 0.0], [2.2, 1.6, 1.1], [55.0],
                         bank_pull=0.0,
                         label="3 skips: bank 75/65/55 deg dives, pull at 2.2/1.6/1.1 g, then dive out"),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarise(res: RunResult) -> dict:
    tr = res.traj
    return {
        "profile": res.prof.name,
        "ok": tr.ok,
        "end": tr.end_reason,
        "passes": tr.n_passes,
        "flight_min": tr.t[-1] / 60.0,
        "downrange_km": tr.x[-1] / 1000.0,
        "apogee_km": res.apogee_km,
        "peak_g": res.peak_g,
        "peak_qdyn_kPa": res.peak_qdyn / 1000.0,
        "peak_q_Wcm2": res.hottest_dry_q / 1e4,
        "dry_Tmax_K": res.hottest_dry_t,
        "wet_Tmax_K": res.hottest_wet_t,
        "coolant_kg": res.coolant_total,
        "heat_load_MJm2": max(r.heat_load for r in res.dry) / 1e6,
    }


def print_table(results: List[RunResult]):
    print()
    hdr = (f"{'profile':<10} {'ok':<3} {'pass':>4} {'min':>6} {'range km':>9} "
           f"{'apog km':>8} {'peak g':>7} {'qdyn kPa':>8} {'peak q W/cm2':>12} "
           f"{'Tmax dry K':>10} {'Tmax wet K':>10} {'coolant kg':>10} {'load MJ/m2':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        s = summarise(r)
        print(f"{s['profile']:<10} {'Y' if s['ok'] else 'N':<3} {s['passes']:>4d} "
              f"{s['flight_min']:>6.1f} {s['downrange_km']:>9.0f} {s['apogee_km']:>8.0f} "
              f"{s['peak_g']:>7.2f} {s['peak_qdyn_kPa']:>8.1f} {s['peak_q_Wcm2']:>12.1f} "
              f"{s['dry_Tmax_K']:>10.0f} {s['wet_Tmax_K']:>10.0f} "
              f"{s['coolant_kg']:>10.1f} {s['heat_load_MJm2']:>10.1f}")
    print()
    for r in results:
        if not r.traj.ok:
            print(f"  note: {r.prof.name} ended with '{r.traj.end_reason}'")
    print("  (peak q / heat load are for the hottest zone, cold wall, no blowing;")
    print("   Tmax dry = radiative-equilibrium wall temp with no sweating;")
    print("   Tmax wet = same, but with sweating on nose tip / flap edges / hinge gaps)")


def save_csv(res: RunResult, path: str):
    g = res.grid
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["t_s", "alt_km", "v_mps", "gamma_deg", "downrange_km", "qdyn_Pa", "load_g"]
        for r in res.wet:
            cols += [f"{r.zone.name}: q_cold_Wcm2", f"{r.zone.name}: T_dry_K",
                     f"{r.zone.name}: T_wet_K", f"{r.zone.name}: mdot_kgm2s"]
        w.writerow(cols)
        dry = {r.zone.name: r for r in res.dry}
        for i in range(g.t.size):
            row = [f"{g.t[i]:.1f}", f"{g.h[i]/1000:.3f}", f"{g.v[i]:.1f}",
                   f"{math.degrees(g.gamma[i]):.3f}", f"{g.x[i]/1000:.2f}",
                   f"{g.qdyn[i]:.2f}", f"{g.load_g[i]:.3f}"]
            for r in res.wet:
                row += [f"{r.q_cold[i]/1e4:.3f}", f"{dry[r.zone.name].temp[i]:.1f}",
                        f"{r.temp[i]:.1f}", f"{r.mdot[i]:.5f}"]
            w.writerow(row)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_detail(res: RunResult, cool_name: str, path: str, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = res.grid
    tmin = g.t / 60.0
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"{res.prof.label or res.prof.name}   |   coolant: {cool_name}", fontsize=13)

    ax[0, 0].plot(g.v / 1000, g.h / 1000, lw=1.8)
    ax[0, 0].set(xlabel="Speed (km/s)", ylabel="Altitude (km)", title="Altitude-velocity map")
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(tmin, g.h / 1000, lw=1.8)
    ax[0, 1].set(xlabel="Time (min)", ylabel="Altitude (km)", title="Altitude vs time (skips)")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(tmin, g.load_g, color="tab:red", lw=1.6)
    ax[1, 0].set(xlabel="Time (min)", ylabel="Aero load factor (g)", title="Load factor and dynamic pressure")
    ax[1, 0].grid(alpha=0.3)
    ax2 = ax[1, 0].twinx()
    ax2.plot(tmin, g.qdyn / 1000, color="tab:blue", lw=1.2, alpha=0.7)
    ax2.set_ylabel("Dynamic pressure (kPa)", color="tab:blue")

    for r in res.wet:
        ax[1, 1].plot(tmin, r.q_cold / 1e4, lw=1.5, label=r.zone.name)
    ax[1, 1].set(xlabel="Time (min)", ylabel="Cold-wall heat flux (W/cm$^2$)",
                 title="Convective heating by zone (no blowing)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    dry = {r.zone.name: r for r in res.dry}
    for r in res.wet:
        if r.zone.sweat:
            ln, = ax[2, 0].plot(tmin, dry[r.zone.name].temp, "--", lw=1.2, alpha=0.8,
                                label=f"{r.zone.name} (no sweat)")
            ax[2, 0].plot(tmin, r.temp, "-", lw=1.8, color=ln.get_color(),
                          label=f"{r.zone.name} (sweat)")
    ax[2, 0].axhline(res.wet[0].zone.t_limit, color="k", ls=":", lw=1, label="tile limit")
    ax[2, 0].set(xlabel="Time (min)", ylabel="Wall temperature (K)",
                 title="Wall temperature: dashed = unprotected, solid = sweating")
    ax[2, 0].legend(fontsize=7, ncol=2)
    ax[2, 0].grid(alpha=0.3)

    tot = np.zeros_like(g.t)
    for r in res.wet:
        tot += r.mdot * r.zone.area
    cum = np.cumsum(tot * g.dt)
    ax[2, 1].plot(tmin, tot, color="tab:green", lw=1.6)
    ax[2, 1].set(xlabel="Time (min)", ylabel="Total coolant flow (kg/s)",
                 title=f"Coolant use: total {cum[-1]:.0f} kg")
    ax[2, 1].grid(alpha=0.3)
    ax3 = ax[2, 1].twinx()
    ax3.plot(tmin, cum, color="tab:orange", lw=1.2)
    ax3.set_ylabel("Cumulative coolant (kg)", color="tab:orange")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_compare(results: List[RunResult], zone_name: str, cool_name: str,
                 path: str, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Profile comparison   |   coolant: {cool_name}   |   zone: {zone_name}", fontsize=13)
    for r in results:
        g = r.grid
        nm = r.prof.name
        ax[0, 0].plot(g.v / 1000, g.h / 1000, lw=1.5, label=nm)
        ax[0, 1].plot(g.t / 60, g.h / 1000, lw=1.5, label=nm)
        zr = [z for z in r.dry if z.zone.name == zone_name][0]
        ax[1, 0].plot(g.t / 60, zr.q_cold / 1e4, lw=1.5, label=nm)
    ax[0, 0].set(xlabel="Speed (km/s)", ylabel="Altitude (km)", title="Altitude-velocity map")
    ax[0, 1].set(xlabel="Time (min)", ylabel="Altitude (km)", title="Altitude vs time")
    ax[1, 0].set(xlabel="Time (min)", ylabel="Cold-wall heat flux (W/cm$^2$)",
                 title=f"Heating on {zone_name}")
    names = [r.prof.name for r in results]
    dry_q = [max(z.peak_q for z in r.dry) / 1e4 for r in results]
    cool = [r.coolant_total for r in results]
    xs = np.arange(len(results))
    ax[1, 1].bar(xs - 0.2, cool, width=0.4, color="tab:green", label="coolant (kg)")
    ax[1, 1].set_xticks(xs)
    ax[1, 1].set_xticklabels(names, rotation=20)
    ax[1, 1].set_ylabel("Coolant mass (kg)", color="tab:green")
    axb = ax[1, 1].twinx()
    axb.bar(xs + 0.2, dry_q, width=0.4, color="tab:red", label="peak flux (W/cm2)")
    axb.set_ylabel("Peak hottest-zone flux (W/cm$^2$)", color="tab:red")
    ax[1, 1].set_title("Coolant needed vs peak flux, by profile")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Optimiser: random search + local refinement over skip bank/alpha angles
# --------------------------------------------------------------------------
def optimise(veh, entry, zones, sweat, n_samples: int, n_lobes: int,
             g_limit: float, seed: int = 1):
    """Random search + local refinement over the dive bank angles, the
    n_pull g-triggers, and a single alpha, for an n_lobes-lobe skip
    (n_lobes-1 skips), minimising total coolant mass subject to a peak-g
    limit. The last lobe always dives at bank 0 (no more pull-outs);
    earlier lobes search a bank angle (steeper bank = shallower vertical
    lift = deeper, faster dive before the g-trigger fires) plus the g-level
    at which each one pulls out to bank 0."""
    rng = np.random.default_rng(seed)
    n_free_pull = n_lobes - 1   # last lobe has no pull trigger
    n_free_bank = n_lobes - 1   # last lobe's dive bank is fixed at 0

    def build(x):
        banks = list(x[:n_free_bank]) + [0.0] if n_free_bank > 0 else [0.0]
        pulls = list(x[n_free_bank:n_free_bank + n_free_pull]) if n_free_pull > 0 else [2.0]
        alpha = float(x[-1])
        return Profile("opt", banks, pulls, [alpha])

    def score(x):
        prof = build(x)
        try:
            traj = simulate(veh, prof, entry)
            if not traj.ok:
                return 1e9, None
            grid = build_grid(veh, prof, traj, dt_atm=1.0, dt_space=4.0)
            wet = analyse_heating(veh, grid, zones, sweat)
        except Exception:
            return 1e9, None
        coolant = sum(r.coolant_kg for r in wet)
        pg = grid.load_g.max()
        pen = 0.0
        if pg > g_limit:
            pen += 1e3 * (pg - g_limit)
        if traj.t[-1] > 3600:
            pen += 1e3
        return coolant + pen, (round(coolant, 1), round(pg, 2), traj.n_passes,
                               round(traj.t[-1] / 60.0, 1))

    def rand_x():
        banks = rng.uniform(40, 85, n_free_bank)
        pulls = rng.uniform(0.8, g_limit, n_free_pull) if n_free_pull > 0 else [2.0]
        alpha = rng.uniform(45, 65, 1)
        return np.concatenate([banks, pulls, alpha])

    best_x, best_s, best_info = None, 1e9, None
    for i in range(n_samples):
        x = rand_x()
        s, info = score(x)
        if s < best_s:
            best_x, best_s, best_info = x, s, info
            print(f"  [{i+1}/{n_samples}] score {s:8.1f} kg  info={info}")
    if best_x is None:
        print("  no feasible sample found in the random search; loosen --g-limit")
        return None
    step = 5.0
    for rnd in range(4):
        for _ in range(max(10, n_samples // 6)):
            x = best_x + rng.normal(0, step, best_x.size)
            if n_free_bank > 0:
                x[:n_free_bank] = np.clip(x[:n_free_bank], 20, 88)
            if n_free_pull > 0:
                x[n_free_bank:n_free_bank + n_free_pull] = np.clip(
                    x[n_free_bank:n_free_bank + n_free_pull], 0.3, g_limit)
            x[-1] = np.clip(x[-1], 40, 70)
            s, info = score(x)
            if s < best_s:
                best_x, best_s, best_info = x, s, info
        step *= 0.5
        print(f"  refine round {rnd+1}: best {best_s:.1f} kg  info={best_info}")
    banks_r = [round(float(b), 1) for b in best_x[:n_free_bank]] + [0.0] if n_free_bank > 0 else [0.0]
    pulls_r = [round(float(p), 2) for p in best_x[n_free_bank:n_free_bank + n_free_pull]] if n_free_pull > 0 else []
    alpha_r = round(float(best_x[-1]), 1)
    return Profile("optimised", banks_r, pulls_r or [2.0], [alpha_r],
                   label=f"Optimised {n_lobes}-lobe skip: banks {banks_r}, pull at {pulls_r} g, alpha {alpha_r}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Skip re-entry + transpiration cooling simulator")
    ap.add_argument("--coolant", choices=list(COOLANTS), default="methane")
    ap.add_argument("--t-target", type=float, default=1500.0, help="sweated wall temperature, K")
    ap.add_argument("--blocking", type=float, default=0.5, help="film-theory blocking efficiency 0..1")
    ap.add_argument("--mass", type=float, default=150_000.0, help="entry mass, kg")
    ap.add_argument("--area", type=float, default=400.0, help="reference area, m^2")
    ap.add_argument("--v0", type=float, default=7600.0, help="entry speed, m/s")
    ap.add_argument("--gamma0", type=float, default=-3.0, help="entry flight path angle, deg")
    ap.add_argument("--detail", default=None, help="profile name for a detailed plot + CSV (default: all)")
    ap.add_argument("--optimize", type=int, default=0, help="random-search samples (0 = off)")
    ap.add_argument("--opt-passes", type=int, default=3)
    ap.add_argument("--g-limit", type=float, default=4.0)
    ap.add_argument("--out", default="output")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    veh = Vehicle(mass=args.mass, s_ref=args.area)
    entry = EntryState(v0=args.v0, gamma0_deg=args.gamma0)
    zones = default_zones()
    coolant = COOLANTS[args.coolant]
    sweat = SweatConfig(coolant, t_target=args.t_target, blocking_eff=args.blocking)

    profs = preset_profiles()
    print(f"Vehicle: m = {veh.mass/1e3:.0f} t, S = {veh.s_ref:.0f} m^2, "
          f"beta(alpha=60) = {veh.ballistic_coeff(60):.0f} kg/m^2")
    cl, cd = veh.coeffs(60.0)
    print(f"Aero @ alpha=60: CL = {cl:.2f}, CD = {cd:.2f}, L/D = {cl/cd:.2f}")
    print(f"Entry: {entry.h0/1e3:.0f} km, {entry.v0:.0f} m/s, gamma = {entry.gamma0_deg} deg")
    print(f"Sweat: {coolant.name}, hold walls at {sweat.t_target:.0f} K, blocking efficiency {sweat.blocking_eff}")

    if args.optimize > 0:
        print(f"\nOptimising a {args.opt_passes}-pass skip ({args.optimize} samples) ...")
        best = optimise(veh, entry, zones, sweat, args.optimize, args.opt_passes, args.g_limit)
        if best is not None:
            profs["optimised"] = best
            print("Best profile:", best.label)

    results = []
    for name, p in profs.items():
        print(f"running {name} ...")
        results.append(run_profile(veh, p, entry, zones, sweat))
    print_table(results)

    hottest = max(zones, key=lambda z: z.q_mult / math.sqrt(z.r_nose)).name
    plot_compare(results, hottest, coolant.name, os.path.join(args.out, "compare.png"), args.show)
    print(f"saved {os.path.join(args.out, 'compare.png')}")

    detail_names = [args.detail] if args.detail else [r.prof.name for r in results]
    for r in results:
        if r.prof.name in detail_names:
            plot_detail(r, coolant.name, os.path.join(args.out, f"detail_{r.prof.name}.png"), args.show)
            save_csv(r, os.path.join(args.out, f"history_{r.prof.name}.csv"))
    print(f"saved detail plots and CSV history to ./{args.out}/")


if __name__ == "__main__":
    main()
