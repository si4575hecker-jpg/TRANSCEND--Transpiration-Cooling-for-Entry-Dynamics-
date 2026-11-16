# TRANSCEND: Transpiration Cooling for Entry Dynamics

**Integrated Thermal Protection System (TPS) & Trajectory Optimization for Starship Reusability**

TRANSCEND is a 7,000+ line Python aerothermodynamic simulation framework investigating zero-maintenance structural reusability for Starship-class orbital re-entry vehicles.

During hypersonic atmospheric entry, localized hot spots—such as flap hinge gaps and leading edges—experience heat fluxes that exceed the **1,650 K** durability limit of passive ceramic tiles. TRANSCEND solves this by coupling a **continuous 60° Angle of Attack ($\alpha = 60^\circ$) descent profile** with **zoned active transpiration ("sweat") cooling** fueled directly by residual header-tank liquid methane ($\text{CH}_4$).

---

### Core Engineering Highlights
* **Thermal Ceiling Clamped at 1,500 K:** Maintains a ~200 K safety margin below the ~1,700 K melting threshold of 30X-series stainless steel, preventing yield-strength degradation and deep thermal soaking.
* **Zero Auxiliary Tank Mass Penalty:** Taps existing sub-cooled $\text{CH}_4$ reserves, exploiting latent heat of vaporization and boundary-layer "blowing" effects to push superheated plasma away from the hull.
* **Targeted Zoned Cooling Matrix:** Isolates porous transpiration manifolds to high-flux hot spots (flap leading edges, flap hinge seams, nose cone stagnation zone, and transient belly spikes) rather than cooling the entire underbelly.
* **Crew-Safe Descent Profile:** Replaces high-exposure lifting skips with a steady energy-dissipating descent slope, keeping peak deceleration strictly under **2.5 Gs**.

---

### Where to Find Everything
All simulation code, telemetry data, comparison graphs, and the full technical overview paper are located inside the **`TRANSCEND Prototype FINAL - github/`** directory:

* **`OVERVIEW PAPER.pdf`** — Full technical proposal, thermodynamic phase-change methodology, and metallurgical validation.
* **`skip+sweat(draft 1)/`** — Baseline lifting-skip trajectory coupled with initial transpiration cooling.
* **`sweat+60°(draft 2)/`** — Transition to high-drag $\alpha = 60^\circ$ continuous descent profile.
* **`improved_sweat(draft 3)/`** — Zoned manifold isolation and propellant mass-flow optimization.
* **`sweat+hover(draft 4)/`** — Terminal descent dynamics and structural thermal soak modeling.
* **`FINAL(draft 5)/`** — Complete integrated multi-zone TPS and trajectory optimization suite.

---

### Methodology & Verification
* **Author:** Shaurya Mehta (Class 8, Sanskriti School, New Delhi, India)
* **Verification:** All numerical integration loops, phase-change enthalpy balances, and aerodynamic heating equations were independently verified against fundamental conservation of mass and energy laws.
