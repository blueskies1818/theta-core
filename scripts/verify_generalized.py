#!/usr/bin/env python3
"""Generalized claims test — blind discovery across 7 physics domains."""
from __future__ import annotations
import math, random, statistics, sys, time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.dimensions import Dimension
from src.physics.observations import Observation
from src.physics.search import auto_discover

# ═══════════════════════ Data generation ═══════════════════════

def gen_obs(quantities: dict[str, str],
            invariant_fn: callable,  # (vals, K, rng) -> vals with invariant
            n_obs: int = 6, n_ts: int = 8, noise: float = 0.0003,
            rng: random.Random | None = None) -> list[Observation]:
    """Each observation has its own constant K. Invariant holds within each obs."""
    if rng is None:
        rng = random.Random(42)
    qnames = list(quantities.keys())
    observations = []
    for obs_i in range(n_obs):
        K = 1.0 + rng.random() * 100  # one constant per observation
        timesteps = []
        for _ in range(n_ts):
            vals = {q: 0.5 + rng.random() * 10 for q in qnames}
            vals = invariant_fn(vals, K, rng)
            for q in qnames:
                vals[q] += rng.gauss(0, noise * abs(vals[q]) + noise)
            timesteps.append(vals)
        observations.append(Observation(
            id=f"o{obs_i}", name=f"Obs {obs_i}", description="",
            quantities=dict(quantities), parameters={},
            timesteps=timesteps, known_invariant=None, lean_theorem=""))
    return observations

# ═══════════════════════ Invariant enforcers ═══════════════════════
# Each: (vals, K, rng) -> vals where invariant = K

def inv_product(a, b):
    def fn(v, K, r): v[b] = K / v[a]; return v
    return fn

def inv_ratio(a, b):
    def fn(v, K, r): v[b] = v[a] / K; return v
    return fn

def inv_power_ratio(a, b, pa, pb):
    """a^pa / b^pb = K"""
    def fn(v, K, r): v[b] = (v[a]**pa / K) ** (1.0 / pb); return v
    return fn

def inv_product_sq(a, b):
    """a * b^2 = K"""
    def fn(v, K, r): v[b] = math.sqrt(K / v[a]); return v
    return fn

def inv_triple(a, b, c, pa=1, pb=1, pc=1):
    """a^pa * b^pb * c^pc = K"""
    def fn(v, K, r):
        v[c] = (K / (v[a]**pa * v[b]**pb)) ** (1.0 / pc); return v
    return fn

# ═══════════════════════ 12 Claims ═══════════════════════

@dataclass
class Claim:
    name: str; domain: str; invariant_form: str; description: str
    quantities: dict[str, str]; generator: callable

def mk(fn, qd, **kw):
    """Helper: make claim generator."""
    def gen(rng):
        return gen_obs(qd, fn, noise=0.0003, rng=rng, **kw)
    return gen

CLAIMS = [
    Claim("Boyle", "Thermo", "P*V", "P*V = const",
          {"P":"Pressure","V":"Volume","m":"Mass","d":"Length"},
          mk(inv_product("P","V"), {"P":"Pressure","V":"Volume","m":"Mass","d":"Length"})),
    Claim("Charles", "Thermo", "V/T", "V/T = const",
          {"V":"Volume","T":"Scalar","P":"Pressure","m":"Mass"},
          mk(inv_ratio("V","T"), {"V":"Volume","T":"Scalar","P":"Pressure","m":"Mass"})),
    Claim("Gay-Lussac", "Thermo", "P/T", "P/T = const",
          {"P":"Pressure","T":"Scalar","V":"Volume","m":"Mass"},
          mk(inv_ratio("P","T"), {"P":"Pressure","T":"Scalar","V":"Volume","m":"Mass"})),
    Claim("Kepler", "Mechanics", "(T^2)/(a^3)", "T^2/a^3 = const",
          {"T":"Time","a":"Length","M":"Mass","v":"Velocity"},
          mk(inv_power_ratio("T","a",2,3), {"T":"Time","a":"Length","M":"Mass","v":"Velocity"})),
    Claim("Pendulum", "Mechanics", "(T^2)/L", "T^2/L = const",
          {"T":"Time","L":"Length","g":"Accel","m":"Mass"},
          mk(inv_power_ratio("T","L",2,1), {"T":"Time","L":"Length","g":"Accel","m":"Mass"})),
    Claim("Newton2", "Mechanics", "F/a", "F/a = const",
          {"F":"Force","a":"Accel","m":"Mass","t":"Time"},
          mk(inv_ratio("F","a"), {"F":"Force","a":"Accel","m":"Mass","t":"Time"})),
    Claim("Coulomb", "EM", "F*(r^2)", "F*r^2 = const",
          {"F":"Force","r":"Length","q1":"Scalar","q2":"Scalar"},
          mk(inv_product_sq("F","r"), {"F":"Force","r":"Length","q1":"Scalar","q2":"Scalar"})),
    Claim("LC", "EM", "(w^2)*L*C", "w^2*L*C = const",
          {"w":"Scalar","L":"Energy","C":"Scalar","R":"Scalar"},
          mk(inv_triple("w","L","C",2,1,1), {"w":"Scalar","L":"Energy","C":"Scalar","R":"Scalar"})),
    Claim("EscapeV", "Grav", "(v^2)*r", "v^2*r = const",
          {"v":"Velocity","r":"Length","M":"Mass","t":"Time"},
          mk(lambda v,K,r: inv_power_ratio("v","r",2,-1)(v,K,r),
             {"v":"Velocity","r":"Length","M":"Mass","t":"Time"})),
    Claim("Wave", "Waves", "f*L", "f*lambda = const",
          {"f":"Scalar","L":"Length","v":"Velocity","A":"Length"},
          mk(inv_product("f","L"), {"f":"Scalar","L":"Length","v":"Velocity","A":"Length"})),
    Claim("Ohm", "Circuits", "V/I", "V/I = const",
          {"V":"Energy","I":"Scalar","R":"Scalar","T":"Scalar"},
          mk(inv_ratio("V","I"), {"V":"Energy","I":"Scalar","R":"Scalar","T":"Scalar"})),
    Claim("Kinetic", "Mechanics", "K/(v^2)", "K/v^2 = const",
          {"K":"Energy","v":"Velocity","m":"Mass","t":"Time"},
          mk(inv_power_ratio("K","v",1,2), {"K":"Energy","v":"Velocity","m":"Mass","t":"Time"})),
]

# ═══════════════════════ Runner ═══════════════════════

def normalize(e): return e.replace(" ","").replace("**","^")

def run(seeds=None):
    if seeds is None: seeds = [42, 1039, 2036]
    T = 0.90
    print("="*70); print("GENERALIZED TEST — 12 claims, 7 domains"); print("="*70)
    results = []
    for i, c in enumerate(CLAIMS):
        print(f"\n{'─'*70}\n[{i+1:2d}/12] [{c.domain:15s}] {c.name}")
        print(f"  Expected: {c.description}  |  Form: {c.invariant_form}")
        sr = []
        for seed in seeds:
            rng = random.Random(seed); obs = c.generator(rng)
            qd = {}
            for o in obs:
                for qn, qs in o.quantities.items():
                    if qn not in qd:
                        try: qd[qn] = Dimension.named(qs)
                        except: qd[qn] = Dimension.scalar()
            t0 = time.time()
            d = auto_discover(quantities=qd, observations=obs,
                              known_invariant=None, discovery_threshold=T)
            t = time.time()-t0
            ok = d.score >= T; ex = normalize(d.expression) == normalize(c.invariant_form)
            sr.append((seed, d.expression, d.score, ok, ex, t))
            st = "EXACT" if ex else ("PASS" if ok else "FAIL")
            print(f"  seed={seed}: {st:5s}  {d.score:.4f}  {d.expression[:45]:45s}  {t:.1f}s")
        scores = [s[2] for s in sr]; mn = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores)>1 else 0
        pr = sum(1 for s in sr if s[3])/len(sr)
        er = sum(1 for s in sr if s[4])/len(sr)
        v = "VERIFIED" if pr >= 0.6 else "FAILED"
        exps = ", ".join(s[1][:25] for s in sr)
        print(f"  → {v}  mean={mn:.4f}±{std:.3f}  pass={pr:.0%}  exact={er:.0%}  top: {exps}")
        results.append({"c":c,"v":v=="VERIFIED","pr":pr,"er":er,"mn":mn,"best":[s[1] for s in sr if s[3]]})
    # Scorecard
    print(f"\n{'='*70}\nSCORECARD\n{'='*70}")
    vf = [r for r in results if r["v"]]; fl = [r for r in results if not r["v"]]
    print(f"  Verified: {len(vf)}/{len(results)}  Failed: {len(fl)}/{len(results)}\n")
    for r in vf:
        c=r["c"]; b=r["best"][0] if r["best"] else "???"
        print(f"    ✓ [{c.domain:15s}] {c.name:15s} pass={r['pr']:.0%} got={b}")
    for r in fl:
        c=r["c"]; b=r["best"][0] if r["best"] else "nothing"
        print(f"    ✗ [{c.domain:15s}] {c.name:15s} pass={r['pr']:.0%} got={b}")
    print(f"\n  TOTAL: {len(vf)}/{len(results)} ({len(vf)/len(results)*100:.0f}%)")
    return results

if __name__ == "__main__":
    run()
