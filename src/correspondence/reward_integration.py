"""Correspondence-layer reward integration for the GRPO training loop.

Wires the frontier map (Phase 2.5) and failure coordinates (Phase 2.7)
into the explorer trainer's reward computation.

This is what makes the compass actually steer — without this module,
the frontier map and failure points are just data structures that
nothing uses.

Architecture:
    Proof result → classify into frontier zone → zone reward multiplier
    Proof result → evaluate against failure points → bonus/penalty
    → modified_reward = base_reward * zone_multiplier + failure_modifier

Usage in explorer_trainer.py:
    from src.correspondence.reward_integration import CorrespondenceRewardModifier

    modifier = CorrespondenceRewardModifier(frontier_map, failure_coords)
    modified_rewards = modifier.apply(rewards, proofs, theorem_statements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from src.correspondence.frontier import (
    FrontierMap,
    ZoneType,
    load_frontier_map,
)
from src.correspondence.failure_points import (
    FailureCoordinateSystem,
    FailurePoint,
    FailureSeverity,
    load_failure_coordinates,
)


@dataclass
class CorrespondenceRewardModifier:
    """Modifies proof rewards based on frontier map and failure coordinates.

    Three mechanisms:
    1. Zone multiplier: proofs in breakdown zones get higher rewards
    2. Failure resolution bonus: proofs that resolve known failures get bonus
    3. Failure reproduction penalty: proofs that reproduce known failures get penalized

    The combined effect pulls the explorer toward the breakdown zone — the
    frontier where new physics must exist — while punishing it for simply
    copying known broken theories.
    """

    frontier_map: FrontierMap
    failure_coords: FailureCoordinateSystem

    # Scaling factors to calibrate correspondence reward against
    # the base reward (which is typically 0-2 range from proof checker)
    zone_multiplier_scale: float = 1.0    # How strongly zones affect reward
    failure_bonus_scale: float = 0.5       # Scale of failure resolution bonus
    failure_penalty_scale: float = 0.3     # Scale of failure reproduction penalty

    # Tracking
    stats: dict = field(default_factory=lambda: {
        "total_modifications": 0,
        "breakdown_hits": 0,
        "established_hits": 0,
        "uncertain_hits": 0,
        "failure_resolutions": 0,
        "failure_reproductions": 0,
        "zone_distribution": {},
    })

    def apply(
        self,
        rewards: "torch.Tensor",
        proofs: list[str],
        theorem_statements: list[str],
        energy_scale: float | None = None,
        gauge_group: str | None = None,
    ) -> "torch.Tensor":
        """Apply correspondence-layer modifications to a batch of rewards.

        Args:
            rewards: [batch_size] tensor of base rewards from proof checker.
            proofs: List of proof strings.
            theorem_statements: List of theorem statements.
            energy_scale: Optional energy scale hint (GeV).
            gauge_group: Optional gauge group hint.

        Returns:
            Modified rewards tensor, same shape as input.
        """
        import torch
        modified = rewards.clone()
        batch_size = rewards.size(0)

        for i in range(batch_size):
            base_r = rewards[i].item()
            proof = proofs[i] if i < len(proofs) else ""
            statement = theorem_statements[i] if i < len(theorem_statements) else ""

            # Skip failed proofs (reward < 1.0 = not a valid proof)
            # Only modify successful proofs — failed ones are already penalized
            if base_r < 1.0:
                continue

            # ---- Step 1: Classify the proof's mathematical territory ----
            # Determine the frontier zone based on what the theorem is about
            zone = self._classify_proof(statement, proof, energy_scale, gauge_group)
            zone_multiplier = zone.reward_multiplier if zone else 1.0

            # Track zone distribution
            zone_name = zone.name if zone else "unclassified"
            self.stats["zone_distribution"][zone_name] = (
                self.stats["zone_distribution"].get(zone_name, 0) + 1
            )

            if zone and zone.zone_type == ZoneType.BREAKDOWN:
                self.stats["breakdown_hits"] += 1
            elif zone and zone.zone_type == ZoneType.ESTABLISHED:
                self.stats["established_hits"] += 1
            elif zone and zone.zone_type == ZoneType.UNCERTAIN:
                self.stats["uncertain_hits"] += 1

            # ---- Step 2: Evaluate against failure coordinates ----
            resolved, reproduced = self._check_failure_points(statement, proof)

            if resolved:
                self.stats["failure_resolutions"] += 1
            if reproduced:
                self.stats["failure_reproductions"] += 1

            failure_modifier = self.failure_coords.estimate_reward_modifier(
                statement, resolved, reproduced
            )

            # ---- Step 3: Combine ----
            # Zone multiplier stretches/shrinks the base reward
            # Failure modifier is additive (bonus for resolving, penalty for reproducing)
            zone_effect = 1.0 + self.zone_multiplier_scale * (zone_multiplier - 1.0)
            failure_effect = self.failure_bonus_scale * max(0, failure_modifier) \
                             - self.failure_penalty_scale * max(0, -failure_modifier)

            modified[i] = base_r * zone_effect + failure_effect
            self.stats["total_modifications"] += 1

        return modified

    def _classify_proof(
        self,
        statement: str,
        proof: str,
        energy_scale: float | None,
        gauge_group: str | None,
    ) -> "FrontierZone | None":
        """Classify a proof/theorem into a frontier zone.

        Strategy (in priority order):
        1. Keyword classification with specific zone name matching — most reliable
        2. Condition-based classification with explicit (non-heuristic) conditions
        3. Broad keyword fallback by zone type
        """
        statement_lower = statement.lower()
        proof_lower = proof.lower()
        combined = statement_lower + " " + proof_lower

        # ── Step 1: Keyword classification (try first — most specific) ──
        # The _keyword_classify method checks specific zone-keyword mappings
        # first, then falls back to broad type-based matching. A match from
        # the specific zone keywords is the strongest signal we have.
        keyword_zone = self._keyword_classify(combined)
        if keyword_zone is not None:
            # Check if any of this zone's specific keywords appear in the text
            # (confirms the match is from specific keywords, not broad fallback)
            zone_specific_kw = {
                "planck_breakdown": ["planck scale", "planck energy", "planck mass"],
                "black_hole_singularity": ["black hole", "schwarzschild", "event horizon"],
                "big_bang_singularity": ["big bang", "initial singularity", "cosmological singularity"],
                "qft_divergence": ["landau pole", "uv divergence", "qft divergence"],
                "gr_qft_incompatibility": ["gr and qft", "general relativity and quantum",
                                           "gr-qft", "mutually incompatible"],
                "dark_matter": ["dark matter", "wimp", "galactic rotation", "missing mass"],
                "dark_energy": ["dark energy", "cosmological constant problem", "vacuum energy"],
                "inflation": ["inflation", "inflaton", "slow-roll", "primordial power"],
                "quantum_gravity": ["quantum gravity", "quantum geometry", "loop quantum gravity"],
                "standard_model": ["standard model", "electroweak", "higgs mechanism"],
                "gr_classical": ["general relativity", "einstein field", "schwarzschild"],
                "qed": ["quantum electrodynamics", "qed", "u(1) gauge"],
                "thermodynamics": ["thermodynamics", "second law", "carnot", "entropy"],
            }
            zone_kw_list = zone_specific_kw.get(keyword_zone.name, [])
            has_specific_match = any(kw in combined for kw in zone_kw_list)
            if has_specific_match:
                return keyword_zone

        # ── Step 2: Build explicit conditions ──
        # Only include conditions that come from actual parameters,
        # not from heuristic keyword extraction (that's what keyword classify is for)
        conditions = {}

        if energy_scale is not None:
            conditions["energy_scale"] = energy_scale
        if gauge_group is not None:
            conditions["gauge_group"] = gauge_group

        # Extract energy scale from explicit GeV/TeV mentions
        import re
        if "tev" in combined:
            conditions["energy_scale"] = 1e3  # ~1 TeV
        elif "gev" in combined:
            match = re.search(r'(\d+\.?\d*)\s*gev', combined)
            if match:
                conditions["energy_scale"] = float(match.group(1))

        # Only set Planck-scale energy if Planck is explicitly mentioned
        if any(w in combined for w in ("planck scale", "planck mass", "planck energy",
                                        "planck length", "m_planck", "e_planck")):
            conditions["energy_scale"] = 1.22e19

        # Only set curvature=singularity if specifically about singularities
        if any(w in combined for w in ("singularity at", "singularity in",
                                        "curvature singularity", "singularity theorem")):
            conditions["curvature"] = "singularity"

        # If we have explicit conditions, try condition-based classification
        if conditions:
            zone = self.frontier_map.classify(**conditions)
            if zone is not None and zone.name != "unknown_territory":
                return zone

        # ── Step 3: Return keyword match (even if not exact zone name) ──
        if keyword_zone is not None:
            return keyword_zone

        # ── Step 4: Default to uncertain ──
        zones = self.frontier_map.get_zones_by_type(ZoneType.UNCERTAIN)
        return zones[0] if zones else None

    def _keyword_classify(self, text: str) -> "FrontierZone | None":
        """Classification based on keyword content in the text.

        Scores each zone by how many of its specific keywords appear in the
        text, then returns the highest-scoring zone. Falls back to broad
        type-based matching only when no specific keywords match.

        Returns the best-matching zone, or None if no keywords found.
        """
        # ── Specific zone keyword scoring ──
        zone_keywords = {
            # Breakdown zones
            "planck_breakdown": ["planck scale", "planck energy", "planck mass",
                                 "planck length", "m_planck", "e_planck",
                                 "planck breakdown"],
            "black_hole_singularity": ["black hole", "schwarzschild", "kerr black",
                                       "event horizon", "hawking radiation",
                                       "black hole information", "ringdown"],
            "big_bang_singularity": ["big bang", "initial singularity",
                                     "cosmological singularity"],
            "qft_divergence": ["landau pole", "uv divergence", "qft divergence",
                               "ultraviolet divergence", "qft uv"],
            "gr_qft_incompatibility": ["gr and qft", "general relativity and quantum",
                                       "gr-qft", "incompatibility between gr",
                                       "mutually incompatible", "incompatible",
                                       "general relativity and quantum field theory",
                                       "gr and quantum field theory"],
            # Uncertain zones
            "quantum_gravity": ["quantum gravity", "quantum geometry",
                                "spacetime foam", "loop quantum gravity",
                                "causal dynamical triangulation"],
            "dark_matter": ["dark matter", "wimp", "axion", "galactic rotation",
                            "missing mass", "non-baryonic", "dm particle"],
            "dark_energy": ["dark energy", "cosmological constant problem",
                            "lambda cdm", "vacuum energy", "quintessence"],
            "inflation": ["inflation", "inflaton", "slow-roll", "primordial",
                          "inflationary", "ekpyrotic"],
            # Established zones
            "standard_model": ["standard model", "su(3)×su(2)×u(1)", "electroweak",
                               "higgs mechanism", "yang-mills", "glashow",
                               "weinberg", "salam"],
            "gr_classical": ["general relativity", "einstein field equation",
                             "einstein's theory", "friedmann equation"],
            "qed": ["quantum electrodynamics", "qed", "dirac equation",
                    "fermionic", "u(1) gauge", "photon propagator"],
            "thermodynamics": ["thermodynamics", "second law", "carnot",
                               "heat engine", "free energy",
                               "statistical mechanics", "partition function"],
        }

        # Score each zone by keyword matches
        zone_scores = {}
        for zone_name, keywords in zone_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                zone_scores[zone_name] = score

        if zone_scores:
            # Return the zone with the most keyword matches
            best_zone_name = max(zone_scores, key=zone_scores.get)
            for z in self.frontier_map.zones:
                if z.name == best_zone_name:
                    return z

        # ── Broad type matches (lower confidence) ──
        # Only reached when no specific zone keywords matched

        # Breakdown zone generic keywords
        if any(w in text for w in ("singularity", "breakdown", "non-renormalizable",
                                     "uv completion", "ultraviolet complete")):
            zones = self.frontier_map.get_zones_by_type(ZoneType.BREAKDOWN)
            if zones:
                for z in zones:
                    if "singularity" in text and "singularity" in z.name:
                        return z
                return zones[0]

        # Established zone generic keywords
        if any(w in text for w in ("standard model", "einstein", "qed",
                                     "maxwell", "newtonian", "classical")):
            zones = self.frontier_map.get_zones_by_type(ZoneType.ESTABLISHED)
            if zones:
                for z in zones:
                    if z.name.replace("_", " ") in text:
                        return z
                return zones[0]

        # Uncertain zone generic keywords
        if any(w in text for w in ("quantum gravity", "dark", "inflation",
                                     "beyond standard model", "new physics")):
            zones = self.frontier_map.get_zones_by_type(ZoneType.UNCERTAIN)
            if zones:
                for z in zones:
                    if z.name.replace("_", " ") in text:
                        return z
                return zones[0]

        return None

    def _check_failure_points(
        self, statement: str, proof: str
    ) -> tuple[set[str], set[str]]:
        """Check a proof against known failure points.

        Returns:
            (resolved_failure_names, reproduced_failure_names)
        """
        resolved = set()
        reproduced = set()

        combined = (statement + " " + proof).lower()
        words = set(combined.split())

        for fp in self.failure_coords.failure_points:
            fp_name_lower = fp.name.lower().replace("_", " ")
            fp_desc_lower = fp.description.lower()

            # Resolution keywords — must match as whole phrases in text
            resolved_keywords = [
                "resolve", "resolves", "resolved", "resolution",
                "solution", "solves", "solved",
                "avoids", "avoided", "regularizes", "regularized",
                "uv complete", "uv-complete",
                "non-singular", "nonsingular", "bounce",
                "no singularity", "explain", "explains", "explained",
            ]

            # ── Check if the failure point is explicitly mentioned ──
            # High-confidence: name match or failing_theory match
            fp_named = (
                fp_name_lower in combined
                or any(
                    thm.lower().replace("_", " ") in combined
                    for thm in fp.related_theorems
                )
                or any(
                    theory.lower() in combined
                    for theory in fp.failing_theories
                )
                or fp.regime.value.replace("_", " ") in combined
            )

            # Medium-confidence: description keyword overlap (≥5 hits)
            fp_described = False
            if not fp_named:
                desc_words = [
                    w for w in fp_desc_lower.split()
                    if len(w) > 4
                    and w not in ("their", "these", "those", "which", "where",
                                  "about", "must", "with")
                ]
                hits = sum(1 for w in desc_words if w in combined)
                fp_described = hits >= 5  # Raised from 3 → 5 for fewer false positives

            fp_mentioned = fp_named or fp_described

            if not fp_mentioned:
                continue

            # ── Determine resolution vs reproduction ──
            is_resolving = any(kw in combined for kw in resolved_keywords)
            # Also: standalone "finite" without "infinite" = resolution signal
            if not is_resolving and "finite" in words and "infinite" not in words:
                is_resolving = True

            is_reproducing = (
                fp.severity == FailureSeverity.CATASTROPHIC
                and any(
                    w in words
                    for w in ("singularity", "divergence", "divergences",
                              "infinite", "infinity", "infinities")
                )
            )

            # Resolution requires explicit naming (not just description overlap)
            if is_resolving and fp_named:
                resolved.add(fp.name)
            elif is_reproducing and fp_named:
                reproduced.add(fp.name)

        return resolved, reproduced

    def get_stats(self) -> dict:
        """Return current modification statistics."""
        return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        for key in self.stats:
            if isinstance(self.stats[key], dict):
                self.stats[key] = {}
            else:
                self.stats[key] = 0


# ---------------------------------------------------------------------------
# Era-gated reward modifier (temporal gating)
# ---------------------------------------------------------------------------


class EraGatedRewardModifier:
    """Wraps CorrespondenceRewardModifier with passive era-gated discovery tracking.

    When training with a historical era cutoff (e.g., ≤1904), this wrapper:
    1. Applies the standard correspondence modifier (zone multipliers + failure bonuses)
    2. PASSIVELY scans proofs for "future" physics concepts unknown at the cutoff
    3. Reports discoveries in training logs — but does NOT modify rewards

    This is the honest temporal gating test: the explorer gets NO hints about
    what physics comes next. It only sees pre-era knowledge through the
    correspondence layer. If it spontaneously generates proofs touching
    post-era concepts (Lorentz transformations, wave functions, etc.),
    that's genuine evidence the architecture produces real discovery.

    The era tracker is a MONITOR, not a teacher.
    """

    def __init__(
        self,
        base_modifier: CorrespondenceRewardModifier,
        era_tracker: "EraTracker | None" = None,
    ):
        self.base = base_modifier
        self.tracker = era_tracker

    def apply(
        self,
        rewards: "torch.Tensor",
        proofs: list[str],
        theorem_statements: list[str],
        energy_scale: float | None = None,
        gauge_group: str | None = None,
    ) -> "torch.Tensor":
        """Apply correspondence modifier. Passively track era discoveries.

        Rewards are NOT modified by era tracking — discoveries are
        observed, not incentivized. This prevents inflated results.
        """
        # Step 1: Standard correspondence modifier (the only reward signal)
        modified = self.base.apply(
            rewards, proofs, theorem_statements,
            energy_scale=energy_scale,
            gauge_group=gauge_group,
        )

        # Step 2: Passive discovery tracking (no reward modification)
        if self.tracker is not None:
            self.tracker.scan_batch(proofs, theorem_statements)

        return modified

    def get_stats(self) -> dict:
        """Return combined stats from correspondence modifier + era tracker."""
        stats = self.base.get_stats()
        if self.tracker is not None:
            stats["era"] = self.tracker.era_name
            stats["era_cutoff_year"] = self.tracker.cutoff_year
            stats["era_discovery_rate"] = self.tracker.get_discovery_rate()
            stats["era_top_discoveries"] = self.tracker.get_top_discoveries(5)
            stats["era_total_discoveries"] = self.tracker.total_discoveries
        return stats

    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        self.base.reset_stats()
        if self.tracker is not None:
            self.tracker.reset_counts()

    @property
    def frontier_map(self):
        return self.base.frontier_map

    @property
    def failure_coords(self):
        return self.base.failure_coords


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_modifier(
    frontier_path: str = "configs/frontier_map.yaml",
    failure_path: str = "configs/failure_coordinates.yaml",
) -> CorrespondenceRewardModifier:
    """Create a reward modifier from the default frontier map and failure coordinates.

    This is the standard setup — loads the canonical physics zones and
    failure coordinates from YAML and builds the modifier.

    Returns:
        CorrespondenceRewardModifier ready to plug into the training loop.
    """
    frontier_map = load_frontier_map(frontier_path)
    failure_coords = load_failure_coordinates(failure_path)

    return CorrespondenceRewardModifier(
        frontier_map=frontier_map,
        failure_coords=failure_coords,
    )
