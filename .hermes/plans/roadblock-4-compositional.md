# Roadblock #4: Compositional 4+ Variable Search

## Goal
Discover invariants involving 4+ interacting variables. Simple search
only handles up to 3-var templates. Beam search composes bottom-up and
should handle 4+ vars — but does it actually work?

## Test Plan
Generate synthetic data for 4-var invariants that simple search CANNOT
find and test whether the current beam search pipeline discovers them.

### Test cases
1. a*b + c*d (sum of products — e.g., energy = kinetic + potential terms)
2. (a*b)/(c*d) (ratio of products — e.g., dimensionless ratios)
3. a*b*c*d (product of all four)
4. (a+b)/(c+d) (ratio of sums)

### Success criteria
- If beam search finds ≥3/4 → roadblock #4 is solved by #1 changes
- If beam search finds <3/4 → need targeted fix

## Approach (if needed)
If beam search fails, the fix is a guided compositional search:
1. Run simple search to find 2-var sub-invariants
2. Compose pairs of 2-var expressions with +, -, *, /
3. Score composed expressions against data
4. Cross-validate as usual

This is O(k²) where k is number of discovered 2-var expressions,
much smaller than O(n⁴) for 4-var enumeration.
