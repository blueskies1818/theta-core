# Roadblock #2: Null-Hypothesis Baseline

## Goal
System must know when it found nothing. Currently, if fed data with no
conserved quantity, it still returns a high-scoring coincidence. This is
fundamentally dishonest — an independent reviewer would flag it immediately.

## Approach

### Null-hypothesis check
After auto_discover finds a candidate expression:
1. Generate N = 50 random expressions using the same quantity names
2. Score all random expressions against the data
3. Get the 95th percentile of random scores
4. If discovered score < random_95th_percentile + margin (0.05), reject
   the discovery and return empty result

### Random expression generation
Uses the same structural templates the system already has (products, ratios,
powers, etc.) but with SHUFFLED variable assignments. This ensures random
expressions use the same structural vocabulary as real discoveries, making
the comparison fair.

### Integration point
In auto_discover, after all pipelines complete and best_result is chosen:
- If best_result.is_discovery, run null-hypothesis check
- If it fails, return empty SearchResult with score=0.0

### Edge cases
- If < 3 quantity names, null check is unreliable → accept discovery
- If no discovery found at all (score < threshold), skip check
- If too few observations for meaningful statistics (< 4), skip check

## Success criteria
- Feeding null data (random unrelated measurements) → system returns empty
- Current 28 claims still pass (their scores are well above random baseline)
- No performance regression on verify scripts

## Files
- src/physics/search.py — _null_hypothesis_check function, integration in auto_discover
