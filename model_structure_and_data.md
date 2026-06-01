# Model Structure and Data Use
## Autonomous Mathematical Physics AI — Detailed Technical Specification

*Companion document to: Autonomous Mathematical Physics AI — System Design*

---

## Document Purpose

The system design document covers what the system does and why. This document covers how — specifically the internal structure of each model component, how experimental data from any domain is ingested, formatted, and used, and how all of these pieces connect into a functioning training loop. The goal is enough specificity that an engineer or physicist reading this can identify the concrete work required at each stage.

---

## Part 1 — Model Structure

### 1.1 The Three-Component Architecture

The system is not one model. It is three models with distinct architectures connected by well-defined interfaces. This is a deliberate design choice — no single architecture is suited to all three jobs, and monolithic systems trying to do everything at once sacrifice depth in every dimension to achieve breadth in none.

The three components are:

- **The Mathematical Explorer** — proposes candidate mathematical structures and proof steps
- **The Physical Prediction Scorer** — evaluates candidate structures against experimental observation data
- **The Translation Layer** — converts formal outputs into natural language for human physicists

Data flows in one direction through the system during a training iteration: Explorer → Proof Checker → Scorer → Reward Signal → Explorer update. The Translation Layer operates asynchronously, converting flagged outputs for human review independently of the training loop.

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │  Mathematical│     │    Lean 4    │                  │
│  │   Explorer   │────▶│    Proof     │                  │
│  │  (GNN + MCTS)│     │   Checker   │                  │
│  └──────┬───────┘     └──────┬───────┘                  │
│         │                   │ verified structures        │
│         │ reward signal     ▼                           │
│         │            ┌──────────────┐                  │
│         └────────────│   Physical   │                  │
│                      │  Prediction  │◀── experimental  │
│                      │   Scorer     │    data corpus   │
│                      └──────┬───────┘                  │
│                             │ flagged anomalies         │
└─────────────────────────────┼───────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Translation     │     human
                    │  Layer (LLM)     │────▶physicists
                    └──────────────────┘
```

---

### 1.2 Component 1 — The Mathematical Explorer

#### Architecture: Graph Neural Network with Monte Carlo Tree Search

The explorer is the core of the system and its architecture is the most important design decision in the whole project. It is emphatically not a standard transformer.

The reason is structural. A transformer processes sequences — it is optimized for finding patterns in ordered token streams. Formal mathematics has sequential surface syntax but its deep structure is a directed acyclic graph. Theorems depend on other theorems. Definitions reference other definitions. Structures inherit from other structures. The logical relationships between mathematical objects form a web, not a line.

A graph neural network operates natively on this structure. Nodes in the graph are mathematical objects — theorems, definitions, structures, propositions. Edges are logical relationships — implies, depends on, is a special case of, is equivalent to given. The GNN learns representations of mathematical objects that encode their position in this relational web, not just their syntactic form.

The Monte Carlo Tree Search component handles the exploration problem. At each step the explorer must decide: given the current state of a proof or candidate structure, what is the most promising next move? MCTS builds a search tree of possible moves, uses the GNN to evaluate promising branches, and allocates more search effort toward regions of the tree that appear productive. This is exactly what AlphaGo Zero used — the GNN is the evaluation function, MCTS is the search strategy.

#### What A Move Looks Like

In Go, a move is placing a stone at a coordinate. In formal mathematics, a move is a proof tactic applied to the current proof state. In Lean 4, tactics look like this:

```lean
-- Current proof state: goal is to show E^2 = (pc)^2 + (mc^2)^2
-- Explorer selects tactic: ring
-- Proof checker evaluates: valid, goal closed

example (E p m c : ℝ) : E^2 = (p*c)^2 + (m*c^2)^2 := by
  ring  -- ← this is the move
```

More complex moves involve introducing hypotheses, applying known lemmas from the Mathlib library, constructing new mathematical objects, or proposing new definitions. The explorer learns which moves tend to make progress in which situations — the same way AlphaGo Zero learned which moves tend to improve board position.

#### The Exploration Frontier — Where The Explorer Searches

The explorer doesn't search all of mathematics randomly. It maintains a formal frontier map — a machine-readable description of where current theoretical physics has solid footing, where it becomes uncertain, and where it breaks entirely. This map has three zones:

**Established zone** — theorems and structures that are formally proven and physically confirmed. GR in the weak field limit. QFT at energies below the electroweak scale. These are training anchors, not exploration targets. The explorer learns from them but is not rewarded for reproving them.

**Uncertain zone** — regions where theoretical frameworks exist but make predictions with limited experimental confirmation, or where multiple competing frameworks make different predictions. Early universe cosmology. Quantum gravity phenomenology. Dark matter interaction structure. These are productive exploration territory — candidate structures here can be scored against available data.

**Breakdown zone** — mathematically precise conditions where current theories produce infinities, non-renormalizable divergences, or singularity theorems. Planck scale interactions. Black hole interiors. These are the primary targets. A candidate structure that remains consistent and finite in the breakdown zone while reproducing established results in the established zone is the system's primary objective.

The reward contribution from each zone is weighted differently. Finding something consistent in the breakdown zone contributes more to reward than finding something in the uncertain zone, which contributes more than reproving something in the established zone. This creates a gradient that pulls the explorer toward the frontier.

#### Parameter Scale and Justification

**Target: 1–7 billion parameters, most likely 3 billion for first serious training run.**

Justification from empirical precedent: AlphaProof operates at exactly 3 billion parameters and solves IMO-level mathematical problems — harder than most research mathematicians encounter in daily work. The mathematical explorer's task is more specialized than AlphaProof's (narrower domain, formal physics rather than all mathematics) but involves larger and more complex structures. 3 billion is a reasonable starting anchor with a plan to measure the scaling curve empirically before committing to larger runs.

Why not larger: The explorer's job is learned heuristic search, not knowledge storage. Knowledge lives in the Mathlib library, which is external to the model. The model needs enough capacity to learn which search strategies work in which situations — this is a pattern recognition problem over a constrained formal domain, not the open-ended knowledge compression problem that drives frontier language models to hundreds of billions of parameters.

Why not smaller: Structures at the GR-QFT interface involve genuinely complex mathematical objects — fiber bundles, spinor fields, connections on principal bundles, non-abelian gauge theories. Representing these with sufficient fidelity to make useful predictions requires meaningful model capacity. Below roughly 1 billion parameters, performance on complex formal structures degrades significantly.

---

### 1.3 Component 2 — The Physical Prediction Scorer

#### Architecture: Multimodal Transformer with Domain-Specific Encoders

The scorer's job is to take a formally verified mathematical structure and ask: what does this structure predict about physical measurements, and how close are those predictions to what was actually observed?

This requires two distinct capabilities. First, interpreting a formal mathematical structure as a physical theory — understanding what its equations say about measurable quantities. Second, comparing those predictions against heterogeneous physical data across wildly different experimental modalities.

The architecture reflects this. A domain-specific encoder for each measurement modality converts raw physical data into a common representation space. A large transformer then operates in this common space, learning to compare mathematical structure predictions against encoded physical observations.

#### Domain-Specific Encoders

Each encoder is specialized for its data modality and is effectively a separate trained model that feeds into the shared transformer:

**Time Series Encoder** — handles gravitational wave strain data, pulsar timing arrays, particle decay curves. Uses 1D convolutional layers followed by attention mechanisms. Performs Fourier analysis internally to extract frequency content. Output: fixed-dimension vector encoding the temporal structure and spectral properties of the signal.

**Spatial Field Encoder** — handles cosmic microwave background maps, galaxy survey data, detector images. Uses spherical harmonic decomposition for sky maps, standard 2D/3D convolutions for planar detector data. Output: fixed-dimension vector encoding spatial correlation structure and power spectrum properties.

**Spectroscopic Encoder** — handles atomic and molecular spectra, absorption lines, emission profiles. Uses learned peak detection and line identification. Particularly important because spectral line positions encode direct information about quantum energy levels, connecting to QFT predictions. Output: fixed-dimension vector encoding line positions, widths, and relative intensities.

**Discrete Event Encoder** — handles particle collision records, cosmic ray events, neutrino detections. These are not continuous signals but catalogs of discrete events with associated properties. Uses set-based neural network architecture (permutation invariant, order doesn't matter) to encode the statistical distribution of events. Output: fixed-dimension vector encoding event rate, energy distribution, and conservation law verification results.

**Thermodynamic/Chemical Encoder** — handles titration curves, calorimetric measurements, phase diagrams, reaction rate data. Uses curve parameterization to extract critical points, inflection points, scaling exponents. Output: fixed-dimension vector encoding equilibrium structure and thermodynamic relationships.

All encoders map their respective data into the same dimensional space — this is the common representation the transformer operates on.

#### Parameter Scale and Justification

**Target: 10–30 billion parameters, initialized from existing pretrained scientific model.**

Why larger than the explorer: The scorer cannot offload verification to an external checker. It must learn the mapping from formal mathematical structure to physical prediction entirely in its weights, across heterogeneous modalities with no shortcuts. This is a genuine large-scale representation learning problem.

Why initialized from existing model: Scientific language models pretrained on physics and chemistry literature already have substantial implicit knowledge of how mathematical structures relate to physical phenomena. Fine-tuning from this starting point is dramatically more efficient than training from scratch. The fine-tuning task is well-defined — given a formal structure and experimental metadata, predict the measurement outcome — and can be trained with a clear automated loss signal.

---

### 1.4 Component 3 — The Translation Layer

#### Architecture: Fine-tuned Large Language Model

The translation layer is the most conventional component. It takes formal mathematical outputs — verified structures, proof trees, anomaly flags with their formal property descriptions — and produces natural language that human physicists can read and act on.

This is a well-defined translation task. Existing large language models are already capable of explaining mathematics in natural language. The required fine-tuning teaches the model the specific translation task: given a Lean 4 formal object with associated physical metadata, produce a description that accurately conveys the physical implications to a domain physicist.

#### Automated Translation Verification

The translation layer's outputs are verifiable in a way that normal language model outputs are not. When the translator claims "this structure predicts a force-carrying boson with spin-1 and zero rest mass coupling to the electromagnetic field," that claim can be checked formally. The proof checker can verify whether the formal structure actually has those mathematical properties.

This enables an automated correctness signal for training the translation layer — every translation claim that can be formalized gets verified, and the verification result provides a training signal. Translations that make claims the formal structure doesn't support are penalized. This is a substantially cleaner training setup than most language model fine-tuning, where correctness signals are expensive or subjective.

#### Experimental Proposal Generation

For anomalous flagged solutions — solution families with no matching known physical entity — the translation layer has a secondary task: generating experimental proposals. Given the formal properties of an unmatched solution, it produces natural language descriptions of what experimental conditions would make the entity detectable if it is real.

This works well for solutions near known physics — entities similar to known particles but with different quantum numbers, or similar to known fields but with different coupling constants. For these, the translator has template reasoning available from the literature on experimental proposals for predicted-but-undetected particles.

This works poorly for solutions that are radically unlike existing physics. If the formal structure implies an entity that doesn't interact via any known force or exists in additional spatial dimensions, the translator has no template to reason from. The failure mode in this case is the translator producing plausible-sounding but physically meaningless proposals. The mitigation is a confidence score on experimental proposals — the translator expresses uncertainty when reasoning about entities far outside existing experimental physics vocabulary, flagging these for human physicist engagement rather than producing false confident proposals.

#### Parameter Scale

**Target: 7–70 billion parameters, initialized from existing frontier language model.**

The wide range reflects genuine uncertainty about how much capacity is needed for high-quality formal-to-natural translation of genuinely novel mathematical physics. Starting at the lower end and measuring quality is the right approach. The translation task doesn't require frontier model scale — but if the explorer is producing genuinely novel structures with no existing physical vocabulary, the translator may need substantial capacity to handle the conceptual bridging work.

---

## Part 2 — Data Architecture

### 2.1 The Core Problem

Experimental data across physics, chemistry, and related sciences shares almost nothing at the surface level. A gravitational wave strain time series and a pH titration curve look completely different, are measured in completely different units, operate at completely different physical scales, and were produced by completely different instruments for completely different theoretical purposes.

Naive approaches fail here. You cannot concatenate these into a single array and feed them to a model. You cannot train separate models for each experimental domain — the whole point is cross-domain physical reasoning. You need a data architecture that preserves the physical meaning of each measurement while abstracting away the measurement-specific format, so the model can find relationships across domains that human scientists working in silos might miss.

The solution is a two-layer architecture. Layer 1 converts heterogeneous raw data into a physically meaningful common format. Layer 2 extracts formal mathematical objects from that format. The model sees only Layer 2 outputs — never raw experimental data directly.

---

### 2.2 Layer 1 — Physical Encoding

Every experiment processed by the system passes through a standardized metadata schema before any numerical processing occurs. This schema captures everything about the experimental context that the model needs to correctly interpret the numbers.

#### The Metadata Schema

```json
{
  "experiment_id": "unique_identifier",
  "experiment_class": "one of: gravitational, electromagnetic, strong_force, 
                        weak_force, thermodynamic, quantum_chemical, 
                        cosmological, condensed_matter",
  "physical_domain": "which theoretical framework applies",
  "measurement_target": "what physical quantity is being observed",
  "physical_regime": {
    "energy_scale": "value in eV or joules",
    "length_scale": "value in meters",
    "time_scale": "value in seconds",
    "gravitational_strength": "ratio to flat spacetime",
    "quantum_effects": "significant | negligible | dominant"
  },
  "units": {
    "independent_variable": "unit string",
    "dependent_variable": "unit string"
  },
  "symmetries_preserved": ["list of symmetry group names"],
  "conservation_laws_verified": ["list of conservation laws confirmed in this experiment"],
  "known_systematic_errors": [
    {
      "source": "description of error source",
      "magnitude": "quantified uncertainty",
      "correction_applied": true
    }
  ],
  "theoretical_prediction": {
    "framework": "GR | QFT | QCD | standard_model | etc",
    "predicted_values": "formal expression or numerical array",
    "confidence": "estimated theoretical uncertainty"
  },
  "raw_data_location": "path or URI to raw measurement file",
  "preprocessing_pipeline": "which Layer 1 pipeline processes this data",
  "holdout_status": "training | validation | future_experiment"
}
```

This schema does something crucial: it encodes what a physicist knows when they look at a graph. The axis labels, the units, the relevant theoretical framework, the known noise sources, the symmetries that should be preserved — all of this interpretive context is made explicit and machine-readable before any numerical processing happens.

The physical regime fields are particularly important. They tell the model what region of the theoretical landscape this experiment is probing — which part of the candidate structure's predictions to compare against this data. A gravitational wave experiment sits in the weak-field GR regime. A proton collision event sits in the high-energy QCD regime. A titration sits in the quantum chemistry regime. These are different predictions from the same candidate structure, and the regime fields tell the scorer which ones to compute and compare.

#### Calibration and Uncertainty Quantification

Every real measurement has systematic errors — instrument drift, finite resolution, environmental interference. Uncalibrated data teaches the model to predict instrument artifacts rather than physical reality.

The solution is to represent every measurement not as a point value but as a probability distribution over possible true values, given the known instrument characteristics. This is standard experimental physics practice — propagation of uncertainties — formalized into the data pipeline.

A gravitational wave strain measurement at time t is not a single number h(t). It is a Gaussian distribution N(μ, σ²) where μ is the measured value and σ² encodes the combined uncertainty from shot noise, thermal noise, seismic noise, and calibration uncertainty. The model fits to these distributions, and a candidate structure's prediction is evaluated against the distribution rather than the point value.

This prevents reward hacking through systematic error exploitation — a structure cannot score well by predicting the drift pattern of a specific detector, because the uncertainty quantification has already absorbed that drift into the error bars.

---

### 2.3 Layer 2 — Mathematical Object Extraction

Layer 2 converts physically encoded data into formal mathematical objects. This is where the real abstraction happens. The model never sees raw numbers — it sees mathematical relationships extracted from those numbers.

Four classes of mathematical objects are extracted from every experiment regardless of domain:

#### Symmetries and Their Breaking

The system checks what mathematical transformations leave the data invariant. If a collision experiment shows that outcomes are the same regardless of which direction you orient the detector, that is spatial rotational symmetry — a formal mathematical statement about the SO(3) group. If it shows invariance under time shifts, that is time translation symmetry — U(1) in the relevant group.

More interesting is where symmetry breaks. The weak nuclear force breaks parity symmetry — it distinguishes left-handedness from right-handedness in a way other forces don't. This parity violation is a formal mathematical property, not a descriptive statement. It is represented as the absence of a specific group element from the symmetry group of the relevant interaction.

Symmetries and their breaking are extracted as group-theoretic objects — formal mathematical structures that are directly comparable across experimental domains. A candidate structure that predicts the wrong symmetry group for electroweak interactions is immediately incompatible with decades of particle physics measurements, and this incompatibility is detectable at the formal level without comparing raw numbers.

#### Conservation Laws and Violations

From the symmetry analysis, Noether's theorem provides the conserved quantities automatically: every continuous symmetry implies a conservation law. Energy conservation follows from time translation symmetry. Momentum conservation follows from spatial translation symmetry. Charge conservation follows from U(1) gauge symmetry.

These become hard constraints encoded as formal theorems. A candidate structure that breaks charge conservation is immediately incompatible with every experiment in the corpus where charge conservation has been verified — which is essentially all of them. This constraint propagates automatically through the formal system without requiring explicit comparison to each experiment individually.

Apparent violations are particularly interesting. If a measurement appears to violate a conservation law — like the apparent violation of energy conservation in early beta decay measurements, which led to the prediction of the neutrino — this is flagged as a high-priority anomaly. Either the measurement has an unaccounted systematic error, or the conservation law has an exception, or there is an undetected participant carrying the missing quantity. All three possibilities are interesting.

#### Scaling Relations

How do measured quantities relate to each other quantitatively? These relations are power laws and logarithmic laws — mathematical relationships that carry deep structural information.

Gravitational wave strain scales as 1/r with distance. This is not just a number — it is a statement about the dimensionality of space and the structure of the wave equation. A candidate structure that predicts 1/r² scaling for gravitational waves is immediately wrong, and this wrongness is detectable as a scaling law mismatch without comparing individual strain values.

Spectral line positions in hydrogen scale as 1/n² - 1/m² for integer n and m. This Rydberg relation is a mathematical fingerprint of quantum mechanics applied to the hydrogen atom. A candidate structure at the QFT level must reproduce this relation in the appropriate limit — it is a hard consistency check.

Scaling relations near critical points in thermodynamic experiments follow universal power laws with critical exponents that depend only on the dimensionality and symmetry of the system, not on microscopic details. These exponents are mathematical invariants — the same exponent appears in magnets, superconductors, and liquid-gas transitions if they share the same symmetry class. A candidate structure must predict the correct universality class for each experimental regime.

#### Anomaly Residuals

After subtracting the best available theoretical prediction from the experimental measurement, what remains? This residual is the most information-dense part of the data for the purpose of discovering new physics.

The residual is not noise — noise has been characterized and accounted for in the uncertainty quantification step. The residual is the part of the measurement that existing theory cannot explain. It is encoded as a formal object: the magnitude of the deviation, its functional form, the physical regime in which it appears, and the confidence level at which it is statistically significant.

High-value residuals — large deviation, high statistical significance, reproducible across independent experiments — are the primary targets for candidate structures. A structure that reduces a high-value residual while maintaining consistency with everything else scores very highly. These residuals are the mathematical coordinates of where new physics is most likely to be found.

---

### 2.4 Domain-Specific Preprocessing Pipelines

Layer 1 uses different preprocessing pipelines for different measurement types. These pipelines are domain-specific engineering work requiring physicists who understand both the instruments and the relevant theory. They sit entirely below Layer 2 and are invisible to the model.

#### Gravitational Wave Pipeline

Raw input: HDF5 files containing strain time series at 16,384 Hz, one per detector arm, from LIGO, Virgo, KAGRA.

Processing steps:
1. Glitch removal — detector artifacts from environmental disturbances are identified and excised using known glitch morphologies
2. Whitening — the noise power spectral density is measured from off-source data and divided out, flattening the noise floor across frequencies
3. Matched filtering — the whitened data is convolved with a bank of template waveforms from known GR predictions to identify candidates
4. Residual extraction — verified signals have their best-fit GR template subtracted, leaving the residual that GR cannot explain
5. Symmetry extraction — the signal's polarization properties are analyzed to extract the tensor structure of the gravitational wave
6. Uncertainty quantification — the noise model is propagated through all steps to produce calibrated uncertainty on all extracted quantities

Output: Metadata schema + Layer 2 mathematical objects (symmetry tensors, scaling relations, residual characterization)

#### Spectroscopic Pipeline

Raw input: Wavelength-indexed intensity arrays from optical, UV, X-ray, or radio spectrometers.

Processing steps:
1. Background subtraction — instrument background and sky background removed
2. Wavelength calibration — reference lines used to convert pixel positions to precise wavelengths
3. Line identification — known spectral lines matched to observed peaks, unmatched peaks flagged
4. Line profile fitting — each line fitted with Voigt profile to extract position, width, and intensity
5. Theoretical comparison — fitted line positions compared to quantum mechanical predictions for the relevant atomic/molecular system
6. Residual extraction — deviations from predicted positions and intensities encoded as formal anomaly objects

Output: Metadata schema + Layer 2 mathematical objects (energy level structure, transition probabilities, symmetry of quantum states, residuals from QED predictions)

#### Thermodynamic and Chemical Pipeline

Raw input: Tabular data — measurements of thermodynamic state variables (temperature, pressure, concentration, pH, enthalpy) as functions of control variables (volume, titrant volume, reaction coordinate).

Processing steps:
1. Curve parameterization — functional form fitted to the data (sigmoid for titration curves, power law near critical points, Arrhenius for reaction rates)
2. Critical point extraction — inflection points, equivalence points, phase transition temperatures identified with uncertainty
3. Scaling exponent extraction — near critical points, the scaling exponents of diverging quantities are measured and compared to theoretical universality class predictions
4. Conservation law verification — mass balance, charge balance, energy balance checked across the measurement range
5. Residual extraction — deviations from ideal theoretical behavior (ideal solution, ideal gas, standard model reaction mechanism) encoded as anomaly objects

Output: Metadata schema + Layer 2 mathematical objects (symmetry group of the relevant equilibrium, conservation law status, scaling exponents, chemical potential structure, residuals from quantum chemistry predictions)

#### Particle Collision Pipeline

Raw input: Event records from particle detectors — each record contains particle tracks, energy deposits in calorimeter cells, timing information for each collision event.

Processing steps:
1. Track reconstruction — particle trajectories reconstructed from detector hits using Kalman filter algorithms
2. Particle identification — track curvature in magnetic field, energy deposit pattern, and timing used to identify particle species
3. Invariant mass reconstruction — for each combination of identified final-state particles, invariant mass computed from four-momenta
4. Conservation law verification — energy-momentum conservation verified for each event; events with large apparent violation flagged as potentially indicating undetected particles
5. Cross-section calculation — event rates normalized to beam luminosity to produce production cross-sections as a function of center-of-mass energy
6. Standard model comparison — predicted cross-sections from QFT calculations compared to measured values; residuals extracted

Output: Metadata schema + Layer 2 mathematical objects (symmetry group of interaction vertices, conservation law status, resonance structure of invariant mass distributions, scaling of cross-sections with energy, residuals from Standard Model predictions)

---

### 2.5 The Regime Map

Every experiment is located on a two-dimensional regime map before being used in scoring. The axes are energy scale and gravitational field strength. The location on this map determines which theoretical framework applies and therefore which part of a candidate structure's predictions to compare against this data.

```
                    Strong Gravity
                         │
    Black hole           │          Neutron star
    interiors            │          surfaces
    [GR + QFT           │          [GR + QFT 
     required]          │           coupled]
                         │
─────────────────────────┼──────────────── Energy Scale
    Low energy           │          High energy
                         │
    Classical            │          Particle
    mechanics            │          colliders
    [Newtonian]          │          [QFT dominant]
                         │
                    Weak Gravity
```

The regime map serves two functions. First, it tells the scorer which subset of a candidate structure's predictions to compute for a given experiment — you don't compute quantum gravity corrections when scoring a room-temperature titration experiment. Second, it tells the explorer where the interesting frontier is — the upper-left quadrant of the map (strong gravity + high energy) is precisely where GR and QFT are simultaneously necessary, where they conflict, and where a unified structure would show its distinctive predictions.

Experiments in the upper-left quadrant receive the highest weight in the reward signal. Data from that regime is rare — it comes from gravitational wave observations of black hole mergers, from cosmological observations of the early universe, from planned next-generation experiments — but it is the most informative about the structure the system is trying to find.

---

### 2.6 Holdout Strategy

#### Why Standard Random Holdout Fails

Holding out a random fraction of measurements within a domain does not work for physical data. Measurements within a domain are not independent samples from a distribution — they are measurements of the same underlying physical process. A model trained on 80% of LIGO gravitational wave events can interpolate to the remaining 20% without having learned anything new about GR. This is not generalization, it is interpolation, and it provides no evidence that the model has captured genuine physical structure.

#### Domain-Level Holdout

The correct holdout unit is an entire experimental domain. The system trains on gravitational wave data, particle collision data, and spectroscopic data. It holds out cosmological survey data entirely. A candidate structure that captures the genuine physics underlying all domains should predict cosmological observations even without training on them — because the same mathematical structure governs all physical phenomena. If it does, that is strong evidence. If it only predicts trained domains, that is overfitting to domain-specific patterns.

The domain holdout assignments should be determined before training begins and not changed. Changing holdout assignments after observing training performance is a form of information leakage.

#### Future Experiment Holdout

The strongest possible validation is against experiments that do not yet exist. Before training begins, the system formally commits to treating specific planned experimental results as holdout data — next-generation gravitational wave detectors, planned collider upgrades, upcoming space telescope surveys.

When those experiments report results, the system's predictions are compared against them without any additional training. This is the analog of a registered pre-registered clinical trial in medicine — the prediction is made and committed to before the result is known, eliminating any possibility of post-hoc fitting.

This validation method has no analog in standard machine learning and is one of the strongest scientific validation standards available. It is worth the organizational complexity of setting it up.

---

### 2.7 How Data Enters The Training Loop — End to End

Putting all of the above together, here is the complete flow from raw experimental measurement to reward signal contribution in a single training iteration:

**Step 1 — Metadata loading.** The experiment's metadata schema is loaded. Physical regime is determined from energy scale and gravitational strength fields. Holdout status is checked — if this experiment is in the holdout set, it is not used in this iteration.

**Step 2 — Layer 1 preprocessing.** The appropriate domain-specific pipeline runs on the raw data file. Calibration corrections are applied. Uncertainty quantification produces probability distributions over true values rather than point measurements.

**Step 3 — Layer 2 extraction.** Symmetries are identified and their group-theoretic representation encoded. Conservation law status is verified. Scaling relations are fitted and their exponents extracted. Residuals from best available theoretical predictions are computed and characterized.

**Step 4 — Explorer proposes candidate structure.** The mathematical explorer, guided by MCTS, proposes a new mathematical structure or a modification of a promising existing one. The proposal is expressed in Lean 4 formal syntax.

**Step 5 — Proof checker verification.** The Lean 4 proof checker evaluates internal consistency. If the structure contains a logical contradiction or violates any established Mathlib theorem, the check fails immediately and a strong negative reward signal is returned to the explorer. No further processing occurs for this candidate.

**Step 6 — Correspondence check.** If internally consistent, the structure is checked against the formal theorems encoding the correspondence requirements — does it reduce to GR in the appropriate limit? Does it reduce to QFT in the appropriate limit? These are additional formal verification steps. Failure produces a negative reward and halts processing.

**Step 7 — Physical prediction computation.** The scorer computes what the candidate structure predicts for an experiment with this metadata in this physical regime. The prediction is expressed as a probability distribution over the Layer 2 mathematical objects — predicted symmetry group, predicted conservation law structure, predicted scaling exponents, predicted residual magnitude and form.

**Step 8 — Comparison and scoring.** The predicted Layer 2 objects are compared to the actual extracted Layer 2 objects from the experimental data. The comparison is statistical — the predicted distributions are evaluated against the observed values, properly weighted by experimental uncertainty. The score reflects how well the predictions match across all extracted mathematical objects simultaneously.

**Step 9 — Simplicity penalty.** The raw prediction score is adjusted downward by a function of the candidate structure's complexity — number of free parameters, number of independent terms in the Lagrangian, size of the symmetry group required. A structure that fits as well as a simpler one but requires more mathematical machinery is scored lower.

**Step 10 — Reward aggregation.** The final reward signal combines the prediction score (positive), the simplicity penalty (negative), and a curiosity bonus (positive, proportional to the novelty of the proposed structure relative to previously explored regions). This aggregate reward updates the explorer's parameters via the reinforcement learning algorithm.

**Step 11 — Anomaly flagging.** Independently of the reward computation, the solution enumeration process runs on any structure that passed verification. Unmatched solution families are logged with their formal property descriptions and queued for the translation layer.

**Step 12 — Translation.** Asynchronously from the main training loop, the translation layer processes flagged anomalies and generates natural language descriptions and experimental proposals. Verified structures above a score threshold are also translated to provide human physicists with a running account of the most promising candidates.

---

## Part 3 — Open Problems

These are the genuinely unsolved problems in building this system. They are not reasons not to build it — they are the hard parts that require research rather than engineering.

**Formal encoding of QFT.** Encoding general relativity in Lean 4 is substantially advanced. Encoding quantum field theory in full generality — with renormalization, running coupling constants, spontaneous symmetry breaking — is a harder and less complete task. This is active research in mathematical physics and formal verification communities. The system's capability in the quantum domain is limited by the completeness of this encoding.

**Reward hacking at the physical scoring stage.** A sufficiently capable optimizer will find mathematical structures that maximize the scoring function through means other than capturing genuine physical structure. The uncertainty quantification and domain holdout strategy mitigate this but do not eliminate it. Identifying and patching reward hacking behavior requires ongoing human oversight of what the system is actually finding.

**Curiosity reward calibration.** The balance between exploiting known productive regions of mathematical space and exploring genuinely novel regions is not well-characterized for this domain. Too much exploitation produces deeper results in narrow areas. Too much exploration produces many shallow results with no follow-through. The right calibration is probably dynamic — favoring exploration early in training and exploitation as promising structures are identified — but the specific schedule is an empirical question.

**Translation failure for genuinely novel physics.** When the explorer finds structures with no analogy in existing physics, the translation layer fails to produce useful experimental proposals. This failure is detectable but not automatically correctable. It requires human physicists with sufficient mathematical sophistication to engage directly with the formal outputs. Identifying and cultivating that human capacity is an organizational challenge as much as a technical one.

**Layer 1 coverage.** The preprocessing pipelines described here cover major experimental modalities in high-energy physics, gravitational wave astronomy, and spectroscopy. Many potentially informative experimental domains — condensed matter physics, biological systems, quantum information experiments — require additional pipelines that do not yet exist in the system. Each new domain requires domain experts to design and validate its pipeline, which is a continuing investment.

---

*This document is a companion to: Autonomous Mathematical Physics AI — System Design*
*Both documents should be read together for complete system understanding.*
