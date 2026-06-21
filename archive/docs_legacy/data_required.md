# Data Requirements for Physical Correspondence (Phase 2.6 + Phase 3)

**Date:** 2026-06-03
**Updated:** 2026-06-03 (reorganized chronologically)

**Purpose:** Every experimentally verified result the system must be capable of reproducing. Organized by historical era so the model can be temporally gated — restrict it to pre-1905 data and see whether it discovers special relativity; restrict it to pre-1960 data and see whether it predicts the Standard Model structure.

---

## Design Principle: Chronological Gating

The system should be evaluable at any historical cutoff. Given only data available in:

| Era | Cutoff year | Should be able to... | Key missing theory |
|-----|-------------|---------------------|-------------------|
| Classical | 1860 | Reproduce Newtonian mechanics, Coulomb's law, wave optics | Maxwell's equations |
| Classical crisis | 1900 | Reproduce EM + thermodynamics, struggle with blackbody | Relativity, QM |
| Pre-relativity | 1904 | Hit contradictions in electrodynamics of moving bodies | Special relativity |
| Pre-GR | 1914 | Hit contradictions in Mercury + light and gravity | General relativity |
| Early QM | 1925 | Reproduce atomic spectra, struggle with interpretation | Matrix/wave mechanics |
| Pre-QED | 1946 | Hit Lamb shift, anomalous magnetic moment | QED |
| Pre-SM | 1965 | Hit parity violation, neutrino puzzles, hadron zoo | Electroweak, QCD |
| Pre-precision | 1990 | See hints of neutrino mass, dark matter, dark energy | ν masses, DM, Λ |
| Modern | 2012 | See Higgs, GW, Hubble tension, g-2 anomaly | Beyond SM |
| Present | 2026 | All of the above | Still open |

This directly tests whether the explorer can rediscover theoretical advances from experimental constraints — the strongest possible validation that it's doing real science rather than memorization.

---

## Era 1 — Pre-1800: The Classical Foundation

Data available before 1800. If a structure cannot reproduce these, it's wrong regardless of what else it predicts.

### 1.1 Celestial Mechanics

| # | Experiment / Measurement | Value | Source | Format |
|---|---|---|---|---|
| C1 | Kepler's laws (planetary orbital periods, semi-major axes, eccentricities) | T² ∝ a³ for all known planets | Astronomical records (Tycho Brahe through Kepler, ~1609) | Table, ~50 numbers |
| C2 | Precession of the equinoxes | ~50.3"/year | Hipparchus through Newton | Single number |
| C3 | Moon's orbital irregularities (evection, variation, annual equation) | Multiple periodic terms | Newton, Euler, Clairaut (~1750) | Table, ~10 numbers |
| C4 | Tidal amplitudes and phases (Bay of Fundy, Brest, etc.) | Site-specific | Royal Society records | Table, ~20 numbers |

### 1.2 Terrestrial Mechanics

| # | Experiment / Measurement | Value | Source | Format |
|---|---|---|---|---|
| G1 | Galileo's inclined plane acceleration | g = 9.8 m/s² (from inclined plane ratio) | Galileo, *Two New Sciences* (1638) | Single number |
| G2 | Pendulum period vs length (T = 2π√(L/g)) | Linear with √L, independent of mass | Huygens, *Horologium Oscillatorium* (1673) | Table, ~10 rows |
| G3 | Conservation of momentum in collisions | m₁v₁ + m₂v₂ = constant | Huygens, Wallis, Wren (1668) | Table, ~10 rows |

### 1.3 Optics

| # | Experiment / Measurement | Value | Source | Format |
|---|---|---|---|---|
| O1 | Snell's law of refraction (n sin θ = constant for interface) | n_water ≈ 1.33, n_glass ≈ 1.5-1.6 | Snell (1621), Descartes (1637) | Table, ~20 materials |
| O2 | Speed of light (finite, measured by Römer from Jupiter's moons) | c ≈ 2.2×10⁸ m/s (first estimate) | Römer (1676) | Single number |
| O3 | Newton's rings (interference pattern radii) | Ring spacing proportional to √m | Newton, *Opticks* (1704) | Table, ~10 rows |
| O4 | Solar spectrum with Fraunhofer dark lines | ~574 lines catalogued | Fraunhofer (1814) | Table, ~500 wavelengths |

### 1.4 Electrostatics and Magnetostatics

| # | Experiment / Measurement | Value | Source | Format |
|---|---|---|---|---|
| E1 | Coulomb's inverse-square law (force between charged spheres) | F ∝ q₁q₂/r² | Coulomb (1785) | Binary test (exponent = 2.00 ± 0.01) |
| E2 | Magnetic dipole inverse-cube law | F ∝ 1/r³ for dipoles | Coulomb, Michell | Binary test |
| E3 | Electrostatic series (triboelectric ordering) | Ordering of ~30 materials | Multiple sources through 1700s | Ordered list |

### 1.5 Fluid Mechanics and Thermodynamics (early)

| # | Experiment / Measurement | Value | Source | Format |
|---|---|---|---|---|
| T1 | Boyle's law (PV = constant at fixed T) | Product constant to ~1% for air | Boyle (1662), Mariotte (1676) | Table, ~10 rows |
| T2 | Charles's law (V ∝ T at fixed P) | Linear with T, extrapolates to zero at -273°C | Charles (1787), Gay-Lussac (1802) | Table, ~10 rows |
| T3 | Latent heat of fusion/vaporization for water | 334 J/g, 2260 J/g | Black, Watt (~1760) | Table, ~10 substances |
| T4 | Speed of sound in air | ~343 m/s at 20°C | Multiple, 1600s-1700s | Table, ~10 conditions |

**Era 1 total:** ~650 data points. All are constants or small tables encodable in a Python dict.

---

## Era 2 — 1800-1865: Classical Physics Matures

Maxwell's equations published 1861-1865. A cutoff at 1860 tests whether the system can discover them.

### 2.1 Electromagnetism (pre-Maxwell synthesis)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| EM1 | Oersted's discovery: current deflects compass needle | Oersted (1820) | Qualitative → quantitative |
| EM2 | Ampère's force law between current-carrying wires (F ∝ I₁I₂/r) | Ampère (1820-1825) | Table, ~20 measurements |
| EM3 | Biot-Savart law for magnetic field around a wire | Biot, Savart (1820) | Table, ~10 measurements |
| EM4 | Faraday's electromagnetic induction (changing B → EMF) | Faraday (1831) | Table, ~30 configurations |
| EM5 | Faraday's law of electrolysis (mass ∝ charge × equivalent weight) | Faraday (1834) | Table, ~20 substances |
| EM6 | Ohm's law (V = IR) across materials | Ohm (1827) | Table, ~50 materials |
| EM7 | Wheatstone bridge precision resistance measurements | Wheatstone (1843) | Table, ~30 measurements |
| EM8 | Lenz's law (induced current opposes flux change) | Lenz (1834) | Qualitative verification |
| EM9 | Speed of electrical signals in wires (Fizeau, Wheatstone) | Fizeau (1849), Wheatstone | Single number (close to c) |
| EM10 | Dielectric constants of materials | Faraday (1837) | Table, ~30 materials |

### 2.2 Wave Optics

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| WO1 | Young's double-slit interference (fringe spacing vs wavelength) | Young (1801) | Table, ~10 wavelengths |
| WO2 | Fresnel diffraction patterns (Poisson spot at center of disc shadow) | Fresnel, Arago (1818) | Binary confirmation of wave theory |
| WO3 | Fizeau rotating cogwheel measurement of c | Fizeau (1849) → c ≈ 3.13×10⁸ m/s | Single number |
| WO4 | Foucault rotating mirror measurement of c | Foucault (1862) → c ≈ 2.98×10⁸ m/s | Single number |
| WO5 | Polarization by reflection (Brewster's angle) | Brewster (1815) | Table, ~20 materials |

### 2.3 Thermodynamics and Kinetic Theory

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| TD1 | Carnot efficiency (η = 1 - T_cold/T_hot) | Carnot (1824) | Theoretical + water/steam data |
| TD2 | Joule's mechanical equivalent of heat (4.18 J/cal) | Joule (1843-1850) | Single number ± 1% |
| TD3 | Mayer's mechanical equivalent of heat (from specific heats) | Mayer (1842) | Single number |
| TD4 | Specific heats of gases (C_p, C_v) for air, H₂, O₂, CO₂ | Regnault (~1840s) | Table, ~20 measurements |
| TD5 | Thermal expansion coefficients (solids and liquids) | Multiple sources | Table, ~50 materials |
| TD6 | Thermal conductivity of metals | Multiple sources | Table, ~30 materials |
| TD7 | Dulong-Petit law (C_v ≈ 3R for solids at room temp) | Dulong, Petit (1819) | Table, ~20 elements |

### 2.4 Elasticity and Materials

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| ML1 | Young's modulus for metals, wood, glass | Young (1807) | Table, ~30 materials |
| ML2 | Poisson's ratio measurements | Poisson (1820s) | Table, ~20 materials |
| ML3 | Speed of sound in solids (bars, wires) | Chladni, multiple sources | Table, ~20 materials |

### 2.5 Astronomy (improved)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| A1 | Stellar parallax (first measurements: 61 Cygni, Vega, α Centauri) | Bessel (1838), Henderson, Struve | 3 stars, precise angles |
| A2 | Discovery of Neptune (position predicted from Uranus perturbations) | Galle, Le Verrier, Adams (1846) | Orbital elements |
| A3 | Binary star orbits (testing Newtonian gravity beyond solar system) | Herschel, multiple | Table, ~20 systems |

**Era 2 total:** ~350 data points + qualitative verifications.

---

## Era 3 — 1865-1905: Classical Crisis

Maxwell through Einstein's 1905 papers. This is the key testing era — the system should hit contradictions with classical physics.

### 3.1 Electromagnetism (post-Maxwell)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| MW1 | Hertz's generation and detection of radio waves (λ, reflection, refraction, polarization) | Hertz (1886-1888) | Table, ~20 measurements |
| MW2 | Speed of EM waves = speed of light (in free space and along wires) | Hertz, multiple | Single number confirmation |
| MW3 | **Michelson-Morley: null result for aether drift** | Michelson, Morley (1887) | Δv < 4 km/s (later < 1 km/s with Morley-Miller) |
| MW4 | Trouton-Noble: null torque on moving capacitor | Trouton, Noble (1903) | Null result |
| MW5 | Fizeau's moving water experiment (light dragging) | Fizeau (1851) | Fresnel drag coefficient confirmed |
| MW6 | Zeeman effect (spectral line splitting in magnetic field) | Zeeman (1896) | Table, ~20 lines |
| MW7 | Faraday effect (magnetic rotation of polarization) | Faraday (1845) | Table, ~20 materials |

### 3.2 Thermal Radiation (the blackbody problem)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| BB1 | Stefan-Boltzmann law (total radiated power ∝ T⁴) | Stefan (1879), Boltzmann (1884) | Table, ~10 temperatures |
| BB2 | Wien's displacement law (λ_max T = constant) | Wien (1893) | Table, ~10 temperatures |
| BB3 | **Blackbody spectrum (Lummer-Pringsheim, Rubens-Kurlbaum) — full curve** | Physikalisch-Technische Reichsanstalt (~1897-1901) | Spectrum tables, ~100 frequencies each at ~5 T |
| BB4 | Rayleigh-Jeans law failure at short wavelengths ("ultraviolet catastrophe") | Derived from classical equipartition | Not an experiment per se, but a documented failure |

### 3.3 Atomic Spectra (pre-quantum explanation)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| AS1 | **Balmer series of hydrogen (visible lines)** | Balmer (1885) | 4 lines: Hα 656.3, Hβ 486.1, Hγ 434.0, Hδ 410.2 nm |
| AS2 | Rydberg formula (1/λ = R(1/n₁² - 1/n₂²)) generalized | Rydberg (1888) | Fits H with R = 109677 cm⁻¹ |
| AS3 | Lyman series (UV), Paschen series (IR), Brackett, Pfund | Later discovery but same formula | All fit Rydberg formula |
| AS4 | Alkali metal spectra (Li, Na, K — series structure) | Liveing, Dewar, Rydberg | Table, ~100 lines per element |
| AS5 | Pickering series (He⁺ — initially misidentified as H) | Pickering (1896) | Table, ~20 lines |

### 3.4 Cathode Rays and Electron Discovery

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| CR1 | **Thomson's e/m measurement (cathode ray deflection)** | J.J. Thomson (1897) | e/m ≈ 1.76×10¹¹ C/kg |
| CR2 | **Millikan oil drop experiment (quantized charge)** | Millikan (1909-1913) | e = 1.592×10⁻¹⁹ C (modern: 1.602) |
| CR3 | Kaufmann-Bucherer-Neumann: electron mass increase with velocity | Kaufmann (1901), Bucherer (1908) | Table, ~10 velocities → test of relativistic mass |
| CR4 | X-ray discovery and properties (penetration, ionization, diffraction) | Röntgen (1895), Barkla | Qualitative + early measurements |

### 3.5 Photoelectric Effect

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| PE1 | Lenard: photoelectric current vs light intensity and frequency | Lenard (1902) | Key observation: stopping voltage independent of intensity, depends on frequency |
| PE2 | Stopping voltage vs frequency for multiple metals | Later refined by Millikan (1916) | Table, ~10 frequencies per metal, ~5 metals |

### 3.6 Radioactivity (new phenomenon)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| R1 | Becquerel: uranium salts fog photographic plates | Becquerel (1896) | Qualitative discovery |
| R2 | Curie: radioactivity of U, Th, Po, Ra (activity per gram) | M. Curie (1898-1903) | Table, ~10 elements |
| R3 | Rutherford: alpha, beta, gamma classification by penetration | Rutherford (1899) | Table, 3 radiation types |
| R4 | Rutherford-Soddy: exponential decay law, half-lives | Rutherford, Soddy (1902) | Table, ~15 isotopes |

**Era 3 total:** ~500 data points. This era contains the crucial *null results* and *crises* that classical physics cannot explain — Michelson-Morley, blackbody UV catastrophe, photoelectric threshold, electron mass-velocity — which are the strongest tests of whether the explorer can discover new physics.

---

## Era 4 — 1905-1930: Relativity + Old Quantum Theory

Einstein's 1905 papers through the formulation of modern quantum mechanics.

### 4.1 Special Relativity Tests

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| SR1 | Rossi-Hall: muon lifetime dilation (cosmic ray muons reach sea level) | Rossi, Hall (1941) — idea tested earlier | Table, ~5 altitudes |
| SR2 | Ives-Stilwell: transverse Doppler effect in canal rays | Ives, Stilwell (1938) | Table, ~5 velocities |
| SR3 | Kennedy-Thorndike: null test of time dilation + length contraction | Kennedy, Thorndike (1932) | Null result |
| SR4 | E = mc²: mass defect in nuclear reactions (Cockcroft-Walton) | Cockcroft, Walton (1932) | Li⁷ + p → 2α, mass-energy balance |

### 4.2 General Relativity Tests

| # | Experiment / Measurement | Source | Format | Use |
|---|---|---|---|---|
| GR1 | **Perihelion precession of Mercury (43"/century excess)** | Leverrier (1859), Newcomb (1882), Einstein (1915) | 43.0 ± 0.5 "/century | Critical: unexplained by Newtonian gravity |
| GR2 | **Eddington 1919 eclipse: light deflection by Sun** | Eddington, Crommelin (1919) | 1.98" ± 0.16" (Sobral), 1.61" ± 0.40" (Principe) | First GR confirmation |
| GR3 | Gravitational redshift: solar spectral lines | Adams (1925), refined by later measurements | ~2×10⁻⁶ fractional shift on Sun | Disputed at the time, confirmed later |
| GR4 | Gravitational redshift: Sirius B (white dwarf) | Adams (1925) | ~20 km/s equivalent shift | Larger effect, more convincing |
| GR5 | Pound-Rebka: laboratory gravitational redshift | Pound, Rebka (1959) | ~2.5×10⁻¹⁵ fractional shift, confirmed to 10% | First lab test |
| GR6 | Shapiro delay: radar echoes from Venus/Mercury delayed near Sun | Shapiro (1964-1968) | ~200 μs excess delay | Precision Solar System test |

### 4.3 Atomic Structure (Bohr-Sommerfeld model)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| QM1 | **Franck-Hertz: quantized energy levels in mercury vapor** | Franck, Hertz (1914) | Current vs voltage steps at 4.9 V intervals | Direct evidence for discrete energy levels |
| QM2 | Stern-Gerlach: silver atom beam splits in magnetic field (spin quantization) | Stern, Gerlach (1922) | Two spots, not a smear | Direct evidence for spin |
| QM3 | Compton scattering: X-ray wavelength shift Δλ = (h/mc)(1-cosθ) | Compton (1923) | Table, ~10 angles → h/mc confirmed |
| QM4 | Davisson-Germer: electron diffraction from nickel crystal | Davisson, Germer (1927) | Table, ~10 angles → λ = h/p confirmed |
| QM5 | G.P. Thomson: electron diffraction through thin films | Thomson (1927) | Diffraction rings → wave nature of electrons |
| QM6 | Fine structure of hydrogen (Sommerfeld) | Sommerfeld (1916) | Doublet splitting, α = 1/137 confirmed |
| QM7 | Stark effect (spectral line splitting in electric field) | Stark (1913) | Table, ~20 hydrogen lines |
| QM8 | Paschen-Back effect (strong-field Zeeman splitting) | Paschen, Back (1912) | Table, ~20 lines |

### 4.4 Nuclear Physics Emerges

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| N1 | Rutherford scattering: α-particles from gold foil (nuclear model) | Rutherford, Geiger, Marsden (1909-1913) | Angular distribution → 1/sin⁴(θ/2), ~1/10000 backscattered |
| N2 | Moseley's law: √(ν) ∝ Z for Kα X-ray lines | Moseley (1913-1914) | Table, ~30 elements → atomic number concept |
| N3 | Chadwick: neutron discovery (α on Be → penetrating radiation) | Chadwick (1932) | Mass ~1.0087 u, neutral |
| N4 | Aston: mass spectra (whole-number rule, binding energy curves) | Aston (1919-1925) | Table, ~200 isotopes |
| N5 | Cockcroft-Walton: first artificial nuclear transmutation | Cockcroft, Walton (1932) | Table, ~5 reactions |

### 4.5 Expanding Universe

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| H1 | **Hubble's law (v = H₀ d)** | Hubble (1929) | Original: H₀ ≈ 500 km/s/Mpc; 24 galaxies, linear with scatter |
| H2 | Slipher's redshift measurements of spiral nebulae | Slipher (1912-1925) | Table, ~40 galaxies, mostly redshifts |
| H3 | Hubble's galaxy morphological classification | Hubble (1926) | Classification scheme, not numeric |

**Era 4 total:** ~500 data points. The crucial prediction targets are Mercury's precession (43"/century), Eddington's light deflection, Compton scattering, and Stern-Gerlach.

---

## Era 5 — 1930-1960: QED + Nuclear + Particles

The era that built the foundation for the Standard Model.

### 5.1 Quantum Electrodynamics (precision)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| QED1 | **Lamb shift: 2S_½ - 2P_½ in hydrogen (1057 MHz)** | Lamb, Retherford (1947) | 1057.86 ± 0.10 MHz | QED vacuum polarization test |
| QED2 | **Electron anomalous magnetic moment (g-2)** | Kusch, Foley (1948) → later refined | a_e = (g-2)/2 = 0.00115965218059(13) | Most precise QED test |
| QED3 | Positronium spectrum (e⁺e⁻ bound state, hyperfine splitting) | Deutsch (1951) | 203.39 GHz | Pure QED bound state |
| QED4 | Muonium hyperfine splitting | Hughes (1960) | 4463 MHz | μ⁺e⁻, tests lepton universality |
| QED5 | Delbrück scattering (photon-photon scattering in Coulomb field) | Multiple | Small cross-section | QED nonlinearity |

### 5.2 Weak Interactions

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| W1 | Beta decay spectra (continuous → neutrino hypothesis) | Chadwick (1914), Ellis, Wooster (1927) | Continuous electron spectrum → Pauli proposes neutrino (1930) |
| W2 | Fermi's theory of beta decay: transition rates vs Q-value | Fermi (1934) | Table, ~20 isotopes |
| W3 | **Parity violation in Co-60 beta decay** | Wu et al. (1957) | Asymmetric electron emission → parity violated maximally |
| W4 | **Parity violation in π→μ→e chain** | Garwin, Lederman, Weinrich (1957) | Muon spin → decay electron asymmetry |
| W5 | Goldhaber: neutrino helicity measurement | Goldhaber, Grodzins, Sunyar (1958) | Neutrino is left-handed |
| W6 | Reines-Cowan: neutrino detection (reactor ν̄_e + p → n + e⁺) | Reines, Cowan (1956) | Cross-section ~6×10⁻⁴⁴ cm² |
| W7 | Muon decay spectrum (Michel parameters, V-A structure) | Multiple (1950s) | ρ = 0.75 → V-A interaction confirmed |

### 5.3 Strong Interactions and Hadrons

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| S1 | Yukawa: pion mass prediction from nuclear force range | Yukawa (1935) | m_π ≈ 200 m_e predicted → π discovered 1947 |
| S2 | Powell: pion discovery in photographic emulsions | Powell, Occhialini (1947) | m_π ≈ 273 m_e |
| S3 | Discovery of strange particles (K⁰, Λ⁰, Σ, Ξ) | Multiple (1947-1955) | Table, ~20 particles, masses, lifetimes |
| S4 | Dalitz plot for τ (K⁺→3π) → τ-θ puzzle → parity violation | Dalitz (1953-1956) | τ and θ same mass, lifetime → same particle with parity violation |
| S5 | Fermi-Yang, Sakata models (early hadron classification) | Fermi, Yang (1949), Sakata (1956) | Failed models but historically important |
| S6 | Hofstadter: electron scattering off nucleons (form factors) | Hofstadter (1955) | Proton radius ~0.84 fm → nucleons have internal structure |

### 5.4 Nuclear Physics (detailed)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| NP1 | Bethe-Weizsäcker semi-empirical mass formula coefficients | Bethe, Weizsäcker (1935-1936) | 5-term formula, fit to ~300 nuclei |
| NP2 | Shell model magic numbers (2, 8, 20, 28, 50, 82, 126) | Goeppert-Mayer, Jensen (1949-1950) | Table, ~10 key nuclear properties at magic numbers |
| NP3 | Nuclear magnetic moments (deviations from Schmidt lines) | Multiple | Table, ~100 isotopes, shows collective effects |
| NP4 | Fission cross-sections (²³⁵U, ²³⁹Pu thermal and fast neutron) | Multiple (1939-1945) | Table, ~10 energies per isotope |
| NP5 | B²FH: elemental abundance curve (stellar nucleosynthesis evidence) | Burbidge, Burbidge, Fowler, Hoyle (1957) | Abundance vs mass number, ~100 points |

### 5.5 Condensed Matter — Emergent Quantum Phenomena

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| CM1 | Superconductivity: critical temperature, critical field, isotope effect | Kamerlingh Onnes (1911), Meissner (1933), isotope effect (1950) | Table, ~20 superconductors |
| CM2 | Superfluidity of He-4 (λ transition at 2.17 K, viscosity → 0) | Kapitsa, Allen, Misener (1937) | Key measurement: viscosity vanishes below λ point |
| CM3 | Quantized vortices in superfluid He-4 | Hall, Vinen (1956) | Circulation quantized in units of h/m |
| CM4 | de Haas-van Alphen effect (oscillatory magnetization → Fermi surface) | de Haas, van Alphen (1930) | Table, ~5 metals → Fermi surface mapping |
| CM5 | Cyclotron resonance in semiconductors | Multiple (1950s) | Effective masses for Si, Ge, etc. |

### 5.6 Quantum Foundations

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| QF1 | Einstein-Podolsky-Rosen argument | EPR (1935) | Theoretical argument |
| QF2 | Bohm's EPR variant (spin correlations) | Bohm (1951) | Spin singlet state correlations |
| QF3 | Bell's theorem (local hidden variables predict different correlations) | Bell (1964) | Inequality, not yet tested in this era |
| QF4 | Electron double-slit (one at a time, build up interference) | Merli, Missiroli, Pozzi (1974) — but conceptually pre-1960 | Interference with single electrons |

**Era 5 total:** ~600 data points. The strongest prediction targets: Lamb shift (QED vacuum), electron g-2 (QED precision), parity violation (weak interaction structure), and the hadron spectrum (early evidence for quarks).

---

## Era 6 — 1960-1990: The Standard Model Era

The Standard Model is assembled and tested. Quarks, W/Z, gluons, and three generations.

### 6.1 Quark Evidence

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| Q1 | **Deep inelastic scattering: Bjorken scaling → pointlike constituents** | SLAC-MIT (1968-1969) | Structure functions F₂(x,Q²) → scaling at large Q² |
| Q2 | Callan-Gross relation (F₂ = 2xF₁ → spin-½ partons) | SLAC (1969) | Ratio measured → evidence for spin-½ quarks |
| Q3 | J/ψ discovery (c\bar{c} bound state, narrow resonance at 3.1 GeV) | Ting (BNL), Richter (SLAC) (1974) | Mass 3096.9 MeV, width 93 keV → charm quark |
| Q4 | Υ discovery (b\bar{b} bound state at 9.46 GeV) | Lederman (Fermilab, 1977) | Mass 9460 MeV → bottom quark |
| Q5 | Three-jet events (gluon evidence) | TASSO at PETRA (1979) | Jet angular distributions → spin-1 gluon |
| Q6 | Running of α_s (asymptotic freedom confirmation) | Multiple experiments (1980s) | α_s(M_Z) decreasing with energy |
| Q7 | Top quark discovery (DØ/CDF at Tevatron) | Fermilab (1995) | m_t = 176 ± 13 GeV (initial), now 172.5 GeV |

### 6.2 Electroweak Unification

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| EW1 | **Neutral current discovery (ν_μ + N → ν_μ + X, no muon)** | Gargamelle (CERN, 1973) | σ_NC/σ_CC ≈ 0.21 ± 0.03 → Z boson evidence |
| EW2 | W boson discovery (UA1/UA2 at SPS collider) | CERN (1983) | m_W = 80.4 GeV, Γ_W ~ 2.1 GeV |
| EW3 | Z boson discovery | CERN (1983) | m_Z = 91.2 GeV, Γ_Z ~ 2.5 GeV |
| EW4 | **LEP-I: Z lineshape (σ vs √s scan at Z pole)** | LEP (1989-1995) | Precise m_Z, Γ_Z, σ_had — [HEPData](https://hepdata.net) JSON, ~10 MB |
| EW5 | LEP-I: number of light neutrino species (N_ν = 2.984 ± 0.008) | LEP (1990) | From invisible Z width → exactly 3 generations |
| EW6 | LEP-I: forward-backward asymmetries (A_FB for e⁺e⁻ → f\bar{f}) | LEP (1990s) | sin²θ_W from multiple channels |
| EW7 | SLD: polarized e⁻ beam → A_LR measurement | SLAC (1990s) | sin²θ_W = 0.23098 ± 0.00026 |
| EW8 | Weinberg angle from neutrino-nucleon scattering | CHARM, CDHS, CCFR | sin²θ_W consistent with LEP |

### 6.3 CP Violation

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| CP1 | **CP violation discovery: K_L → π⁺π⁻ decay** | Cronin, Fitch (1964) | Branching ratio ~2×10⁻³ → CP violated at ~0.2% level in kaons |
| CP2 | K_L-K_S mass difference (Δm_K) | Multiple | 3.48×10⁻¹² MeV → tiny but measurable |
| CP3 | ε'/ε in kaon system (direct CP violation) | NA31, NA48, KTeV (1988-2001) | ε'/ε ≈ 1.7×10⁻³ → confirms direct CP violation |
| CP4 | B⁰-B̄⁰ mixing (Δm_d) | ARGUS (1987) | Δm_d ≈ 0.5 ps⁻¹ → top quark mass upper bound before discovery |

### 6.4 Neutrino Physics

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| NU1 | Homestake solar neutrino experiment (ν_e capture on ³⁷Cl) | Davis (1968-1990s) | ~0.5 atoms/day vs ~1.5 atoms/day predicted → solar neutrino problem |
| NU2 | Kamiokande: real-time solar neutrino detection (directional) | Kamiokande (1987) | Confirmed deficit, direction confirmed from Sun |
| NU3 | IMB + Kamiokande: supernova 1987A neutrino burst | IMB, Kamiokande (1987) | 25 events over ~13 seconds → neutrino astrophysics born |
| NU4 | SNO: charged-current, neutral-current, elastic scattering separation | SNO (2001-2002) | **ν_e → ν_μ,τ flavor transformation confirmed** (2001) → neutrino mass |
| NU5 | Super-Kamiokande: atmospheric neutrino oscillations | Super-K (1998) | ν_μ disappearance, zenith angle dependence → Δm²_atm confirmed |
| NU6 | KamLAND: reactor ν̄_e disappearance (long baseline) | KamLAND (2002) | Δm²_sol, θ_12 from reactor neutrinos |
| NU7 | DONUT: direct tau neutrino detection | DONUT (Fermilab, 2000) | 4 ν_τ events → third neutrino confirmed |

### 6.5 CMB and Cosmology

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| CMB1 | **Penzias-Wilson: CMB discovery (isotropic 3.5 K excess)** | Penzias, Wilson (1965) | T ≈ 3.5 ± 1 K (initial), now 2.72548 ± 0.00057 K |
| CMB2 | COBE FIRAS: perfect blackbody spectrum at T = 2.725 K | COBE (1990) | Residuals < 0.005% of peak — most perfect blackbody ever measured |
| CMB3 | COBE DMR: CMB temperature anisotropies (ΔT/T ~ 10⁻⁵) | COBE (1992) | Angular power spectrum, first detection of primordial fluctuations |

### 6.6 Precision GR Tests

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| PGR1 | **Hulse-Taylor binary pulsar (PSR B1913+16): orbital decay** | Hulse, Taylor (1974-1982) | dP/dt = -2.422×10⁻¹², matches GR prediction to 0.2% |
| PGR2 | Lunar Laser Ranging: equivalence principle, time variation of G | LLR (1969-present) | Nordtvedt parameter η = 4.4×10⁻⁴, Ḡ/G < 10⁻¹²/yr |
| PGR3 | Viking/Mars: Shapiro delay to ~0.1% | Viking (1976) | γ PPN parameter = 1.000 ± 0.002 |
| PGR4 | Cassini: Shapiro delay to ~10⁻⁵ | Cassini (2003) | γ-1 = (2.1 ± 2.3)×10⁻⁵ |
| PGR5 | Double pulsar PSR J0737-3039 (5 independent GR tests in one system) | Kramer et al. (2006) | Orbital decay, Shapiro delay, precession, etc. |

**Era 6 total:** ~800 data points. The key prediction targets: Bjorken scaling (quarks), neutral currents (Z boson), CP violation, neutrino oscillations, and the Z lineshape (N_ν=3).

---

## Era 7 — 1990-2010: Precision Cosmology + Flavor

### 7.1 CMB Precision

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| CMB4 | BOOMERanG/MAXIMA: first acoustic peak → Ω_total ≈ 1 (flat universe) | (2000) | ℓ_peak ≈ 200 → flat geometry |
| CMB5 | WMAP 1-year, 3-year, 5-year, 7-year, 9-year results | WMAP (2003-2013) | TT, TE power spectra — [LAMBDA](https://lambda.gsfc.nasa.gov) FITS, ~500 MB full mission |
| CMB6 | DASI: CMB polarization detection | DASI (2002) | E-mode polarization → reionization τ ~ 0.17 |
| CMB7 | ACBAR, CBI, VSA: small-scale CMB | Multiple | ℓ up to ~3000 → SZ effect, lensing |

### 7.2 Dark Energy Evidence

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| DE1 | **High-z SNIa: cosmic acceleration (Ω_Λ > 0)** | Riess (1998), Perlmutter (1999) | ~50 SNIa, Hubble diagram → q₀ < 0 |
| DE2 | SDSS BAO: baryon acoustic peak at ~150 Mpc | SDSS (2005) | BAO scale at z=0.35 → standard ruler |
| DE3 | 2dFGRS: galaxy power spectrum shape | 2dF (2001) | Ω_m h ≈ 0.2, baryon fraction |
| DE4 | Weak lensing: cosmic shear 2-point correlations | CFHTLS, COSMOS | σ_8(Ω_m/0.3)^0.5 constraints |

### 7.3 B Factories (CP Violation in B System)

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| BF1 | sin 2β measurement from B⁰→J/ψ K_S | BaBar, Belle (2001) | sin 2β ≈ 0.7 → CKM phase confirmed |
| BF2 | B⁰-B̄⁰ mixing Δm_s measurement | CDF (2006) | Δm_s = 17.77 ps⁻¹ |
| BF3 | Direct CP violation in B⁰→K⁺π⁻ | BaBar, Belle | A_CP ≈ -0.1 → direct CPV in B decays |
| BF4 | B→τν, B→μν, B→Dτν — lepton universality tests | BaBar, Belle | Early hints of lepton non-universality (R(D), R(D*)) |

### 7.4 Neutrino Precision

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| NU8 | MINOS: ν_μ disappearance (accelerator long baseline) | MINOS (2006) | Δm²_atm confirmed with accelerator beam |
| NU9 | T2K: ν_μ→ν_e appearance (θ_13 indication) | T2K (2011) | First hint θ_13 ≠ 0 |
| NU10 | Daya Bay: θ_13 measurement (reactor ν̄_e disappearance) | Daya Bay (2012) | sin² 2θ_13 = 0.092 ± 0.017 → non-zero at 5.2σ |
| NU11 | RENO, Double Chooz: θ_13 confirmation | (2012) | Consistent with Daya Bay |

### 7.5 Hadron Spectroscopy

| # | Experiment / Measurement | Source | Format |
|---|---|---|---|
| HS1 | PDG: complete hadron mass tables (light unflavored, strange, charm, bottom) | PDG annual | Table, ~500 hadrons with masses, widths, quantum numbers |
| HS2 | Lattice QCD: hadron mass spectrum from first principles | Multiple collaborations | Reference values for m_π, m_K, m_N, m_Ω from lattice |
| HS3 | Exotic hadrons (X(3872), Y(4260), Z(4430) — tetraquarks/pentaquarks) | Belle, BaBar, LHCb | Table, ~30 exotic candidates |

**Era 7 total:** ~600 data points.

---

## Era 8 — 2010-Present: Modern Era

### 8.1 LHC Results

| # | Experiment / Measurement | Source | Format | Use |
|---|---|---|---|---|
| LHC1 | **Higgs boson discovery (γγ and ZZ→4ℓ channels)** | ATLAS + CMS (2012) | m_H = 125.09 ± 0.24 GeV — [HEPData](https://hepdata.net) JSON, ~5 MB | Higgs mechanism confirmed |
| LHC2 | Higgs couplings (κ_framework: κ_Z, κ_W, κ_t, κ_τ, κ_b, κ_μ) | ATLAS + CMS Run 2 combination | κ values consistent with SM within ~10% | Higgs sector verification |
| LHC3 | Higgs width (indirect: Γ_H < 1.1 GeV from off-shell production) | CMS (2014) | Upper limit | Consistency check |
| LHC4 | Higgs self-coupling (HH→bbγγ, HH→4b) — first constraints | ATLAS + CMS (2022) | κ_λ within [-1.2, 7.2] (95% CL) | Higgs potential shape |
| LHC5 | Drell-Yan, diboson, tri-boson cross-sections (13 TeV) | ATLAS + CMS | Multiple final states, ~50 measurements | EW verification at highest energy |
| LHC6 | Searches: SUSY, Z', W', leptoquarks, vector-like quarks, etc. (exclusion limits) | ATLAS + CMS | Exclusion contours, ~100 publication results | What is NOT there — constraints on BSM |

### 8.2 Gravitational Wave Astronomy

| # | Experiment / Measurement | Source | Format | Use |
|---|---|---|---|---|
| GW1 | **GW150914: first BBH merger (strain time series)** | LIGO O1 (2015) — [GWOSC](https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/) | HDF5, ~50 MB | GR in dynamical strong-field regime |
| GW2 | GW170817: BNS merger + EM counterpart (kilonova AT2017gfo, GRB 170817A) | LIGO/Virgo + Fermi/INTEGRAL + optical/NIR/radio (2017) | HDF5 + multi-wavelength | Speed of gravity = c, r-process nucleosynthesis confirmed, Hubble constant measurement |
| GW3 | GW190521: intermediate-mass black hole (85 M⊙ + 66 M⊙ → 142 M⊙) | LIGO/Virgo O3 (2020) | HDF5 | IMBH existence, pair-instability mass gap |
| GW4 | GW170814: first three-detector detection (LIGO H+L + Virgo) | LIGO/Virgo (2017) | HDF5 | Sky localization via triangulation |
| GW5 | GWTC-3 catalog (90 confident BBH + BNS + NSBH events through O3b) | LIGO/Virgo/KAGRA (2021) — [GWOSC](https://gwosc.org) | HDF5 + JSON catalog, ~2 GB total | Population properties: mass, spin, redshift distributions |

### 8.3 CMB (Planck)

| # | Experiment / Measurement | Source | Format | Use |
|---|---|---|---|---|
| CM1 | **Planck 2018: TT, TE, EE power spectra + lensing** | [Planck Legacy Archive](https://pla.esac.esa.int) | FITS + likelihood code, ~10 MB for spectra | Cosmological parameter precision era |
| CM2 | Planck: cosmological parameters (6-parameter ΛCDM + extensions) | Planck 2018 | Table: H₀=67.36, Ω_m=0.315, etc. | Baseline cosmology |
| CM3 | Planck: constraints on N_eff, Σm_ν, running of n_s, tensor-to-scalar ratio r | Planck 2018 | Parameter constraints, ~20 extensions | CMB constraints on BSM physics |
| CM4 | ACT/SPT: high-resolution CMB (small angular scales, ℓ up to ~5000) | ACT, SPT (2018-2023) | FITS, ~1 GB | SZ cluster counts, σ_8 constraint |

### 8.4 Large-Scale Structure

| # | Experiment / Measurement | Source | Format | Use |
|---|---|---|---|---|
| LSS1 | SDSS DR16 eBOSS: BAO + RSD at multiple redshifts | SDSS (2020) | ASCII tables, ~1 MB | Growth of structure, fσ_8(z) |
| LSS2 | DES Y3: 3×2pt correlation functions (cosmic shear, galaxy-galaxy lensing, clustering) | DES (2021) | [DES Data](https://des.ncsa.illinois.edu/releases/y3a2) FITS, ~50 MB | σ_8 tension with Planck |
| LSS3 | KiDS-1000: weak lensing | KiDS (2020) | FITS | S_8 = σ_8(Ω_m/0.3)^0.5 constraint |
| LSS4 | DESI DR1: BAO at z=0.5-3.5 (first year, 2024) | [DESI](https://data.desi.lbl.gov) | FITS, ~10 GB | Most precise BAO to date, dark energy evolution |

### 8.5 Anomalies and Tensions (current open problems)

| # | Measurement | Value | Significance | Source |
|---|---|---|---|---|
| AN1 | Muon g-2 anomaly (FNAL 2023 + BNL) | a_μ(exp) - a_μ(SM) = (249 ± 48)×10⁻¹¹ | 5.1σ depending on SM prediction | Fermilab Muon g-2 |
| AN2 | Hubble tension (SH0ES vs Planck) | H₀ = 73.0 ± 1.0 (SH0ES) vs 67.4 ± 0.5 (Planck) | 5σ | [Riess 2022](https://arxiv.org/abs/2112.04510) |
| AN3 | S_8 tension (weak lensing vs CMB) | S_8 from DES/KiDS ≈ 0.76, Planck ≈ 0.83 | 2-3σ | Multiple surveys |
| AN4 | W boson mass (CDF 2022) | m_W = 80,433.5 ± 9.4 MeV | 7σ from SM global fit | CDF, but ATLAS agrees with SM |
| AN5 | b→sμμ anomalies (B→K*μμ, B_s→φμμ) | Lepton universality in rare B decays | ~3σ in some observables | LHCb, Belle II |
| AN6 | Xenon 1T excess (electronic recoil at ~2.4 keV) | Excess at 3.3σ | Could be solar axions, tritium background | XENON1T (2020) |

### 8.6 Direct Detection Constraints

| # | Experiment | Constraint | Source |
|---|---|---|---|
| DD1 | LUX-ZEPLIN (LZ): spin-independent WIMP-nucleon cross-section | σ_SI < 9.2×10⁻⁴⁸ cm² at 36 GeV | LZ (2023) |
| DD2 | XENONnT: similar WIMP limits + electronic recoil | Comparable sensitivity | XENON (2023) |
| DD3 | ADMX: axion-photon coupling (g_aγγ) | g_aγγ < 10⁻¹⁵ GeV⁻¹ in 2-4 μeV range | ADMX (2021) |
| DD4 | Neutrinoless double beta decay: m_ββ limits | m_ββ < 0.036-0.156 eV (depending on nuclear matrix elements) | KamLAND-Zen 800 (2023) |
| DD5 | Neutron EDM: |d_n| | < 1.8×10⁻²⁶ e·cm | PSI nEDM (2020) |
| DD6 | Proton decay: τ(p→e⁺π⁰) | > 2.4×10³⁴ years | Super-Kamiokande |

**Era 8 total:** ~900 data points + distributional comparisons.

---

## Summary: All Data by Category

| Category | Eras | Approximate data points | File downloads needed |
|----------|------|------------------------|---------------------|
| Celestial mechanics | 1 | 80 | None (published tables) |
| Terrestrial mechanics | 1 | 30 | None |
| Optics (classical + wave) | 1-2 | 550 | None (published tables) |
| Electromagnetism | 2-3 | 150 | None |
| Thermodynamics | 1-3 | 150 | None |
| Blackbody radiation | 3 | 60 (curves) | None (digitized from papers) |
| Atomic spectra | 3-4 | 250 | None (NIST ASD) |
| Relativity (SR + GR tests) | 4, 6 | 50 | None (published) |
| Quantum mechanics (foundations) | 4 | 100 | None |
| Nuclear physics | 4-5 | 200 | None (AME tables) |
| QED precision | 5 | 20 | None (published values) |
| Weak interactions | 5 | 50 | None |
| Hadron physics / quarks | 5-7 | 100 | None (PDG) |
| Neutrino physics | 6-7 | 40 | None (NuFIT) |
| CMB | 6-8 | 30 (spectra) | FITS files, ~20 MB |
| Dark energy / cosmology | 7-8 | 40 (spectra + tables) | FITS + ASCII, ~60 MB |
| Flavor / CP violation | 6-7 | 30 | None (PDG) |
| LHC Higgs + EW | 8 | 20 | JSON from HEPData, ~10 MB |
| Gravitational waves | 8 | 15 events × 2 detectors | HDF5, ~2 GB |
| Large-scale structure | 8 | 20 (spectra + corr. functions) | FITS, ~60 GB |
| Anomalies / tensions | 8 | 15 | Published values |
| Direct detection limits | 8 | 15 | Published exclusion curves |
| Condensed matter | 5 | 80 | None (published) |
| **TOTAL** | **1-8** | **~3,000 data points + distributional comparisons** | **~62 GB (full) / ~100 MB (initial)** |

---

## Implementation Plan

### Phase A — Constants (no download, code only)
Status: **Can start now**
File to create: `src/data/physical/constants.py`

Encode all Era 1-8 constants as Python dataclasses:
- PDG particle properties (masses, charges, spins, widths)
- CODATA fundamental constants
- Cosmological parameters (Planck 2018)
- Atomic spectral lines (H, He, alkali metals)
- Nuclear binding energies + magic numbers
- CKM and PMNS matrix elements
- Neutrino oscillation parameters
- Anomalies (g-2, H₀ tension, W mass, etc.)

### Phase B — Spectral/Curve Data (small downloads, ~100 MB)
Status: **Can start after Phase A**

Download and parse:
- CMB power spectra (FITS from Planck, ~20 MB)
- LEP Z-pole measurements (JSON from HEPData, ~10 MB)
- LHC Higgs couplings (JSON from HEPData, ~5 MB)
- Sne Ia Hubble diagram (ASCII, ~30 MB)
- BAO distance measurements (ASCII, ~5 MB)
- Blackbody spectra (digitized historical curves)
- GW event strain data (HDF5 from GWOSC, ~50 MB per event × 2 events initially)

### Phase C — Full Survey Data (large, on demand)
Status: **After Phase B validation**

- Full GWTC-3 catalog (~2 GB)
- Planck maps (if needed beyond power spectra, ~5 GB)
- ATLAS/CMS Open Data subsamples (~20 GB)
- SDSS/DESI spectra (subsamples, ~10 GB)

---

## Temporal Gating Implementation

Each data point should carry a `discovery_year` field. The evaluator can then filter:

```python
# Train on pre-1905 data only
available_data = [d for d in DATA if d.discovery_year <= 1904]

# Evaluate: does the system's structure predict...
# - Mercury perihelion precession of 43"/century? (was known, should reproduce)
# - Michelson-Morley null result? (was known, should reproduce)
# - Special relativity? (was NOT known — DISCOVERY test)
# - General relativity? (was NOT known — DISCOVERY test)
```

This transforms the system evaluation from "can it fit known physics?" to "can it discover physics the way humans did, from the same experimental constraints?" — which is a much stronger test of whether the architecture does genuine science.

---

*Generated 2026-06-03. Reorganized chronologically by era with temporal gating as the evaluation framework.*
