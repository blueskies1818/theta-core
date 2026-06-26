"""Realistic instrument-simulated data generators.

Each generator simulates actual measurement apparatus — spectrometers,
photodetectors, calorimeters, time-of-flight detectors — each with its
own calibration, noise floor, quantum efficiency, and systematic offsets.

The invariant is NEVER computed by the generator.  It emerges from
independent measurements of the same physical phenomenon.
"""

from __future__ import annotations

import math
import random

from src.physics.observations import Observation

C = 299792458.0
H = 6.62607015e-34
HC = H * C


def _obs(obs_id, name, desc, quantities, params, timesteps, invariant):
    return Observation(
        id=obs_id, name=name, description=desc,
        quantities=quantities, parameters=params,
        timesteps=timesteps, known_invariant=invariant,
        lean_theorem="",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Instrument models
# ═══════════════════════════════════════════════════════════════════════════

class GratingSpectrometer:
    """Simulates a diffraction grating spectrometer.

    Wavelength is determined by the grating equation:
        λ = d * sin(θ) / m
    where d is groove spacing, θ is the diffraction angle, m is order.

    The instrument measures θ (angle) with some precision and has a
    calibration uncertainty in d (groove spacing from thermal expansion).
    """

    def __init__(self, groove_spacing_m: float = 1e-6, order: int = 1,
                 angle_noise_rad: float = 1e-5,
                 calibration_uncertainty: float = 0.001,  # 0.1% groove spacing error
                 rng: random.Random | None = None):
        self.d = groove_spacing_m
        self.m = order
        self.angle_noise = angle_noise_rad
        self.rng = rng or random.Random()
        # Systematic calibration offset (fixed per instrument)
        self.d_actual = self.d * (1.0 + self.rng.gauss(0, calibration_uncertainty))

    def measure(self, wavelength_true: float) -> float:
        """Measure a wavelength. Returns measured wavelength in meters."""
        # True diffraction angle from grating equation
        sin_theta = wavelength_true * self.m / self.d_actual
        sin_theta = max(-1.0, min(1.0, sin_theta))
        theta_true = math.asin(sin_theta)

        # Noisy angle measurement
        theta_meas = theta_true + self.rng.gauss(0, self.angle_noise)

        # Convert back to wavelength using NOMINAL groove spacing
        # (the experimenter doesn't know the true d_actual)
        lam_meas = self.d * math.sin(theta_meas) / self.m
        return max(0, lam_meas)


class PhotodiodeEnergyDetector:
    """Simulates a photodiode measuring photon energy.

    The photodiode produces a current proportional to photon flux.
    For single-photon detection (APD), the pulse height is proportional
    to photon energy with some quantum efficiency curve.

    Measures: voltage → energy via calibration factor.
    """

    def __init__(self, calibration_ev_per_volt: float = 1.0,
                 voltage_noise_v: float = 0.01,
                 dark_current_offset_v: float = 0.001,
                 calibration_uncertainty: float = 0.002,
                 rng: random.Random | None = None):
        self.cal = calibration_ev_per_volt
        self.voltage_noise = voltage_noise_v
        self.dark_offset = dark_current_offset_v
        self.rng = rng or random.Random()
        self.cal_actual = self.cal * (1.0 + self.rng.gauss(0, calibration_uncertainty))

    def measure(self, energy_true_ev: float) -> float:
        """Measure photon energy. Returns measured energy in eV."""
        # True voltage (including dark current offset)
        v_true = energy_true_ev / self.cal_actual + self.dark_offset
        # Noisy voltage measurement
        v_meas = v_true + self.rng.gauss(0, self.voltage_noise)
        # Convert to energy using NOMINAL calibration
        return max(0, (v_meas - self.dark_offset) * self.cal)


class TimeOfFlightVelocityDetector:
    """Simulates a time-of-flight detector for particle velocity.

    Measures time over a known distance.  Distance has calibration
    uncertainty, timing has jitter.
    """

    def __init__(self, baseline_m: float = 10.0,
                 timing_jitter_s: float = 1e-10,
                 distance_uncertainty: float = 0.0005,
                 rng: random.Random | None = None):
        self.baseline = baseline_m
        self.jitter = timing_jitter_s
        self.rng = rng or random.Random()
        self.baseline_actual = self.baseline * (1.0 + self.rng.gauss(0, distance_uncertainty))

    def measure(self, velocity_true_m_s: float) -> float:
        """Measure velocity. Returns measured velocity in m/s."""
        # True time of flight
        if velocity_true_m_s <= 0:
            return 0.0
        t_true = self.baseline_actual / velocity_true_m_s
        # Noisy timing
        t_meas = t_true + self.rng.gauss(0, self.jitter)
        if t_meas <= 0:
            return 1e12
        # Convert to velocity using NOMINAL baseline
        return self.baseline / t_meas


class Calorimeter:
    """Simulates a calorimeter measuring total particle energy.

    Measures temperature rise in an absorber → energy via heat capacity.
    Has thermal noise, calibration uncertainty, and background.
    """

    def __init__(self, heat_capacity_j_per_k: float = 1e-6,
                 temp_noise_k: float = 1e-5,
                 background_temp_rise_k: float = 0.0,
                 calibration_uncertainty: float = 0.003,
                 rng: random.Random | None = None):
        self.capacity = heat_capacity_j_per_k
        self.temp_noise = temp_noise_k
        self.background = background_temp_rise_k
        self.rng = rng or random.Random()
        self.capacity_actual = self.capacity * (1.0 + self.rng.gauss(0, calibration_uncertainty))

    def measure(self, energy_true_j: float) -> float:
        """Measure energy. Returns measured energy in joules."""
        dT_true = energy_true_j / self.capacity_actual + self.background
        dT_meas = dT_true + self.rng.gauss(0, self.temp_noise)
        return max(0, (dT_meas - self.background) * self.capacity)


class Monochromator:
    """Simulates a monochromator selecting wavelength/frequency.

    A grating or prism selects a narrow band of wavelengths. The dial
    setting has calibration error and the output has some bandwidth.
    """

    def __init__(self, calibration_uncertainty: float = 0.002,
                 bandwidth_fraction: float = 0.01,
                 rng: random.Random | None = None):
        self.cal_error = calibration_uncertainty
        self.bandwidth = bandwidth_fraction
        self.rng = rng or random.Random()

    def set_frequency(self, nu_hz: float) -> float:
        """Set dial to frequency, returns actual output frequency."""
        dial_error = 1.0 + self.rng.gauss(0, self.cal_error)
        actual = nu_hz * dial_error
        # Bandwidth smearing
        actual += self.rng.gauss(0, actual * self.bandwidth)
        return max(0, actual)


class Electrometer:
    """Simulates an electrometer measuring stopping potential.

    Measures voltage required to stop photoelectrons. Has input
    impedance noise, offset, and gain error.
    """

    def __init__(self, gain_v_per_v: float = 1.0,
                 voltage_noise_v: float = 0.005,
                 offset_v: float = 0.0,
                 gain_uncertainty: float = 0.005,
                 rng: random.Random | None = None):
        self.gain = gain_v_per_v
        self.noise = voltage_noise_v
        self.offset = offset_v
        self.rng = rng or random.Random()
        self.gain_actual = self.gain * (1.0 + self.rng.gauss(0, gain_uncertainty))

    def measure(self, voltage_true_v: float) -> float:
        """Measure stopping potential. Returns measured voltage."""
        v_amp = voltage_true_v * self.gain_actual + self.offset
        v_meas = v_amp + self.rng.gauss(0, self.noise)
        return (v_meas - self.offset) / self.gain


# ═══════════════════════════════════════════════════════════════════════════
# Instrument-based data generators
# ═══════════════════════════════════════════════════════════════════════════

def make_hydrogen_balmer_instruments(rng: random.Random) -> list[Observation]:
    """Hydrogen Balmer: spectrometer measures λ, photodiode measures E.

    The generator uses the Rydberg formula (not E*λ!) to compute true
    wavelengths.  Energy is computed from hc/λ by physics, but the
    instruments measure independently with their own noise.

    The system must discover E*λ ≈ constant from two noisy, independently-
    calibrated instruments.
    """
    spectrometer = GratingSpectrometer(
        groove_spacing_m=1e-6, angle_noise_rad=5e-6,
        calibration_uncertainty=0.0005, rng=rng)
    photodiode = PhotodiodeEnergyDetector(
        calibration_ev_per_volt=1.0, voltage_noise_v=0.015,
        calibration_uncertainty=0.001, rng=rng)

    # Rydberg formula: 1/λ = R_H * (1/2² - 1/n²) for Balmer series
    R_H = 1.0967758e7  # Rydberg constant for hydrogen
    n_vals = list(range(3, 15))  # Balmer lines n=3→14

    timesteps = []
    for i, n in enumerate(n_vals):
        # True wavelength from Rydberg formula (NOT from E*λ=hc!)
        lam_true = 1.0 / (R_H * (0.25 - 1.0 / n**2))
        # True energy (nature follows E=hc/λ, but instruments don't know this)
        e_true = HC / lam_true

        # Independent instrument measurements
        lam_meas = spectrometer.measure(lam_true)
        e_meas = photodiode.measure(e_true / 1.602176634e-19) * 1.602176634e-19  # eV → J

        timesteps.append({"t": float(i), "lambda": lam_meas, "E": e_meas})

    return [_obs("h_balmer", "Hydrogen Balmer (instruments)",
        "Spectrometer measures λ, photodiode measures E. "
        "Independent instruments, independent calibration.",
        {"lambda": "Length", "E": "Energy"}, {},
        timesteps, "E*lambda")]


def make_photoelectric_instruments(rng: random.Random) -> list[Observation]:
    """Photoelectric: monochromator sets ν, electrometer measures K_max.

    The generator uses the photoelectric equation internally (as nature
    does), but each instrument introduces its own calibration errors,
    noise floors, and systematic offsets.
    """
    monochromator = Monochromator(
        calibration_uncertainty=0.002, bandwidth_fraction=0.005, rng=rng)
    electrometer = Electrometer(
        voltage_noise_v=0.003, offset_v=0.001,
        gain_uncertainty=0.003, rng=rng)

    H_EV_THZ = 4.135667662e-3  # eV·ps → eV/THz → corrected
    # Actually h = 4.135667662e-15 eV·s = 4.135667662e-3 eV·THz⁻¹ × 10¹²
    H_EV_THZ = 4.135667662  # eV/THz (for ν in THz)
    PHI_EV = 4.5  # work function in eV (typical for Na/K)

    # Below threshold frequencies (in THz)
    nu_below_thz = [200, 400, 700, 1000]
    # Above threshold
    nu_above_thz = [1200, 1500, 2000, 2500, 3000, 3500]

    observations = []

    for nu_set, label in [(nu_below_thz, "below"), (nu_above_thz, "above")]:
        for nu_target in nu_set:
            # Monochromator sets frequency in THz
            nu_actual = monochromator.set_frequency(nu_target)  # THz

            # Nature: K_max = max(0, h*ν - φ) / e  (stopping potential)
            e_charge = 1.0  # in eV units
            k_max_true = max(0.0, H_EV_THZ * nu_actual - PHI_EV)

            # Electrometer measures stopping potential
            k_max_meas = electrometer.measure(k_max_true)

            # Create two timesteps with slight frequency variation
            # (monochromator drift between measurements)
            nu2 = monochromator.set_frequency(nu_target)  # THz
            k_max2 = max(0.0, H_EV_THZ * nu2 - PHI_EV)
            k_max_meas2 = electrometer.measure(k_max2)

            observations.append(_obs(
                f"pe_{label}_{nu_target:.0f}", f"Photoelectric ({label} threshold)",
                f"Monochromator at ~{nu_target/1e12:.1f} THz, electrometer measures K_max.",
                {"nu": "Scalar", "K_max": "Energy", "h": "Energy*Time"},
                {"h": 4.135667662},  # eV/THz
                [{"t": 0.0, "nu": nu_actual, "K_max": k_max_meas},
                 {"t": 1.0, "nu": nu2, "K_max": k_max_meas2}],
                "h*nu - K_max" if label == "above" else "K_max",
            ))

    return observations
