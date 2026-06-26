"""Additional instrument simulators for relativistic claims."""

from __future__ import annotations

import math
import random


class Thermocouple:
    """Type-K thermocouple measuring temperature via Seebeck voltage."""

    def __init__(self, sensitivity_uv_per_k: float = 41.0,
                 voltage_noise_uv: float = 0.5,
                 cold_junction_error_k: float = 0.5,
                 calibration_uncertainty: float = 0.002,
                 rng: random.Random | None = None):
        self.sensitivity = sensitivity_uv_per_k
        self.noise = voltage_noise_uv
        self.cj_error = cold_junction_error_k
        self.rng = rng or random.Random()
        self.sens_actual = self.sensitivity * (1 + self.rng.gauss(0, calibration_uncertainty))

    def measure(self, temp_true_k: float) -> float:
        """Measure temperature. Returns Kelvin."""
        v_true = temp_true_k * self.sens_actual
        v_meas = v_true + self.rng.gauss(0, self.noise)
        return v_meas / self.sensitivity + self.rng.gauss(0, self.cj_error)


class BlackbodySpectrometer:
    """Spectrometer measuring peak wavelength of blackbody radiation."""

    def __init__(self, wavelength_noise_frac: float = 0.005,
                 calibration_uncertainty: float = 0.003,
                 rng: random.Random | None = None):
        self.noise_frac = wavelength_noise_frac
        self.rng = rng or random.Random()
        self.cal_factor = 1.0 + self.rng.gauss(0, calibration_uncertainty)

    def measure(self, lambda_peak_true: float) -> float:
        """Measure peak wavelength. Returns meters."""
        lam_meas = lambda_peak_true * self.cal_factor
        lam_meas += self.rng.gauss(0, lam_meas * self.noise_frac)
        return max(0, lam_meas)


class LaserRangefinder:
    """Phase-shift laser rangefinder measuring distance."""

    def __init__(self, distance_noise_m: float = 0.1,
                 calibration_ppm: float = 5.0,
                 rng: random.Random | None = None):
        self.noise = distance_noise_m
        self.rng = rng or random.Random()
        self.cal = 1.0 + self.rng.gauss(0, calibration_ppm * 1e-6)

    def measure(self, distance_true_m: float) -> float:
        """Measure distance. Returns meters."""
        d_meas = distance_true_m * self.cal + self.rng.gauss(0, self.noise)
        return max(0, d_meas)


class AtomicClock:
    """Cesium atomic clock measuring time intervals."""

    def __init__(self, jitter_s: float = 1e-12,
                 drift_ppb: float = 0.01,
                 rng: random.Random | None = None):
        self.jitter = jitter_s
        self.rng = rng or random.Random()
        self.drift = 1.0 + self.rng.gauss(0, drift_ppb * 1e-9)

    def measure(self, time_true_s: float) -> float:
        """Measure time interval. Returns seconds."""
        return max(0, time_true_s * self.drift + self.rng.gauss(0, self.jitter))


class PhotonCounter:
    """Single-photon counter measuring discrete energy levels."""

    def __init__(self, efficiency: float = 0.8,
                 dark_count_rate: float = 10.0,
                 integration_time_s: float = 1.0,
                 rng: random.Random | None = None):
        self.efficiency = efficiency
        self.rng = rng or random.Random()
        self.dark_counts = dark_count_rate * integration_time_s

    def measure(self, energy_true_ev: float) -> float:
        """Measure photon energy. Returns eV."""
        # Detection probability
        if self.rng.random() > self.efficiency:
            return 0.0
        # Dark count noise
        bg = self.rng.gauss(0, self.dark_counts * energy_true_ev * 0.01)
        return max(0, energy_true_ev + bg)
