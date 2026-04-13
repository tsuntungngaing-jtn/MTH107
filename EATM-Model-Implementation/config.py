"""
Global physical and expectancy parameters for the EATM implementation.

Values follow the Expectancy-Adjusted Thermal Model (EATM) specification
documented in the accompanying study (metabolic baselines, time constants,
expectancy gain, and boundary-defense thresholds).
"""

from __future__ import annotations

# Expectancy and PMV mapping
LINEAR_EXPECTANCY_GAIN_G: float = 0.303

# Time constants (seconds)
TAU_PHYSIOLOGICAL_S: float = 900.0  # thermal / metabolic inertia (approx. 15 min)
TAU_EXPECTANCY_S: float = 72.0  # rapid psychological expectancy decay in transit

# Metabolic rate baselines (met)
METABOLIC_PLATFORM_MET: float = 1.6
METABOLIC_TRAIN_MET: float = 1.2

# Anthropometry (DuBois area normalization when required by other modules)
PARTICIPANT_MASS_KG: float = 70.0
PARTICIPANT_HEIGHT_M: float = 1.75  # typical adult stature 

# Skin wettedness and boundary defense
CRITICAL_SKIN_WETTEDNESS: float = 0.25  # ISO 7730 sticky-discomfort threshold
BOUNDARY_DEFENSE_STEEPNESS_K: float = 44.0  # logistic steepness (high-fidelity setting)

# Asymmetric retention of expectancy after a SET step-change
ALPHA_RETENTION_COOLING_STEP: float = 0.6  # Delta SET < 0 (relief / cooling)
ALPHA_RETENTION_WARMING_STEP: float = 0.4  # Delta SET > 0 (warming stress)

# Refined skin wettedness split (dimensionless weights)
BASELINE_SKIN_WETTEDNESS: float = 0.06
ACTIVE_SKIN_WETTEDNESS_WEIGHT: float = 0.94

# Initial heat-storage rate scale (W/m^2); calibrate with scenario energy budget
DEFAULT_HEAT_STORAGE_INITIAL_WM2: float = 35.0

# Stefan-Boltzmann constant (W / (m^2 * K^4))
STEFAN_BOLTZMANN: float = 5.670374419e-8

# Long-wave emissivity band for clothing / skin (dimensionless)
EMISSIVITY_CLOTHING: float = 0.93

# Radiative view-factor parameters (crowd shielding and saturation)
VIEW_FACTOR_MAX_PP: float = 0.85
VIEW_FACTOR_INTERPERSONAL_RATE_KP: float = 0.35  # density-driven saturation of F_p-p
VIEW_FACTOR_WALL_BASELINE: float = 0.45
VIEW_FACTOR_WALL_DECAY_KW: float = 0.40  # occlusion of wall radiation with density (spec)

# Environmental attenuation (SI-friendly tuning constants)
FLOW_ATTENUATION_GAMMA: float = 0.40  # exp(-gamma * D) in local velocity mapping (spec)
MIN_LOCAL_AIR_VELOCITY_MS: float = 0.001  # epsilon floor on local velocity (m/s)
METABOLIC_VAPOR_COUPLING_ETA: float = 0.022  # calibrated so 1 + eta*9 ≈ 1.20 at peak density

# Clothing compression / moisture correction (Icl,final = Icl,0 * f_clo * f_wet)
CLOTHING_COMPRESSION_K: float = 0.10  # scales crowding-induced clo reduction with D (m^2 / person)
CLOTHING_WETNESS_DECAY_ALPHA: float = 3.0  # steepness for f_wet(omega) logistic floor

# Metabolic rate for vapor coupling normalization (met)
METABOLIC_TRAIN_REFERENCE_MET: float = METABOLIC_TRAIN_MET

# Newton-Raphson defaults for coupled SB solves
NEWTON_TOLERANCE: float = 1e-6
NEWTON_MAX_ITERATIONS: int = 60

# ISO 7730 metabolic conversion (W/m^2 per met) for skin-temperature proxy
METABOLIC_W_PER_M2_PER_MET: float = 58.15

# Dynamic PMV bracket (ISO 7730 / CBE kernel): PMV_dyn = (a*exp(b*m)+c0)*L, m in W/m^2
PMV_DYNAMIC_SENSITIVITY_A: float = 0.303  # matches classic 0.303 gain (report G alignment)
PMV_DYNAMIC_EXPONENT_PER_WM2: float = -0.036  # argument multiplies absolute metabolic rate (W/m^2)
PMV_DYNAMIC_BRACKET_INTERCEPT: float = 0.028

# GB 50157 peak crowding reference (persons per square metre)
MAX_OCCUPANT_DENSITY_PER_M2: float = 9.0

# Clothing compression lower bound (dimensionless) to avoid non-physical clo collapse
CLOTHING_COMPRESSION_FLOOR: float = 0.7
