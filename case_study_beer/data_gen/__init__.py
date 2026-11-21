"""Synthetic data generators for the beer retail case study."""
from .synthetic import make_multi_arm_data, mix_assignment, true_best_arm
from .beer_promo import make_beer_promo_data

__all__ = ["make_multi_arm_data", "mix_assignment", "true_best_arm", "make_beer_promo_data"]
