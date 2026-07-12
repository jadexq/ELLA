"""ELLA v1: subcellular RNA localization analysis.

Over-dispersed nonhomogeneous Poisson model with radial expression-gradient
kernels; the alternative fit uses a bounded-Newton solver (deterministic global
optimum). This makes ``ELLA`` a real importable package so ``pip install -e .``
registers a working ``from ELLA import ELLA`` (no sys.path juggling needed).
"""

from ELLA.ELLA import ELLA, newton_fit_gene, polygon_boundary_radius, subset_prepared

__all__ = ["ELLA", "newton_fit_gene", "polygon_boundary_radius", "subset_prepared"]
