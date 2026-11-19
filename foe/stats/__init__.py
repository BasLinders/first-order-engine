"""
FOE's native statistics layer.

This is the only package in FOE that may import third-party statistical code,
and it imports only numerical primitives (scipy.special, scipy.optimize). The
engines import their distributions and tests from here, so all statistical
logic and conventions live in FOE. tests/test_dependency_boundary.py enforces
this.
"""
