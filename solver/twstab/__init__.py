"""twstab: spectral-stability labels for travelling waves of 1D PDEs.

Design document: ../DESIGN.md.  Implemented: local systems (class L, parabolic of even order) including
partly parabolic systems, cross-diffusion and continuation of the wave in a
parameter; constraint equations (reducible nonlocal terms); multi-pulses; free-boundary problems
with a Stefan condition (class F: Dirichlet, Neumann and Robin conditions, components without
diffusion); discrete and distributed time delays (class D); convolution integrals with a general kernel
(class N), with or without diffusion; equations with second-order time derivatives.
"""
__version__ = "2.18.0"

from .batch import read_list, solve          # noqa: E402  (a list of equations in, a list of labels out)
