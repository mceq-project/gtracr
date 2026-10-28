"""Rigidity-aware escape criterion shared by Trajectory and GMRC.

A back-traced trajectory is allowed once it reaches the escape radius

    r_esc = max(escape_radius, k * C_St),   C_St = RE * sqrt(B0 * RE * |q| / p),

within the time budget

    t_max = max(max_time, N * r_esc / (beta * c)),

where C_St is the Störmer length for momentum p and B0 the dipole strength
sqrt(g10^2 + g11^2 + h11^2) of the active field model. At high rigidity the
fixed escape_radius / max_time apply unchanged; at low rigidity the sphere
and the budget grow so that slowly escaping trajectories are not counted as
forbidden. k = N = 0 restores the fixed criterion.
"""

from gtracr._libgtracr import DEFAULT_ESCAPE_FACTOR, DEFAULT_TIME_FACTOR

__all__ = [
    "DEFAULT_ESCAPE_FACTOR",
    "DEFAULT_TIME_FACTOR",
    "resolve_escape_scaling",
    "set_escape_scaling",
]


def resolve_escape_scaling(escape_factor=None, time_factor=None):
    """Return (escape_factor, time_factor) with ``None`` replaced by the defaults."""
    k = DEFAULT_ESCAPE_FACTOR if escape_factor is None else float(escape_factor)
    n = DEFAULT_TIME_FACTOR if time_factor is None else float(time_factor)
    if k < 0 or n < 0:
        raise ValueError("escape_factor and time_factor must be non-negative")
    return k, n


def set_escape_scaling(tracer, escape_factor=None, time_factor=None):
    """Apply the escape criterion to a C++ ``TrajectoryTracer``."""
    tracer.set_escape_scaling(*resolve_escape_scaling(escape_factor, time_factor))
    return tracer
