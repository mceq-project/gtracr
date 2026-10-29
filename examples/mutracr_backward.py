"""Backward source integration, including a signed derivative column.

Run: python examples/mutracr_backward.py
No GPU, external cascade database or Python stepping callback is required.
"""

import numpy as np

from gtracr import MuonTracer
from gtracr.mutracr import BACKWARD_FATES

tracer = MuonTracer(bfield="dipole", atmosphere="usstd", h_escape=1.128e7)
x = [[0.0, 0.0, tracer.r_ground + 1e6]]
p = [[0.0, 0.0, -2.0]]  # physical forward momentum, even when tracing backward

# Toy intensity source per cm: nominal and signed response columns, separate
# charges, altitude and local zenith direction. Replace with cascade sources.
h = np.linspace(0, 1.128e7, 100)
mu = np.linspace(-1, 1, 25)
energy = np.geomspace(0.11, 1e3, 101)  # TOTAL GeV
base = (
    np.exp(-(((h[:, None, None] - 1.7e6) / 8e5) ** 2))
    * (1 - 0.2 * mu[None, :, None])
    * energy[None, None, :] ** -2.7
)
values = np.stack([base, -0.3 * base], axis=-1)
values = np.stack([values, 0.9 * values], axis=0)
source = dict(h_cm=h, cos_theta=mu, e_mu=energy, values=values)
result = tracer.trace_backward(x, p, [-1], bank=False, source=source)
print("fate:", BACKWARD_FATES[int(result["fate"][0])])
print("survival to observation:", result["survival"][0])
print("phase-space focusing:", result["jacobian"][0])
print("nominal and signed source integral:", result["source_integral"][0])
