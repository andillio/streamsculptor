import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import Potential


class LinearModifier:
    def __init__(self, scale=1.0):
        self.scale = scale

    def potential(self, xyz, t):
        return self.scale * xyz[0]


def test_release_model_uses_potential_modifier():
    class SimplePotential(Potential):
        def __init__(self):
            super().__init__(units=None, params={})

        def potential(self, xyz, t):
            return 0.0

    pot = SimplePotential()
    x = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    modifier = LinearModifier(scale=0.1)

    lead, trail, v_lead, v_trail = pot.release_model(
        x=x,
        v=v,
        Msat=1.0,
        i=0,
        t=0.0,
        seed_num=0,
        kval_arr=1.0,
        potential_modifier=modifier,
    )

    assert lead.shape == (3,)
    assert trail.shape == (3,)
    assert v_lead.shape == (3,)
    assert v_trail.shape == (3,)
    assert not np.allclose(lead, trail)


def test_gen_stream_scan_accepts_potential_modifier():
    class SimplePotential(Potential):
        def __init__(self):
            super().__init__(units=None, params={})

        def potential(self, xyz, t):
            return 0.0

    pot = SimplePotential()
    ts = jnp.array([0.0, 1.0])
    prog_w0 = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    Msat = jnp.array([1.0, 1.0])
    modifier = LinearModifier(scale=0.1)

    lead, trail = pot.gen_stream_scan(
        ts=ts,
        prog_w0=prog_w0,
        Msat=Msat,
        seed_num=0,
        kval_arr=1.0,
        potential_modifier=modifier,
    )

    assert lead.shape[0] == 1
    assert trail.shape[0] == 1
    assert lead.shape[1] == 6
    assert trail.shape[1] == 6
