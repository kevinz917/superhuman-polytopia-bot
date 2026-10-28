import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from polytopia_bot.gpu import random

DATA=json.loads((Path(__file__).parent/'fixtures/dotnet-rng.json').read_text())


@jax.jit
def samples(seed):
    rng=random.initialize(seed)
    def run(fn):
        return jax.lax.scan(lambda r,_:fn(r),rng,None,length=128)[1]
    return run(random.next_int),run(lambda r:random.bounded(r,196)),run(random.unit_float)


@pytest.mark.parametrize('case',DATA['cases'],ids=lambda c:str(c['seed']))
def test_device_rng_matches_microsoft_runtime(case):
    actual=samples(jnp.int32(case['seed']))
    for got,key in zip(actual,('samples','bounded','unit_float'),strict=True):
        np.testing.assert_array_equal(got,np.asarray(case[key],dtype=got.dtype))


def test_bounded_rng_visits_both_halves_of_the_map():
    _,values,_=samples(jnp.int32(7))
    assert (values<98).any() and (values>=98).any()
    assert values.min()>=0 and values.max()<196
