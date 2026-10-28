from functools import partial
import gzip
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from polytopia_bot.gpu import round_limit
from polytopia_bot.gpu.mapgen import layout, reset
from polytopia_bot.gpu.actions import legal_masks
from polytopia_bot.gpu.rules import load_rules

DATA=json.loads(gzip.decompress((Path(__file__).parent/'fixtures/corrected-drylands.json.gz').read_bytes()))
generate=jax.jit(layout,static_argnames=('size',))


@pytest.mark.parametrize('case',DATA['cases'],ids=lambda c:f"{c['size']}-{c['seed']}")
def test_gpu_layout_matches_corrected_sampling_reference(case):
    result=generate(jnp.int32(case['seed']),size=case['size'])
    for key,value in zip(result._fields,result,strict=True):
        np.testing.assert_array_equal(value,case[key],err_msg=key)


@pytest.mark.parametrize('size',[11,14,18])
def test_procedural_reset_has_complete_bardur_start_and_legal_actions(size):
    start=jax.jit(partial(reset,size=size))(jnp.int32(7))
    assert (start.tiles.terrain>=3).sum()>=int(size*size*.94)
    np.testing.assert_array_equal(start.players.stars,[5,5])
    assert start.players.tech[:,0].all() and start.players.tech[:,15].all()
    assert start.players.tech.sum()==4
    assert (start.cities.owner>0).sum()==2 and (start.units.owner>0).sum()==2
    assert (start.tiles.structure==47).sum()==4
    assert start.reference_max_turns==round_limit(size) and start.core_supported
    assert not start.terminated and not start.truncated
    for player in (1,2):
        at=int(jnp.argmax(start.units.owner==player))
        assert start.units.kind[at]==2 and start.units.health[at]==10
        assert start.units.home[at]==at+1
        assert start.cities.territory[at].sum()==9
        assert (start.tiles.explored[player-1]& (start.tiles.resource==1)).sum()>=2
    masks=jax.jit(partial(legal_masks,rules=load_rules(),size=size))(start)
    assert masks.move.any() and masks.harvest.any() and masks.research.any() and masks.end
