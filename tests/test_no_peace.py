"""no_peace (training harness, LearnerConfig.no_peace -> RustGames(no_peace=True)): the mover can never propose or
accept peace; a pending request can only be declined. Off by default, so the rules' legal set is unchanged."""
import numpy as np
import pytest

SIZE = 14
DIPLOMATIC = 14
DECLINE, ACCEPT, OFFER = 1, 2, 4


def _game(no_peace):
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(SIZE, 1, no_peace=no_peace)
    g.batch.reset(np.array([31], np.int32))
    return g


def _load(no_peace, edit):
    d = {k: np.array(v) for k, v in _game(no_peace).batch.export().items()}
    edit(d)
    g = _game(no_peace)
    g.batch.import_({k: np.ascontiguousarray(v) for k, v in d.items()})
    return g


def _diplomatic(g):
    from polytopia_bot.rl.rust_engine import split_masks
    return np.asarray(split_masks(g.batch.legal_masks(), SIZE)[DIPLOMATIC])[0].reshape(-1)


def _can_offer(d):
    mover = int(d['current_player'][0])
    d['players.known'][0] = True
    d['players.tech'][0, mover - 1, 7] = True        # the tech that allows a peace treaty


def _pending(d):
    _can_offer(d)
    mover = int(d['current_player'][0])
    d['players.peace_offer'][0, mover - 1, 2 - mover] = True   # the other seat asked the mover


@pytest.mark.parametrize('no_peace', [False, True])
def test_offer_peace(no_peace):
    m = _diplomatic(_load(no_peace, _can_offer))
    assert bool(m[OFFER]) is (not no_peace)


@pytest.mark.parametrize('no_peace', [False, True])
def test_pending_request_can_only_be_declined(no_peace):
    m = _diplomatic(_load(no_peace, _pending))
    assert m[DECLINE] and bool(m[ACCEPT]) is (not no_peace)
    # nothing else is legal while the request is pending (resigning is off in training)
    from polytopia_bot.rl.rust_engine import split_masks
    g = _load(no_peace, _pending)
    others = [np.asarray(x)[0].any() for f, x in enumerate(split_masks(g.batch.legal_masks(), SIZE)) if f != DIPLOMATIC]
    assert not any(others)
