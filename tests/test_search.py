"""Turn search on engine-rs (rl/search.py): raw greedy turns, robust determinized search, the belief invariant."""
import jax
import jax.numpy as jnp
import numpy as np

SIZE = 14


def _params(seed=0):
    from polytopia_bot.rl.distributed import _template_params
    template = _template_params(SIZE)
    keys = jax.random.split(jax.random.key(seed), len(jax.tree.leaves(template)))
    leaves = [jax.random.normal(k, a.shape) * .2 for k, a in zip(keys, jax.tree.leaves(template))]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


def _start(seed=5):
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.search import first
    g = RustGames(SIZE, 1)
    g.reset(np.array([seed], np.int32))
    return first(g.states())


def test_raw_turns_are_legal_and_alternate():
    from polytopia_bot.rl.search import TurnSearcher
    raw = TurnSearcher(_params(), size=SIZE, lanes=1)
    s, key = _start(), jax.random.key(0)
    for _ in range(6):
        mover = int(s.current_player)
        key, draw = jax.random.split(key)
        s, plans = raw.play_turn(s, draw)
        assert plans and all(int(p.best) == 0 for p in plans)
        assert int(plans[-1].actions[int(plans[-1].count) - 1][0]) == 15   # the turn ends with END
        assert int(s.current_player) == 3 - mover or bool(s.terminated | s.truncated)


def test_raw_turn_is_the_greedy_policy():
    """K=1 plays network.sample(deterministic=True) at every decision."""
    from polytopia_bot.rl import network
    from polytopia_bot.rl.search import Lanes, TurnSearcher, _Nets, one
    params = _params(1)
    s = _start(7)
    _, plans = TurnSearcher(params, size=SIZE, lanes=1).play_turn(s, jax.random.key(3))
    lanes, nets = Lanes(SIZE, 1), _Nets(SIZE)
    lanes.load(one(s))
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    layout = view_layout(SIZE)
    for row in plans[0].actions[:int(plans[0].count)]:
        masks = jax.tree.map(lambda m: jnp.asarray(m[0]), lanes.masks())
        x = view_features(lanes.views()[0], layout, SIZE)
        choice = network.sample(params, x, masks, jax.random.key(0), size=SIZE, deterministic=True)[0]
        assert [int(choice.family), int(choice.source), int(choice.target)] == [int(v) for v in row]
        assert lanes.step(*([int(v)] for v in row))[0] == 0


def test_robust_determinized_search_plays_legal_turns():
    from polytopia_bot.rl.search import TurnSearcher
    searcher = TurnSearcher(_params(2), size=SIZE, lanes=4, depth=1, rollout_temperature=1., replies=8,
                            determinize=True)
    s, key = _start(9), jax.random.key(1)
    for _ in range(4):
        mover = int(s.current_player)
        key, draw = jax.random.split(key)
        s, plans = searcher.play_turn(s, draw)
        assert plans
        for p in plans:
            assert 0 <= int(p.best) < 4 and p.scores.shape == (4,) and np.isfinite(p.scores).all()
        assert int(s.current_player) == 3 - mover or bool(s.terminated | s.truncated)


def test_belief_keeps_the_searchers_view():
    """kernels.observe(determinize(s), me) == kernels.observe(s, me), read through engine-rs's observe."""
    from polytopia_bot.rl.belief import build_belief
    from polytopia_bot.rl.search import Lanes, TurnSearcher, one
    raw = TurnSearcher(_params(3), size=SIZE, lanes=1)
    s, key = _start(11), jax.random.key(4)
    for _ in range(8):                       # some play first, so there is fog with something in it
        key, draw = jax.random.split(key)
        s, _ = raw.play_turn(s, draw)
    me = int(s.current_player)
    pair = Lanes(SIZE, 2)
    belief = build_belief(size=SIZE)
    for k in range(3):
        sample = jax.tree.map(np.asarray, belief(s, jax.random.key(k), 3))
        pair.load(jax.tree.map(lambda a, b: np.stack([np.asarray(a), np.asarray(b)]), s, sample))
        views = pair.views(me)
        assert np.array_equal(views[0], views[1])
        assert pair.masks().end[1]           # the sampled world is a playable position
