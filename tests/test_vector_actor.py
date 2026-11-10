"""Vectorized actor (rl/vector_actor.py) and the engine pieces it relies on.

1. GameBatch.set_threads: a tick stepped on worker threads equals the sequential tick, output for output, over
   random legal games with resets (classic seeds and explicit layouts), dense and packed masks.
2. Light states: the post-tick scalars equal the full export's fields.
3. RowStore + segment emission: the vectorized collector emits exactly the rows and segments SegmentCollector
   emits for the same decision stream.
"""
import numpy as np
import pytest

SIZE = 14
END_FAMILY = 15


def _random_choices(masks, rng, size=SIZE, end_bias=.2):
    from polytopia_bot.rl.actions import shapes
    shp = shapes(size)
    games = np.asarray(masks.end).shape[0]
    out = [np.zeros(games, np.int32) for _ in range(3)]
    for g in range(games):
        mats = [np.asarray(m[g]).reshape(shp[f]) for f, m in enumerate(masks)]
        legal = [f for f in range(17) if mats[f].any()]
        f = END_FAMILY if END_FAMILY in legal and rng.random() < end_bias else int(rng.choice(legal))
        s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
        out[0][g], out[1][g], out[2][g] = f, s, int(rng.choice(np.flatnonzero(mats[f][s])))
    return out


@pytest.mark.parametrize('packed,layouts', [(False, False), (True, False), (True, True)])
def test_threaded_tick_equals_sequential(packed, layouts):
    from polytopia_bot.ruleset import rules_json
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.rl.rust_engine import split_masks
    from polytopia_engine_rs import GameBatch
    games = 12
    a = GameBatch(SIZE, games, rules_json(), ENGINE_REVISION, 30, True, 10_000, 2, True)   # 30 rounds: resets happen
    b = GameBatch(SIZE, games, rules_json(), ENGINE_REVISION, 30, True, 10_000, 2, True)
    b.set_threads(4)
    rng = np.random.default_rng(0)
    seeds = np.arange(games, dtype=np.int32) + 100
    lay = None
    if layouts:
        from polytopia_bot.maps.stream import MapSource
        src = MapSource('classic', SIZE, players=2)
        mb = src.batch(seeds)[0]
        i32 = lambda x: np.ascontiguousarray(np.asarray(x), np.int32)
        lay = (i32(mb.terrain), i32(mb.structure), i32(mb.resource), i32(mb.capitals))
        a.reset_layouts(seeds, *lay)
        b.reset_layouts(seeds, *lay)
    else:
        a.reset(seeds)
        b.reset(seeds)
    masks = a.legal_masks(packed=False)
    resets = 0
    for step in range(400):
        f, s, t = _random_choices(split_masks(masks, SIZE), rng)
        nxt = (seeds + 1000 + step).astype(np.int32)
        oa = a.tick(f, s, t, nxt, True, lay, packed)
        ob = b.tick(f, s, t, nxt, True, lay, packed)
        assert set(oa) == set(ob)
        for k in oa:
            np.testing.assert_array_equal(np.asarray(oa[k]), np.asarray(ob[k]), err_msg=f'{k} at step {step}')
        resets += int(np.asarray(oa['done']).sum())
        assert not np.asarray(oa['status']).any()
        masks = a.legal_masks(packed=False)
    assert resets > 0
    ea, eb = a.export(), b.export()
    for k in ea:
        np.testing.assert_array_equal(np.asarray(ea[k]), np.asarray(eb[k]), err_msg=k)


def test_light_states_match_the_export():
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    games = 8
    full = RustGames(SIZE, games, max_turns=25, both_views=True)
    light = RustGames(SIZE, games, max_turns=25, both_views=True, light_states=True)
    seeds = np.arange(games, dtype=np.int32) + 7
    sf, pf = full.reset(seeds)
    sl, pl = light.reset(seeds)
    rng = np.random.default_rng(1)
    for step in range(250):
        choice = _random_choices(pf.masks, rng)
        tf, _ = full.tick(sf, choice, seeds + step + 1)
        tl, _ = light.tick(sl, choice, seeds + step + 1)
        for a, b in ((tf.states.current_player, tl.states.current_player), (tf.states.turn, tl.states.turn),
                     (tf.states.players.score, tl.states.players.score),
                     (tf.states.players.killed_turn, tl.states.players.killed_turn),
                     (tf.states.players.resigned_turn, tl.states.players.resigned_turn)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(tf.prepared.view), np.asarray(tl.prepared.view))
        sf, pf, sl = tf.states, tf.prepared, tl.states


def _canonical(batch, segs):
    """{segment: (row ids in pos order, terminal, reward, boot view id, boot opp id)} from a message, row ids read
    from logp (unique per row) — independent of row order."""
    out = {}
    order = np.lexsort((batch['pos'], batch['seg']))
    seg, logp = batch['seg'][order], batch['logp'][order]
    for j in range(len(segs['length'])):
        ids = tuple(int(x) for x in logp[seg == j])
        assert len(ids) == int(segs['length'][j])
        key = (ids, bool(segs['terminal'][j]), float(segs['reward'][j]), int(segs['boot_view'][j, 0]),
               int(segs['boot_opp_view'][j, 0]))
        out[ids] = key
    # per-row fields travel with the row
    for k in ('choice', 'version'):
        np.testing.assert_array_equal(batch[k][:, 0] if batch[k].ndim > 1 else batch[k], batch['logp'].astype(np.int64) % 97)
    np.testing.assert_array_equal(batch['view'][:, 0], batch['logp'].astype(np.int64))
    np.testing.assert_array_equal(batch['opp_view'][:, 0], -batch['logp'].astype(np.int64))
    return out


def test_row_store_emits_what_segment_collector_emits():
    from polytopia_bot.rl.distributed import SegmentCollector, stack_rows
    from polytopia_bot.rl.vector_actor import RowStore
    lanes, players, segment, width = 10, 2, 4, 3
    rng = np.random.default_rng(5)
    ref = SegmentCollector(lanes, segment, players)
    widths = dict(family=3, source=25, target=25)
    store = RowStore(lanes, players, segment, widths, capacity=1 << 12)
    ref_out, vec_out = {}, {}
    next_id = 1
    for tick in range(300):
        movers = np.flatnonzero(rng.random(lanes) < .8)
        seats = rng.integers(1, players + 1, lanes)
        view = np.zeros((lanes, width), np.int16)
        opp = np.zeros((lanes, width), np.int16)
        ids = np.arange(next_id, next_id + len(movers))
        next_id += len(movers)
        view[movers, 0], opp[movers, 0] = ids, -ids
        fields = dict(family_mask=np.zeros((len(movers), 3), np.uint8), source_mask=np.zeros((len(movers), 25), np.uint8),
                      target_mask=np.zeros((len(movers), 25), np.uint8),
                      choice=np.stack([ids % 97, ids * 0, ids * 0], 1).astype(np.int32),
                      logp=ids.astype(np.float32), version=(ids % 97).astype(np.int32), end_turn=np.zeros(len(movers), bool))
        for r, i in enumerate(movers):
            ref.add(int(i), int(seats[i]), dict(view=view[i].copy(), opp_view=opp[i].copy(),
                                                **{k: v[r] for k, v in fields.items()}))
        store.add(movers.astype(np.int64), seats[movers].astype(np.int64), view, opp, movers.astype(np.int32), fields)
        for i in np.flatnonzero(rng.random(lanes) < .05):   # games end
            terminated = bool(rng.random() < .5)
            rewards = rng.choice([-1., 1.], players).astype(np.float32)
            final = np.zeros((players, width), np.int16)
            final[:, 0] = 900000 + tick * 10 + np.arange(players)
            ref.finish(int(i), rewards, terminated, final)
            store.finish(int(i), rewards, terminated, final)
        if tick % 37 == 36:
            rows, segs = ref.take()
            if rows:
                b = stack_rows(rows)
                ref_out.update(_canonical(b, stack_rows(segs)))
            job = store.take()
            if job is not None:
                vec_out.update(_canonical(*job.build()))
            store.release()
    assert ref_out == vec_out and len(ref_out) > 50


def test_coded_messages_equal_view_codec_encode():
    """MessageJob.build_coded (engine-rs coo_deltas from the tick arrays) == view_codec.encode of the dense message
    in segment order, entry for entry; and decodes back to the dense views."""
    from polytopia_bot.rl import view_codec
    from polytopia_bot.rl.vector_actor import RowStore
    lanes, players, segment, width = 9, 2, 5, 40
    rng = np.random.default_rng(11)
    store = RowStore(lanes, players, segment, dict(family=3, source=25, target=25), capacity=1 << 12)
    checked = 0
    base_view = rng.integers(-3, 4, (lanes, width)).astype(np.int16)
    for tick in range(200):
        movers = np.flatnonzero(rng.random(lanes) < .85)
        seats = rng.integers(1, players + 1, lanes)
        base_view += (rng.random((lanes, width)) < .05) * rng.integers(-2, 3, (lanes, width)).astype(np.int16)
        view = base_view.copy()
        opp = (-base_view + tick % 3).astype(np.int16)
        m = len(movers)
        fields = dict(family_mask=rng.integers(0, 255, (m, 3)).astype(np.uint8), source_mask=np.zeros((m, 25), np.uint8),
                      target_mask=np.zeros((m, 25), np.uint8), choice=rng.integers(0, 9, (m, 3)).astype(np.int32),
                      logp=rng.normal(size=m).astype(np.float32), version=np.zeros(m, np.int32),
                      end_turn=rng.random(m) < .2)
        store.add(movers.astype(np.int64), seats[movers].astype(np.int64), view, opp, movers.astype(np.int32), fields)
        for i in np.flatnonzero(rng.random(lanes) < .04):
            final = rng.integers(-3, 4, (players, width)).astype(np.int16)
            store.finish(int(i), rng.choice([-1., 1.], players).astype(np.float32), bool(rng.random() < .5), final)
        if tick % 23 == 22:
            job = store.take()
            if job is None:
                continue
            import copy
            coded_batch, coded_segs = copy.copy(job).build_coded(2)
            dense_batch, dense_segs = job.build()
            order = np.lexsort((dense_batch['pos'], dense_batch['seg']))
            ordered = {k: v[order] for k, v in dense_batch.items()}
            ref_batch, ref_segs = view_codec.encode(ordered, dense_segs)
            assert set(ref_batch) == set(coded_batch) and set(ref_segs) == set(coded_segs)
            for k in ref_batch:
                np.testing.assert_array_equal(ref_batch[k], coded_batch[k], err_msg=k)
            for k in ref_segs:
                np.testing.assert_array_equal(ref_segs[k], coded_segs[k], err_msg=k)
            back, back_segs = view_codec.decode_numpy(coded_batch, coded_segs)
            np.testing.assert_array_equal(back['view'], ordered['view'])
            np.testing.assert_array_equal(back['opp_view'], ordered['opp_view'])
            np.testing.assert_array_equal(back_segs['boot_view'], dense_segs['boot_view'])
            checked += 1
            store.release(max_age=40)
    assert checked >= 5
