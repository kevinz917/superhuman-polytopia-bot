"""Map sources for actors and arenas: a weighted mix of map types, deterministic per seed.

A mix is a dict {type: weight}; types are the variant presets plus 'classic' (the engine-rs generator, a port
of the game's Drylands map generator, i.e. exactly the boards GameBatch.reset(seed) plays).
"""
import numpy as np

from .state import stack
from .variants import PRESETS, generate

TYPES = ('classic',) + tuple(PRESETS)
MIXES = {
    'classic': {'classic': 1.},
    'uniform': {t: 1. for t in TYPES},
    # Proposed training default (see docs/map-variants-2026-10-19.md).
    # Tilted toward water maps, where the current bot most often fails to finish games.
    'proposed-v1': {'classic': .15, 'drylands': .05, 'lakes': .1, 'pangea': .1, 'continents': .12,
                    'archipelago': .12, 'waterworld': .08, 'highlands': .1, 'ridge': .1, 'jungle': .08},
    # Four-seat training default (docs/four-player-2026-10-24.md): land-heavy maps, no water worlds or ridge.
    'four-v1': {'classic': .25, 'drylands': .15, 'lakes': .2, 'pangea': .15, 'continents': .1, 'highlands': .08,
                'jungle': .07},
}


def parse_mix(mix):
    if isinstance(mix, str):
        if mix in MIXES:
            mix = MIXES[mix]
        elif mix in TYPES:
            mix = {mix: 1.}
        else:
            raise ValueError(f'unknown map mix {mix!r}')
    if not mix or any(t not in TYPES for t in mix) or any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError('invalid map mix')
    total = sum(mix.values())
    return {t: w / total for t, w in mix.items() if w > 0}


class MapSource:
    def __init__(self, mix='classic', size=14, players=2):
        """players=4: four-capital maps. 'classic' always comes from engine-rs (mapgen::layout_players)."""
        self.mix, self.size, self.players = parse_mix(mix), size, players
        self.types = list(self.mix)
        self.weights = np.array([self.mix[t] for t in self.types])
        self._classic = None

    def type_for(self, seed):
        """Map type is a pure function of the seed, so arenas can replay exact boards."""
        if len(self.types) == 1:
            return self.types[0]
        u = np.random.default_rng([int(seed), 77]).random()
        return self.types[int(np.searchsorted(np.cumsum(self.weights), u * self.weights.sum(), side='right'))]

    def layout(self, seed, kind=None):
        kind = kind or self.type_for(seed)
        if kind == 'classic':
            if self._classic is None:
                self._classic = self._classic_generator()
            return kind, self._classic(np.int32(seed))
        layout, _ = generate(seed, kind, size=self.size, players=self.players)
        return kind, layout

    def _classic_generator(self):
        from importlib.resources import files
        from polytopia_engine_rs import GameBatch
        from polytopia_bot.gpu import ENGINE_REVISION
        from .variants import Layout
        rules = files('polytopia_bot.gpu').joinpath('rules.json').read_text()
        batch = GameBatch(self.size, 1, rules, ENGINE_REVISION, None, True, 10_000, self.players)
        def layout(seed):
            d = batch.layout(int(seed))
            return Layout(*(np.asarray(d[k], np.int32) for k in ('terrain', 'structure', 'resource', 'capitals')))
        return layout

    def batch(self, seeds):
        rows, kinds = [], []
        for s in seeds:
            kind, layout = self.layout(s)
            rows.append((int(s), layout))
            kinds.append(kind)
        return stack(rows), kinds
