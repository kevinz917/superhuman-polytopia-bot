"""Native two-player diplomacy. Embassies are relation state, never terrain."""
import jax
import jax.numpy as jnp
from . import economy, military, structures
from .kernels import distances


def protected(s, owner, other):
    p, q = jnp.clip(owner - 1, 0, 1), jnp.clip(other - 1, 0, 1)
    return (other > 0) & ((owner == other) | s.players.peace[p, q] | (s.players.truce_breaker[p, q] != 0))


def legal(s, a):
    p, q = s.current_player - 1, 2 - s.current_player
    pl = s.players
    other = q + 1
    capital = jnp.argmax((s.tiles.capital == other) & (s.cities.owner == other))
    has_capital = jnp.any((s.tiles.capital == other) & (s.cities.owner == other))
    eligible = (a.source == other) & pl.known[p, q] & (pl.killed_turn[q] == 0) & (pl.resigned_turn[q] == 0)
    ability = a.parameter
    return eligible & (
        ((ability == 18) & pl.peace[p, q])
        | ((ability == 19) & pl.peace_offer[p, q] & ((a.target == 0) | (a.target == 1)))
        | ((ability == 20) & pl.tech[p, 38] & ~pl.embassy[p, q] & has_capital & (a.target == capital)
           & (pl.stars[p] >= 5) & (s.turn * 2 + p >= pl.hostile_until[p]))
        | ((ability == 21) & pl.tech[p, 7] & ~pl.peace[p, q] & ~pl.peace_offer[q, p] & (pl.truce_breaker[p, q] == 0))
        | ((ability == 22) & pl.embassy[p, q]))


def hostile(s):
    """Attacks/conversions destroy both embassies and reset pacifism."""
    p = s.current_player - 1
    return s._replace(players=s.players._replace(
        embassy=jnp.zeros_like(s.players.embassy),
        attacked=s.players.attacked.at[p].set(True),
        peace_offer=jnp.zeros_like(s.players.peace_offer),
        hostile_until=s.players.hostile_until.at[p].set(s.turn * 2 + p + 2)))


def capital_vision(s, *, size):
    p = s.current_player - 1
    other = 3 - s.current_player
    reveal = (s.tiles.capital == other) & s.players.known[p, other - 1] & s.players.tech[p, 38]
    return economy.discover(s, reveal)


def start_turn(s, owner):
    breaker = s.players.truce_breaker
    return s._replace(players=s.players._replace(truce_breaker=jnp.where(breaker == owner, 0, breaker)))


def execute(s, a, rules, *, size):
    p, q = s.current_player - 1, 2 - s.current_player
    def break_peace(x):
        pl = x.players
        x = x._replace(players=pl._replace(peace=pl.peace.at[p,q].set(False).at[q,p].set(False),
            truce_breaker=pl.truce_breaker.at[p,q].set(x.current_player).at[q,p].set(x.current_player)))
        remove = (x.units.owner == x.current_player) & (x.tiles.owner == q + 1)
        x = jax.lax.fori_loop(0, size * size, lambda i, v: jax.lax.cond(remove[i], lambda v: military.remove(v,i,rules), lambda v:v,v), x)
        own = x.units.owner == x.current_player
        return x._replace(units=x.units._replace(moved=x.units.moved | own, attacked=x.units.attacked | own))
    def respond(x):
        accepted = a.target == 1
        return x._replace(players=x.players._replace(
            peace=x.players.peace.at[p,q].set(accepted).at[q,p].set(accepted),
            peace_offer=x.players.peace_offer.at[p,q].set(False).at[q,p].set(False)))
    def embassy(x):
        pl = x.players
        x = x._replace(players=pl._replace(embassy=pl.embassy.at[p,q].set(True),
            stars=pl.stars.at[p].add(-5), known=pl.known.at[q,p].set(True)))
        return economy.discover(x, distances(size)[a.target] <= 1)
    def offer(x):
        return x._replace(players=x.players._replace(peace_offer=x.players.peace_offer.at[q,p].set(True)))
    def destroy(x):
        return x._replace(players=x.players._replace(embassy=x.players.embassy.at[p,q].set(False)))
    s = jax.lax.switch(a.parameter - 18, (break_peace, respond, embassy, offer, destroy), s)
    s = structures.connections(s, s.current_player, size=size,native=True)
    return structures.connections(s, 3 - s.current_player, size=size,native=True)
