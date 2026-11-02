"""Reference-ordered shortest paths for exploration during land movement.

Destination reachability is a parallel relaxation in kernels.py. Path replay
must also match Rust BinaryHeap's equal-cost behavior; arbitrary shortest paths
can expose different fog. This correctness implementation retains that heap
ordering on device. The heap capacity is bounded by directed board edges.
"""
import jax
import jax.numpy as jnp
from .kernels import distances, neighbors, movement_data
from .rules import CREEP, SCOUT


def _push(heap, count, index, cost):
    ids, costs = heap
    ids, costs = ids.at[count].set(index), costs.at[count].set(cost)
    def cond(v):
        ids, costs, at = v
        return (at > 0) & (costs[at] < costs[(at - 1) // 2])
    def body(v):
        ids, costs, at = v
        parent = (at - 1) // 2
        ids = ids.at[at].set(ids[parent]).at[parent].set(ids[at])
        costs = costs.at[at].set(costs[parent]).at[parent].set(costs[at])
        return ids, costs, parent
    ids, costs, _ = jax.lax.while_loop(cond, body, (ids, costs, count))
    return (ids, costs), count + 1


def _pop(heap, count):
    ids, costs = heap
    index, cost = ids[0], costs[0]
    count = count - 1
    last_id, last_cost = ids[count], costs[count]
    # Rust pop uses sift_down_to_bottom followed by sift_up, choosing the right
    # child when keys tie. A normal priority-queue stable sort does not match it.
    def cond(v):
        ids, costs, at = v
        return 2 * at + 1 < count
    def body(v):
        ids, costs, at = v
        left = 2 * at + 1
        right = left + 1
        child = jnp.where((right < count) & (costs[right] <= costs[left]), right, left)
        return ids.at[at].set(ids[child]), costs.at[at].set(costs[child]), child
    ids, costs, at = jax.lax.while_loop(cond, body, (ids, costs, jnp.int32(0)))
    ids, costs = ids.at[at].set(last_id), costs.at[at].set(last_cost)
    def upcond(v):
        ids, costs, at = v
        return (at > 0) & (costs[at] < costs[(at - 1) // 2])
    def up(v):
        ids, costs, at = v
        p = (at - 1) // 2
        return ids.at[at].set(ids[p]).at[p].set(ids[at]), costs.at[at].set(costs[p]).at[p].set(costs[at]), p
    ids, costs, _ = jax.lax.while_loop(upcond, up, (ids, costs, at))
    return (ids, costs), count, index, cost


def path_details(s, source, target, rules, *, size):
    n = size * size
    nb, board = neighbors(size)
    t, u = s.tiles, s.units
    passable, terminal, road, budget = movement_data(s, source, rules, size=size)
    passable &= (u.owner == 0) | (rules.native & ((u.owner == s.current_player) | u.invisible))
    inf = jnp.int32(1_000_000)
    heap = (jnp.zeros(8 * n + 1, jnp.int32).at[0].set(source), jnp.full(8 * n + 1, inf).at[0].set(0))
    dist = jnp.full(n, inf).at[source].set(0)
    parent = jnp.full(n, -1, jnp.int32)
    def cond(v):
        heap, count, dist, parent, found = v
        return (count > 0) & ~found
    def body(v):
        heap, count, dist, parent, found = v
        heap, count, at, cost = _pop(heap, count)
        found = at == target
        expand = ~found & (cost <= dist[at]) & ((at == source) | ~terminal[at]) & (cost < budget)
        def edges(k, v):
            heap, count, dist, parent = v
            dest = nb[at, k]
            newcost = cost + jnp.where(road[at] & road[dest] & ~rules.skills[u.kind[source], CREEP], 1, 2)
            valid = board[at, k] & passable[dest] & (dest != source) & (newcost < dist[dest])
            def add(v):
                heap, count, dist, parent = v
                heap, count = _push(heap, count, dest, newcost)
                return heap, count, dist.at[dest].set(newcost), parent.at[dest].set(at)
            return jax.lax.cond(valid, add, lambda v: v, v)
        heap, count, dist, parent = jax.lax.cond(expand,
            lambda v: jax.lax.fori_loop(0, 8, edges, v), lambda v: v, (heap, count, dist, parent))
        return heap, count, dist, parent, found
    _, _, _, parent, found = jax.lax.while_loop(cond, body, (heap, jnp.int32(1), dist, parent, jnp.bool_(False)))
    mask = jnp.zeros(n, jnp.bool_).at[source].set(True).at[target].set(True)
    rank = jnp.full(n, -1, jnp.int32)
    def back_cond(v):
        at, mask, rank, depth = v
        return (at >= 0) & (at != source)
    def back(v):
        at, mask, rank, depth = v
        return parent[at], mask.at[at].set(True), rank.at[at].set(depth), depth+1
    _, mask, rank, depth = jax.lax.while_loop(back_cond, back, (jnp.where(found, target, -1), mask, rank, jnp.int32(0)))
    rank=rank.at[source].set(depth)
    return mask, parent, rank, found


def path_reveal(s, source, target, rules, *, size):
    mask, _, _, _ = path_details(s,source,target,rules,size=size)
    return reveal_path(s,source,mask,rules,size=size)


def reveal_path(s,source,mask,rules,*,size):
    return jnp.any(mask[:, None] & (distances(size) <= jnp.where((s.tiles.terrain == 4) | rules.skills[s.units.kind[source], SCOUT], 2, 1)[:, None]), axis=0)
