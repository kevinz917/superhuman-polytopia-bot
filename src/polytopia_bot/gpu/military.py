"""Ordinary land unit actions, ported from the pinned Polyfish implementation."""
import jax
import jax.numpy as jnp
from .economy import discover
from .kernels import combat, distances, max_health
from .rules import FLOAT, CARRY, SCOUT, HIDE, INDEPENDENT, SURPRISE, SPLASH, STOMP, INFILTRATE


def exhaust(s, source):
    return s._replace(units=s.units._replace(moved=s.units.moved.at[source].set(True), attacked=s.units.attacked.at[source].set(True)))


def remove(s, target, rules, killer=-1, *, casualty=True):
    u, p = s.units, s.players
    owner = jnp.clip(u.owner[target] - 1, 0, 1)
    k = jnp.clip(killer, 0, u.owner.size - 1)
    kp = jnp.clip(u.owner[k] - 1, 0, 1)
    p = p._replace(score=p.score.at[owner].add(-5 * (rules.unit_cost[u.kind[target]] + jnp.where(u.passenger[target] > 0, rules.unit_cost[u.passenger[target]], 0)) * ~u.converted[target]),
                   casualties=p.casualties.at[owner].add(jnp.asarray(casualty,jnp.int32)), kills=p.kills.at[kp].add((killer >= 0).astype(jnp.int32) if hasattr(killer, 'dtype') else int(killer >= 0)))
    resource = jnp.where(u.poison[target] & (s.tiles.structure[target] == 0), 7, s.tiles.resource[target])
    u = u._replace(kills=u.kills.at[k].add(jnp.asarray(killer >= 0, jnp.int32)))
    u = jax.tree.map(lambda x: x.at[target].set(jnp.zeros((), x.dtype)), u)
    return s._replace(units=u, players=p, tiles=s.tiles._replace(resource=s.tiles.resource.at[target].set(resource)))


def relocate(s, source, target, rules, *, size, involuntary=False, reveal=None):
    u, t = s.units, s.tiles
    owner, kind = u.owner[source], u.kind[source]
    attacked = jnp.bool_(True)
    units = jax.tree.map(lambda x: x.at[target].set(x[source]).at[source].set(jnp.zeros((), x.dtype)), u)
    units = units._replace(moved=units.moved.at[target].set(True), attacked=units.attacked.at[target].set(attacked),
                           prev_x=units.prev_x.at[target].set(source % size), prev_y=units.prev_y.at[target].set(source // size))
    if reveal is None:
        d = distances(size)
        reveal = (d[source] <= jnp.where((t.terrain[source] == 4) | rules.skills[kind, SCOUT], 2, 1)) | (d[target] <= jnp.where((t.terrain[target] == 4) | rules.skills[kind, SCOUT], 2, 1))
    s = discover(s._replace(units=units), reveal, owner)
    s = jax.lax.cond(rules.skills[kind, STOMP], lambda x: area_damage(x,target,target,rules,size=size,stomp=True), lambda x:x, s)
    unit=s.units
    embark=(t.structure[target]==8)&~rules.skills[kind,FLOAT]
    special_land=rules.native & ((kind==39)|(kind==40)|(kind==45)) & (t.terrain[target]>2)
    disembark=rules.skills[kind,CARRY]&((u.passenger[source]>0)|special_land)&(t.terrain[target]>2)&~embark
    new_kind=jnp.where(embark,jnp.where(kind==12,45,jnp.where(kind==38,39,jnp.where(kind==37,40,43))),jnp.where(disembark,u.passenger[source],kind))
    new_kind=jnp.where(special_land,jnp.where(kind==39,38,jnp.where(kind==40,37,12)),new_kind)
    passenger=jnp.where(embark & (kind!=12) & (kind!=38) & (kind!=37),kind,jnp.where(disembark,0,u.passenger[source]))
    score_before=rules.unit_cost[kind]+jnp.where(u.passenger[source] > 0, rules.unit_cost[u.passenger[source]], 0)
    score_after=rules.unit_cost[new_kind]+jnp.where(passenger > 0, rules.unit_cost[passenger], 0)
    dash_skill=rules.dash[new_kind] | (~rules.native & (passenger>0) & rules.dash[passenger])
    dash=~jnp.asarray(involuntary)&~u.moved[source]&~u.attacked[source]&dash_skill
    unit=unit._replace(kind=unit.kind.at[target].set(new_kind),passenger=unit.passenger.at[target].set(passenger),
        attacked=unit.attacked.at[target].set(jnp.where(dash,False,unit.attacked[target] | disembark)))
    hidden=rules.native & rules.skills[new_kind,HIDE] & ~jnp.asarray(involuntary)
    # Newly hiding Cloaks cannot infiltrate until their next turn.
    unit=unit._replace(invisible=unit.invisible.at[target].set(unit.invisible[target]|hidden),
        attacked=unit.attacked.at[target].set(unit.attacked[target] | (hidden & ~u.invisible[source])))
    return s._replace(units=unit,players=s.players._replace(score=s.players.score.at[owner-1].add(5*(score_after-score_before)*~u.converted[source])))


def attack_mask(s, source, rules, *, size):
    from .diplomacy import protected
    u = s.units
    own = (u.owner[source] == s.current_player) & ~u.attacked[source] & (u.health[source] > 0)
    dash=rules.dash[u.kind[source]] | (~rules.native & (u.passenger[source]>0) & rules.dash[u.passenger[source]])
    can = own & (~u.moved[source] | dash) & (rules.unit_attack[u.kind[source]] >= 0)
    can &= ~rules.native | (rules.unit_attack[u.kind[source]] > 0) | (u.kind[source] == 10) | rules.skills[u.kind[source],INFILTRATE]
    d = distances(size)[source]
    # Reference get_enemy_at also includes hidden units; do not silently change its rules.
    normal=(u.owner == 3-s.current_player) & ~u.invisible & ~(rules.native & protected(s,s.current_player,u.owner))
    city=(s.cities.owner == 3-s.current_player) & ~protected(s,s.current_player,s.cities.owner)
    city &= ~((u.owner != 0)&(u.owner != s.cities.owner)) & (s.cities.infiltrated != s.turn*2+s.current_player)
    targets=jnp.where(rules.native & rules.skills[u.kind[source],INFILTRATE],city,normal)
    return can & (d > 0) & (d <= rules.unit_range[u.kind[source]]) & targets & (~rules.native | s.tiles.explored[s.current_player-1])


def attack(s, source, target, rules, *, size):
    u, t, c, p = s.units, s.tiles, s.cities, s.players
    kind, defender = u.kind[source], u.kind[target]
    terrain = t.terrain[target]
    tech = p.tech[u.owner[target] - 1]
    city_bonus = jnp.where((c.owner[target] == u.owner[target]) & rules.fortify[defender],
                           jnp.where(c.rewards[target, 1] > 0, 4., 1.5), 1.)
    bonus = jnp.where((terrain == 1) | (terrain == 2), jnp.where(tech[12], 1.5, 1.),
            jnp.where(terrain == 5, jnp.where(tech[18], 1.5, 1.),
            jnp.where(terrain == 4, jnp.where(tech[20], 1.5, 1.), city_bonus)))
    bonus = bonus * jnp.where(u.poison[target], .5, 1.)
    hp = max_health(s, rules)
    damage, retaliation, _ = combat(rules.unit_attack[kind], u.health[source], hp[source],
        rules.unit_defense[defender] * jnp.where(u.poison[target], .7, 1.), u.health[target], hp[target], bonus)
    dead = u.health[target] - damage <= 0
    retaliate = ~rules.skills[kind, SURPRISE] & ~rules.stiff[defender] & (distances(size)[source, target] <= rules.unit_range[defender])
    # Source updates attack direction even when retaliation subsequently kills the attacker.
    s = s._replace(players=p._replace(attacked=p.attacked.at[s.current_player - 1].set(True)),
        units=u._replace(health=u.health.at[target].add(-damage), boosted=u.boosted.at[target].set(False),
            prev_x=u.prev_x.at[source].set(2 * (source % size) - target % size),
            prev_y=u.prev_y.at[source].set(2 * (source // size) - target // size), last_attack=u.last_attack.at[source].set(target + 1)))
    s = jax.lax.cond(rules.skills[kind,SPLASH],lambda x:area_damage(x,source,target,rules,size=size),lambda x:x,s)
    s = jax.lax.cond(dead, lambda x: remove(x, target, rules, source),
                     lambda x: x._replace(units=x.units._replace(health=x.units.health.at[source].add(-jnp.where(retaliate, retaliation, 0)))), s)
    attacker_dead = s.units.health[source] <= 0
    s = jax.lax.cond(attacker_dead, lambda x: remove(x, source, rules, target), lambda x: x, s)
    can_enter = ((terrain == 3) | (terrain == 5) | ((terrain == 4) & p.tech[s.current_player - 1, 20]) | ((terrain <= 2) & (rules.skills[kind,FLOAT] | (t.structure[target]==48) | ((t.structure[target]==8)&(t.owner[target]==s.current_player)))))
    advance = dead & ~attacker_dead & (rules.unit_range[kind] < 2) & can_enter
    s = jax.lax.cond(advance, lambda x: relocate(x, source, target, rules, size=size, involuntary=True), lambda x: x, s)
    at = jnp.where(advance, target, source)
    def finish(x):
        u = x.units
        persist=rules.persist[kind] | (~rules.native & (u.passenger[at]>0) & rules.persist[u.passenger[at]])
        escape=rules.escape[kind] | (~rules.native & (u.passenger[at]>0) & rules.escape[u.passenger[at]])
        return x._replace(units=u._replace(attacks_performed=u.attacks_performed.at[at].add(1),
            attacked=u.attacked.at[at].set(jnp.where(dead & persist, jnp.where(rules.native, False, u.attacked[at]), True)),
            moved=u.moved.at[at].set(~escape)))
    return jax.lax.cond(attacker_dead, lambda x: x, finish, s)


def convert(s, source, target, rules, *, size):
    """Native Mind Bender attack; conversion never deals retaliatory damage."""
    u, p = s.units, s.players
    old = u.owner[target] - 1
    from .structures import capital_index
    cap,exists=capital_index(s,s.current_player)
    fallback=jnp.argmin(jnp.where(s.cities.owner==s.current_player,s.cities.order,jnp.iinfo(jnp.int32).max))
    home=jnp.where(rules.skills[u.kind[target],INDEPENDENT],0,jnp.where(exists,cap,fallback)+1)
    cost = rules.unit_cost[u.kind[target]] + jnp.where(u.passenger[target] > 0, rules.unit_cost[u.passenger[target]], 0)
    p = p._replace(score=p.score.at[old].add(-5 * cost * ~u.converted[target]),
                   attacked=p.attacked.at[s.current_player - 1].set(True))
    u = u._replace(owner=u.owner.at[target].set(s.current_player), converted=u.converted.at[target].set(True),
                   home=u.home.at[target].set(home), moved=u.moved.at[target].set(True), attacked=u.attacked.at[target].set(True))
    s=exhaust(s._replace(units=u, players=p), source)
    return discover(s,distances(size)[target]<=jnp.where((s.tiles.terrain[target]==4)|rules.skills[u.kind[target],SCOUT],2,1))


def summon_mask(s, rules):
    u, c = s.units, s.cities
    n = c.owner.size
    supported_owner=jnp.where(rules.native,u.owner==s.current_player,u.owner>0)
    counts = jnp.sum(supported_owner[:, None] & (u.home[:, None] == jnp.arange(1, n + 1)), axis=0)
    available = (c.owner == s.current_player) & (counts <= c.level) & (u.owner == 0)
    kind = jnp.arange(128)
    tech = s.players.tech[s.current_player - 1, rules.unit_tech] | (kind == 2)
    unlocked = tech & (rules.unit_cost >= 1) & (rules.unit_tech < 127)
    upgrade=(kind==41)|(kind==42)|(kind==44)
    affordable=s.players.stars[s.current_player-1]>=rules.unit_cost
    friendly=(s.tiles.owner==s.current_player)|((s.tiles.owner>0)&s.players.peace[s.current_player-1,jnp.clip(s.tiles.owner-1,0,1)])
    upgrade_allowed=(u.owner==s.current_player)&(u.kind==43)&(~rules.native|friendly)
    return ((available[:,None]&~upgrade) | upgrade_allowed[:,None]&upgrade) & unlocked & affordable


def summon(s, source, kind, rules, *, size, costs=True):
    u = jax.tree.map(lambda x: x.at[source].set(jnp.zeros((), x.dtype)), s.units)
    u = u._replace(owner=u.owner.at[source].set(s.current_player), kind=u.kind.at[source].set(kind),
                   health=u.health.at[source].set(rules.unit_health[kind]), home=u.home.at[source].set(source + 1),
                   moved=u.moved.at[source].set(True), attacked=u.attacked.at[source].set(True),
                   prev_x=u.prev_x.at[source].set(-1), prev_y=u.prev_y.at[source].set(-1), created=u.created.at[source].set(s.turn))
    p = s.current_player - 1
    s = s._replace(units=u, players=s.players._replace(stars=s.players.stars.at[p].add(-rules.unit_cost[kind] * costs),
                                                      score=s.players.score.at[p].add(5 * rules.unit_cost[kind])))
    return discover(s, distances(size)[source] <= jnp.where(s.tiles.terrain[source] == 4, 2, 1))


def promote(s, source, rules):
    u = s.units
    return s._replace(units=u._replace(veteran=u.veteran.at[source].set(True), health=u.health.at[source].set(max_health(s, rules)[source] + 5)))


def disband(s, source, rules):
    kind=s.units.kind[source]
    passenger=s.units.passenger[source]
    underlying=jnp.where(passenger>0,passenger,jnp.where(kind==39,38,jnp.where(kind==40,37,jnp.where(kind==45,12,kind))))
    refund = rules.unit_cost[jnp.where(rules.native,underlying,kind)] // 2
    s = remove(s, source, rules,casualty=~rules.native)
    return s._replace(players=s.players._replace(stars=s.players.stars.at[s.current_player - 1].add(refund)))


def heal_others(s,source,rules,*,size):
    u=s.units
    allies=(distances(size)[source]==1)&(u.owner==s.current_player)
    hp=jnp.where(allies&~u.poison,jnp.minimum(u.health+4,max_health(s,rules)),u.health)
    return exhaust(s._replace(units=u._replace(health=hp,poison=u.poison&~allies)),source)


def push(s,source,rules,*,size):
    """Reference alternating push directions, including its center tie breaks."""
    u=s.units
    owner=u.owner[source]
    x,y=source%size,source//size
    center=size//2
    cx,cy=jnp.sign(center-x),jnp.sign(center-y)
    cy=jnp.where((cx==0)&(cy==0),1,cy)
    prev_valid=(u.prev_x[source]>=0)&(u.prev_y[source]>=0)&((u.prev_x[source]!=x)|(u.prev_y[source]!=y))
    flip=jnp.where(owner!=s.current_player,-1,1)
    mx,my=jnp.sign(x-u.prev_x[source])*flip,jnp.sign(y-u.prev_y[source])*flip
    la=u.last_attack[source]-1
    ranged=(rules.unit_range[u.kind[source]]>1)&u.attacked[source]&~(rules.escape[u.kind[source]]&u.moved[source])
    ax,ay=jnp.sign(la%size-x)*flip,jnp.sign(la//size-y)*flip
    use_move=prev_valid & ((la<0)|u.moved[source])
    dx=jnp.where((la>=0)&ranged,ax,jnp.where(use_move,mx,cx))
    dy=jnp.where((la>=0)&ranged,ay,jnp.where(use_move,my,cy))
    dirs=jnp.array([[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]])
    base=jnp.argmax((dirs[:,0]==dx)&(dirs[:,1]==dy))
    offsets=jnp.array([0,-1,1,-2,2,-3,3,4])
    choices=dirs[(base+offsets)%8]+jnp.array([x,y])
    for a,b in ((1,2),(3,4),(5,6)):
        dista=jnp.sum((choices[a]-center)**2);distb=jnp.sum((choices[b]-center)**2)
        va=jnp.abs(choices[a,0]-choices[a,1]);vb=jnp.abs(choices[b,0]-choices[b,1])
        swap=(distb<dista)|((distb==dista)&(vb<va))
        aa,bb=choices[a],choices[b]
        choices=choices.at[a].set(jnp.where(swap,bb,aa)).at[b].set(jnp.where(swap,aa,bb))
    indices=jnp.clip(choices[:,1]*size+choices[:,0],0,size*size-1)
    terrain=s.tiles.terrain[indices]
    valid=jnp.all((choices>=0)&(choices<size),axis=-1)&(u.owner[indices]==0)
    valid &= (terrain==3)|(terrain==5)|((terrain==4)&s.players.tech[jnp.clip(owner-1,0,1),20])|((terrain<=2)&(rules.skills[u.kind[source],FLOAT]|(s.tiles.structure[indices]==48)|((s.tiles.structure[indices]==8)&(s.tiles.owner[indices]==owner))))
    target=indices[jnp.argmax(valid)]
    def occupied(x):
        return jax.lax.cond(jnp.any(valid),lambda x:relocate(x,source,target,rules,size=size,involuntary=True),lambda x:remove(x,source,rules),x)
    return jax.lax.cond(owner>0,occupied,lambda x:x,s)


def superunit(s,source,rules,*,size):
    return summon(push(s,source,rules,size=size),source,jnp.int32(12),rules,size=size,costs=False)


def defense_bonuses(s,rules):
    u,t,c=s.units,s.tiles,s.cities
    tech=s.players.tech[jnp.clip(u.owner-1,0,1)]
    city=jnp.where((c.owner==u.owner)&rules.fortify[u.kind],jnp.where(c.rewards[:,1]>0,4.,1.5),1.)
    bonus=jnp.where((t.terrain==1)|(t.terrain==2),jnp.where(tech[:,12],1.5,1.),
          jnp.where(t.terrain==5,jnp.where(tech[:,18],1.5,1.),jnp.where(t.terrain==4,jnp.where(tech[:,20],1.5,1.),city)))
    return bonus*jnp.where(u.poison,.5,1.)


def area_damage(s,source,center,rules,*,size,stomp=False):
    from .diplomacy import protected, hostile
    u=s.units
    hp=max_health(s,rules)
    damage=combat(rules.unit_attack[u.kind[source]],u.health[source],hp[source],
                  rules.unit_defense[u.kind]*jnp.where(u.poison,.7,1.),u.health,hp,defense_bonuses(s,rules))[:,2]
    # Public release 2.16.4 floors all splash, not only Juggernaut Stomp.
    damage=jnp.where(rules.native | jnp.asarray(stomp),jnp.floor(damage),damage)
    targets=(distances(size)[center]==1)&(u.owner!=0)&(u.owner!=u.owner[source])
    targets &= ~(rules.native & protected(s,u.owner[source],u.owner))
    damage=jnp.where(targets,damage,0)
    dead=targets&(u.health-damage<=0)
    s=s._replace(units=u._replace(health=u.health-damage))
    def body(i,x):
        return jax.lax.cond(dead[i],lambda x:remove(x,i,rules,source),lambda x:x,x)
    s=jax.lax.fori_loop(0,size*size,body,s)
    return jax.lax.cond(rules.native & jnp.any(targets & (damage>0)),hostile,lambda x:x,s)


def upgrade(s,source,kind,rules,*,size):
    from .rules import SCOUT
    u,p=s.units,s.players
    before=max_health(s,rules)[source]
    old=u.kind[source]
    u=u._replace(kind=u.kind.at[source].set(kind))
    s=s._replace(units=u)
    hp=jnp.maximum(max_health(s,rules)[source]-(before-u.health[source]),1)
    s=s._replace(units=u._replace(health=u.health.at[source].set(hp)),players=p._replace(
        stars=p.stars.at[s.current_player-1].add(-rules.unit_cost[kind]),
        score=p.score.at[s.current_player-1].add(5*(rules.unit_cost[kind]-rules.unit_cost[old]))))
    return discover(s,distances(size)[source]<=jnp.where(rules.skills[kind,SCOUT],2,1))
