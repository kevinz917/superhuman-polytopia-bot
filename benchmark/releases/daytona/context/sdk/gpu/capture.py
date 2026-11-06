"""Village/city ownership transfers and native elimination metadata."""
import jax
import jax.numpy as jnp
from . import economy, military, structures
from .kernels import distances
from .rules import INDEPENDENT


def capture_mask(s, rules):
    from .diplomacy import protected
    u,t=s.units,s.tiles
    idle=(u.owner==s.current_player)&~u.moved&~u.attacked
    return idle & (((t.structure==1)&(t.owner!=s.current_player)&~(rules.native & protected(s,s.current_player,t.owner))) | (t.structure==2) |
                   ((t.resource==8)&s.players.tech[s.current_player-1,14]))


def city(s,source,rules,*,size):
    before=s
    c,t,p=s.cities,s.tiles,s.players
    owner=s.current_player
    old=t.owner[source]
    enemy=old>0
    old_order=c.order[source]
    old_territory=c.territory[source]
    order=jnp.max(jnp.where(c.owner==owner,c.order,0))+1
    # Neutral village construction stores the complete square, including tiles
    # which claim_territory cannot take from an existing owner.
    area=distances(size)[source]<=1
    membership=jnp.where(enemy,old_territory,area)
    transfer=100+(c.level[source]-1)*50+c.population[source]*5+c.rewards[source,2]*250
    score=jnp.where(enemy,transfer,100)
    c=c._replace(owner=c.owner.at[source].set(owner),
        level=c.level.at[source].set(jnp.where(enemy,c.level[source],1)),
        population=c.population.at[source].set(jnp.where(enemy,c.population[source],0)),
        progress=c.progress.at[source].set(jnp.where(enemy,c.progress[source],0)),
        production=c.production.at[source].set(jnp.where(enemy,c.production[source],1)),
        border=c.border.at[source].set(jnp.where(enemy,c.border[source],1)),
        rewards=c.rewards.at[source].set(jnp.where(enemy,c.rewards[source],0)),
        connected=c.connected.at[source].set(enemy & c.connected[source]),
        territory=c.territory.at[source].set(membership),
        order=jnp.where(enemy & (c.owner==old) & (c.order>old_order),c.order-1,c.order).at[source].set(order))
    p=p._replace(score=p.score.at[owner-1].add(score).at[jnp.clip(old-1,0,1)].add(jnp.where(enemy,-score,0)))
    t=t._replace(capital=t.capital.at[source].set(jnp.where(rules.native,t.capital[source],jnp.where(t.capital[source]>0,owner,0))))
    s=s._replace(cities=c,players=p,tiles=t)
    embassy_target = jnp.clip(old - 1, 0, 1)
    s=s._replace(players=s.players._replace(embassy=s.players.embassy.at[:,embassy_target].set(
        s.players.embassy[:,embassy_target] & ~(rules.native & enemy & (t.capital[source] > 0)))))
    eliminated=enemy & ~jnp.any(c.owner==old)
    def eliminate(x):
        p=x.players
        x=x._replace(players=p._replace(killed_turn=p.killed_turn.at[old-1].set(jnp.where(rules.native,jnp.maximum(x.turn,1),x.turn)),killer=p.killer.at[old-1].set(owner)))
        def body(i,x):
            return jax.lax.cond(x.units.owner[i]==old,lambda x:military.remove(x,i,rules),lambda x:x,x)
        return jax.lax.fori_loop(0,size*size,body,x)
    s=jax.lax.cond(eliminated,eliminate,lambda x:x,s)
    s=s._replace(terminated=s.terminated | (rules.native & eliminated))
    claimed=jnp.where(enemy,membership | (jnp.arange(size*size)==source),area & (t.owner==0))
    s=economy.claim(s,source,claimed,rules=rules)
    s=jax.lax.cond(rules.native,lambda x:structures.adjust_building_population(before,x,rules,size=size),lambda x:x,s)
    s=economy.discover(s,jnp.where(enemy,old_territory,area))
    s=structures.connections(s,owner,size=size,native=rules.native)
    s=jax.lax.cond(enemy,lambda x:structures.connections(x,old,size=size,native=rules.native),lambda x:x,s)
    s=military.exhaust(s,source)
    s=s._replace(units=s.units._replace(invisible=s.units.invisible.at[source].set(False)))
    home=jnp.where(rules.native & (s.units.owner!=owner) & (s.units.home==source+1),0,s.units.home)
    home=home.at[source].set(jnp.where(rules.native & rules.skills[s.units.kind[source],INDEPENDENT],0,source+1))
    return s._replace(units=s.units._replace(home=home))


def starfish(s,source):
    s=military.exhaust(s,source)
    s=s._replace(units=s.units._replace(invisible=s.units.invisible.at[source].set(False)))
    return s._replace(tiles=s.tiles._replace(resource=s.tiles.resource.at[source].set(0)),
                      players=s.players._replace(stars=s.players.stars.at[s.current_player-1].add(8)))


def ruin(s,source,rules,*,size):
    """Unhinted seeded land-ruin rewards; RNG state remains on the device."""
    from .discovery import hash_word, explorer
    p=s.current_player-1
    tech=s.players.tech[p]
    ids=jnp.arange(128)
    # The source free-tech path is prerequisite based (unlike ResearchMove's
    # forward/backward graph), and includes Waterways/Rituals for Bardur.
    eligible=rules.free_tech & ~tech & jnp.where(rules.free_requires>=0,tech[jnp.clip(rules.free_requires,0,127)],rules.tech_tier==1)
    cap,has_cap=structures.capital_index(s,s.current_player)
    nearby=(distances(size)[source]<=2)&(jnp.arange(size*size)!=source)
    can_explore=(s.tiles.terrain[source]!=4)&jnp.any(nearby&~s.tiles.explored[p])
    options=jnp.array([True,jnp.any(eligible),has_cap,can_explore,True])
    seed=hash_word(s.initial_seed,s.seed)
    draw=(seed%jnp.sum(options,dtype=jnp.uint32)).astype(jnp.int32)
    choice=jnp.argmax(options&(jnp.cumsum(options)-1==draw))
    s=military.exhaust(s,source)
    s=s._replace(units=s.units._replace(invisible=s.units.invisible.at[source].set(False)))
    s=s._replace(seed=seed,tiles=s.tiles._replace(structure=s.tiles.structure.at[source].set(0),
                        structure_level=s.tiles.structure_level.at[source].set(0),structure_turn=s.tiles.structure_turn.at[source].set(0)))
    def stars(x):
        return x._replace(players=x.players._replace(stars=x.players.stars.at[p].add(10)))
    def unlock(x):
        seed=hash_word(x.initial_seed,x.seed)
        draw=(seed%jnp.maximum(jnp.sum(eligible,dtype=jnp.uint32),1)).astype(jnp.int32)
        kind=jnp.argmax(eligible&(jnp.cumsum(eligible)-1==draw))
        return x._replace(seed=seed,players=x.players._replace(tech=x.players.tech.at[p,kind].set(True),
                  tech_turn=x.players.tech_turn.at[p,kind].set(x.turn),score=x.players.score.at[p].add(100*rules.tech_tier[kind])))
    def unit(x):
        water=x.tiles.terrain[source]<=2
        x=military.summon(military.push(x,source,rules,size=size),source,jnp.where(water,44,11),rules,size=size,costs=False)
        x=military.promote(x,source,rules)
        # The reference attaches the Warrior passenger after pricing the reward.
        return x._replace(units=x.units._replace(passenger=x.units.passenger.at[source].set(jnp.where(water,2,0)),
            home=x.units.home.at[source].set(jnp.where(rules.native,0,x.units.home[source]))))
    return jax.lax.switch(choice,(stars,unlock,lambda x:economy.add_population(x,cap,3),
                                 lambda x:economy.discover(x,explorer(x,source,size=size)),unit),s)
