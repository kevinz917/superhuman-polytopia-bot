"""Buildings, monument conditions, temple growth and road networks."""
import jax
import jax.numpy as jnp
from .kernels import distances, neighbors
from .economy import economic_tiles, in_territory, add_population, add_population_vector


def temple(kind):
    return (kind >= 17) & (kind <= 20)


def scores(s, rules):
    return jnp.where(temple(s.tiles.structure), 100 * s.tiles.structure_level, rules.structures.score[s.tiles.structure])


def production_levels(s, rules, *, size):
    """Native multiplier levels depend on current adjacent owned improvements."""
    t=s.tiles
    adjacent=(distances(size)==1)&(t.owner[:,None]==t.owner[None,:])&(t.owner[:,None]>0)
    compatible=rules.structures.adjacent[t.structure[:,None],t.structure[None,:]]
    count=jnp.sum(adjacent&compatible,axis=-1,dtype=jnp.int32)
    multiplier=jnp.any(rules.structures.adjacent[t.structure],axis=-1)&(t.structure!=50)
    levels=jnp.where(multiplier,count,t.structure_level)
    market=jnp.minimum(jnp.sum(jnp.where(adjacent&compatible,levels[None,:],0),axis=-1,dtype=jnp.int32),8)
    return jnp.where(t.structure==50,market,levels)


def population_contribution(s,rules,*,size):
    kinds=s.tiles.structure
    multiplier=jnp.any(rules.structures.adjacent[kinds],axis=-1)
    return rules.structures.population[kinds]*jnp.where(multiplier,production_levels(s,rules,size=size),1)


def adjust_building_population(before,after,rules,*,size):
    delta=population_contribution(after,rules,size=size)-population_contribution(before,rules,size=size)
    membership=(jnp.arange(size*size)[:,None]==after.tiles.city[None,:]) & (after.cities.owner[:,None]==after.tiles.owner[None,:])
    amount=jnp.sum(membership*delta[None,:],axis=-1,dtype=jnp.int32)
    after=after._replace(tiles=after.tiles._replace(structure_level=production_levels(after,rules,size=size)))
    after=add_population_vector(after,amount,owner=1,native=True)
    return add_population_vector(after,amount,owner=2,native=True)


def capital_index(s, owner):
    c = s.cities
    valid = (c.owner == owner) & (s.tiles.capital == owner)
    return jnp.argmin(jnp.where(valid, c.order, jnp.iinfo(jnp.int32).max)), jnp.any(valid)


def connections(s, owner, *, size, native=False):
    t,c = s.tiles,s.cities
    n=size*size
    cap,exists = capital_index(s,owner)
    nb, board=neighbors(size)
    friendly=(t.owner==0)|(t.owner==owner)|(jnp.asarray(native)&s.players.peace[owner-1,jnp.clip(t.owner-1,0,1)])
    nodes=(t.road | (t.structure==48) | (t.structure==8) | ((c.owner>0)&friendly)) & friendly
    ports=t.structure==8
    water=(t.terrain==1)|(t.terrain==2)
    d=distances(size)
    water_seen=jnp.eye(n,dtype=jnp.bool_)&ports[:,None]
    def water_cond(v):
        seen,changed=v
        return changed
    def water_step(v):
        seen,_=v
        expandable=seen&(d<4)
        new=seen | (jnp.any(expandable[:,nb]&board[None,:,:],axis=-1)&water[None,:]&friendly[None,:])
        return new,jnp.any(new!=seen)
    water_seen,_=jax.lax.while_loop(water_cond,water_step,(water_seen,jnp.any(ports)))
    port_edges=water_seen&ports[:,None]&ports[None,:]&(d<=4)
    reached=jnp.zeros(n,jnp.bool_).at[cap].set(exists)
    def condition(v):
        seen,changed=v
        return changed
    def body(v):
        seen,_=v
        out=seen | (nodes & jnp.any(seen[nb] & board,axis=-1)) | jnp.any(port_edges & seen[:,None],axis=0)
        return out,jnp.any(out!=seen)
    reached,_=jax.lax.while_loop(condition,body,(reached,jnp.bool_(True)))
    own=(c.owner==owner)&(jnp.arange(n)!=cap)&exists
    delta=jnp.where(own,reached.astype(jnp.int32)-c.connected.astype(jnp.int32),0)
    connected=jnp.where(own,reached,c.connected)
    # add_population deliberately follows reference POV behavior, even when
    # refreshing the opponent's disconnected network after capturing a city.
    amount=delta.at[cap].add(jnp.sum(delta,dtype=jnp.int32))
    s=s._replace(cities=c._replace(connected=connected))
    return add_population_vector(s,amount,owner=jnp.where(native,owner,s.current_player),native=native)


def monument_conditions(s, owner, rules, *, size):
    p, c = s.players, s.cities
    pi = owner - 1
    own = c.owner == owner
    count = jnp.sum(own & c.connected)
    corners = jnp.array([0, size-1, size*size-1, size*size-size])
    return (jnp.zeros(128, jnp.bool_)
        .at[23].set(p.tech[pi,21] & (p.pacifist[pi]>=5))
        .at[24].set(jnp.all(p.tech[pi] | ~rules.tech_exists))
        .at[25].set(p.tech[pi,4] & (count>=5))
        .at[26].set(p.tech[pi,5] & (p.stars[pi]>=100))
        .at[27].set(p.kills[pi]>=10)
        .at[28].set(jnp.any(own & (c.level>=5)))
        .at[29].set(jnp.all(s.tiles.explored[pi,corners]) & s.fog))


def remember_monuments(s, rules, *, size):
    earned = jax.vmap(lambda owner: monument_conditions(s,owner,rules,size=size))(jnp.array([1,2]))
    return s._replace(players=s.players._replace(monuments=s.players.monuments | earned))


def build_mask(s,rules,*,size):
    t,c,p=s.tiles,s.cities,s.players
    pi=s.current_player-1
    r=rules.structures
    types=jnp.arange(128)
    econ=economic_tiles(s)
    territory=in_territory(s)
    own_cities=c.owner==s.current_player
    present=c.territory.astype(jnp.int32) @ jax.nn.one_hot(t.structure,128,dtype=jnp.int32)>0
    tribe_has=jnp.any(present & own_cities[:,None],axis=0)
    unlocked=jnp.any(p.tech[pi,:,None] & r.unlock,axis=0) & r.allowed & (r.cost>=0) & (r.resource==0)
    unlocked &= (p.stars[pi]>=r.cost) & (~r.per_tribe | ~tribe_has)
    own_members=c.territory & own_cities[:,None]
    city_room=(own_members.T.astype(jnp.int32) @ (~present | ~r.per_city).astype(jnp.int32))>0
    free=(t.structure==0)[:,None]
    terrain=r.terrain[:,jnp.clip(t.terrain,0,15)].T
    adj=(distances(size)==1)
    adj_allowed=r.adjacent[:,t.structure].T
    neighbors_own=adj & (t.owner==s.current_player)[None,:]
    has_adj=(neighbors_own.astype(jnp.int32) @ adj_allowed.astype(jnp.int32))>0
    adj_ok=~jnp.any(r.adjacent,axis=-1) | has_adj
    empty= jnp.where(types==3,~t.road[:,None] & (c.owner==0)[:,None],free)
    standard=econ[:,None] & city_room & unlocked & terrain & empty & adj_ok
    # Resource builds use resource settings, not the terrain-building restrictions.
    res_tech=jnp.where(t.resource==2,8,jnp.where(t.resource==5,23,0))
    res_build=jnp.where(t.resource==2,5,jnp.where(t.resource==5,21,jnp.where(t.resource==7,37,0)))
    resources=econ[:,None] & free & (res_build[:,None]==types) & (res_build>0)[:,None] & p.tech[pi,res_tech,None] & (p.stars[pi]>=r.cost)
    count=jnp.sum(own_cities & c.connected)
    monuments=jnp.zeros(128,jnp.bool_).at[23].set(p.tech[pi,21] & (p.pacifist[pi]>=5))
    monuments=monuments.at[24].set(p.tech[pi,22] & (jnp.sum(p.tech[pi])>=25))
    monuments=monuments.at[25].set(count>4).at[26].set(p.tech[pi,5] & (count>4))
    monuments=monuments.at[27].set(p.kills[pi]>9).at[28].set(jnp.any(own_cities & (c.level>4)))
    corners=jnp.array([0,size-1,size*size-1,size*size-size])
    monuments=monuments.at[29].set(jnp.all(t.explored[pi,corners]) & s.fog)
    monuments=jnp.where(rules.native,p.monuments[pi] | monument_conditions(s,s.current_player,rules,size=size),monuments)
    monuments &= ~tribe_has & ~p.built[pi]
    special=econ[:,None] & free & terrain & monuments
    neutral=(t.owner==0)&t.explored[pi]&~t.road&((s.units.owner==0)|(s.units.owner==s.current_player)|(rules.native & s.units.invisible))
    standard |= neutral[:,None] & (types==3) & unlocked & terrain
    out=standard | resources | special
    # Bridge requires opposing land neighbors, using the reference fog terrain.
    ids=jnp.arange(size*size);x,y=ids%size,ids//size
    vis_terrain=jnp.where(t.explored[pi],t.terrain,0)
    water=(vis_terrain==1)|(vis_terrain==2)
    vertical=(y>0)&(y<size-1)&~water[jnp.clip(ids-size,0,size*size-1)]&~water[jnp.clip(ids+size,0,size*size-1)]
    horizontal=(x>0)&(x<size-1)&~water[jnp.clip(ids-1,0,size*size-1)]&~water[jnp.clip(ids+1,0,size*size-1)]
    return out.at[:,48].set(out[:,48]&(vertical|horizontal))


def build(s,target,kind,rules,*,size):
    before=s
    t,p=s.tiles,s.players
    r=rules.structures
    road=kind==3
    pi=s.current_player-1
    t=t._replace(structure=t.structure.at[target].set(jnp.where(road,t.structure[target],kind)),
                 structure_level=t.structure_level.at[target].set(jnp.where(road,t.structure_level[target],1)),
                 structure_turn=t.structure_turn.at[target].set(jnp.where(road,t.structure_turn[target],s.turn)),
                 road=t.road.at[target].set(t.road[target]|road))
    s=s._replace(tiles=t,players=p._replace(stars=p.stars.at[pi].add(-jnp.maximum(r.cost[kind],0))))
    s=jax.lax.cond((kind==3)|(kind==8)|(kind==48),lambda x:connections(x,x.current_player,size=size,native=rules.native),lambda x:x,s)
    score=jnp.where(temple(kind),100,r.score[kind])
    owner=t.owner[target]
    s=s._replace(players=s.players._replace(
        score=s.players.score.at[jnp.clip(owner-1,0,1)].add(jnp.where(owner>0,score,0)),
        built=s.players.built.at[pi,kind].set(s.players.built[pi,kind]|(score>0))))
    count=jnp.sum((distances(size)[target]==1)&r.adjacent[kind,t.structure],dtype=jnp.int32)
    pop=r.population[kind]*jnp.where(jnp.any(r.adjacent[kind]),count,1)
    return jax.lax.cond(rules.native,lambda x:adjust_building_population(before,x,rules,size=size),lambda x:add_population(x,t.city[target],pop),s)


def destroy_mask(s,rules):
    kind=s.tiles.structure
    return economic_tiles(s)&(kind>0)&(kind!=1)&(kind!=2)&s.players.tech[s.current_player-1,jnp.where(rules.native,3,2)]


def destroy(s,target,rules,*,size):
    before=s
    t,p=s.tiles,s.players
    kind=t.structure[target]
    score=scores(s,rules)[target]
    owner=t.owner[target]
    s=s._replace(tiles=t._replace(structure=t.structure.at[target].set(0),structure_level=t.structure_level.at[target].set(0),
                     structure_turn=t.structure_turn.at[target].set(0),road=t.road.at[target].set(t.road[target]&(kind!=3)&~(rules.native&(kind==48)))),
        players=p._replace(score=p.score.at[jnp.clip(owner-1,0,1)].add(jnp.where(owner>0,-score,0))))
    s=jax.lax.cond(rules.native,lambda x:adjust_building_population(before,x,rules,size=size),
        lambda x:add_population(x,t.city[target],-rules.structures.population[kind]),s)
    return jax.lax.cond((kind==3)|(kind==8)|(kind==48),lambda x:connections(x,x.current_player,size=size,native=rules.native),lambda x:x,s)


def start_turn(s,owner,rules):
    t=s.tiles
    expected=jnp.clip(1+(s.turn-t.structure_turn)//jnp.where(rules.native,2,3),1,5)
    new=jnp.where(temple(t.structure)&(t.owner==owner),jnp.maximum(expected,t.structure_level),t.structure_level)
    return s._replace(tiles=t._replace(structure_level=new),players=s.players._replace(score=s.players.score.at[owner-1].add(jnp.sum(new-t.structure_level,dtype=jnp.int32)*100)))


def income(s,owner,rules,*,size):
    t,c,u=s.tiles,s.cities,s.units
    adj=distances(size)==1
    r=rules.structures
    market=(t.structure==50)
    count=jnp.sum(adj & r.adjacent[50,t.structure],axis=-1,dtype=jnp.int32)
    count=jnp.where(rules.native,production_levels(s,rules,size=size),count)
    bonus=jnp.sum(c.territory * (market*count*r.stars[50])[None,:],axis=-1,dtype=jnp.int32)
    # Workshop/Park increments are already stored in city.production.
    production=c.production+((t.capital==c.owner)&(t.capital>0))+jnp.where(rules.native,0,c.rewards[:,3]+c.rewards[:,2])+bonus
    production=jnp.where(c.progress<0,jnp.maximum(production+c.progress,0),production)
    siege=((u.owner!=0)&(u.owner!=c.owner)&~(rules.native & u.invisible)) | (rules.native & c.riot)
    p=owner-1
    embassy_count=jnp.sum(s.players.embassy[p].astype(jnp.int32)+s.players.embassy[:,p].astype(jnp.int32))
    embassy_income=embassy_count*jnp.where(s.players.peace[p,1-p],4,2)
    return jnp.sum(jnp.where((c.owner==owner)&~siege,production,0),dtype=jnp.int32)+jnp.where(rules.native,embassy_income,0)
