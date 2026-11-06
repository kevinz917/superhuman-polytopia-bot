"""Device xxHash32 and the reference's deterministic city explorer."""
import jax
import jax.numpy as jnp
from .kernels import distances, neighbors


def hash_word(seed, word):
    """Polyfish hash::get_hash for one little-endian i32 word, wrapping u32."""
    return hash_words(seed,(word,))


def hash_words(seed, words):
    """Standard xxHash32 short-input path, up to three little-endian words."""
    if len(words)>3:
        raise ValueError('short xxHash input must be below 16 bytes')
    h=jnp.asarray(seed,jnp.uint32)+jnp.uint32(374761393)+jnp.uint32(4*len(words))
    for word in words:
        h=h+jnp.asarray(word,jnp.uint32)*jnp.uint32(3266489917)
        h=((h<<jnp.uint32(17))|(h>>jnp.uint32(15)))*jnp.uint32(668265263)
    h=(h^(h>>jnp.uint32(15)))*jnp.uint32(2246822519)
    h=(h^(h>>jnp.uint32(13)))*jnp.uint32(3266489917)
    return h^(h>>jnp.uint32(16))


def explorer(s,source,*,size):
    n=size*size
    d=distances(size)
    nb,valid=neighbors(size)
    p=s.current_player-1
    initial=s.tiles.explored[p]
    tech=s.players.tech[p]
    def body(_,v):
        current,previous,visible,stopped=v
        fog=~visible
        count=jnp.clip(jnp.sum((d<=2)&fog[None,:],axis=-1),1,4)
        light=jnp.any((d<=2)&fog[None,:]&(s.tiles.structure==47)[None,:],axis=-1)
        score=jnp.where(fog,183-count*10-jnp.where(light,33,0),1000)
        def spread(_,score):
            return jnp.minimum(score,jnp.min(jnp.where(valid,score[nb]+100,1000),axis=-1))
        score=jax.lax.fori_loop(0,3,spread,score)
        indices=nb[current]
        # Real execution has _areYouSure=true, so simulated-visible cells
        # use their actual terrain even before explorers are committed.
        terrain=s.tiles.terrain[indices]
        passable=(terrain!=0)&((terrain!=4)|tech[20])&((terrain!=1)|tech[13])&((terrain!=2)|tech[14])
        allowed=valid[current]&(~visible[indices]|passable)
        values=jnp.where(allowed,score[indices]+(indices==previous),10000)
        best=jnp.min(values)
        candidates=allowed&(values==best)
        count=jnp.sum(candidates,dtype=jnp.int32)
        draw=(hash_word(s.seed,current)%jnp.maximum(count,1).astype(jnp.uint32)).astype(jnp.int32)
        which=jnp.argmax(candidates&(jnp.cumsum(candidates)-1==draw))
        target=indices[which]
        stop=stopped|(count==0)|(target==current)
        next_visible=visible|((d[target]<=1)&~stop)
        return jnp.where(stop,current,target),jnp.where(stop,previous,current),next_visible,stop
    _,_,visible,_=jax.lax.fori_loop(0,12,body,(source,jnp.int32(-1),initial,jnp.bool_(False)))
    return visible & ~initial
