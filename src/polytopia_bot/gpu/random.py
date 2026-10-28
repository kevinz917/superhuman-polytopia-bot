"""Device .NET seeded System.Random stream for procedural generation.

A .NET sample has 31 random bits. It must NOT be passed unchanged into a
32-bit-uniform RNG interface: that biases bounded draws toward half the map.
Reference algorithm: Polyfish dotnet_rng.rs; independent Microsoft runtime
vectors test this port, including signed-seed overflow behavior.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

MAX=jnp.int32(2147483647)


class RandomState(NamedTuple):
    values: jax.Array
    current: jax.Array
    lag: jax.Array


def initialize(seed):
    seed=jnp.asarray(seed,jnp.int32)
    absolute=jnp.where(seed==jnp.int32(-2147483648),MAX,jnp.abs(seed))
    first=jnp.int32(161803398)-absolute
    values=jnp.zeros(56,jnp.int32).at[55].set(first)
    def fill(i,carry):
        a,previous,next_value=carry
        at=(21*i)%55
        a=a.at[at].set(next_value)
        delta=previous-next_value
        return a,next_value,jnp.where(delta<0,delta+MAX,delta)
    values,_,_=jax.lax.fori_loop(1,55,fill,(values,first,jnp.int32(1)))
    def warm(_,a):
        def subtract(i,a):
            delta=a[i]-a[1+(i+30)%55]
            return a.at[i].set(jnp.where(delta<0,delta+MAX,delta))
        return jax.lax.fori_loop(1,56,subtract,a)
    values=jax.lax.fori_loop(0,4,warm,values)
    return RandomState(values,jnp.int32(0),jnp.int32(21))


def next_int(rng):
    i=jnp.where(rng.current==55,1,rng.current+1)
    j=jnp.where(rng.lag==55,1,rng.lag+1)
    value=rng.values[i]-rng.values[j]
    value=jnp.where(value==MAX,value-1,value)
    value=jnp.where(value<0,value+MAX,value)
    return RandomState(rng.values.at[i].set(value),i,j),value


def bounded(rng,upper):
    """Exact Next(0, upper) for map-sized positive bounds <= 65,535.

    Compute the 64-bit product/division with 32-bit limbs, so CUDA does not
    require enabling global JAX float64/int64 mode or lose integer precision.
    """
    rng,value=next_int(rng)
    a=value.astype(jnp.uint32)
    b=jnp.asarray(upper,jnp.uint32)
    lower=(a&jnp.uint32(65535))*b
    middle=(a>>jnp.uint32(16))*b+(lower>>jnp.uint32(16))
    high=middle>>jnp.uint32(16)
    low=(middle<<jnp.uint32(16))|(lower&jnp.uint32(65535))
    quotient=(high<<jnp.uint32(1))|(low>>jnp.uint32(31))
    remainder=(low&jnp.uint32(2147483647))+quotient
    quotient += (remainder>=jnp.uint32(2147483647)).astype(jnp.uint32)
    return rng,quotient.astype(jnp.int32)


def unit_float(rng):
    """Float32 cast of NextDouble, including upward-shifted halfway cases."""
    rng,value=next_int(rng)
    bits=value.astype(jnp.uint32)
    shift=jnp.maximum(8-jax.lax.clz(bits),0)
    quantum=jnp.uint32(1)<<shift.astype(jnp.uint32)
    tie=(shift>0)&((bits&(quantum-1))==(quantum>>jnp.uint32(1)))&(((bits//quantum)&1)==0)
    result=value.astype(jnp.float32)*jnp.float32(1/2147483648.)
    result=jnp.where(tie,jnp.nextafter(result,jnp.float32(jnp.inf)),result)
    return rng,result
