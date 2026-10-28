//! Validation-only pinned generator with explicit .NET sampling semantics.
//! The vendor source remains unchanged. This is NOT the commercial generator.
pub use polyfish::*;
use std::io::{self,BufRead};
use std::ops::{Range,RangeInclusive};
struct NativeRandom(dotnet_rng::DotNetRandom);
trait DrawRange {type Value;fn draw(self,rng:&mut dotnet_rng::DotNetRandom)->Self::Value;}
impl DrawRange for Range<i32>{type Value=i32;fn draw(self,r:&mut dotnet_rng::DotNetRandom)->i32{r.next_range(self.start,self.end)}}
impl DrawRange for Range<usize>{type Value=usize;fn draw(self,r:&mut dotnet_rng::DotNetRandom)->usize{r.next_range(self.start as i32,self.end as i32)as usize}}
impl DrawRange for RangeInclusive<i32>{type Value=i32;fn draw(self,r:&mut dotnet_rng::DotNetRandom)->i32{r.next_range(*self.start(),*self.end()+1)}}
impl DrawRange for Range<f32>{type Value=f32;fn draw(self,r:&mut dotnet_rng::DotNetRandom)->f32{self.start+r.next_float()*(self.end-self.start)}}
impl NativeRandom {
    fn seed_from_u64(seed:u64)->Self{Self(dotnet_rng::DotNetRandom::new(seed as i32))}
    fn random<T:From<f32>>(&mut self)->T{T::from(self.0.next_float())}
    fn random_range<R:DrawRange>(&mut self,range:R)->R::Value{range.draw(&mut self.0)}
}
mod corrected { include!(concat!(env!("OUT_DIR"),"/mapgen.rs")); }
fn main(){
    for line in io::stdin().lock().lines(){
        let value:serde_json::Value=serde_json::from_str(&line.unwrap()).unwrap();
        let size=match value["size"].as_i64().unwrap(){11=>types::MapSize::Tiny,14=>types::MapSize::Small,18=>types::MapSize::Large,_=>panic!("unsupported size")};
        let state=corrected::generate(corrected::MapGenSettings{size,map_type:types::MapType::Drylands,
            tribes:vec![types::TribeType::Bardur,types::TribeType::Bardur],seed:value["seed"].as_i64().unwrap(),version:115,symmetric:false});
        println!("{}",serde_json::to_string(&state).unwrap());
    }
}
