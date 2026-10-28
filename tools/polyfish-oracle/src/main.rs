//! Validation-only oracle. Never called by the device transition loop.
use polyfish::{Game, *};
use polyfish::mapgen::{self, MapGenSettings};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use strum::IntoEnumIterator;

const REV: &str = "a9f90ab54607f5633556d02422d7457f548cc58f";

fn rules() -> Value {
    let units: Vec<_> = UnitType::iter().map(|u| {
        let s = polyfish::settings::units::get_unit_setting(u);
        let mut skills: Vec<_> = s.skills.iter().map(|s| format!("{s:?}")).collect();
        skills.sort();
        json!({"id":u as i8,"name":format!("{u:?}"),"cost":s.cost,
            "attack":s.attack,"defense":s.defense,"health":s.health,
            "movement":s.movement,"range":s.range,"skills":skills})
    }).collect();
    let techs: Vec<_> = TechnologyType::iter().map(|t| {
        let s = polyfish::settings::technology::get_technology_setting(t);
        json!({"id":t as i8,"name":format!("{t:?}"),"tier":s.tier.unwrap_or(1),
            "structures": s.unlocks_structure.into_iter().chain(s.unlocks_special_structures.iter().copied()).map(|x|x as i8).collect::<Vec<_>>(),
            "units":polyfish::settings::technology::get_unlocked_units(t,TribeType::Bardur).iter().map(|x|*x as i8).collect::<Vec<_>>(),
            "requires":s.requires.map(|t|t as i8),"next":s.next.iter().map(|t|*t as i8).collect::<Vec<_>>()})
    }).collect();
    let structures: Vec<_> = StructureType::iter().map(|t| {
        let s=polyfish::settings::structures::get_structure_setting(t);
        json!({"id":t as i8,"name":format!("{t:?}"),"cost":s.cost,"population":s.reward_pop,"score":s.reward_score,"stars":s.reward_stars,
          "terrain":s.terrain_types.iter().map(|x|*x as i8).collect::<Vec<_>>(),"adjacent":s.adjacent_types.iter().map(|x|*x as i8).collect::<Vec<_>>(),
          "resource":s.resource_type.map(|x|x as i8),"per_city":s.limited_per_city,"per_tribe":s.limited_per_tribe,"tribe":s.tribe_type.map(|x|x as i8)})
    }).collect();
    json!({"engine_revision":REV,"units":units,"technologies":techs,"structures":structures})
}

fn run(v: Value) -> Result<Value, String> {
    match v["op"].as_str().ok_or("missing op")? {
        "rules" => Ok(rules()),
        "combat" => {
            let values = v["cases"].as_array().ok_or("missing cases")?;
            let results: Result<Vec<_>,String> = values.iter().map(|c| {
                let a = c.as_array().ok_or("case must be an array")?;
                if a.len()!=7 { return Err("combat case requires 7 floats".into()); }
                let nums: Result<Vec<_>,String> = a.iter().map(|n|n.as_f64().map(|n|n as f32).ok_or("invalid float".into())).collect();
                let n=nums?;
                let r=polyfish::actions::units::calculate_combat(n[0],n[1],n[2],n[3],n[4],n[5],n[6]);
                Ok(json!([r.attack_damage,r.defense_damage,r.splash_damage]))
            }).collect();
            Ok(json!({"results":results?}))
        }
        "reset" => {
            let size = match v["size"].as_i64().unwrap_or(14) {
                11 => MapSize::Tiny, 14 => MapSize::Small,
                _ => return Err("oracle reset supports size 11 or 14".into()),
            };
            let state=mapgen::generate(MapGenSettings {
                size,map_type:MapType::Drylands,tribes:vec![TribeType::Bardur,TribeType::Bardur],
                seed:v["seed"].as_i64().unwrap_or(1), version:115,symmetric:false,
            });
            let mut g=Game{state};
            if !v["raw"].as_bool().unwrap_or(false) { g.post_load(); }
            Ok(json!({"state":g.state,"legal":g.legal_moves().iter().map(|m|m.serialize()).collect::<Vec<_>>()}))
        }
        "query"|"step"|"explore"|"researchable" => {
            // Direct deserialize: from_json/post_load recalculates exploration and
            // scores and would hide replay errors. The caller supplies a full state.
            let mut g=Game {state:serde_json::from_value(v["state"].clone()).map_err(|e|e.to_string())?};
            if v["op"]=="researchable" {
                let tribe=&g.state.tribes[&g.state.settings.current_player_turn_id];
                return Ok(json!(polyfish::settings::technology::get_researchable_techs(&tribe.tech_vanilla,tribe.tribe_type).iter().map(|t|json!({"id":*t as i8,"name":format!("{t:?}")})).collect::<Vec<_>>()));
            }
            if v["op"]=="step" {
                let legal=g.legal_moves();
                let m=legal.iter().find(|m|m.serialize()==v["action"]).ok_or("action not in complete legal set")?;
                if g.play_move(m.as_ref()).is_none() {return Err("move execution failed".into());}
            }
            if v["op"]=="explore" {
                g.state.settings._are_you_sure=true;
                let p=v["player"].as_i64().ok_or("missing player")? as i32;
                let _=polyfish::actions::update_exploration(&mut g.state,p);
                g.state.settings._are_you_sure=false;
            }
            Ok(json!({"state":g.state,"legal":g.legal_moves().iter().map(|m|m.serialize()).collect::<Vec<_>>()}))
        }
        _ => Err("unknown op".into()),
    }
}

fn main() {
    let stdin=io::stdin();
    let mut out=io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let result=line.map_err(|e|e.to_string()).and_then(|l|serde_json::from_str(&l).map_err(|e|e.to_string())).and_then(run);
        let response=match result {Ok(v)=>json!({"ok":true,"result":v}),Err(e)=>json!({"ok":false,"error":e})};
        writeln!(out,"{response}").unwrap(); out.flush().unwrap();
    }
}
