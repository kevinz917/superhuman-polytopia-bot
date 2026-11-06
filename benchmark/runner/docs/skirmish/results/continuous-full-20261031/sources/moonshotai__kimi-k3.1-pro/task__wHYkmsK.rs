// Heuristic Bardur skirmish bot.  Only the current public observation and legal
// catalog are used; all coordinates are derived from the advertised board size.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn i(v: &Value, k: &str) -> i32 { v[k].as_i64().unwrap_or(-1) as i32 }
fn num(v: &Value, k: &str) -> f64 { v[k].as_f64().unwrap_or(0.0) }
fn dist(a: i32, b: i32, s: i32) -> i32 {
    ((a % s - b % s).abs()).max((a / s - b / s).abs())
}
fn stats(kind: i32) -> (f64, f64, f64, i32, i32) { // hp, attack, defense, range, cost
    match kind {
        2 => (10.,2.,2.,1,2), 3 => (10.,2.,1.,1,3), 4 => (10.,3.5,1.,1,8),
        5 => (15.,1.,3.,1,3), 8 => (10.,4.,0.,3,8), 9 => (10.,2.,1.,2,3),
        10 => (10.,0.,1.,1,5), 11 => (15.,3.,3.,1,5), 12 => (40.,5.,4.,1,10),
        37 => (10.,2.,2.,1,2), 38 => (5.,0.,0.5,1,8), 39 => (5.,0.,0.5,1,2),
        40 => (10.,2.,2.,1,2), 41 => (10.,3.,2.,3,15), 42 => (10.,2.,1.,2,5),
        43 => (10.,0.,1.,2,0), 44 => (10.,3.,3.,1,5), 45 => (40.,4.,4.,1,10),
        _ => (10.,2.,1.,1,3),
    }
}
fn max_hp(u: &Value) -> f64 {
    let k = if i(u,"passenger") > 0 { i(u,"passenger") } else { i(u,"kind") };
    stats(k).0 + if u["veteran"].as_bool().unwrap_or(false) { 5. } else { 0. }
}
fn combat(ah: f64, am: f64, aa: f64, dh: f64, dm: f64, dd: f64) -> (f64,f64) {
    let af=aa*ah/am.max(1.); let df=dd*dh/dm.max(1.); let total=af+df;
    if total<=0. { return (0.,0.) }
    let attack=((af/total)*aa*4.5+0.000001+0.5).floor();
    let ret=if dh>attack { ((df/total)*dd*4.5+0.000001+0.5).floor() } else {0.};
    (attack,ret)
}

fn choose(obs: &Value) -> String {
    let legal=obs["legal_actions"].as_array().unwrap();
    let v=&obs["visible_state"];
    let p=obs["player"].as_i64().unwrap_or(1) as i32;
    let s=v["board_size"].as_i64().unwrap_or(14) as i32;
    let n=s*s; let turn=i(v,"turn"); let stars=i(v,"stars");
    let units=v["units"].as_array().cloned().unwrap_or_default();
    let cities=v["cities"].as_array().cloned().unwrap_or_default();
    let tiles=v["tiles"].as_array().cloned().unwrap_or_default();
    let own_cities=cities.iter().filter(|c| i(c,"owner")==p).count() as i32;
    let own_units=units.iter().filter(|u| i(u,"owner")==p).count() as i32;
    let enemy_units=units.iter().filter(|u| {let o=i(u,"owner");o>0&&o!=p}).count() as i32;
    let tech=|k:i32| v["technology"].as_array().map(|a|a.iter().any(|x|x.as_i64()==Some(k as i64))).unwrap_or(false);
    let unit_at=|t:i32| units.iter().find(|u|i(u,"tile")==t);
    let tile_at=|t:i32| tiles.iter().find(|x|i(x,"tile")==t);
    let terrain=|t:i32| tile_at(t).map(|x|i(x,"terrain")).unwrap_or(-1);
    let structure=|t:i32| tile_at(t).map(|x|i(x,"structure")).unwrap_or(-1);
    let owner=|t:i32| tile_at(t).map(|x|i(x,"owner")).unwrap_or(-1);
    let water_known=tiles.iter().filter(|t|t["known"].as_bool()==Some(true)&&(i(t,"terrain")==1||i(t,"terrain")==2)).count() as i32;
    let known=tiles.iter().filter(|t|t["known"].as_bool()==Some(true)).count().max(1) as i32;
    let watery=water_known*3>known || tech(10) || tech(13);
    let late=turn>75;

    // Targets are deliberately ordered: expansion first, then known enemy cities,
    // then fog.  Nearby enemy units become targets when they threaten our cities.
    let foreign_cities:Vec<i32>=cities.iter().filter(|c|i(c,"owner")!=p).map(|c|i(c,"tile")).collect();
    let villages:Vec<i32>=foreign_cities.iter().copied().filter(|&t|owner(t)==0).collect();
    let hostile_cities:Vec<i32>=foreign_cities.iter().copied().filter(|&t|owner(t)>0).collect();
    let ruins:Vec<i32>=tiles.iter().filter(|t|t["known"].as_bool()==Some(true)&&i(t,"structure")==2).map(|t|i(t,"tile")).collect();
    let unknown:Vec<i32>=tiles.iter().filter(|t|t["known"].as_bool()!=Some(true)).map(|t|i(t,"tile")).collect();
    let enemy_tiles:Vec<i32>=units.iter().filter(|u|{let o=i(u,"owner");o>0&&o!=p}).map(|u|i(u,"tile")).collect();
    let own_city_tiles:Vec<i32>=cities.iter().filter(|c|i(c,"owner")==p).map(|c|i(c,"tile")).collect();
    let nearest=|from:i32, list:&Vec<i32>| -> i32 {list.iter().map(|&t|dist(from,t,s)).min().unwrap_or(20)};
    let threatened=|city:i32| enemy_tiles.iter().any(|&e|dist(city,e,s)<=3);

    let mut best:&Value=&legal[0]; let mut best_score=-1e100;
    for a in legal {
        let c=&a["command"]; let k=i(c,"kind"); let src=i(c,"source"); let dst=i(c,"target"); let par=i(c,"parameter");
        let mut sc=0.;
        match k {
            11 => sc=-1e6,
            10 => sc=-1000.,
            9 => { // mandatory city rewards
                sc=10000.;
                sc+=match par {3=>120.,4=>if own_cities<=1 {60.}else{100.},7=>110.,1=>20.,8=>95.,5=>65.,6=>150.,2=>30.,_=>0.};
            },
            8 => {sc=5000.; if structure(src)==2 {sc+=200.} if owner(src)>0&&owner(src)!=p {sc+=1000.}},
            2 => {
                sc=800.;
                if let (Some(au),Some(du))=(unit_at(src),unit_at(dst)) {
                    let ak=i(au,"kind"); let dk=i(du,"kind");
                    let ah=num(au,"health"); let dh=num(du,"health");
                    let am=max_hp(au); let dm=max_hp(du);
                    let aa=stats(ak).1; let mut dd=stats(dk).2;
                    let tr=terrain(dst);
                    let city_bonus=if own_city_tiles.contains(&dst)&&i(du,"owner")==owner(dst) {1.5}else{1.};
                    // Conservative estimate of enemy terrain technology.
                    let bonus=if own_city_tiles.contains(&dst){city_bonus}else{1.};
                    dd*=bonus;
                    let (damage,ret)=combat(ah,am,aa,dh,dm,dd);
                    let lethal=dh-damage<=0.01;
                    let suicide=ah-ret<=0.01 && !lethal;
                    let range=stats(ak).3;
                    let retaliation=if range<=dist(src,dst,s)&&stats(dk).3>=dist(src,dst,s)&&stats(dk).1>0.{ret}else{0.};
                    sc+=damage*35.+stats(dk).4 as f64*4.;
                    if lethal {sc+=500.+stats(dk).4 as f64*25.;}
                    if suicide {sc-=1200.;}
                    sc-=retaliation*22.;
                    if structure(dst)==1 {sc+=100.;}
                    if dh<=3. {sc+=80.;}
                    if ah<=3.&&!lethal {sc-=250.;}
                }
            },
            5 => sc=650., // cheap population is the backbone of the opening
            6 => {
                sc=match par {
                    8=>if watery {470.}else{250.}, // port
                    5|21=>390., 12=>360., 6|13|22=>300., 50=>330.,
                    48=>if watery {180.}else{40.},3=>75.,
                    17|18|19|20=>if late {150.}else{30.},
                    23..=29=>500., _=>40.
                };
                // Prefer productive coasts and do not pave the whole map.
                if par==3 {sc-=own_cities as f64*3.;}
            },
            7 => {
                let priority=match par {
                    1=>560.,             // Riding: mobile expansion
                    10=>if watery {510.}else{-200.},
                    16=>360.,            // Forestry
                    6=>if tiles.iter().any(|t|i(t,"resource")==6&&owner(i(t,"tile"))==p){180.}else{-200.},
                    4=>180., 2=>if own_cities>=3 {260.}else{-100.}, 3=>if own_cities>=3 {380.}else{-100.}, // road / knight line
                    13=>if water_known*3>known&&tech(10) {1600.}else if watery {430.}else{-200.},
                    44=>if water_known*3>known&&tech(13) {800.}else if watery {200.}else{-200.},
                    18=>100., 20=>-100., 23=>120., 24=>250.,
                    8=>100.,9=>80.,7=>80.,17=>100.,5=>100.,12=>160.,14=>120.,
                    _=>45.
                };
                sc=priority;
                // Expansion pays for technology; do not research a broad tree with one city.
                if own_cities==1&&turn>12&&par!=1&&par!=10&&par!=16 {sc-=100.;}
            },
            4 => {
                let cost=stats(par).4;
                let mut base=match par {
                    4=>430.,11=>390.,3=>470.,2=>300.,5=>260.,9=>280.,
                    44=>if watery {370.}else{100.},42=>if watery {350.}else{100.},
                    41=>if watery {300.}else{80.},_=>80.
                };
                if structure(src)==1 {base+=20.;}
                // Keep an opening technology fund; large treasuries should become army.
                if stars<8 && own_cities<=2 && par!=3 {base-=40.;}
                if !tech(1)&&turn<3&&stars<5 {base-=1000.;}
                if stars-cost<2 && turn<25 {base-=60.;}
                if own_units>=own_cities*3+3 {base-=100.;}
                if enemy_tiles.iter().any(|&e|dist(src,e,s)<=3){base+=130.;}
                sc=base;
            },
            3 => {
                sc=match par {
                    16=>900., // promotion is free and heals
                    7=> {
                        if let Some(u)=unit_at(src) {let hp=num(u,"health"); if hp<=4.{360.}else{20.}} else{0.}
                    },
                    2=>120.,1=>20.,3=>20.,9=>180.,19=>if i(c,"target")==1&&own_units*2<enemy_units{700.}else{-500.},
                    20=>100.,8|18|21|22=>-10000.,4=>-10000.,_=>0.
                };
            },
            1 => {
                sc=100.;
                let dest_village=villages.contains(&dst);
                let dest_ruin=ruins.contains(&dst);
                let dest_enemy=hostile_cities.contains(&dst);
                if dest_village {sc+=900.;}
                if dest_ruin {sc+=500.;}
                if dest_enemy {sc+=700.;}
                // A threatened home city takes precedence over speculative exploration.
                let defensive=own_city_tiles.iter().any(|&c|threatened(c));
                let (old_d,new_d)=if defensive&&!enemy_tiles.is_empty() {
                    (nearest(src,&enemy_tiles),nearest(dst,&enemy_tiles))
                } else if !villages.is_empty() {
                    (nearest(src,&villages),nearest(dst,&villages))
                } else if !hostile_cities.is_empty() {
                    (nearest(src,&hostile_cities),nearest(dst,&hostile_cities))
                } else if !ruins.is_empty()&&nearest(src,&ruins)<=4 {
                    (nearest(src,&ruins),nearest(dst,&ruins))
                } else if !unknown.is_empty() {
                    (nearest(src,&unknown),nearest(dst,&unknown))
                } else if !enemy_tiles.is_empty() {
                    (nearest(src,&enemy_tiles),nearest(dst,&enemy_tiles))
                } else {(0,0)};
                sc+=(old_d-new_d) as f64*55.;
                if hostile_cities.is_empty() && villages.is_empty() {
                    if let Some(cap)=tiles.iter().find(|t|i(t,"capital")==p).map(|t|i(t,"tile")) {
                        let gx=s-1-cap%s; let gy=s-1-cap/s; let goal=gy*s+gx;
                        sc+=(dist(src,goal,s)-dist(dst,goal,s)) as f64*90.;
                    }
                }
                if new_d>=old_d&&!dest_village&&!dest_ruin&&!dest_enemy {sc-=100.;}
                // Reveal value: ending beside more unknown tiles is useful.
                let reveal=|t:i32| unknown.iter().filter(|&&u|dist(t,u,s)<=1).count() as i32;
                sc+=(reveal(dst)-reveal(src)) as f64*4.;
                if let Some(u)=unit_at(src) {
                    let hp=num(u,"health");
                    if hp<=4. {sc-=enemy_tiles.iter().filter(|&&e|dist(dst,e,s)<=1).count() as f64*350.;}
                    if i(u,"kind")==43 {sc-=30.;} // upgrade rafts before sailing where possible
                }
            },
            _=>sc=0.
        }
        // Stable, deterministic tie break without relying on catalog order.
        if sc>best_score {best_score=sc;best=a;}
    }
    best["action_id"].as_str().unwrap().to_owned()
}

fn main() {
    let stdin=io::stdin(); let mut stdout=io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let obs:Value=serde_json::from_str(&line.unwrap()).unwrap();
        writeln!(stdout,"{}",json!({"action_id":choose(&obs)})).unwrap();
        stdout.flush().unwrap();
    }
}
