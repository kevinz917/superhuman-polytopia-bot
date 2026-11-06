use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::collections::HashMap;

#[derive(Clone)]
struct Act { id: String, k: i32, s: i32, t: i32, p: i32 }
#[derive(Clone, Default)]
struct Unit { owner: i32, kind: i32, hp: f64, veteran: bool, moved: bool, attacked: bool, home: i32 }
#[derive(Default)]
struct Memory { enemy_seen: bool, last_turn: i32, starting_capital: i32 }

fn iv(v: &Value, k: &str) -> i32 { v[k].as_i64().unwrap_or(-1) as i32 }
fn d(size: i32, a: i32, b: i32) -> i32 {
    if a < 0 || b < 0 { return 99; }
    ((a%size-b%size).abs()).max((a/size-b/size).abs())
}
fn stats(k: i32) -> (f64,f64,f64,f64,i32,i32) { // hp, attack, defense, cost, move, range
    match k {
        2 => (10.,2.,2.,2.,1,1), 3 => (10.,2.,1.,3.,2,1), 4 => (10.,3.5,1.,8.,3,1),
        5 => (15.,1.,3.,3.,1,1), 8 => (10.,4.,0.,8.,1,3), 9 => (10.,2.,1.,3.,1,2),
        10 => (10.,0.,1.,5.,1,1), 11 => (15.,3.,3.,5.,1,1), 12 => (40.,5.,4.,10.,1,1),
        37 => (10.,2.,2.,2.,1,1), 38 => (5.,0.,0.5,8.,2,1), 39 => (5.,0.,0.5,2.,2,1),
        40 => (10.,2.,2.,2.,2,1), 41 => (10.,3.,2.,15.,2,3), 42 => (10.,2.,1.,5.,3,2),
        43 => (10.,0.,1.,0.,2,2), 44 => (10.,3.,3.,5.,3,1), 45 => (40.,4.,4.,10.,2,1),
        _ => (10.,2.,1.,3.,1,1)
    }
}
fn is_boat(k:i32)->bool { matches!(k,39|40|41|42|43|44|45) }
fn naval_kind(k:i32)->bool { matches!(k,39|40|41|42|43|44|45) }

fn tech_value(tech: i32, turn: i32, water: f64, known: &Vec<i32>, enemy: bool) -> f64 {
    let has = |x| known.contains(&x);
    let early = if turn < 18 { 1.25 } else if turn < 45 { 1.0 } else { 0.55 };
    let mut v = match tech {
        1 => 150.,                         // Riding
        16 => 130.,                        // Forestry
        6 => 112.,                         // Organization
        10 => 65. + 105.*water,            // Fishing
        20 => 52.,                         // Climbing
        4 => if has(1) {112.} else {0.},   // Roads
        2 => if has(1) {90.} else {0.},
        3 => if has(2) {145.} else {0.},
        18 => 92.,                         // Archery
        8 => if has(6) {165.} else {0.},
        9 => if has(8) {85.} else {0.},
        23 => if has(20) {85.} else {0.},
        24 => if has(23) {135.} else {0.},
        17 => if has(16) {115.} else {0.},
        13 => if has(10) {55.+75.*water} else {0.},
        44 => if has(10) {50.+70.*water} else {0.},
        7 => if has(6) {65.} else {0.},
        14 => if has(44) {35.+65.*water} else {0.},
        12 => if has(44) {45.} else {0.},
        19 => if has(18) {35.} else {0.},
        5 => if has(4) {50.} else {0.},
        21 => if has(20) {25.} else {0.},
        22 => if has(21) {40.} else {0.},
        _ => 0.
    };
    if enemy && matches!(tech,3|24|17|18) { v += 25.; }
    v * early
}

fn choose(obs: &Value, mem: &mut Memory) -> String {
    let legal_v = obs["legal_actions"].as_array().expect("legal_actions");
    let mut legal = Vec::new();
    for a in legal_v {
        let c=&a["command"];
        legal.push(Act{id:a["action_id"].as_str().unwrap().to_owned(),k:iv(c,"kind"),s:iv(c,"source"),t:iv(c,"target"),p:iv(c,"parameter")});
    }
    if legal.is_empty() { return "0".into(); }
    let v=&obs["visible_state"];
    let me=obs["player"].as_i64().unwrap_or(1) as i32;
    let size=iv(v,"board_size"); let n=size*size; let turn=iv(v,"turn");
    let stars=iv(v,"stars");
    let known:Vec<i32>=v["technology"].as_array().map(|a|a.iter().filter_map(|x|x.as_i64().map(|z|z as i32)).collect()).unwrap_or_default();
    let mut terrain=vec![-1;n as usize]; let mut owner=vec![-1;n as usize]; let mut structure=vec![-1;n as usize];
    let mut resource=vec![0;n as usize]; let mut explored=vec![false;n as usize]; let mut capital=vec![0;n as usize];
    let mut water_count=0.0f64; let mut known_count=0.0f64;
    if let Some(ts)=v["tiles"].as_array() { for t in ts {
        let i=iv(t,"tile") as usize; if i>=n as usize {continue}
        explored[i]=t["known"].as_bool().unwrap_or(false);
        if explored[i] { terrain[i]=iv(t,"terrain"); owner[i]=iv(t,"owner"); structure[i]=iv(t,"structure");
            resource[i]=iv(t,"resource"); capital[i]=iv(t,"capital"); known_count+=1.; if terrain[i]<=2 {water_count+=1.} }
    }}
    let water=water_count/known_count.max(1.);
    let mut units:HashMap<i32,Unit>=HashMap::new();
    if let Some(us)=v["units"].as_array() { for u in us {
        let t=iv(u,"tile"); units.insert(t,Unit{owner:iv(u,"owner"),kind:iv(u,"kind"),hp:u["health"].as_f64().unwrap_or(10.),
            veteran:u["veteran"].as_bool().unwrap_or(false),moved:u["moved"].as_bool().unwrap_or(false),
            attacked:u["attacked"].as_bool().unwrap_or(false),home:iv(u,"home")});
    }}
    let mut own_cities=Vec::new(); let mut neutral_cities=Vec::new(); let mut enemy_cities=Vec::new();
    if let Some(cs)=v["cities"].as_array() { for c in cs { let t=iv(c,"tile"); let o=iv(c,"owner");
        if o==me {own_cities.push(t)} else if o==0 {neutral_cities.push(t)} else {enemy_cities.push(t)} }
    }
    let enemies:Vec<i32>=units.iter().filter(|(_,u)|u.owner>0&&u.owner!=me).map(|(&t,_)|t).collect();
    let friendlies:Vec<i32>=units.iter().filter(|(_,u)|u.owner==me).map(|(&t,_)|t).collect();
    if !enemy_cities.is_empty()||!enemies.is_empty() {mem.enemy_seen=true;}
    if mem.starting_capital<0 || mem.last_turn>turn || (turn==0&&mem.starting_capital==0) {
        mem.starting_capital=own_cities.first().copied().unwrap_or(-1); mem.enemy_seen=false;
    }
    mem.last_turn=turn;
    let ruins:Vec<i32>=(0..n).filter(|&i|explored[i as usize]&&structure[i as usize]==2).collect();
    let unknown:Vec<i32>=(0..n).filter(|&i|!explored[i as usize]).collect();
    let enemy_goal=!enemy_cities.is_empty();
    let army=friendlies.len();

    let mut best:Option<(f64,String)>=None;
    let mut offer=|score:f64,a:&Act| {
        // Stable deterministic ordering, with a tiny preference for low action IDs only as a final tie break.
        if best.as_ref().map(|x|score>x.0+1e-9).unwrap_or(true) {best=Some((score,a.id.clone()));}
    };
    for a in &legal {
        let k=a.k; let src=a.s; let dst=a.t; let param=a.p;
        let mut score=match k {
            11=>-1e9, 10=>-1000., 9=>10000., 8=>9000., 2=>0., 5=>0., 4=>0., 6=>0., 7=>0., 1=>0., 3=>0., _=>-5000.
        };
        match k {
            9=> { // city rewards: workshop, stars, border, then giants
                score += match param {3=>300.,7=>260.,5=>if turn<45{250.}else{150.},8=>220.,6=>280.,2=>180.,4=>120.,1=>150.,_=>0.};
            },
            8=> {score += if owner.get(dst as usize)==Some(&0){300.}else{500.};},
            2=> {
                let au=units.get(&src).cloned().unwrap_or_default(); let du=units.get(&dst).cloned().unwrap_or_default();
                let (_,aa,_,acost,_,arange)=stats(au.kind); let (dmax,_,dd,dcost,_,drange)=stats(du.kind);
                let amax=stats(au.kind).0+if au.veteran{5.}else{0.};
                let mut bonus=1.;
                if terrain.get(dst as usize)==Some(&5)&&known.contains(&18){bonus=1.5;} // approximate only
                if enemy_cities.contains(&dst){bonus=1.5;}
                let af=aa*au.hp/amax.max(1.); let df=dd*du.hp/dmax.max(1.)*bonus;
                let damage=if af+df>0.{(af/(af+df)*aa*4.5+0.5).floor()}else{0.};
                let retaliation=if du.hp>damage&&drange>=d(size,src,dst)&&arange<=drange {(df/(af+df).max(0.01)*dd*4.5+0.5).floor()}else{0.};
                let kill=du.hp-damage<=0.; let death=au.hp-retaliation<=0.;
                score=500.+damage*22.+dcost*8.-retaliation*14.-acost*3.;
                if kill {score+=650.+dcost*20.;} if death {score-=if kill{900.+acost*25.}else{2400.+acost*25.};}
                if enemy_cities.contains(&dst){score+=250.;}
                if au.kind==10||au.kind==38 {score+=400.;}
                if au.hp<4.&&retaliation>0. {score-=500.;}
            },
            5=> {score=1000.; if turn>90{score=500.;}},
            7=> {
                let tv=tech_value(param,turn,water,&known,mem.enemy_seen);
                // Keep a small army rather than researching the whole tree before expanding.
                let affordability=if stars>=10||army<own_cities.len()+2{1.}else{0.75};
                score=if tv>0.{280.*affordability+if matches!(param,1|2|3|4){45.}else{0.}}else{-100.};
                if turn>70{score-=180.;}
            },
            4=> {
                let cost=stats(param).3;
                let mut value=match param {
                    2=>110.,3=>145.,4=>if turn>25{210.}else{100.},5=>135.,9=>150.,11=>190.,8=>175.,
                    42=>if water>0.2{180.}else{60.},44=>if water>0.2{170.}else{50.},41=>150.,_=>40.
                };
                if param==2&&turn>45{value-=45.;}
                if naval_kind(param)&&water<0.15{value-=100.;}
                // Produce at the most advanced/front-line available cities, not only the capital.
                if enemy_goal {let md=enemy_cities.iter().map(|&e|d(size,src,e)).min().unwrap_or(20);value+=(20-md)as f64*3.;}
                score=430.+value-army as f64*2.-cost;
                let cats=friendlies.iter().filter(|&&t|units.get(&t).map(|u|u.kind)==Some(8)).count();
                if param==8&&cats>=own_cities.len(){score-=400.;}
                if param==9{score-=900.;}
                if turn<8{score+=35.;}
                if army>=own_cities.len()*5+5{score-=150.;}
            },
            6=> {
                let mut value=match param {
                    8=>if water>0.12{250.}else{80.}, 12=>170.,5=>190.,21=>190.,6=>210.,13=>210.,22=>220.,
                    3=>-250.,48=>85.,50=>150.,17|18|19|20=>40.,23|24|25|26|27|28|29=>170.,_=>20.
                };
                if param==8 {let near_unknown=unknown.iter().filter(|&&u|d(size,dst,u)<=3).count();value+=near_unknown as f64*3.;}
                if turn>65&&matches!(param,3|48|17|18|19|20){value-=50.;}
                score=310.+value;
                if army<own_cities.len()+1&&turn>3{score-=120.;}
            },
            3=> {
                score=match param {
                    16=>1500., 19=>if dst==1{-500.}else{200.}, 7=>0.,9=>300.,1|2|3=>50.,4=>-3000.,8=>-1e8,
                    18|20|21|22=>-2000.,_=>0.
                };
                if param==7 {if let Some(u)=units.get(&src){let max=stats(u.kind).0+if u.veteran{5.}else{0.};
                    let danger=enemies.iter().map(|&e|d(size,src,e)).min().unwrap_or(20);
                    score=180.+(max-u.hp)*25.+if danger>2{150.}else{-100.};
                    if u.hp>=max-1.{score=-200.;}
                }}
            },
            1=> {
                let u=units.get(&src).cloned().unwrap_or_default();
                let explore_src=unknown.iter().filter(|&&x|d(size,src,x)<=1).count() as i32;
                let explore_dst=unknown.iter().filter(|&&x|d(size,dst,x)<=1).count() as i32;
                let mut value=-35.0;
                let mut best_improve=0;
                let mut objective=false;
                // Nearby attacks on our cities outrank every strategic objective.
                let mut threat=false;
                for &e in &enemies {for &c in &own_cities {if d(size,e,c)<=3 {
                    best_improve=best_improve.max(d(size,src,e)-d(size,dst,e)); threat=true; objective=true;
                }}}
                if threat { value += best_improve as f64*48.; }
                let passable_goal=|e:i32| terrain.get(e as usize).map(|&t|t!=4||known.contains(&20)).unwrap_or(false);
                let mut enemy_value=-100000.0;
                if !enemy_cities.is_empty() {
                    let imp=enemy_cities.iter().map(|&e|d(size,src,e)-d(size,dst,e)).max().unwrap_or(0);
                    enemy_value=180.+imp as f64*38.+if enemy_cities.contains(&dst){500.}else{0.}; objective=true;
                } else if mem.enemy_seen&&!enemies.is_empty() {
                    let imp=enemies.iter().map(|&e|d(size,src,e)-d(size,dst,e)).max().unwrap_or(0);
                    enemy_value=imp as f64*30.; objective=true;
                }
                let imp=neutral_cities.iter().chain(ruins.iter()).filter(|&&e|passable_goal(e)).map(|&e|d(size,src,e)-d(size,dst,e)).max().unwrap_or(-20);
                let neutral_value=imp as f64*42.+if neutral_cities.contains(&dst)||ruins.contains(&dst){400.}else{0.};
                if neutral_cities.iter().chain(ruins.iter()).any(|&e|passable_goal(e)){objective=true;}
                value+=enemy_value.max(neutral_value);
                let exploration=(explore_dst-explore_src)as f64*if objective{2.}else{14.}+explore_dst as f64*if objective{0.}else{2.};
                value+=exploration;
                if !objective {
                    let imp=unknown.iter().map(|&e|d(size,src,e)-d(size,dst,e)).max().unwrap_or(0);
                    value+=imp as f64*18.;
                }
                if u.hp<5. {let near=enemies.iter().map(|&e|d(size,dst,e)).min().unwrap_or(20);value+=(near-2)as f64*25.;}
                if matches!(u.kind,8|9|41|42) {let near=enemies.iter().map(|&e|d(size,dst,e)).min().unwrap_or(20);if near<2{value-=180.;}}
                let near_src=enemies.iter().map(|&e|d(size,src,e)).min().unwrap_or(20);
                let near_dst=enemies.iter().map(|&e|d(size,dst,e)).min().unwrap_or(20);
                if (u.kind==3&&u.attacked)||(u.hp<5.&&near_src<=2) {
                    value=350.+(near_dst-near_src)as f64*70.+explore_dst as f64;
                }
                score=value;
            },
            _=>{}
        }
        // Small amount of deterministic variation prevents pathological exact ties without compromising priorities.
        score += ((a.id.parse::<i64>().unwrap_or(0)%17)as f64)*1e-5;
        offer(score,a);
    }
    best.map(|x|x.1).unwrap_or_else(||legal[0].id.clone())
}

fn main() {
    let stdin=io::stdin(); let mut stdout=io::BufWriter::new(io::stdout()); let mut mem=Memory{enemy_seen:false,last_turn:-1,starting_capital:-1};
    for line in stdin.lock().lines() {
        let obs:Value=match serde_json::from_str(&line.unwrap()){Ok(x)=>x,Err(_)=>continue};
        let id=choose(&obs,&mut mem);
        writeln!(stdout,"{}",json!({"action_id":id})).unwrap(); stdout.flush().unwrap();
    }
}
