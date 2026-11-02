use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy)]
struct Cmd { id: usize, kind: i64, src: i64, dst: i64, par: i64 }

struct Bot {
    size: i64,
    cities: HashMap<i64, i64>,       // last known owner, including neutral villages
    ruins: HashSet<i64>,
    terrain: HashMap<i64, i64>,
    previous: HashMap<i64, i64>,     // unit's current tile -> tile it came from
    last_turn: i64,
}
impl Bot {
    fn new() -> Self { Self { size:14, cities:HashMap::new(), ruins:HashSet::new(), terrain:HashMap::new(), previous:HashMap::new(), last_turn:-1 } }
    fn learn(&mut self, obs:&Value) {
        let v=&obs["visible_state"];
        self.size=v["board_size"].as_i64().unwrap_or(14);
        let turn=v["turn"].as_i64().unwrap_or(0);
        if turn < self.last_turn { self.cities.clear(); self.ruins.clear(); self.terrain.clear(); self.previous.clear(); }
        self.last_turn=turn;
        for t in v["tiles"].as_array().into_iter().flatten() {
            if t["known"].as_bool().unwrap_or(false) {
                let i=t["tile"].as_i64().unwrap();
                self.terrain.insert(i,t["terrain"].as_i64().unwrap_or(3));
                let st=t["structure"].as_i64().unwrap_or(0);
                if st==1 { self.cities.insert(i,t["owner"].as_i64().unwrap_or(0)); self.ruins.remove(&i); }
                else if st==2 { self.ruins.insert(i); } else { self.ruins.remove(&i); }
            }
        }
        // Cities are the most reliable memory of ownership changes.
        for c in v["cities"].as_array().into_iter().flatten() {
            self.cities.insert(c["tile"].as_i64().unwrap(),c["owner"].as_i64().unwrap_or(0));
        }
    }
}
fn d(a:i64,b:i64,s:i64)->i64 { ((a%s-b%s).abs()).max((a/s-b/s).abs()) }
fn num(v:&Value,k:&str)->i64 { v[k].as_i64().unwrap_or(-1) }
fn cmd(a:&Value,ix:usize)->Cmd { let c=&a["command"]; Cmd{id:ix,kind:num(c,"kind"),src:num(c,"source"),dst:num(c,"target"),par:num(c,"parameter")} }
fn stat(k:i64)->(f64,f64,f64,f64,i64) { // hp, attack, defense, movement, range
    match k { 2=>(10.,2.,2.,1.,1),3=>(10.,2.,1.,2.,1),4=>(10.,3.5,1.,3.,1),5=>(15.,1.,3.,1.,1),
        8=>(10.,4.,0.,1.,3),9=>(10.,2.,1.,1.,2),10=>(10.,0.,1.,1.,1),11=>(15.,3.,3.,1.,1),12=>(40.,5.,4.,1.,1),
        37=>(10.,2.,2.,1.,1),38=>(5.,0.,0.5,2.,1),39=>(5.,0.,0.5,2.,1),40=>(10.,2.,2.,2.,1),41=>(10.,3.,2.,2.,3),
        42=>(10.,2.,1.,3.,2),43=>(10.,0.,1.,2.,2),44=>(10.,3.,3.,3.,1),45=>(40.,4.,4.,2.,1),_=>(10.,2.,1.,1.,1) }
}
fn maxhp(u:&Value)->f64 { let (h,_,_,_,_)=stat(num(u,"kind")); h+if u["veteran"].as_bool().unwrap_or(false){5.}else{0.} }
fn choose(bot:&mut Bot,obs:&Value)->String {
    bot.learn(obs);
    let legal=obs["legal_actions"].as_array().unwrap();
    let v=&obs["visible_state"]; let p=obs["player"].as_i64().unwrap(); let s=bot.size; let n=s*s;
    let turn=v["turn"].as_i64().unwrap_or(0); let stars=v["stars"].as_i64().unwrap_or(0);
    let tiles=v["tiles"].as_array().unwrap(); let units=v["units"].as_array().unwrap(); let cities=v["cities"].as_array().unwrap();
    let tech:HashSet<i64>=v["technology"].as_array().into_iter().flatten().filter_map(|x|x.as_i64()).collect();
    let um:HashMap<i64,&Value>=units.iter().map(|u|(num(u,"tile"),u)).collect();
    let cm:HashMap<i64,&Value>=cities.iter().map(|c|(num(c,"tile"),c)).collect();
    let own_units:Vec<&Value>=units.iter().filter(|u|num(u,"owner")==p).collect();
    let own_cities:Vec<&Value>=cities.iter().filter(|c|num(c,"owner")==p).collect();
    let enemy_cities:Vec<i64>=bot.cities.iter().filter(|(_,o)|**o>0&&**o!=p).map(|(t,_)|*t).collect();
    let neutral_cities:Vec<i64>=bot.cities.iter().filter(|(_,o)|**o==0).map(|(t,_)|*t).collect();
    let unknown:Vec<i64>=tiles.iter().filter(|t|!t["known"].as_bool().unwrap_or(false)).map(|t|num(t,"tile")).collect();
    let known_water=bot.terrain.values().filter(|t|**t<=2).count() as f64;
    let water_ratio=known_water/(bot.terrain.len().max(1) as f64);
    let threatened:HashSet<i64>=own_cities.iter().filter(|c|units.iter().any(|u|num(u,"owner")!=p&&d(num(u,"tile"),num(c,"tile"),s)<=2)).map(|c|num(c,"tile")).collect();
    let nearest=|from:i64, targets:&Vec<i64>|->i64 {targets.iter().map(|t|d(from,*t,s)).min().unwrap_or(20)};
    let mut best:Option<(f64,Cmd)>=None;
    for (ix,a) in legal.iter().enumerate() {
        let c=cmd(a,ix); let k=c.kind; let mut score=0.;
        match k {
            11=>score=-1e9,
            10=>score=-1000.,
            9=>{ // city rewards
                score=10000.;
                score += match c.par {
                    4=>if own_cities.len()==1 && turn<8 && unknown.len()>30 {220.}else{20.},
                    3=>200.,
                    7=>if stars<8 {100.}else{20.},
                    1=>if threatened.contains(&c.dst){180.}else{25.},
                    8=>120.,5=>65.,6=>150.,2=>10.,_=>0.};
            },
            8=>{score=9000.; if let Some(city)=cm.get(&c.src){if num(city,"owner")>0{score+=1000.;}} if bot.ruins.contains(&c.src){score+=150.;}},
            2=>{ // combat: favor kills and safe/ranged trades
                if let (Some(att),Some(def))=(um.get(&c.src),um.get(&c.dst)) {
                    let ak=num(att,"kind"); let dk=num(def,"kind"); let (_,aa,_,_,range)=stat(ak); let (_,_,dd,_,_)=stat(dk);
                    let ah=att["health"].as_f64().unwrap_or(10.); let dh=def["health"].as_f64().unwrap_or(10.);
                    let af=aa*ah/maxhp(att); let db=if num(def,"owner")==p{1.}else{1.};
                    let df=dd*dh/maxhp(def)*db;
                    let damage=((af/(af+df).max(0.001)*aa*4.5)+0.5).floor();
                    let ret=((df/(af+df).max(0.001)*dd*4.5)+0.5).floor();
                    let kill=damage>=dh; let dies=!kill&&range<=1&&ret>=ah;
                    let urgent=own_cities.iter().any(|ct|num(ct,"tile")==c.dst);
                    let favorable=kill||range>1||damage>ret+0.5;
                    score=4200.+damage*32.-dh*3.+if kill{700.}else{0.}-if dies{700.}else{0.};
                    if !favorable && !urgent { score-=500.; }
                    if range>1{score+=180.;}
                    if urgent{score+=500.;}
                    if enemy_cities.contains(&c.dst){score+=200.;}
                    if dk==12||dk==45{score+=100.;}
                } else {score=3500.;}
            },
            3=>{
                if c.par==16 {score=6000.;}
                else if c.par==19 {score=if c.dst==1{-5000.}else{100.};}
                else if c.par==21||c.par==18||c.par==20||c.par==22 {score=-2000.;}
                else if c.par==7 {score=850.;}
                else if c.par==2 {score=120.;}
                else {score=-1000.;}
            },
            5=>{score=2500.; if turn<15{score+=200.;}},
            7=>{
                // Technology plan: mobility/economy, then water and military branches.
                let pref=match c.par {
                    1=>if turn<25{2350.}else{1800.},
                    6=>2300.,8=>2200.,18=>2050.,16=>300.,
                    10=>if water_ratio>0.15{2150.}else{-1000.},
                    13=>if tech.contains(&10)&&water_ratio>0.15{2000.}else{-1000.},
                    44=>1550.,20=>300.,23=>1400.,24=>1900.,2=>1000.,3=>1800.,
                    4=>-500.,7=>300.,9=>1800.,17=>300.,_=>-1000.};
                score=pref;
                // Do not tech ourselves out of all army production in the opening.
                if own_units.len()==0&&turn>2{score-=300.;}
            },
            6=>{
                score=match c.par {5=>2300.,21=>2250.,12=>2200.,8=>if water_ratio>0.08{2250.}else{700.},
                    6=>2250.,13=>2200.,22=>2250.,3=>350.,48=>650.,50=>1000.,_=>50.};
            },
            4=>{
                let val=match c.par {2=>70.,3=>115.,4=>230.,5=>105.,8=>140.,9=>125.,11=>180.,42=>130.,44=>170.,_=>20.};
                score=2400.+val;
                if own_units.len()<own_cities.len()+1{score+=250.;}
                if threatened.contains(&c.src){score+=500.; if matches!(c.par,2|5|11){score+=250.;} if c.par==3{score-=100.;}}
                // Recruit near the front rather than in a remote safe capital.
                if !enemy_cities.is_empty(){score+=(20-nearest(c.src,&enemy_cities)).max(0) as f64*8.;}
                if c.par==4&&stars<12{score-=150.;}
            },
            1=>{
                score=1000.;
                let u=um.get(&c.src); let kind=u.map(|u|num(u,"kind")).unwrap_or(2);
                let before=c.src; let after=c.dst;
                let hp=u.map(|u|u["health"].as_f64().unwrap_or(10.)).unwrap_or(10.);
                let foes:Vec<i64>=units.iter().filter(|x|num(x,"owner")!=p).map(|x|num(x,"tile")).collect();
                if hp<=5.0 && nearest(before,&foes)<=2 {
                    let escape=(nearest(after,&foes)-nearest(before,&foes)) as f64*180.;
                    let home=(nearest(before,&own_cities.iter().map(|x|num(x,"tile")).collect())-nearest(after,&own_cities.iter().map(|x|num(x,"tile")).collect())) as f64*35.;
                    score+=1200.+escape+home;
                }
                // Concrete expansion/conquest goals outrank generic fog exploration.
                let mut goals:Vec<(i64,f64)>=Vec::new();
                for t in &enemy_cities{goals.push((*t,if turn<15{18.}else{42.}));}
                for t in &neutral_cities{goals.push((*t,if turn<18{120.}else{60.}));}
                for t in &bot.ruins{goals.push((*t,65.));}
                let goal_gain=goals.iter().map(|(t,w)|(d(before,*t,s)-d(after,*t,s)) as f64*w).fold(0f64,f64::max);
                let fog_gain=(nearest(before,&unknown)-nearest(after,&unknown)) as f64*20.;
                score+=if goals.is_empty(){fog_gain}else{goal_gain};
                let newtiles=unknown.iter().filter(|t|d(after,**t,s)<=if kind==42||kind==39{2}else{1}).count();
                score+=newtiles as f64*16.;
                if bot.previous.get(&after)==Some(&before){score-=250.;}
                if bot.previous.get(&before)==Some(&after){score-=500.;}
                // Defend threatened cities; don't strand the last local unit beside an invader.
                if threatened.contains(&after){score+=250.;}
                if threatened.contains(&before)&&d(after,before,s)>1{score-=1200.;}
                for ct in &own_cities {
                    let home=num(ct,"tile");
                    if threatened.contains(&home) && d(before,home,s)<=2 && d(after,home,s)>d(before,home,s) {
                        let guards=own_units.iter().filter(|u|d(num(u,"tile"),home,s)<=2).count();
                        if guards<=2 { score-=if guards<=1{5000.}else{700.}; }
                    }
                }
                // Embarking is useful; boats should not beach unnecessarily.
                let src_water=bot.terrain.get(&before).map(|t|*t<=2).unwrap_or(false);
                let dst_water=bot.terrain.get(&after).map(|t|*t<=2).unwrap_or(false);
                if matches!(kind,40|42|43|44|45)&&src_water&&!dst_water{score-=150.;}
                if after<0||after>=n{score=-1e8;}
            },
            _=>score=0.
        }
        // Stable deterministic tie break, but prefer earlier legal IDs only microscopically.
        score-=ix as f64*1e-5;
        if best.as_ref().map(|b|score>b.0).unwrap_or(true){best=Some((score,c));}
    }
    let c=best.unwrap().1;
    if c.kind==1 { bot.previous.insert(c.dst,c.src); bot.previous.remove(&c.src); }
    legal[c.id]["action_id"].as_str().unwrap().to_owned()
}
fn main(){
    let stdin=io::stdin(); let mut stdout=io::BufWriter::new(io::stdout()); let mut bot=Bot::new();
    for line in stdin.lock().lines(){
        let obs:Value=serde_json::from_str(&line.unwrap()).unwrap();
        let id=choose(&mut bot,&obs);
        writeln!(stdout,"{}",json!({"action_id":id})).unwrap(); stdout.flush().unwrap();
    }
}
