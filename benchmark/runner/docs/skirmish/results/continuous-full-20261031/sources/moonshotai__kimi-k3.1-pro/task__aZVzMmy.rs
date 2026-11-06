// Heuristic Bardur commander.  Every choice is made from the current legal catalog.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::collections::{VecDeque,HashMap};

#[derive(Default)]
struct Bot { intents: HashMap<usize,usize> }

fn i(v: &Value, k: &str) -> i64 { v[k].as_i64().unwrap_or(-1) }
fn f(v: &Value, k: &str) -> f64 { v[k].as_f64().unwrap_or(0.0) }
fn d(size: usize, a: usize, b: usize) -> i64 {
    let (ax, ay) = ((a % size) as i64, (a / size) as i64);
    let (bx, by) = ((b % size) as i64, (b / size) as i64);
    (ax-bx).abs().max((ay-by).abs())
}
fn stats(k: i64) -> (f64,f64,f64,f64,i64,i64) { // hp, attack, defense, cost, move, range
    match k {
        2=>(10.,2.,2.,2.,1,1), 3=>(10.,2.,1.,3.,2,1), 4=>(10.,3.5,1.,8.,3,1),
        5=>(15.,1.,3.,3.,1,1), 8=>(10.,4.,0.,8.,1,3), 9=>(10.,2.,1.,3.,1,2),
        10=>(10.,0.,1.,5.,1,1), 11=>(15.,3.,3.,5.,1,1), 12=>(40.,5.,4.,10.,1,1),
        37=>(10.,2.,2.,2.,1,1), 38=>(5.,0.,0.5,8.,2,1), 39=>(5.,0.,0.5,2.,2,1),
        40=>(10.,2.,2.,2.,2,1), 41=>(10.,3.,2.,15.,2,3), 42=>(10.,2.,1.,5.,3,2),
        43=>(10.,0.,1.,0.,2,2), 44=>(10.,3.,3.,5.,3,1), 45=>(40.,4.,4.,10.,2,1),
        _=>(10.,2.,1.,3.,1,1)
    }
}
fn boat(k:i64)->bool { matches!(k,39|40|41|42|43|44|45) }
fn stiff(k:i64)->bool { matches!(k,8|10|38|39|41|43|45) }
fn adjacent_enemies(size:usize, t:usize, p:i64, units:&[Value])->usize {
    units.iter().filter(|u| i(u,"owner")!=p && d(size,t,i(u,"tile") as usize)<=1).count()
}

// A conservative land-connectivity test: unknown tiles may be crossed, water may not.
fn land_route(size:usize, src:usize, dst:usize, tiles:&[Value], climbing:bool)->bool {
    let n=size*size; let mut seen=vec![false;n]; let mut q=VecDeque::new(); q.push_back(src); seen[src]=true;
    while let Some(a)=q.pop_front() {
        if a==dst { return true; }
        let (x,y)=(a%size,a/size);
        for (dx,dy) in [(-1i64,0i64),(1,0),(0,-1),(0,1)] {
            let (nx,ny)=(x as i64+dx,y as i64+dy);
            if nx<0||ny<0||nx>=size as i64||ny>=size as i64 {continue}
            let b=ny as usize*size+nx as usize; if seen[b] {continue}
            let t=&tiles[b]; let known=t["known"].as_bool().unwrap_or(false);
            let ter=i(t,"terrain");
            if known && (ter<=2 || (ter==4&&!climbing)) {continue}
            seen[b]=true; q.push_back(b);
        }
    }
    false
}

fn choose(bot:&mut Bot,obs:&Value)->String {
    let legal=obs["legal_actions"].as_array().unwrap();
    if legal.len()==1 { return legal[0]["action_id"].as_str().unwrap().to_owned(); }
    let v=&obs["visible_state"]; let p=obs["player"].as_i64().unwrap_or(1);
    let size=v["board_size"].as_u64().unwrap_or(14) as usize; let n=size*size;
    let tiles=v["tiles"].as_array().unwrap(); let units=v["units"].as_array().unwrap();
    let cities=v["cities"].as_array().unwrap(); let tech=v["technology"].as_array().unwrap();
    let has=|k:i64| tech.iter().any(|x|x.as_i64()==Some(k));
    let stars=i(v,"stars"); let turn=i(v,"turn");
    let own:Vec<&Value>=units.iter().filter(|u|i(u,"owner")==p).collect();
    let enemy:Vec<&Value>=units.iter().filter(|u|i(u,"owner")!=p).collect();
    let own_cities:Vec<&Value>=cities.iter().filter(|c|i(c,"owner")==p).collect();
    let objectives:Vec<&Value>=cities.iter().filter(|c|i(c,"owner")!=p).collect();
    let known=tiles.iter().filter(|t|t["known"].as_bool().unwrap_or(false)).count();
    let water=tiles.iter().filter(|t|t["known"].as_bool().unwrap_or(false)&&i(t,"terrain")<=2).count();
    let mountains=tiles.iter().filter(|t|t["known"].as_bool().unwrap_or(false)&&i(t,"terrain")==4).count();
    let watery=water*5>known.max(1) || (known<40&&water>2);
    let army=own.len(); let city_count=own_cities.len();
    let unit_at=|t:usize| units.iter().find(|u|i(u,"tile") as usize==t);
    let threatened=|c:usize| enemy.iter().any(|u|d(size,c,i(u,"tile") as usize)<=4);
    let any_threat=own_cities.iter().any(|c|threatened(i(c,"tile") as usize));
    let enemy_cap_known=objectives.iter().any(|c|i(c,"owner")>0);

    let mut best=&legal[0]; let mut best_score=-1e100f64; let mut best_intent=None;
    for a in legal {
        let c=&a["command"]; let k=i(c,"kind"); let src=i(c,"source"); let dst=i(c,"target"); let par=i(c,"parameter");
        let mut s=0f64; let mut current_intent=None;
        match k {
            9=> { // city rewards
                s=5000.;
                if par==4 { s+=if known<n*4/5 {400.}else{20.}; }
                if par==3 { s+=180.; }
                if par==7 { s+=300.; }
                if par==1 { s+=if threatened(dst as usize){450.}else{0.}; }
                if par==8 { s+=250.; }
                if par==5 { s+=80.; }
                if par==6 { s+=1000.; }
                if par==2 { s+=50.; }
            },
            8=> { // capture city, ruin, or starfish
                s=4000.;
                if tiles[src as usize]["structure"].as_i64()==Some(1) {s+=1000.;}
            },
            2=> { // combat, with exact basic damage model and cautious retaliation estimate
                let su=unit_at(src as usize); let tu=unit_at(dst as usize);
                if let (Some(u),Some(e))=(su,tu) {
                    let (uk,ek)=(i(u,"kind"),i(e,"kind")); let us=stats(uk); let es=stats(ek);
                    let ah=f(u,"health"); let eh=f(e,"health");
                    let am=us.0+if u["veteran"].as_bool().unwrap_or(false){5.}else{0.};
                    let em=es.0+if e["veteran"].as_bool().unwrap_or(false){5.}else{0.};
                    let mut bonus=1.;
                    if i(&tiles[dst as usize],"owner")==i(e,"owner") && cities.iter().any(|c|i(c,"tile")==dst) {bonus=1.5;}
                    let af=us.1*ah/am.max(1.); let df=es.2*eh/em.max(1.)*bonus;
                    let damage=((af/(af+df).max(0.0001)*us.1*4.5)+0.500001).floor().max(0.);
                    let retaliation=if eh>damage && d(size,src as usize,dst as usize)<=es.5 && !stiff(ek) {
                        ((df/(af+df).max(0.0001)*es.2*4.5)+0.500001).floor().max(0.)
                    } else {0.};
                    let lethal=damage>=eh; let dies=retaliation>=ah;
                    s=800.+damage*28.-retaliation*32.+es.3*8.;
                    if lethal {s+=700.+es.3*35.;}
                    if dies&&!lethal {s-=900.;}
                    if cities.iter().any(|c|i(c,"tile")==dst&&i(c,"owner")!=p) {s+=180.;}
                    if uk==9||uk==42||uk==8 {s+=100.;}
                } else {s=600.;}
            },
            1=> {
                let u=unit_at(src as usize); s=-200.;
                if let Some(u)=u {
                    let kind=i(u,"kind"); let su=src as usize; let du=dst as usize;
                    let on_threatened_city=cities.iter().any(|c|i(c,"tile")==src&&i(c,"owner")==p&&threatened(src as usize));
                    if on_threatened_city {s-=3000.;}
                    // Respond to invasions before exploration.
                    let mut defensive:Option<usize>=None; let mut dd=999;
                    for c in &own_cities {let ct=i(c,"tile") as usize;if threatened(ct)&&d(size,su,ct)<dd {dd=d(size,su,ct);defensive=Some(ct);}}
                    if let Some(ct)=defensive {
                        if dd<=5 {s+=250.+(d(size,su,ct)-d(size,du,ct)) as f64*160.;}
                    }
                    // Select the most useful current strategic target. Sticky waypoints prevent exploration oscillation.
                    let remembered=bot.intents.get(&su).copied().filter(|x| {
                        if *x==su || *x>=n {return false;}
                        if !tiles[*x]["known"].as_bool().unwrap_or(false) {return true;}
                        let ter=i(&tiles[*x],"terrain");
                        ter>2 && (ter!=4||has(20)) && (i(&tiles[*x],"structure")==2 || cities.iter().any(|c|i(c,"tile")==*x as i64&&i(c,"owner")!=p))
                    });
                    let mut targets:Vec<(f64,usize)>=Vec::new();
                    for c in &objectives {
                        let t=i(c,"tile") as usize; let enemy_city=i(c,"owner")>0;
                        let mut w=1.0;
                        if enemy_city && !enemy_cap_known {w=1.2;}
                        if enemy_city && (army<city_count*4 || turn<8) {w=4.0;}
                        targets.push((d(size,su,t) as f64*w,t));
                    }
                    for t in tiles {
                        if t["known"].as_bool().unwrap_or(false)&&i(t,"structure")==2 && i(t,"owner")!=p && i(t,"terrain")>2 && (i(t,"terrain")!=4||has(20)) {
                            let x=i(t,"tile") as usize;targets.push((d(size,su,x) as f64*1.15,x));
                        }
                    }
                    if targets.is_empty() || (targets.iter().map(|x|x.0).fold(999.,f64::min)>5.0) {
                        for t in tiles.iter().filter(|t|!t["known"].as_bool().unwrap_or(false)) {
                            let x=i(t,"tile") as usize;let dd=d(size,su,x);let aim=d(size,x,(size/2)*(size+1)) as f64+(6-dd).max(0) as f64*2.0;targets.push((aim*1.35,x));
                        }
                    }
                    targets.sort_by(|a,b|a.0.partial_cmp(&b.0).unwrap());
                    let selected=remembered.or_else(||targets.first().map(|x|x.1));
                    if let Some(target)=selected {
                        let mut actual=target;
                        // Once a channel is known, land armies embark rather than waiting on the beach.
                        if !boat(kind)&&cities.iter().any(|c|i(c,"tile")==target as i64&&i(c,"owner")>0)
                            && !land_route(size,su,target,tiles,has(20)) {
                            let mut ports:Vec<usize>=tiles.iter().filter(|t|t["known"].as_bool().unwrap_or(false)&&i(t,"structure")==8&&i(t,"owner")==p)
                                .map(|t|i(t,"tile") as usize).collect();
                            ports.sort_by_key(|x|d(size,su,*x)*2+d(size,*x,target));
                            if let Some(port)=ports.first(){actual=*port;}
                        }
                        let before=d(size,su,actual); let after=d(size,du,actual);
                        current_intent=Some((su,du,actual));
                        s+=180.+(before-after) as f64*150.;
                        if du==actual {s+=250.;}
                        if after<before {s+=100.;}
                    }
                    // Newly revealed territory and ruins are worth a small detour.
                    if !tiles[du]["known"].as_bool().unwrap_or(false) {s+=120.;}
                    if i(&tiles[du],"structure")==2 {s+=500.;}
                    if cities.iter().any(|c|i(c,"tile")==dst&&i(c,"owner")!=p) {s+=700.;}
                    let danger=adjacent_enemies(size,du,p,units);
                    if danger>0 {s-=danger as f64*if army<4{100.}else{35.};}
                    // Do not shuffle exhausted-looking armies backwards without progress.
                    if d(size,su,du)==0 {s-=1000.;}
                }
            },
            5=> {s=1100.;}, // harvest is the cheapest permanent growth
            7=> {
                s=match par {
                    1=>950., 6=>850., 16=>650., 18=>610.,
                    10=>if watery{1000.}else{300.},
                    13=>if watery{1050.}else{220.},
                    8=>620., 20=>if mountains*4>known{650.}else{430.},
                    23=>500.,24=>560., 4=>300.,2=>250.,3=>520.,
                    7=>820.,9=>300.,44=>if watery{430.}else{100.},
                    5=>200.,17=>350.,14=>if watery{300.}else{100.},
                    22=>180.,21=>180.,19=>160.,12=>if watery{200.}else{80.},
                    _=>100.
                };
                // Early expansion tech is valuable; once war starts, troop quality matters more.
                if turn>15 && matches!(par,3|24) {s+=100.;}
            },
            6=> {
                s=match par {
                    5|21=>720., 12=>650., 8=>if watery{900.}else{400.},
                    6|13|22=>620., 23|24|25|26|27|28|29=>800.,
                    17|18|19|20=>220., 3=>if turn>10{180.}else{30.},
                    48=>250., 50=>500., _=>150.
                };
            },
            4=> {
                s=match par {
                    4=>540.,11=>570.,12=>600.,5=>if any_threat{560.}else{260.},
                    3=>470.,9=>460.,2=>400.,8=>350.,42=>420.,44=>470.,41=>400.,_=>200.
                };
                if any_threat {s+=500.;if par==3{s-=260.;}if par==2{s+=260.;}if par==5{s+=500.;}}
                if army<city_count*4 {s+=650.;}
                if army>=city_count*4+5 && !any_threat {s-=220.;}
                // Prefer front-line production, but all legal cities are useful.
                if enemy_cap_known && objectives.iter().any(|c|d(size,src as usize,i(c,"tile") as usize)<=5) {s+=80.;}
            },
            3=> {
                if par==16 {s=1800.;}
                else if par==7 {
                    if let Some(u)=unit_at(src as usize) {
                        let hp=f(u,"health"); let max=stats(i(u,"kind")).0+if u["veteran"].as_bool().unwrap_or(false){5.}else{0.};
                        s=if hp<max*0.45{650.}else if hp<max*0.75{250.}else{50.};
                    } else {s=100.;}
                } else if par==2 {s=40.;}
                else if par==1||par==3 {s=-100.;}
                else {s=-1000.;}
            },
            10=>s=-500., 11=>s=-100000., _=>s=-500.
        }
        // Stable but nontrivial tie break; action IDs are legal-catalog order.
        let tie=a["action_id"].as_str().unwrap_or("").parse::<i64>().unwrap_or(0) as f64*1e-8;
        if s+tie>best_score {best_score=s+tie;best=a;best_intent=current_intent;}
    }
    if let Some((src,dst,target))=best_intent {bot.intents.remove(&src);bot.intents.insert(dst,target);}
    best["action_id"].as_str().unwrap().to_owned()
}

fn main(){
    let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout());let mut bot=Bot::default();
    for line in stdin.lock().lines(){
        let obs:Value=serde_json::from_str(&line.unwrap()).unwrap();
        writeln!(stdout,"{}",json!({"action_id":choose(&mut bot,&obs)})).unwrap();stdout.flush().unwrap();
    }
}
