use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Default)]
struct Seen { owner: i32, structure: i32, terrain: i32, turn: i32 }
struct Bot {
    size: usize,
    seen: Vec<Seen>,
    enemy_cities: HashSet<usize>,
    own_capital: Option<usize>,
    enemy_capital: Option<usize>,
}
impl Bot {
    fn new() -> Self { Self { size: 14, seen: vec![Seen::default(); 196], enemy_cities: HashSet::new(), own_capital: None, enemy_capital: None } }
    fn learn(&mut self, o: &Value) {
        let v=&o["visible_state"]; let p=i_at(o,"player"); let turn=i_at(v,"turn");
        self.size=i_at(v,"board_size").max(1) as usize;
        if self.seen.len()!=self.size*self.size { self.seen=vec![Seen::default();self.size*self.size]; }
        for t in arr(v,"tiles") {
            if !b_at(t,"known") { continue; }
            let x=i_at(t,"tile") as usize;
            self.seen[x]=Seen{owner:i_at(t,"owner"),structure:i_at(t,"structure"),terrain:i_at(t,"terrain"),turn};
            if i_at(t,"structure")==1 && i_at(t,"owner")>0 && i_at(t,"owner")!=p {
                self.enemy_cities.insert(x);
                if i_at(t,"capital")>0 { self.enemy_capital=Some(x); }
            } else if self.enemy_cities.contains(&x) && i_at(t,"owner")==p { self.enemy_cities.remove(&x); }
            if i_at(t,"capital")==p { self.own_capital=Some(x); }
        }
    }
    fn choose(&mut self,o:&Value)->String {
        self.learn(o);
        let v=&o["visible_state"]; let p=i_at(o,"player"); let n=self.size*self.size;
        let legal=arr(o,"legal_actions");
        let units=arr(v,"units"); let cities=arr(v,"cities"); let tiles=arr(v,"tiles");
        let stars=i_at(v,"stars"); let turn=i_at(v,"turn");
        let tech:HashSet<i32>=arr(v,"technology").iter().map(|x|x.as_i64().unwrap_or(0) as i32).collect();
        let mut um=HashMap::new(); let mut tm=HashMap::new(); let mut cm=HashMap::new();
        for u in units {um.insert(i_at(u,"tile") as usize,u);}
        for t in tiles {if b_at(t,"known"){tm.insert(i_at(t,"tile") as usize,t);}}
        for c in cities {cm.insert(i_at(c,"tile") as usize,c);}
        let own_units=units.iter().filter(|u|i_at(u,"owner")==p).count();
        let enemy_units:Vec<&Value>=units.iter().filter(|u|{let q=i_at(u,"owner");q>0&&q!=p}).collect();
        let own_cities:Vec<&Value>=cities.iter().filter(|c|i_at(c,"owner")==p).collect();
        let city_count=own_cities.len();
        let mut objectives:Vec<(usize,i32)>=Vec::new();
        for (x,t) in &tm {
            let st=i_at(t,"structure"); let ow=i_at(t,"owner");
            if st==1 && ow!=p {objectives.push((*x,if ow>0{1000}else{850}));}
            if st==2 {objectives.push((*x,650));}
        }
        for &x in &self.enemy_cities { if !objectives.iter().any(|q|q.0==x){objectives.push((x,1100));} }
        for u in &enemy_units {objectives.push((i_at(u,"tile") as usize,500));}
        // The exact capital remains a useful strategic heading after it disappears into fog.
        if let Some(x)=self.enemy_capital {if !objectives.iter().any(|q|q.0==x){objectives.push((x,1200));}}
        let land_target=objectives.iter().any(|(x,_)|tm.get(x).map(|t|i_at(t,"terrain")>2).unwrap_or(true));
        let water_seen=self.seen.iter().filter(|s|s.turn>0&&s.terrain<=2).count(); let known_seen=self.seen.iter().filter(|s|s.turn>0).count();
        let water_map = tech.contains(&10) || (water_seen>=2 && water_seen*8>=known_seen) || (!land_target && water_seen>0);
        let mut best:Option<(&Value,f64)>=None;
        for (ix,a) in legal.iter().enumerate() {
            let c=&a["command"]; let k=i_at(c,"kind"); let src=i_at(c,"source"); let dst=i_at(c,"target"); let par=i_at(c,"parameter");
            let mut sc=match k {
                11=>-1e9, 10=>-1000.0, 9=>self.reward_score(par,dst,&enemy_units,&tm,turn),
                8=>5000.0, 2=>self.attack_score(src,dst,&um,&tm,&cm,p),
                5=>650.0,
                7=>self.research_score(par,stars,turn,city_count,water_map,&tech)*0.45+if matches!(par,2|3){1400.0}else{0.0}+if turn==0{700.0}else{0.0},
                4=>self.summon_score(par,src,stars,turn,city_count,own_units,&objectives,&enemy_units),
                6=>self.build_score(par,dst,turn,city_count,water_map),
                1=>self.move_score(src,dst,&um,&tm,&objectives,&enemy_units,p,ix),
                3=>match par {16=>900.0,19=>if dst==0{50.0}else{-2000.0},21=>-3000.0,18=>-3000.0,7=>-50.0,9=>200.0,2=>30.0,8=>-5000.0,_=>-100.0},
                _=>0.0
            };
            // Tiny stable tie break; never enough to change a real preference.
            sc-=ix as f64*1e-7;
            if best.map(|b|sc>b.1).unwrap_or(true){best=Some((a,sc));}
        }
        best.unwrap_or((&legal[0],0.0)).0["action_id"].as_str().unwrap().to_owned()
    }
    fn reward_score(&self,par:i32,city:i32,enemies:&Vec<&Value>,tm:&HashMap<usize,&Value>,turn:i32)->f64 {
        match par {
            6=>6000.0,
            8=>4500.0,
            5=>if turn<35{3600.0}else{2500.0},
            7=>4200.0,
            1=>{let threatened=enemies.iter().any(|u|dist(i_at(u,"tile") as usize,city as usize,self.size)<=2);if threatened{5000.0}else{2500.0}},
            3=>4400.0,
            4=>{let known=tm.len();if turn<20&&known<self.size*self.size*3/4{4600.0}else{3500.0}},
            2=>2000.0,_=>1000.0}
    }
    fn research_score(&self,k:i32,stars:i32,turn:i32,cities:usize,water:bool,tech:&HashSet<i32>)->f64 {
        // Technology before expansion is cheap; core military and transport lines lead.
        let mut s=match k {
            1=>if turn<12{1050.0}else{700.0},
            10=>if water{1300.0}else{400.0},
            13=>if water{1400.0}else{100.0},
            44=>if water{1200.0}else{100.0},
            14=>if water{1250.0}else{100.0},
            16=>800.0,17=>1150.0,18=>850.0,
            4=>650.0,2=>700.0,3=>1350.0,
            6=>100.0,8=>700.0,9=>650.0,
            20=>550.0,23=>500.0,24=>1000.0,
            7=>550.0,5=>500.0,_=>150.0};
        if cities>=4 && turn<8 && (k==16||k==17||k==3||k==24){s-=500.0;}
        if tech.contains(&1)&&k==3{s+=150.0;}
        if stars<=5 && k!=10 && k!=1 {s-=100.0;}
        s
    }
    fn summon_score(&self,k:i32,src:i32,stars:i32,turn:i32,cities:usize,own:usize,objectives:&Vec<(usize,i32)>,enemies:&Vec<&Value>)->f64 {
        let base=match k {12=>1500.0,4=>1150.0,11=>1050.0,8=>1100.0,45=>1400.0,44=>1000.0,42=>900.0,41=>1000.0,9=>800.0,3=>750.0,5=>700.0,2=>620.0,10=>500.0,_=>100.0};
        let front=objectives.iter().map(|(x,_)|dist(src as usize,*x,self.size)).min().unwrap_or(10);
        let threatened=enemies.iter().any(|u|dist(i_at(u,"tile") as usize,src as usize,self.size)<=3);
        let mut s=base - front as f64*12.0;
        if threatened{s+=350.0;}
        if own>cities*5+3 && !threatened{s-=500.0;}
        if turn<8 && k==2{s+=80.0;}
        if stars<=3 && k==2{s+=50.0;}
        s
    }
    fn build_score(&self,k:i32,dst:i32,turn:i32,cities:usize,water:bool)->f64 {
        let mut s=match k {5=>780.0,21=>780.0,12=>620.0,8=>if water{1000.0}else{300.0},6=>700.0,13=>700.0,22=>750.0,50=>650.0,48=>if water{650.0}else{100.0},3=>200.0,17|18|19|20=>if turn>45{400.0}else{50.0},_=>100.0};
        if cities>3 && (k==5||k==21){s+=50.0;}
        if dst<0{s=-1000.0;} s
    }
    fn attack_score(&self,src:i32,dst:i32,um:&HashMap<usize,&Value>,tm:&HashMap<usize,&Value>,cm:&HashMap<usize,&Value>,p:i32)->f64 {
        let a=match um.get(&(src as usize)){Some(x)=>*x,None=>return -1000.0};
        let d=match um.get(&(dst as usize)){Some(x)=>*x,None=>return 3000.0}; // infiltration
        let ak=i_at(a,"kind"); let dk=i_at(d,"kind"); let ah=f_at(a,"health") as f32; let dh=f_at(d,"health") as f32;
        let (aa,am,_ar) = stats(ak); let (_da,dm,dr)=stats(dk); let dd=defense(dk);
        let bonus=if let Some(t)=tm.get(&(dst as usize)) {
            let ter=i_at(t,"terrain"); let city=i_at(t,"structure")==1;
            if city && matches!(dk,2|3|4|5|9){1.5}else if ter==4||ter==5{1.25}else{1.0}
        }else{1.0};
        let af=aa*ah/am; let df=dd*dh/dm*bonus; let total=(af+df).max(0.001);
        let damage=((af/total*aa*4.5)+0.500001).floor().max(0.0);
        let retaliation=if dh>damage && dist(src as usize,dst as usize,self.size)<=dr && !matches!(dk,8|10|43){((df/total*dd*4.5)+0.500001).floor().max(0.0)}else{0.0};
        let kill=dh<=damage; let dies=ah<=retaliation;
        let value=unit_value(dk); let own=unit_value(ak);
        let mut s=500.0+damage as f32*35.0-retaliation as f32*35.0;
        if kill{s+=1800.0+value as f32*20.0;}
        if retaliation==0.0{s+=300.0;}
        if dies{s-=if kill{own as f32*8.0}else{1800.0+own as f32*25.0};}
        if !kill && retaliation>damage*1.2{s-=900.0;}
        if i_at(d,"owner")==p{s-=10000.0;}
        s as f64
    }
    fn move_score(&self,src:i32,dst:i32,um:&HashMap<usize,&Value>,tm:&HashMap<usize,&Value>,objectives:&Vec<(usize,i32)>,enemies:&Vec<&Value>,p:i32,ix:usize)->f64 {
        let s=src as usize; let d=dst as usize; let u=match um.get(&s){Some(x)=>*x,None=>return 0.0};
        let kind=i_at(u,"kind");
        let mut best_gain=-100.0; let mut on_objective: f64=0.0;
        for &(t,weight) in objectives {
            let before=dist(s,t,self.size) as f64; let after=dist(d,t,self.size) as f64;
            let gain=(before-after)*35.0-after*0.7+weight as f64*0.006;
            if gain>best_gain{best_gain=gain;}
            if d==t{on_objective=on_objective.max(weight as f64*0.12);}
        }
        let mut score=100.0+best_gain+on_objective;
        if objectives.is_empty() {score=100.0-(dist(d,self.size*self.size/2,self.size) as f64);}
        // Explore locally even when the nearest strategic goal is far away.
        let mut unexplored=0; let mut known_around=0;
        for (x,t) in tm {if dist(d,*x,self.size)<=1{known_around+=1;}}
        for y in 0..self.size*self.size {if !tm.contains_key(&y)&&dist(d,y,self.size)<=1{unexplored+=1;}}
        score+=unexplored as f64*7.0-known_around as f64*0.2;
        // Avoid walking fragile units beside a stronger visible enemy without attacking it.
        let near=enemies.iter().filter(|e|dist(d,i_at(e,"tile") as usize,self.size)<=1).count();
        if near>0 { score-=if matches!(kind,8|10|42){300.0}else{170.0}*near as f64; }
        let hp=f_at(u,"health");
        if hp<=5.0 {
            let mut retreat: f64=-1000.0;
            for (x,t) in tm { if i_at(t,"structure")==1 && i_at(t,"owner")==p { retreat=retreat.max((dist(s,*x,self.size)-dist(d,*x,self.size)) as f64*70.0-dist(d,*x,self.size) as f64); }}
            if retreat>-1000.0 {score=score*0.25+retreat;}
        }
        if let Some(t)=tm.get(&d){if i_at(t,"structure")==1&&i_at(t,"owner")!=p{score+=150.0;} if i_at(t,"structure")==2{score+=100.0;}}
        score+(ix%7) as f64*0.01
    }
}
fn arr<'a>(v:&'a Value,k:&str)->&'a Vec<Value>{v[k].as_array().unwrap()}
fn i_at(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn f_at(v:&Value,k:&str)->f64{v[k].as_f64().unwrap_or(0.0)}
fn b_at(v:&Value,k:&str)->bool{v[k].as_bool().unwrap_or(false)}
fn dist(a:usize,b:usize,s:usize)->i32{let ax=(a%s)as i32;let ay=(a/s)as i32;let bx=(b%s)as i32;let by=(b/s)as i32;(ax-bx).abs().max((ay-by).abs())}
fn stats(k:i32)->(f32,f32,i32){match k{2=>(2.0,10.0,1),3=>(2.0,10.0,1),4=>(3.5,10.0,1),5=>(1.0,15.0,1),8=>(4.0,10.0,3),9=>(2.0,10.0,2),10=>(0.0,10.0,1),11=>(3.0,15.0,1),12=>(5.0,40.0,1),37=>(2.0,10.0,1),39=>(0.0,5.0,1),40=>(2.0,10.0,1),41=>(3.0,10.0,3),42=>(2.0,10.0,2),43=>(0.0,10.0,2),44=>(3.0,10.0,1),45=>(4.0,40.0,1),_=>(2.0,10.0,1)}}
fn defense(k:i32)->f32{match k{2|40=>2.0,5|11=>3.0,12|45=>4.0,8=>0.0,43=>1.0,44=>3.0,41=>2.0,_=>1.0}}
fn unit_value(k:i32)->i32{match k{12|45=>10,4|8=>8,11=>5,3|5|9=>3,41=>15,42|44=>5,2=>2,_=>3}}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut bot=Bot::new();for line in stdin.lock().lines(){let obs:Value=serde_json::from_str(&line.unwrap()).unwrap();let id=bot.choose(&obs);writeln!(out,"{}",json!({"action_id":id})).unwrap();out.flush().unwrap();}}
