use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn arr(v:&Value,k:&str)->Vec<Value>{v[k].as_array().cloned().unwrap_or_default()}
fn get_tile(tiles:&[Value],i:usize)->Value{tiles.get(i).cloned().unwrap_or(Value::Null)}
fn dist(a:usize,b:usize,s:usize)->i32{((a%s) as i32-(b%s) as i32).abs().max(((a/s) as i32-(b/s) as i32).abs())}
fn score_action(a:&Value, obs:&Value)->f64{
 let c=&a["command"]; let kind=num(c,"kind"); let src=num(c,"source").max(0) as usize; let dst=num(c,"target").max(0) as usize; let p=num(c,"parameter");
 let v=&obs["visible_state"]; let size=num(v,"board_size").max(1) as usize; let n=size*size; let me=num(obs,"player"); let turn=num(v,"turn"); let stars=num(v,"stars");
 let tiles=arr(v,"tiles"); let units=arr(v,"units"); let cities=arr(v,"cities"); let tech:Vec<i32>=arr(v,"technology").iter().filter_map(|x|x.as_i64().map(|x|x as i32)).collect();
 let has=|t:i32|tech.contains(&t);
 let tile=|i:usize|->Value{get_tile(&tiles,i)};
 let ter=|i:usize|num(&tile(i),"terrain"); let struc=|i:usize|num(&tile(i),"structure"); let res=|i:usize|num(&tile(i),"resource"); let owner=|i:usize|num(&tile(i),"owner");
 let owncities:Vec<usize>=cities.iter().filter(|x|num(x,"owner")==me).map(|x|num(x,"tile").max(0) as usize).collect();
 let enemycities:Vec<usize>=cities.iter().filter(|x|num(x,"owner")!=0&&num(x,"owner")!=me).map(|x|num(x,"tile").max(0) as usize).collect();
 let villages:Vec<usize>=tiles.iter().filter(|x|x["known"]==true&&num(x,"structure")==1&&num(x,"owner")!=me).map(|x|num(x,"tile").max(0) as usize).collect();
 let danger=|i:usize|->f64{ units.iter().filter(|u|num(u,"owner")!=me&&dist(i,num(u,"tile").max(0) as usize,size)<=1).map(|u| if num(u,"kind")==8||num(u,"kind")==9 {1.5}else{1.0}).sum()};
 match kind {
  9 => { // City reward
    match p {3=> if turn<30 {105.0}else{65.0},4=> if turn<12 {if tiles.iter().filter(|t|t["known"]==false).count()>100 {75.0}else{48.0}}else{25.0},7=> if turn<50 {104.0}else{52.0},8=> if turn<45 {90.0}else{52.0},5=> if villages.len()>0 {65.0}else{20.0},6=> 65.0,1=>20.0,2=>15.0,_=>0.0}
  }
  8 => { if struc(src)==2 {130.0} else if owner(src)!=0 {1200.0} else {190.0} }
  2 => { // attack estimate damage and kill based public unit definitions
    let attacker=units.iter().find(|u|num(u,"tile")==src as i32).cloned().unwrap_or(Value::Null);
    let defender=units.iter().find(|u|num(u,"tile")==dst as i32).cloned().unwrap_or(Value::Null);
    let ak=num(&attacker,"kind"); let dk=num(&defender,"kind");
    let (aa,ad,am)=stats(ak); let (da,dd,dm)=stats(dk);
    let ah=attacker["health"].as_f64().unwrap_or(10.); let dh=defender["health"].as_f64().unwrap_or(10.);
    let defbonus=if num(&tile(dst),"structure")==1 && owner(dst)==num(&defender,"owner") {1.5} else if ter(dst)==4||ter(dst)==5 {1.5}else{1.};
    let af=aa*ah/am; let df=dd*dh/dm*defbonus;
    let dmg=if aa<=0. {0.} else {((af/(af+df)*aa*4.5)+0.5).floor()};
    let kill=dmg>=dh; let retaliation=if !kill && dist(src,dst,size)<=1 { (df/(af+df)*dd*4.5+0.5).floor() } else {0.};
    let risk=if ah<=retaliation {160.} else if ah-retaliation<4. {45.} else {0.};
    50.+dmg*4.0+if kill {80.}else{0.}-risk + if enemycities.contains(&dst){35.}else{0.} + if (dk==8||dk==9)&&ak==2 {5.}else{0.}
  }
  1 => { // movement; approach objectives / explore, avoid danger
    let mut s=7.0;
    let unit=units.iter().find(|u|num(u,"tile")==src as i32).cloned().unwrap_or(Value::Null);
    let uk=num(&unit,"kind");
    let before=dist(src,dst,size);
    let nearest_before=villages.iter().map(|&x|dist(src,x,size)).min().unwrap_or(12);
    let nearest_after=villages.iter().map(|&x|dist(dst,x,size)).min().unwrap_or(12);
    if nearest_after<nearest_before {s+= (nearest_before-nearest_after) as f64*22.0;}
    // Avoid moving back and forth: recent own moves encode the last position even across decisions.
    if let Some(hist)=obs["remembered_state"]["recent_actions"].as_array(){
      for h in hist.iter().rev().take(8){let hc=&h["command"]; if num(hc,"kind")==1 && num(hc,"source")==dst as i32 && num(hc,"target")==src as i32 {s-=42.;break;}}
    }
    let e_before=enemycities.iter().map(|&x|dist(src,x,size)).min().unwrap_or(20);
    let e_after=enemycities.iter().map(|&x|dist(dst,x,size)).min().unwrap_or(20);
    if e_after<e_before {s+= (e_before-e_after) as f64*3.0;}
    if struc(dst)==2 {s+=75.} if struc(dst)==1&&owner(dst)!=me {s+=110.}
    // Unknown tiles, or known-to-be valuable tiles, advance exploration.
    let neigh_unknown=(0..n).filter(|&i|dist(dst,i,size)<=1&&tile(i)["known"]==false).count();
    s+= neigh_unknown as f64*3.2;
    if res(dst)==1||res(dst)==6 {s+=9.} if ter(dst)==5 && res(dst)==1 {s+=5.}
    if owner(dst)==me {s+=4.} else if owner(dst)>0 {s-=16.}
    s-= danger(dst)*18.0;
    // Keep the capital garrisoned when enemy pressure is near; losing it ends the game.
    let cap=tiles.iter().find(|t|num(t,"capital")==me && num(t,"structure")==1).map(|t|num(t,"tile").max(0) as usize);
    if let Some(cap)=cap { let threat=units.iter().filter(|u|num(u,"owner")!=me).map(|u|dist(cap,num(u,"tile").max(0) as usize,size)).min().unwrap_or(99);
      let own_elsewhere=units.iter().any(|u|num(u,"owner")==me && num(u,"tile")!=cap as i32);
      if src==cap && dst!=cap {s-= if turn<8 {230.} else if own_elsewhere && threat<=4 {150.} else {20.};}
      if dst==cap && threat<=3 {s+=55.;}
    }
    // avoid gratuitous cycling / retreat. Approaching huts strongly.
    if uk==3 {s+=2.};
    if before==0{s-=30.;}
    if turn<3 && owncities.contains(&src) {s+=35.;}
    s.max(-20.)
  }
  4 => { // recruit unit
    let city_count=owncities.len();
    let owned_units=units.iter().filter(|u|num(u,"owner")==me).count();
    let mut val=match p {3=> if villages.len()>0 {if turn<5 {104.}else{92.}}else{75.}, 2=> if turn<5 {100.}else{38.},9=> if turn>10 {59.}else{37.},11=> 52.,8=> 26.,5=> if turn>8 {42.}else{24.},_=>18.};
    if p==3 {val+=18.;} // riders excellent expansion
    if p==2 && turn>8 {val-=15.;}
    if p==2 && (turn<8 || owned_units<2){val+=8.;}
    if city_count>0 && owned_units>city_count*3+2 {val-=25.;}
    if p==5 && turn<20 {val+=12.;} // durable city guard
    val + if enemycities.iter().any(|&x|dist(src,x,size)<7){7.}else{0.}
  }
  7 => { // tech research
    let cities=owncities.len() as i32;
    let cost=4+cities*match p {1|6|10|15|20=>1,2|4|7|8|13|16|18|21|23|44=>2,_=>3};
    let base=match p {
      1=> if !has(1){if turn==0 {42.}else{85.}}else{0.}, // riding
      6=> if !has(6){if turn<20 {70.}else{48.}}else{0.}, // organization
      16=> if !has(16){68.}else{0.}, // forestry
      18=> if turn<30 {43.}else{28.}, // archery
      4=> if turn<35 {39.}else{25.}, // roads
      8=> if turn<30 {40.}else{25.},
      10=> if turn<45 {if tiles.iter().any(|t|t["known"]==true && num(t,"terrain")<=2) {68.}else{36.}}else{18.},
      20=> if turn<25 {32.}else{18.},
      23=> if turn>25 {32.}else{15.},
      17=> if turn>25 {38.}else{18.},
      3=> if turn>30 {36.}else{15.},
      5=> if turn>40 {35.}else{10.},
      _=>20.
    };
    if stars-cost<3 {base-18.}else{base}
  }
  5 => { // harvest population
    if res(dst)==6||res(dst)==1||res(dst)==3 {48.}else{42.}
  }
  6 => { // building
    let building=p;
    let near_enemy=enemycities.iter().any(|&x|dist(dst,x,size)<5);
    match building {
      12=> if turn<75 {43.}else{17.}, // lumberhut
      5=> if turn<75 {54.}else{30.}, // farm
      21=> if turn<75 {53.}else{30.}, // mine
      8=> if turn<45 {if near_enemy {46.}else{24.}}else{12.}, // port
      3=> 8., 17|18|20=>if turn>45 {26.}else{3.},
      _=>15.
    }
  }
  3 => {if p==7 {let u=units.iter().find(|u|num(u,"tile")==src as i32).cloned().unwrap_or(Value::Null); let hp=u["health"].as_f64().unwrap_or(10.); if hp<6. {50.} else if hp<8. {32.} else {8.}}else if p==16 {52.}else{22.}}, // ability, recovery/promote etc
  10 => 0.,
  _=> -100.
 }
}
fn stats(k:i32)->(f64,f64,f64){match k{2=>(2.,2.,10.),3=>(2.,1.,10.),4=>(3.5,1.,10.),5=>(1.,3.,15.),8=>(4.,0.,10.),9=>(2.,1.,10.),11=>(3.,3.,15.),12=>(5.,4.,40.),15=>(3.,1.,15.),_ =>(2.,2.,10.)}}
fn choose(obs:&Value)->String{
 let actions=obs["legal_actions"].as_array().expect("legal_actions");
 let mut best=&actions[0]; let mut bs=f64::NEG_INFINITY;
 for a in actions {let s=score_action(a,obs); if s>bs {bs=s;best=a;}}
 best["action_id"].as_str().unwrap().to_string()
}
fn main(){let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout());for line in stdin.lock().lines(){let line=line.unwrap();let obs:Value=serde_json::from_str(&line).unwrap();writeln!(stdout,"{}",json!({"action_id":choose(&obs)})).unwrap();stdout.flush().unwrap();}}
