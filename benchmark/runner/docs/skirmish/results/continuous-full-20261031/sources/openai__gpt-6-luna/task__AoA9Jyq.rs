use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn dist(a:usize,b:usize,s:usize)->i32{((a%s) as i32-(b%s) as i32).abs().max(((a/s) as i32-(b/s) as i32).abs())}
fn choose(obs:&Value)->String{
 let acts=obs["legal_actions"].as_array().unwrap(); let v=&obs["visible_state"]; let size=v["board_size"].as_u64().unwrap_or(14) as usize; let n=size*size; let me=obs["player"].as_i64().unwrap_or(1) as i32; let stars=num(v,"stars"); let turn=num(v,"turn");
 let tech:Vec<i32>=v["technology"].as_array().unwrap().iter().filter_map(|x|x.as_i64().map(|n|n as i32)).collect(); let has=|id:i32|tech.contains(&id);
 let mut terrain=vec![-1;n]; let mut structure=vec![-1;n]; let mut owner=vec![-1;n]; let mut resource=vec![-1;n]; let mut capital=vec![0;n];
 for t in v["tiles"].as_array().unwrap(){let i=num(t,"tile").max(0) as usize;if i<n && t["known"]==true {terrain[i]=num(t,"terrain");structure[i]=num(t,"structure");owner[i]=num(t,"owner");resource[i]=num(t,"resource");capital[i]=num(t,"capital");}}
 let mut city_level=vec![0;n]; for c in v["cities"].as_array().unwrap(){let i=num(c,"tile").max(0) as usize;if i<n&&num(c,"owner")==me {city_level[i]=num(c,"level");}}
 let units=v["units"].as_array().unwrap(); let mut uowner=vec![0;n];let mut ukind=vec![0;n];let mut hp=vec![0.;n];
 for u in units {let i=num(u,"tile").max(0) as usize;if i<n{uowner[i]=num(u,"owner");ukind[i]=num(u,"kind");hp[i]=u["health"].as_f64().unwrap_or(10.);}}
 // Objective tiles weighted: villages and ruins. Estimate each unit's strategic destination.
 let mut objectives=Vec::<(usize,f64)>::new();
 for i in 0..n { if structure[i]==1 && owner[i]!=me { objectives.push((i, if owner[i]>0 {1.7}else{1.0})); } else if structure[i]==2 { objectives.push((i,0.45)); } }
 let mut best=-1e30f64; let mut answer=&acts[0];
 for a in acts {let c=&a["command"];let k=num(c,"kind");let src=num(c,"source");let dst=num(c,"target");let p=num(c,"parameter");let si=src.max(0) as usize;let di=dst.max(0) as usize; let mut score=-1000.0;
 match k {
  9=>{ // city rewards
    score=match p {4=> if turn<10 {45.} else {22.},3=> if turn<28{74.}else{52.},7=> if stars<5{36.}else{18.},5=> if objectives.iter().any(|(t,_)|dist(*t,si,size)>3){38.}else{12.},8=> if city_level.get(di).copied().unwrap_or(1)>=3{36.}else{20.},6=> if turn<35{37.}else{29.},1=>8.,2=>24.,_=>0.}; }
  8=>{score=1300.0; if structure[si]==2{score+=25.;} if owner[si]>0{score+=20.;}}
  2=>{ // combat estimate via rough damage formula and counter-risk
    let defender=ukind[di].max(2) as usize;let attacker=ukind[si].max(2) as usize;
    let stats=|u:usize|->(f64,f64,f64){match u{2=>(2.,2.,10.),3=>(2.,1.,10.),4=>(3.5,1.,10.),5=>(1.,3.,15.),8=>(4.,0.,10.),9=>(2.,1.,10.),11=>(3.,3.,15.),12=>(5.,4.,40.),_ =>(2.,2.,10.)}};
    let (aa,_,am)=stats(attacker);let (_,dd,dm)=stats(defender);let ah=hp[si] as f64;let dh=hp[di] as f64;
    let db=if structure[di]==1&&owner[di]==uowner[di]{1.5}else if terrain[di]==5||terrain[di]==4{1.0}else{1.};
    let dmg=((aa*ah/am)/((aa*ah/am)+(dd*dh/dm*db)).max(0.01)*aa*4.5+0.5).floor();
    let retaliate=if dh>dmg {((dd*dh/dm*db)/((aa*ah/am)+(dd*dh/dm*db)).max(0.01)*dd*4.5+0.5).floor()}else{0.};
    score=34.+dmg*3.5+if dh<=dmg{110.}else{0.}-retaliate*3.1;
    let city_threat=(0..n).any(|ci| capital[ci]==me && dist(di,ci,size)<=2) || (0..n).any(|ci|city_level[ci]>0 && dist(di,ci,size)<=1);
    if city_threat {score+=220.;}
    if dh<=dmg && structure[di]==1 {score+=100.;}
    if ah<=retaliate {score-=105.;}
  }
  7=>{score=match p {6=> if !has(6){if turn<3 {96.} else {60.}}else{0.},16=>if !has(16){if turn<7 {if has(1){25.}else{0.}} else {55.}}else{0.},1=>if !has(1){if turn<8 {24.} else {45.}}else{0.},20=>if !has(20)&&resource.iter().any(|&r|r==5){32.}else{0.},18=>if !has(18){15.}else{0.},10=>if !has(10)&&resource.iter().any(|&r|r==3){40.}else{0.},4=>if !has(4){if turn>25{24.}else{8.}}else{0.},_=>8.}; if stars<=6 {score-=10.;} }
  5=>{score= if p==di as i32 && resource[di]==6 {78.}else{68.}; if stars<=3{score+=5.;} }
  6=>{score=match p {12=>if turn<45{22.}else{15.},5=>45.,21=>40.,8=>if turn<35{25.}else{18.},3=>12.,17|18|19|20=>15.,_=>20.}; if p==12 {score+= if city_level.get(di).copied().unwrap_or(0)==0 {0.}else{0.};} }
  4=>{score= if p==3 { // Riders give reach, but city capacity and economy take priority
       if turn<28 { if stars>=5 {78.} else {20.} } else {48.}
     } else if p==2 { let enemy_near=(0..n).any(|i|uowner[i]>0&&uowner[i]!=me&&(0..n).any(|ci|capital[ci]==me&&dist(i,ci,size)<=4));
       if enemy_near && stars>=2 {220.} else if (capital[si]==me && (0..n).all(|i|capital[i]!=me || uowner[i]!=me)) && stars>=2 {210.} else if turn<18{ if stars>=2 {110.}else{18.} }else{55.} } else if p==9 {38.}else{10.};
     if stars<2{score-=50.;}
  }
  1=>{ // movement toward objectives, map discovery, forward deployment
    let from=si;let to=di;score=5.;
    let cap_here=(0..n).any(|ci|capital[ci]==me&&ci==from);
    let enemy_near_cap=(0..n).any(|ci|capital[ci]==me&&(0..n).any(|j|uowner[j]>0&&uowner[j]!=me&&dist(j,ci,size)<=5));
    if cap_here&&enemy_near_cap{score-=65.;}
    let oldbest=objectives.iter().map(|(t,w)| (dist(from,*t,size) as f64)/w).fold(99.,f64::min);
    let newbest=objectives.iter().map(|(t,w)| (dist(to,*t,size) as f64)/w).fold(99.,f64::min);
    if !objectives.is_empty(){score+= (oldbest-newbest)*18.; if dist(to,objectives.iter().min_by_key(|(t,_)|dist(to,*t,size)).unwrap().0,size)>0 {}}
    // Reward exploration of unknown tiles around destination and getting out toward center/opponent
    let mut reveal=0.; for j in 0..n{if terrain[j]<0 && dist(to,j,size)<=1{reveal+=1.;}}
    score+=reveal*1.7;
    if let Some(hist)=obs["remembered_state"]["recent_actions"].as_array(){
      for h in hist.iter().rev().take(3){ let hc=&h["command"]; if num(hc,"kind")==1 && num(hc,"source")==di as i32 && num(hc,"target")==from as i32 {score-=48.;break;} }
    }
    let center=(size/2)*size+size/2; score+=(dist(from,center,size)-dist(to,center,size)) as f64*0.45;
    // capture objectives strongly pursue but don't step into enemy unit
    if uowner[to]!=0 {score-=50.;}
    // Defend own cities: screen enemy approaches and avoid abandoning threatened capital.
    let mut defense_need=0.;
    for ci in 0..n {if city_level[ci]>0 {let d=dist(to,ci,size); let enemy_d=(0..n).filter(|&j|uowner[j]>0&&uowner[j]!=me).map(|j|dist(j,ci,size)).min().unwrap_or(99);
      if enemy_d<=5 && d<=2 {defense_need+= if d<=1 {24.} else {10.};}
    }} score+=defense_need;
    // avoid visible combat threat: adjacent enemies, especially if low health
    let mut threat=0.; for j in 0..n{if uowner[j]>0&&uowner[j]!=me&&dist(to,j,size)<=1{threat+=if hp[j]<7.{8.}else{4.};}}
    score-=threat;
    if terrain[to]==4&&!has(20){score-=4.;}
  }
  3=>{score=match p {7=> if hp[si]<6. {34.}else{10.},16=>50.,8=>-5.,9=>12.,_=>-20.};}
  10=>score=0.,
  _=>{}
 }
 // make village approaches prioritize advancing units that can capture
 if k==1 && !objectives.is_empty(){ let closeness=objectives.iter().map(|(t,_)|dist(di,*t,size)).min().unwrap_or(8); if closeness==1 && structure[di]==1 {score+=35.;} }
 if score>best {best=score;answer=a;}
 }
 answer["action_id"].as_str().unwrap().to_owned()
}
fn main(){let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout());for line in stdin.lock().lines(){let obs:Value=serde_json::from_str(&line.unwrap()).unwrap();writeln!(stdout,"{}",json!({"action_id":choose(&obs)})).unwrap();stdout.flush().unwrap();}}
