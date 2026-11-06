use serde_json::{json,Value};
use std::io::{self,BufRead,Write};
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn ar(v:&Value)->Vec<Value>{v.as_array().cloned().unwrap_or_default()}
fn choose(obs:&Value)->String{
 let legal=ar(&obs["legal_actions"]); let vis=&obs["visible_state"]; let sz=num(vis,"board_size").max(1) as usize; let n=sz*sz; let me=num(obs,"player"); let turn=num(vis,"turn"); let stars=num(vis,"stars");
 let mut tiles=vec![Value::Null;n]; for t in ar(&vis["tiles"]){let i=num(&t,"tile").max(0) as usize;if i<n{tiles[i]=t;}}
 let mut units=vec![Value::Null;n];for u in ar(&vis["units"]){let i=num(&u,"tile").max(0) as usize;if i<n{units[i]=u;}}
 let mut cities=vec![Value::Null;n];for c in ar(&vis["cities"]){let i=num(&c,"tile").max(0) as usize;if i<n{cities[i]=c;}}
 let tech:Vec<i32>=ar(&vis["technology"]).iter().map(|x|x.as_i64().unwrap_or(-1) as i32).collect();let has=|x:i32|tech.contains(&x);
 let dist=|a:usize,b:usize| -> i32 {((a%sz) as i32-(b%sz) as i32).abs().max(((a/sz) as i32-(b/sz) as i32).abs())};
 let mut best=&legal[0];let mut bs=f64::NEG_INFINITY;
 for a in &legal {let c=&a["command"];let k=num(c,"kind");let s=num(c,"source");let t=num(c,"target");let p=num(c,"parameter");let ti=if t>=0&&(t as usize)<n{t as usize}else{0};let si=if s>=0&&(s as usize)<n{s as usize}else{0};let mut v=-100.0;
 match k {
  10=>v=0.0, 11=>v=-10000.0,
  9=>{v=match p{3=>20.0,4=> if turn<10{11.0}else{7.0},7=>12.0,1=>7.0,5=>14.0,8=>11.0,6=> if turn>18{24.0}else{6.0},2=>9.0,_=>0.0}; if p==7&&stars<8{v+=3.0} },
  8=>v=250.0,
  2=>{let target=&units[ti];let hp=target["health"].as_f64().unwrap_or(10.0);let typ=num(target,"kind");let (atk,def,mh)=stats(num(&units[si],"kind"));let (_eatk,edef,emh)=stats(typ);let ah=units[si]["health"].as_f64().unwrap_or(10.0);let damage=((atk*ah/mh)/(atk*ah/mh+edef*hp/emh)*atk*4.5+0.5).floor();let retaliation=((def*hp/emh)/(atk*ah/mh+def*hp/emh)*def*4.5+0.5).floor();let kill=damage>=hp; v=damage*2.4+if kill{24.0}else{0.0}-if !kill&&hp>damage{retaliation*1.9}else{0.0};if typ==12{v+=2.0}; if v<0.0{v=-1.0};},
  7=>{v=match p{16=> if !has(16){18.0}else{1.0},6=>if !has(6){20.0}else{1.0},1=>if !has(1){24.0}else{0.0},10=>if !has(10){12.0}else{0.0},18=>if !has(18){6.0}else{0.0},20=>if !has(20){5.0}else{0.0},8=>if !has(8){9.0}else{0.0},13=>if !has(13){17.0}else{0.0},44=>if !has(44){10.0}else{0.0},_=>2.0}; if p==16&&turn>20{v=8.0}; if p==6&&has(16){v=3.0}; v-=if stars<6{12.0}else if stars<8{4.0}else{0.0};},
  5=>v=if num(&tiles[ti],"resource")==1{13.0}else{12.0},
  6=>{v=match p{12=>29.0,5=>15.0,21=>12.0,6=>6.0,8=> if turn>12 {3.0}else{-5.0},17|18|19|20=> if turn>25{8.0}else{2.0},3=>3.0, _=>4.0}; if p==12 {let adj=(0..n).filter(|&j|dist(ti,j)==1&&num(&tiles[j],"structure")==12).count();v+=adj as f64*1.8;} if p==5{let farms=(0..n).filter(|&j|dist(ti,j)==1&&num(&tiles[j],"structure")==5).count();v+=farms as f64*2.0;} },
  4=>{v=match p{2=>10.0,3=> if turn<12{14.0}else{12.0},9=>10.0,11=>12.0,8=>10.0,_=>7.0};if p==3&&turn<4{v=8.0;}},
  1=>{let from=si;let to=ti; let mut unknown=0;for y in (to/sz).saturating_sub(1)..=((to/sz+1).min(sz-1)){for x in (to%sz).saturating_sub(1)..=((to%sz+1).min(sz-1)){let ix=y*sz+x;if tiles[ix]["known"]==false{unknown+=1;}}}
    let mut dfmin=99; let mut dtmin=99; for j in 0..n {let st=num(&tiles[j],"structure");let own=num(&tiles[j],"owner");if st==1&&own!=me {dfmin=dfmin.min(dist(from,j));dtmin=dtmin.min(dist(to,j));}} let goal_gain=if dfmin<99 {(dfmin-dtmin) as f64*10.0 - dtmin as f64*0.3}else{0.0};
        let mut near_enemy=20;for j in 0..n{if num(&units[j],"owner")>0&&num(&units[j],"owner")!=me{near_enemy=near_enemy.min(dist(to,j));}}
    let mut unknown_from=0; for y in (from/sz).saturating_sub(1)..=((from/sz+1).min(sz-1)){for x in (from%sz).saturating_sub(1)..=((from%sz+1).min(sz-1)){if tiles[y*sz+x]["known"]==false{unknown_from+=1;}}} v=8.0+goal_gain+(unknown-unknown_from) as f64*1.2;
    if goal_gain>0.0{v+=3.0};
    // Moves toward combat are valuable but do not suicidal-charge.
    if near_enemy<=2{v-=3.0;} for past in ar(&obs["remembered_state"]["recent_actions"]) {let pc=&past["command"]; if num(pc,"kind")==1&&num(pc,"target")==to as i32 {v-=16.0;} if num(pc,"kind")==1&&num(pc,"source")==to as i32 {v-=10.0;}}
    if num(&tiles[to],"structure")==1&&num(&tiles[to],"owner")!=me{v+=2.0;}
  },
  3=>{v=match p {4=>-12.0,7=>{let hp=units[si]["health"].as_f64().unwrap_or(10.0);if hp<6.0{9.0}else{1.0}},16=>8.0,8=>-10.0,9=>5.0,_=>0.0};},
  _=>{}
 }
 // avoid building / training if poor economy; reward potential strategic unit spawns
 if k==4 {if p==2 {v+= if turn<6{4.0}else{0.0};}if p==3 {v+=8.0;} if p==9 {v+=1.0;} }
 if k==1 { // sources must still be useful: prioritize unmoved explorers over completely spent moves naturally
   if num(&units[si],"kind")==2 && turn<20 {v+=0.2;}
 }
 if v>bs {bs=v;best=a;}
 }
 best["action_id"].as_str().unwrap_or("0").to_string()
}
fn stats(k:i32)->(f64,f64,f64){match k{2=>(2.0,2.0,10.0),3=>(2.0,1.0,10.0),4=>(3.5,1.0,10.0),5=>(1.0,3.0,15.0),8=>(4.0,0.0,10.0),9=>(2.0,1.0,10.0),10=>(0.0,1.0,10.0),11=>(3.0,3.0,15.0),12=>(5.0,4.0,40.0),_ =>(2.0,2.0,10.0)}}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());for line in stdin.lock().lines(){let obs:Value=serde_json::from_str(&line.unwrap()).unwrap();writeln!(out,"{}",json!({"action_id":choose(&obs)})).unwrap();out.flush().unwrap();}}
