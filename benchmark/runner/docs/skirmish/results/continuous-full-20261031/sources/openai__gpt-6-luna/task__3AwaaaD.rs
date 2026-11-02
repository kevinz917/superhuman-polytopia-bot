use serde_json::{json,Value}; use std::io::{self,BufRead,Write};
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32} fn arr(v:&Value)->&Vec<Value>{v.as_array().unwrap()}
fn choose(obs:&Value)->String{
 let vis=&obs["visible_state"];let size=num(vis,"board_size") as usize;let n=size*size;let me=num(obs,"player");let tiles=arr(&vis["tiles"]);let units=arr(&vis["units"]);let cities=arr(&vis["cities"]);let actions=arr(&obs["legal_actions"]);
 let targets:Vec<usize>=cities.iter().filter(|c|num(c,"owner")!=me).map(|c|num(c,"tile") as usize).chain(tiles.iter().filter(|t|t["known"]==false).map(|t|num(t,"tile") as usize)).collect();
 let targets=if targets.is_empty(){vec![n/2]}else{targets};
 let distance=|tile:usize|->i32{targets.iter().map(|&t|((tile%size).abs_diff(t%size).max((tile/size).abs_diff(t/size))) as i32).min().unwrap_or(99)};
 let mut best_i=0usize;let mut best=f64::NEG_INFINITY;
 for (i,a) in actions.iter().enumerate(){let c=&a["command"];let(k,src,dst,p)=(num(c,"kind"),num(c,"source"),num(c,"target"),num(c,"parameter"));
  let score:f64=match k{11=>-10000.,10=>-1000.,9=>1000.+match p{3=>40.,6=>12.,7=>if num(vis,"stars")<8{10.}else{0.},_=>0.},8=>950.,2=>{let hp=units.iter().find(|u|num(u,"tile")==dst).map(|u|u["health"].as_f64().unwrap_or(10.)).unwrap_or(10.);let defender=units.iter().find(|u|num(u,"tile")==dst);let attacker=units.iter().find(|u|num(u,"tile")==src);let mut sc=300.-hp;if let (Some(a),Some(d))=(attacker,defender){let atk:f64=match num(a,"kind"){2=>2.,3=>2.,4=>3.5,5=>1.,8=>4.,9=>2.,11=>3.,12=>5.,_=>2.};let def:f64=match num(d,"kind"){2=>2.,3|4|9=>1.,5|11=>3.,8=>0.,12=>4.,_=>2.};let ah=a["health"].as_f64().unwrap_or(10.);let dh=d["health"].as_f64().unwrap_or(10.);let dmg=(atk/(atk+def)*atk*4.5+0.5).floor();let retaliation=if dh>dmg{(def/(atk+def)*def*4.5+0.5).floor()}else{0.};sc+=dmg*2.-retaliation*5.;if ah<=retaliation {sc-=80.;}} sc},5=>155.,4=>100.+if matches!(p,2|3|4){20.}else{0.}-units.len() as f64,6=>80.+if p==12{15.}else{0.},7=>50.+if matches!(p,1|2|3|4){15.}else{0.}+if p==1 && num(vis,"turn")<10 {15.}else{0.},3=>if p==8{-10000.}else{10.},1=>30.+15.*(distance(src as usize)-distance(dst as usize)) as f64+if distance(dst as usize)<distance(src as usize){4.}else{0.},_=>0.};
  if score>best || (score==best && a["action_id"].as_str().unwrap()>actions[best_i]["action_id"].as_str().unwrap()){best=score;best_i=i;}
 }
 actions[best_i]["action_id"].as_str().unwrap().to_owned()
}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());for line in stdin.lock().lines(){let o:Value=serde_json::from_str(&line.unwrap()).unwrap();writeln!(out,"{}",json!({"action_id":choose(&o)})).unwrap();out.flush().unwrap();}}
