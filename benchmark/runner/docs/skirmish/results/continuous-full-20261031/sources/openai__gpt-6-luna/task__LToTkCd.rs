use serde_json::{json,Value};
use std::io::{self,BufRead,Write};
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn dist(a:usize,b:usize,s:usize)->i32{((a%s) as i32-(b%s) as i32).abs().max(((a/s) as i32-(b/s) as i32).abs())}
fn choose(obs:&Value)->String{
 let vis=&obs["visible_state"]; let size=vis["board_size"].as_u64().unwrap_or(14) as usize; let n=size*size;
 let player=num(obs,"player"); let turn=num(vis,"turn"); let stars=num(vis,"stars");
 let tiles=vis["tiles"].as_array().unwrap(); let mut owner=vec![-1;n];let mut st=vec![-1;n];let mut terr=vec![-1;n];let mut res=vec![0;n];let mut known=vec![false;n];
 for t in tiles {let i=num(t,"tile") as usize;if i<n{known[i]=t["known"].as_bool().unwrap_or(false);if known[i]{owner[i]=num(t,"owner");st[i]=num(t,"structure");terr[i]=num(t,"terrain");res[i]=num(t,"resource");}}}
 let mut units=vec![None::<(i32,i32,f64,bool,bool)>;n]; for u in vis["units"].as_array().unwrap(){let i=num(u,"tile") as usize;if i<n{units[i]=Some((num(u,"owner"),num(u,"kind"),u["health"].as_f64().unwrap_or(10.),u["moved"].as_bool().unwrap_or(true),u["attacked"].as_bool().unwrap_or(true)));}}
 let tech:Vec<i32>=vis["technology"].as_array().unwrap().iter().filter_map(|x|x.as_i64().map(|v|v as i32)).collect();let has=|x:i32|tech.contains(&x);
 let cities=vis["cities"].as_array().unwrap();let mut mycities=Vec::new();for c in cities{if num(c,"owner")==player{mycities.push(num(c,"tile") as usize)}}
 let enemy_units:Vec<usize>=(0..n).filter(|&i|units[i].map(|x|x.0!=player).unwrap_or(false)).collect();let recent=obs["remembered_state"]["recent_actions"].as_array();let lastmove=recent.and_then(|a|a.last()).map(|r|(&r["command"]["kind"],num(&r["command"],"source"),num(&r["command"],"target")));let recent_dest:Vec<i32>=recent.map(|a|a.iter().rev().take(5).filter(|r|num(&r["command"],"kind")==1).map(|r|num(&r["command"],"target")).collect()).unwrap_or_default();let army=units.iter().filter(|u|u.map(|x|x.0==player).unwrap_or(false)).count();
 let acts=obs["legal_actions"].as_array().unwrap();let mut best=&acts[0];let mut bestscore=-1e9f64;
 for a in acts {let c=&a["command"];let k=num(c,"kind");let src=num(c,"source");let dst=num(c,"target");let p=num(c,"parameter");let mut q=-500.;
 match k {
  8=>{let i=src.max(0) as usize;if i<n{q= if st[i]==2 {145.} else if owner[i]==0 {125.} else {165.}; if st[i]==1 && owner[i]>0 && owner[i]!=player {q=190.}}},
  9=>{q=match p{3=>105.,4=> if turn<10 {60.}else{46.},7=>70.,8=>75.,5=>65.,6=>68.,2=>55.,1=>20.,_=>0.};},
  2=>{let i=src.max(0) as usize;let j=dst.max(0) as usize;if i<n&&j<n{if let Some((_,uk,hp,_,_))=units[i]{if let Some((_,dk,dh,_,_))=units[j]{let stats=|u:i32|->(f64,f64,f64){match u{2=>(2.,2.,10.),3=>(2.,1.,10.),4=>(3.5,1.,10.),5=>(1.,3.,15.),8=>(4.,0.,10.),9=>(2.,1.,10.),11=>(3.,3.,15.),12=>(5.,4.,40.),_ =>(2.,2.,10.)}};let (aa,_,ah)=stats(uk);let (_da,dd,dhm)=stats(dk);let af=aa*(hp/ah).max(0.1);let df=dd*(dh/dhm).max(0.1)*if terr[j]==5||terr[j]==4{1.5}else if st[j]==1{1.5}else{1.};let dmg=(af/(af+df)*aa*4.5+0.5).floor();let retaliation=(df/(af+df)*dd*4.5+0.5).floor();let kill=dh<=dmg;let loss=if kill{0.}else{retaliation};q=55.+dmg*3.+if kill{55.}else{0.}-loss*6.;if hp<=loss{q-=45.} if dist(i,j,size)==1&&uk==2{q-=2.};}}}},
  1=>{let i=src.max(0) as usize;let j=dst.max(0) as usize;if i<n&&j<n{let mut target_bonus=0.; if st[j]==1&&owner[j]!=player{target_bonus=115.}else if st[j]==2{target_bonus=60.}
    let mut nearest=100;let mut nearestv=None;for z in 0..n{if known[z]&&st[z]==1&&owner[z]!=player{let d=dist(j,z,size);if d<nearest{nearest=d;nearestv=Some(z)}}}
    let oldd=nearestv.map(|z|dist(i,z,size)).unwrap_or(8);let newd=nearestv.map(|z|dist(j,z,size)).unwrap_or(8);
    let explored_gain=(0..n).filter(|&z|!known[z]&&dist(j,z,size)<=1).count() as f64;
    let frontier=(0..n).filter(|&z|!known[z]&&dist(j,z,size)<=2).count() as f64;
    let oldfront=(0..n).filter(|&z|!known[z]&&dist(i,z,size)<=2).count() as f64;
    let enemynear=enemy_units.iter().map(|&z|dist(j,z,size)).min().unwrap_or(8); if enemynear<=1{target_bonus-=155.;}else if enemynear<=2{target_bonus-=70.;} else if enemynear<=3{target_bonus-=32.;}
    q=target_bonus+12.0+(oldd-newd) as f64*5.5+explored_gain*1.2+(frontier-oldfront).max(0.)*0.45+frontier*0.04;
    // advance the explorer and units toward neutral settlements
    if turn<25 {q+=2.0;} if units[i].map(|x|x.1==3).unwrap_or(false){q+=3.;}
    if owner[j]==player&&st[j]==0 {q-=4.;}
    // avoid pointless oscillation, prefer discovering fresh tiles
    if explored_gain==0.&&newd>=oldd{q-=18.;} if let Some((k,ls,lt))=lastmove {if *k==1 && ls==j as i32 && lt==i as i32 {q-=90.;}} if recent_dest.iter().filter(|&&z|z==j as i32).count()>0 && st[j]!=1 && st[j]!=2 {q-=12.;}
  }},
  5=>{q= if stars<6{48.}else if stars<12{39.}else{25.};},
  6=>{let i=dst.max(0) as usize;if i<n{q=match p{12=>30.,13=>38.,5=> if res[i]==2{46.}else{24.},21=> if res[i]==5{47.}else{20.},8=> if turn>20{25.}else{12.},17|18|19|20=>8.,50=>25.,_=>-10.};if p==12&&has(16){q+=9.;}if p==13{q+=if turn<35{8.}else{0.};}}},
  7=>{q=match p{6=> if !has(6){if turn<3{48.}else if turn<10{40.}else{30.}}else{0.},10=> if !has(10){if (0..n).any(|i|known[i]&&res[i]==3){52.}else if turn>12{32.}else{18.}}else{0.},20=>if (0..n).any(|i|known[i]&&terr[i]==4){38.}else{0.},1=>if turn<30{if turn<3{10.}else{48.}}else{34.},16=>if has(15){if (0..n).any(|i|known[i]&&terr[i]==5){44.}else{22.}}else{0.},8=>if has(6){if (0..n).any(|i|known[i]&&res[i]==2){48.}else{37.}}else{0.},23=>if has(20)&&(0..n).any(|i|known[i]&&res[i]==5){48.}else{0.},18=> if has(15){if (0..n).any(|i|known[i]&&res[i]==1){43.}else{29.}}else{0.},4=>if has(1){if turn<30{33.}else{28.}}else{0.},_=>if turn<12{5.}else{15.}}; if turn>60{q-=10.}},
  4=>{q= if army<mycities.len()+2 {62.}else if army<mycities.len()*2{42.}else{15.};q+=match p{2=>if turn<25{22.}else{12.},3=>0.,9=>if army<mycities.len(){10.}else{0.},11=>18.,_=>0.};if p==3&&!has(1){q-=60.};if p==9&&!has(18){q-=60.};if p==11&&!has(24){q-=30.};if stars<5{q-=8.;}},
  3=>{q=match p{7=>-10.,16=>15.,9=>if src>=0&&units[src as usize].map(|x|x.2<10.).unwrap_or(false){28.}else{10.},_=>-15.};},
  10=>q=-5.,
  _=>{}
 }
 // stable soft priority; prefer actions that unlock/claim before passing
 if q>bestscore{bestscore=q;best=a;}
 }
 best["action_id"].as_str().unwrap().to_string()
}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());for line in stdin.lock().lines(){let s=line.unwrap();let o:Value=serde_json::from_str(&s).unwrap();writeln!(out,"{}",json!({"action_id":choose(&o)})).unwrap();out.flush().unwrap();}}
