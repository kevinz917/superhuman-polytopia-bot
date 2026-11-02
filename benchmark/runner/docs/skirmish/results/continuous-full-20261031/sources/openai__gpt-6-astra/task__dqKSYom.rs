#![allow(dead_code)]
use serde_json::{Value,json};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn iv(v:&Value,k:&str)->i32 {v[k].as_i64().unwrap_or(0) as i32}
fn bv(v:&Value,k:&str)->bool {v[k].as_bool().unwrap_or(false)}
#[derive(Clone,Copy,Default)] struct Stat {atk:f64,def:f64,hp:f64,cost:f64,mv:i32,range:i32,dash:bool,stiff:bool,fort:bool,boat:bool}
fn stat(k:i32)->Stat {
 let (a,d,h,c,m,r,da,st,fo,bo)=match k {
 2=>(2.,2.,10.,2.,1,1,true,false,true,false),3=>(2.,1.,10.,3.,2,1,true,false,true,false),
 4=>(3.5,1.,10.,8.,3,1,true,false,true,false),5=>(1.,3.,15.,3.,1,1,false,false,true,false),
 8=>(4.,0.,10.,8.,1,3,false,true,false,false),9=>(2.,1.,10.,3.,1,2,true,false,true,false),
 10=>(0.,1.,10.,5.,1,1,false,true,false,false),11=>(3.,3.,15.,5.,1,1,true,false,false,false),
 12=>(5.,4.,40.,10.,1,1,false,false,false,false),37=>(2.,2.,10.,2.,1,1,true,false,true,false),
 38=>(0.,0.5,5.,8.,2,1,true,true,false,false),39=>(0.,0.5,5.,8.,2,1,true,true,false,true),
 40=>(2.,2.,10.,2.,2,1,true,false,false,true),41=>(3.,2.,10.,15.,2,3,false,true,false,true),
 42=>(2.,1.,10.,5.,3,2,true,false,false,true),43=>(0.,1.,10.,2.,2,2,false,true,false,true),
 44=>(3.,3.,10.,5.,3,1,true,false,false,true),45=>(4.,4.,40.,10.,2,1,false,true,false,true),
 _=>(2.,2.,10.,3.,1,1,true,false,false,false)};
 Stat{atk:a,def:d,hp:h,cost:c,mv:m,range:r,dash:da,stiff:st,fort:fo,boat:bo}
}
#[derive(Clone,Default)] struct Tile{known:bool,ter:i32,owner:i32,st:i32,res:i32,road:bool,cap:i32,sl:i32}
#[derive(Clone,Default)] struct Unit{pos:usize,owner:i32,k:i32,hp:f64,max:f64,moved:bool,attacked:bool,kills:i32,home:i32,poison:bool}
impl Unit {fn val(&self)->f64 {let v=match self.k {12|45=>22.,8=>11.,41=>19.,4=>11.,11=>8.,10=>7.,43=>4.,42|44=>9.,5=>5.,_=>stat(self.k).cost+1.5};v+if self.max>stat(self.k).hp {1.5}else{0.}}}
#[derive(Clone,Default)] struct City{pos:usize,owner:i32,level:i32,prog:i32,prod:i32,rewards:[i32;9],connected:bool}
#[derive(Clone)] struct Act{id:String,k:i32,s:usize,t:usize,p:i32}
struct Board {size:usize,n:usize,me:i32,turn:i32,stars:f64,tiles:Vec<Tile>,units:Vec<Unit>,umap:Vec<Option<usize>>,cities:Vec<City>,tech:[bool;128],acts:Vec<Act>,adj:Vec<Vec<usize>>,dist:Vec<i32>,town:Vec<Option<usize>>,own:Vec<usize>,enemy:Vec<usize>,income:f64}
#[derive(Default)] struct Memory {turn:i32,visits:Vec<i32>,lastpos:Vec<usize>,town:Vec<i32>,steps:i32,enemy_arch:bool,chosen_goal:Vec<usize>}
impl Board {
 fn read(o:&Value)->Self {
  let v=&o["visible_state"];let size=iv(v,"board_size") as usize;let n=size*size; let me=iv(o,"player");
  let mut b=Board{size,n,me,turn:iv(v,"turn"),stars:iv(v,"stars") as f64,tiles:vec![Tile::default();n],units:vec![],umap:vec![None;n],cities:vec![],tech:[false;128],acts:vec![],adj:vec![vec![];n],dist:vec![0;n*n],town:vec![None;n],own:vec![],enemy:vec![],income:0.};
  for t in v["technology"].as_array().unwrap() {b.tech[t.as_u64().unwrap_or(0).min(127) as usize]=true;}
  for t in v["tiles"].as_array().unwrap() {let i=iv(t,"tile") as usize;b.tiles[i]=Tile{known:bv(t,"known"),ter:iv(t,"terrain"),owner:iv(t,"owner"),st:iv(t,"structure"),res:iv(t,"resource"),road:iv(t,"road")>0,cap:iv(t,"capital"),sl:iv(t,"structure_level")};}
  for u in v["units"].as_array().unwrap(){let k=iv(u,"kind");let pa=iv(u,"passenger");let hp=u["health"].as_f64().unwrap_or(10.);let max=(stat(if pa>0{pa}else{k}).hp+if bv(u,"veteran"){5.}else{0.}).max(hp);let idx=b.units.len();let pos=iv(u,"tile") as usize;b.umap[pos]=Some(idx);b.units.push(Unit{pos,owner:iv(u,"owner"),k,hp,max,moved:bv(u,"moved"),attacked:bv(u,"attacked"),kills:iv(u,"kills"),home:iv(u,"home"),poison:bv(u,"poison")});if iv(u,"owner")==me {b.own.push(idx)}else{b.enemy.push(idx)} }
  for c in v["cities"].as_array().unwrap(){let mut r=[0;9];if let Some(arr)=c["rewards"].as_array(){for (i,x) in arr.iter().enumerate().take(9){r[i]=x.as_i64().unwrap_or(0) as i32;}}let ct=City{pos:iv(c,"tile") as usize,owner:iv(c,"owner"),level:iv(c,"level"),prog:iv(c,"progress"),prod:iv(c,"production"),rewards:r,connected:bv(c,"connected")};if ct.owner==me {b.income+=ct.prod as f64+if b.tiles[ct.pos].cap==me{1.}else{0.};}b.cities.push(ct);}
  for a in o["legal_actions"].as_array().unwrap(){let c=&a["command"];b.acts.push(Act{id:a["action_id"].as_str().unwrap().into(),k:iv(c,"kind"),s:iv(c,"source").max(0) as usize,t:iv(c,"target").max(0) as usize,p:iv(c,"parameter")});}
  for i in 0..n {for j in 0..n {let d=((i%size) as i32-(j%size) as i32).abs().max(((i/size) as i32-(j/size) as i32).abs());b.dist[i*n+j]=d;if d==1{b.adj[i].push(j);}}}
  b
 }
 fn d(&self,a:usize,b:usize)->i32 {self.dist[a*self.n+b]}
 fn water(&self,p:usize)->bool {self.tiles[p].known && self.tiles[p].ter<=2}
 fn bonus(&self,u:&Unit,p:usize)->f64 {
 let t=&self.tiles[p];let own=u.owner==self.me;
 let mut bo=if t.ter==4 || (t.ter==5 && (if own{self.tech[18]}else{self.turn>12})) || (t.ter<=2 && t.known && own && self.tech[12]) {1.5}else{1.};
 if t.st==1&&t.owner==u.owner&&stat(u.k).fort {bo=1.5;if own {if self.cities.iter().any(|c|c.pos==p&&c.rewards[1]>0){bo=4.;}}}
 if u.poison {bo*=0.5;}bo
 }
 fn combat(&self,a:&Unit,ap:usize,d:&Unit,dp:usize)->(f64,f64) {
  if a.k==10 {return(d.hp,0.)}let sa=stat(a.k);let sd=stat(d.k);let af=sa.atk*a.hp/a.max;let dd=sd.def*if d.poison{0.7}else{1.};let df=dd*d.hp/d.max*self.bonus(d,dp);let total=(af+df).max(0.001);
  let dam=(af/total*sa.atk*4.5+0.500001).floor(); let ret=if dam>=d.hp||sd.stiff||self.d(ap,dp)>sd.range||a.k==37||a.k==40 {0.}else{(df/total*dd*4.5+0.500001).floor()};(dam,ret)
 }
 fn near_enemy(&self,p:usize)->i32 {self.enemy.iter().map(|&i|self.d(p,self.units[i].pos)).min().unwrap_or(30)}
 fn friendly(&self,p:usize,r:i32)->f64 {self.own.iter().filter(|&&i|self.d(p,self.units[i].pos)<=r).map(|&i| {let u=&self.units[i];u.val()*u.hp/u.max}).sum()}
}
#[derive(Clone)] struct Goal {pos:usize,kind:i32,value:f64,dist:Vec<i32>}
// Reverse strategic shortest paths on a two-layer land/sea graph. Fog is only a planning uncertainty.
fn routes(b:&Board,target:usize)->Vec<i32> {
 let n=b.n;let mut ds=vec![100000;n*2];let mut heap=BinaryHeap::new();
 for layer in 0..2 {ds[target+layer*n]=0;heap.push(Reverse((0,target+layer*n)));}
 while let Some(Reverse((d,node)))=heap.pop(){if d!=ds[node]{continue}let p=node%n;let mode=node/n;
  for &q in &b.adj[p] {for prev in 0..2 {
   let tq=&b.tiles[q];let tp=&b.tiles[p];let pw=b.water(p);let qw=b.water(q);
   if prev==0 && qw && tq.st!=48 {continue}if prev==1 && !qw && tq.known {continue}
   if tp.ter==4 && !b.tech[20] {continue}
   if pw&&tp.ter==2&&!b.tech[13]{continue}
   let enterport=tp.st==8&&tp.owner==b.me;
   let next=if pw&&tp.st!=48 {if prev==1||enterport{1}else{continue}}else{0};
   if next!=mode {continue}
   let mut cost=if pw {7}else{10};if tp.ter==5||tp.ter==4 {cost=13;}if !tp.known {cost=15;}if !pw&&(tq.road||tq.st==1)&&(tp.road||tp.st==1){cost=6;}
   if enterport&&prev==0 {cost+=9;}if prev==1&&mode==0{cost+=5;}
   if let Some(ui)=b.umap[p] {if b.units[ui].owner!=b.me{cost+=18;}else{cost+=1;}}
   let ix=q+prev*n; if d+cost<ds[ix]{ds[ix]=d+cost;heap.push(Reverse((d+cost,ix)));}
  }}
 }ds
}
struct Tactical {reach:Vec<Vec<bool>>,threat:Vec<Vec<usize>>,moves:Vec<Vec<usize>>,attackers:Vec<Vec<usize>>}
fn tactical(b:&Board)->Tactical {
 let mut moves=vec![vec![];b.units.len()];let mut attackers=vec![vec![];b.n];
 for a in &b.acts {if a.k==1 {if let Some(i)=b.umap[a.s]{moves[i].push(a.t)}}if a.k==2 {if let Some(i)=b.umap[a.s]{attackers[a.t].push(i)}}}
 let mut reach=vec![vec![false;b.n];b.units.len()];let mut threat=vec![vec![];b.n];
 for &ui in &b.enemy {
  let u=&b.units[ui];let st=stat(u.k);let mut ds=vec![999;b.n];let mut heap=BinaryHeap::new();ds[u.pos]=0;heap.push(Reverse((0,u.pos)));reach[ui][u.pos]=true;
  if st.dash {while let Some(Reverse((cost,p)))=heap.pop(){if cost!=ds[p]||cost>=st.mv*2{continue}if p!=u.pos&&(b.tiles[p].ter==4||b.adj[p].iter().any(|&t|b.umap[t].map_or(false,|j|b.units[j].owner==b.me))){continue}
    for &q in &b.adj[p] {let t=&b.tiles[q];if !t.known {continue}if b.water(q)&&!st.boat&&t.st!=48&&!(t.st==8&&t.owner==u.owner){continue}
    if let Some(j)=b.umap[q]{if b.units[j].owner!=u.owner{continue}}
    let road=(t.road||t.st==1)&&(b.tiles[p].road||b.tiles[p].st==1)&&t.owner!=b.me&&b.tiles[p].owner!=b.me;
    let mut nd=cost+if road{1}else{2};if t.ter==5&&!road{nd=nd.max(st.mv*2);}if nd>st.mv*2{continue}if nd<ds[q]{ds[q]=nd;heap.push(Reverse((nd,q)));}
    if b.umap[q].is_none() {reach[ui][q]=true;}
   }
  }}
  for p in 0..b.n {if (st.atk>0.||u.k==10) && (b.d(u.pos,p)<=st.range || (st.dash&&(0..b.n).any(|q|reach[ui][q]&&b.d(q,p)<=st.range))) {threat[p].push(ui);}}
 }
 Tactical{reach,threat,moves,attackers}
}
fn danger(b:&Board,tc:&Tactical,u:&Unit,p:usize,ignore:Option<usize>)->f64 {
 let mut hits=vec![];for &ei in &tc.threat[p]{if Some(ei)==ignore{continue}let e=&b.units[ei];let (damage,_)=b.combat(e,e.pos,u,p);let mut x=damage;
  if e.k==10 {x=u.hp;}
  // Nearby ready allies constrain the opponent's ability to concentrate safely.
  let support=b.own.iter().filter(|&&j|b.units[j].pos!=u.pos&&b.d(b.units[j].pos,e.pos)<=stat(b.units[j].k).range+1).count();
  x*=if support>=3{0.72}else if support>=1 {0.88}else{1.};hits.push(x);
 }hits.sort_by(|a,b|b.partial_cmp(a).unwrap());let sum:f64=hits.iter().take(4).enumerate().map(|(i,h)|h*if i==0{1.}else if i==1{0.85}else{0.65}).sum();sum
}
fn loss_value(u:&Unit,damage:f64)->f64 {let frac=(damage/u.hp.max(1.)).min(1.); u.val()*(damage.min(u.hp)/u.max*0.6 +if frac>=0.98 {0.75}else{frac*0.12})}
fn knight_chain(b:&Board,u:&Unit,pos:usize,used:&mut Vec<bool>,depth:i32)->f64 {
 if depth==0 {return 0.}let mut best:f64=0.;
 for &ei in &b.enemy {let e=&b.units[ei];if used[ei]||b.d(pos,e.pos)!=1 {continue}let t=&b.tiles[e.pos];if b.water(e.pos)&&t.st!=48 {continue}
  let (d,_)=b.combat(u,pos,e,e.pos);if d>=e.hp {used[ei]=true;let v=e.val()*5.+8.+knight_chain(b,u,e.pos,used,depth-1)*0.8;used[ei]=false;best=best.max(v);}
 }best
}
fn attack_score(b:&Board,tc:&Tactical,ui:usize,p:usize,ei:usize,moving:bool)->f64 {
 let u=&b.units[ui];let e=&b.units[ei];let (dam,ret)=b.combat(u,p,e,e.pos);let kill=dam>=e.hp;let mut after=u.clone();after.hp-=ret;
 if u.k==38||u.k==39 {return 125.}
 let mut sc=8.0*(e.val()*(dam.min(e.hp)/e.max*0.65+if kill{0.8}else{0.})-loss_value(u,ret)*1.1);
 if u.k==10 {sc+=e.val()*12.;}
 let mut follow=0.;for &j in &b.own {if j==ui||b.units[j].attacked{continue}let a=&b.units[j];let st=stat(a.k);let can=b.d(a.pos,e.pos)<=st.range||st.dash&&tc.moves[j].iter().any(|&t|b.d(t,e.pos)<=st.range);if can {follow+=b.combat(a,a.pos,e,e.pos).0*0.85;}}
 if !kill && follow+dam>=e.hp {sc+=e.val()*3.5;}else if !kill && dam<e.hp*0.2 && ret>dam {sc-=12.;}
 if after.hp>0. {let end=if kill&&stat(u.k).range==1 {e.pos}else{p};let risk=danger(b,tc,&after,end,if kill{Some(ei)}else{None});let oldrisk=danger(b,tc,u,u.pos,None);
  let netloss=(loss_value(&after,risk)-loss_value(u,oldrisk)*if moving{0.15}else{0.65}).max(0.);
  sc-=netloss*4.5;if u.k==3 {sc+=4.;}
 }
 if b.tiles[e.pos].st==1 {sc+=if b.tiles[e.pos].owner==b.me{50.}else{12.};}
 if kill {sc+=12.;if u.k==4 {let mut used=vec![false;b.units.len()];used[ei]=true;sc+=knight_chain(b,u,e.pos,&mut used,3);}}
 if u.k==41 {for &j in &b.enemy {let victim=&b.units[j];if j!=ei&&b.d(victim.pos,e.pos)==1 {let splash=(b.combat(u,p,victim,victim.pos).0*0.5).floor();sc+=victim.val()*(splash.min(victim.hp)/victim.max*0.65+if splash>=victim.hp{0.8}else{0.})*8.;}}}
 if !kill&&follow+dam<e.hp {sc-=8.;}if !moving {sc+=8.;}sc
}
fn prepare_towns(b:&mut Board,mem:&mut Memory){
 if mem.town.len()!=b.n {mem.town=vec![-1;b.n];mem.visits=vec![0;b.n];mem.lastpos=vec![b.n;b.n];mem.chosen_goal=vec![b.n;b.n];}
 for t in 0..b.n {if b.tiles[t].owner!=b.me{mem.town[t]=-1;continue}let old=mem.town[t];
 let valid=old>=0&&b.cities.iter().any(|c|c.pos==old as usize&&c.owner==b.me);
 if !valid {let city=b.cities.iter().filter(|c|c.owner==b.me&&b.d(c.pos,t)<=if c.rewards[5]>0{2}else{1}).min_by_key(|c|b.d(c.pos,t)).or_else(||b.cities.iter().filter(|c|c.owner==b.me).min_by_key(|c|b.d(c.pos,t)));if let Some(c)=city{mem.town[t]=c.pos as i32;}}
 if mem.town[t]>=0 {b.town[t]=b.cities.iter().position(|c|c.pos==mem.town[t] as usize);}
 }
 if mem.turn!=b.turn{mem.turn=b.turn;mem.steps=0;for &i in &b.own{mem.visits[b.units[i].pos]+=1;}}mem.steps+=1;
}
fn goals(b:&Board,tc:&Tactical)->Vec<Goal> {
 let mut gs:Vec<(usize,i32,f64)>=vec![];
 for c in &b.cities {if c.owner!=b.me {let occupied=b.umap[c.pos].map_or(false,|i|b.units[i].owner==b.me);if !occupied {gs.push((c.pos,if c.owner==0{1}else{2},if c.owner==0{90.}else{58.}));}}
 else {let en=b.near_enemy(c.pos);if en<=4 {let threat=b.enemy.iter().filter(|&&i|b.d(c.pos,b.units[i].pos)<=4).map(|&i|b.units[i].val()*b.units[i].hp/b.units[i].max).sum::<f64>();let own=b.friendly(c.pos,2);if threat>own*0.65||b.umap[c.pos].map_or(false,|i|b.units[i].owner!=b.me) {gs.push((c.pos,4,48.+(threat-own*0.5).min(25.)));}}}
 }
 for p in 0..b.n {let t=&b.tiles[p];if t.st==2&&b.umap[p].map_or(true,|i|b.units[i].owner!=b.me){gs.push((p,3,53.));}}
 // Sparse exploration frontiers, evaluated according to the amount of fog revealed.
 let mut fronts=vec![];
 for p in 0..b.n {let t=&b.tiles[p];if !t.known||t.ter==4&&!b.tech[20]||b.water(p)&&!b.tech[10] {continue}let unseen=(0..b.n).filter(|&q|b.d(p,q)<=if t.ter==4{2}else{1}&&!b.tiles[q].known).count();if unseen==0{continue}
 let center=((p%b.size) as f64-(b.size-1) as f64/2.).abs()+((p/b.size) as f64-(b.size-1) as f64/2.).abs();
 let val=11.+unseen as f64*1.4+(b.size as f64-center)*0.3;
 fronts.push((p,5,val));}
 fronts.sort_by(|a,b|b.2.partial_cmp(&a.2).unwrap());
 let mut selected=vec![];for g in fronts {if selected.iter().all(|&(p,_,_)|b.d(p,g.0)>=2){selected.push(g)}if selected.len()>=22{break}}
 gs.extend(selected);
 if gs.is_empty(){for &i in &b.enemy{gs.push((b.units[i].pos,2,40.));}}
 let _=tc;gs.into_iter().map(|(pos,kind,value)|Goal{pos,kind,value,dist:routes(b,pos)}).collect()
}
fn population_gain(b:&Board,a:&Act)->f64 {
 if a.k==5{return 1.}if a.k!=6{return 0.}let p=a.p;
 let mut pop=match p {5|21=>2.,8|12=>1.,6|13|22=>0.,23..=29=>3.,_=>0.};
 let adj=match p {6=>5,13=>12,22=>21,_=>-1};if adj>=0 {pop=b.adj[a.t].iter().filter(|&&t|b.tiles[t].owner==b.me&&b.tiles[t].st==adj).count() as f64*if p==22{2.}else{1.};}
 if p==12||p==5||p==21 {for &t in &b.adj[a.t] {if b.tiles[t].owner==b.me&&b.tiles[t].st==match p{12=>13,5=>6,_=>22} {pop+=if p==21{2.}else{1.};}}}pop
}
fn buildcost(p:i32)->f64 {match p{3|12=>3.,8=>7.,17|19|20=>20.,18=>15.,23..=29=>0.,_=>5.}}
fn tier(p:i32)->i32 {match p {0=>0,1|6|10|15|20=>1,2|4|7|8|13|16|18|21|23|44=>2,_=>3}}
fn choose(b:&Board,mem:&Memory)->usize {
 let tc=tactical(b);let gs=goals(b,&tc);let nc=b.cities.iter().filter(|c|c.owner==b.me).count();let nu=b.own.len();let ne=b.enemy.len();
 let known=b.tiles.iter().filter(|t|t.known).count();let waters=b.tiles.iter().filter(|t|t.known&&t.ter<=2).count();let waterfrac=waters as f64/known.max(1) as f64;
 let ports:Vec<usize>=(0..b.n).filter(|&p|b.tiles[p].st==8&&b.tiles[p].owner==b.me).collect();let boats=b.own.iter().filter(|&&i|stat(b.units[i].k).boat).count();
 let nearest=b.own.iter().map(|&i|b.near_enemy(b.units[i].pos)).min().unwrap_or(30);
 let threatened=b.cities.iter().filter(|c|c.owner==b.me&&b.near_enemy(c.pos)<=3).count();
 // Unit-goal allocation. Short routes and unclaimed villages precede long campaigns.
 let mut assignment=vec![None;b.units.len()];let mut loads=vec![0.;gs.len()];
 let mut order=b.own.clone();order.sort_by(|&a,&bb|{let da=gs.iter().filter(|g|g.kind==1||g.kind==3).map(|g|g.dist[b.units[a].pos+if stat(b.units[a].k).boat{b.n}else{0}]).min().unwrap_or(9999);let db=gs.iter().filter(|g|g.kind==1||g.kind==3).map(|g|g.dist[b.units[bb].pos+if stat(b.units[bb].k).boat{b.n}else{0}]).min().unwrap_or(9999);da.cmp(&db)});
 for &i in &order {let u=&b.units[i];let mode=if stat(u.k).boat{b.n}else{0};let mut best=-1e9;
  // Do not leave a capturable city or ruin before the next turn.
  if b.tiles[u.pos].st==1&&b.tiles[u.pos].owner!=b.me||b.tiles[u.pos].st==2 {continue}
  for (j,g) in gs.iter().enumerate(){let d=g.dist[u.pos+mode] as f64/10.;if d>1000.{continue}let mut val=g.value/(1.+d*0.26);
   val-=loads[j]*if g.kind==1||g.kind==3{48.}else if g.kind==5{18.}else{5.};
   if g.kind==5 {for (k,gg) in gs.iter().enumerate(){if loads[k]>0.&&k!=j&&b.d(g.pos,gg.pos)<=4 {val-=loads[k]*(5-b.d(g.pos,gg.pos)) as f64*3.;}}}
   if (u.k==8||u.k==41)&&g.kind==5{val*=0.5;}if (u.k==12||u.k==45)&&g.kind==2{val+=5.;}
   if g.kind==4&&u.k==3 {val-=4.;}if g.kind==5&&b.water(g.pos)!=stat(u.k).boat{val-=4.;}
   if mem.chosen_goal[u.pos]==g.pos {val+=1.5;}
   if val>best {best=val;assignment[i]=Some(j);}
  }
  if let Some(j)=assignment[i]{loads[j]+=1.;}
 }
 // City growth projects: available population and minimum marginal cost.
 let mut avail=vec![0.;b.cities.len()];let mut costs=vec![vec![];b.cities.len()];
 for p in 0..b.n {if let Some(ci)=b.town[p] {let t=&b.tiles[p];if t.st!=0{continue}let (pop,cost)=if (t.res==1)||(t.res==6&&b.tech[6])||(t.res==3&&b.tech[10]) {(1.,2.)}else if t.ter==5&&b.tech[16]{(1.,3.)}else if t.res==5&&b.tech[23]||t.res==2&&b.tech[8]{(2.,5.)}else{(0.,0.)};if pop>0.{avail[ci]+=pop;costs[ci].push((cost/pop,pop));}}}
 for a in &b.acts {if a.k==6&&(a.p==13||a.p==6||a.p==22||a.p>=23&&a.p<=29){if let Some(ci)=b.town[a.t]{let pop=population_gain(b,a);if pop>0.{avail[ci]+=pop;costs[ci].push((buildcost(a.p)/pop,pop));}}}}
 let mut grow=vec![0.;b.cities.len()];
 for (i,c) in b.cities.iter().enumerate(){if c.owner!=b.me{continue}let need=(c.level+1-c.prog).max(1) as f64;costs[i].sort_by(|a,b|a.0.partial_cmp(&b.0).unwrap());let mut rem=need;let mut cost=0.;for &(r,p) in &costs[i]{let x=rem.min(p);rem-=x;cost+=x*r;if rem<=0.{break}}
 let base=match c.level {1=>112.,2=>94.,3=>65.,4=>100.,_=>70.};grow[i]=base-cost*2.0-need*1.2;
 if rem>0. {grow[i]-=42.+rem*5.;}if b.near_enemy(c.pos)<=2{grow[i]-=15.;}if b.umap[c.pos].map_or(false,|j|b.units[j].owner!=b.me){grow[i]=if c.level>=4{110.}else{-20.};}
 }
 // Research desirability is compared to spending on construction and troops.
 let countres=|r:i32|->f64{b.tiles.iter().filter(|t|t.owner==b.me&&t.st==0&&t.res==r).count() as f64};
 let forest=b.tiles.iter().filter(|t|t.owner==b.me&&t.st==0&&t.ter==5).count() as f64;
 let mountain=b.tiles.iter().filter(|t|t.known&&t.ter==4).count() as f64;
 let cats=b.own.iter().filter(|&&i|b.units[i].k==8||b.units[i].k==41).count();let riders=b.own.iter().filter(|&&i|b.units[i].k==3).count();let knights=b.own.iter().filter(|&&i|b.units[i].k==4).count();
 let enemies_heavy=b.enemy.iter().filter(|&&i|matches!(b.units[i].k,5|11|12|44|45)).count();
 let mut research=[-200.;128];
 research[1]=if b.turn>=1 {58.+if nu<3{8.}else{0.}}else{-100.};
 research[16]=if forest>=2. {41.+forest.min(10.)*2.7}else{-80.};
 research[6]=if countres(6)>=1. {34.+countres(6).min(6.)*5.}else{-20.};
 research[20]=if b.turn>=5 {24.+mountain.min(12.)*1.5+if gs.iter().all(|g|g.kind==5){5.}else{0.}}else{-30.};
 research[10]=if waterfrac>0.12||countres(3)>=2. {38.+countres(3).min(7.)*3.+waterfrac*28.}else{-50.};
 let mills=b.tiles.iter().filter(|t|t.owner==b.me&&t.st==12).count() as f64;
 let bestmill=(0..b.n).filter(|&t|b.tiles[t].owner==b.me&&b.tiles[t].st==0&&b.tiles[t].ter==3).map(|t|b.adj[t].iter().filter(|&&q|b.tiles[q].owner==b.me&&b.tiles[q].st==12).count()).max().unwrap_or(0);
 research[17]=if b.turn>=5 {30.+mills.min(8.)*2.5+bestmill as f64*5.+if enemies_heavy>0{12.}else{0.}}else{-50.};
 research[18]=if nearest<=3&&nu>=4 {45.+ne.min(6) as f64}else if b.turn>=12{38.}else{-40.};
 research[4]=if nc>=3||riders>=2 {35.+nc.min(7) as f64*2.}else{-40.};
 research[13]=if !ports.is_empty() {70.+boats.min(4) as f64*3.}else if waterfrac>0.3&&b.turn>6{50.}else{-50.};
 research[44]=if boats>=1&&ne>0 {51.}else{-60.};
 research[14]=if boats>=2&&b.stars>=24. {58.}else{-60.};
 research[23]=if countres(5)>=1. {38.+countres(5).min(6.)*6.}else if b.turn>14&&b.income>=12.{46.}else{-40.};
 research[24]=if b.turn>=10&&(b.income>=8.||ne>=4) {76.}else{-70.};
 research[2]=if b.turn>=10&&(b.income>=10.||b.stars>=25.){50.}else{-80.};
 research[3]=if b.turn>=10{78.+ne.min(6) as f64*2.}else{-80.};
 research[7]=if threatened>0&&ne>=4{78.}else if b.turn>15{30.}else{-60.};
 research[8]=if countres(2)>=2.{35.+countres(2).min(6.)*5.}else{-90.};research[9]=if countres(2)>=2.&&b.stars>30.{20.}else{-90.};
 research[5]=if mills>=4.&&b.income>=12.{36.}else{-80.};
 research[12]=if boats>=4{40.}else{-90.};
 research[21]=if b.turn>30&&b.stars>60.{15.}else{-90.};research[22]=if b.turn>30&&b.stars>60.{25.}else{-90.};
 research[38]=if b.turn>40&&b.stars>50.{25.}else{-90.};
 // Never exhaust the opening treasury on a tech before the workshop.
 if b.turn==0 {research.fill(-200.);}
 let mut best_save=0.;
 for k in 0..128 {let cost=(4+nc as i32*tier(k as i32)) as f64*if b.tech[22]{0.667}else{1.};let reachable=match k {1|6|10|20=>true,16|18=>true,17=>b.tech[16],13|44=>b.tech[10],14=>b.tech[13]||b.tech[44],23=>b.tech[20],24=>b.tech[23],2|4=>b.tech[1],3=>b.tech[2],7|8=>b.tech[6],5=>b.tech[4],9=>b.tech[8],12=>b.tech[44],_=>false};
  if !b.tech[k]&&reachable {research[k]-=cost*0.65;if b.stars<cost&&b.stars+b.income>=cost&&research[k]>best_save{best_save=research[k];}}
 }
 let mut best=0;let mut bestscore=-1e15;
 for (ai,a) in b.acts.iter().enumerate(){let mut sc=-1000.;
  match a.k {
  10=>{sc=0.;},11=>{},
  9=>{sc=match a.p {3=>1000.,4=>800.,7=>1000.,1=>800.,6=>1100.,2=>700.,8=>950.,5=>940.,_=>500.};
    if a.p==5 {let useful=(0..b.n).filter(|&t|b.d(a.t,t)==2&&b.tiles[t].owner==0&&(b.tiles[t].res>0||b.tiles[t].ter==5)).count();sc+=useful as f64*5.;}
  },
  8=>{sc=350.;if b.tiles[a.s].st==1 {sc=500.;}},
  2=>{if let (Some(ui),Some(ei))=(b.umap[a.s],b.umap[a.t]) {sc=attack_score(b,&tc,ui,a.s,ei,false);}},
  1=>{if let Some(ui)=b.umap[a.s] {let u=&b.units[ui];let st=stat(u.k);sc=-20.;
    if b.tiles[a.s].st==1&&b.tiles[a.s].owner!=b.me||b.tiles[a.s].st==2 {sc=-400.;}else{
    let newboat=b.water(a.t)&&b.tiles[a.t].st!=48;
    let mode=if st.boat{b.n}else{0};let nm=if newboat{b.n}else{0};
    if let Some(gj)=assignment[ui] {let g=&gs[gj];let old=g.dist[a.s+mode] as f64/10.;let new=g.dist[a.t+nm] as f64/10.;let delta=(old-new).clamp(-3.,4.);
     sc=8.+delta*14.+g.value/(1.+new*0.3)*0.12;
     if delta<=0.{sc-=12.;}if g.kind==4&&b.d(a.t,g.pos)==0{sc+=5.;}
     if matches!(u.k,8|41) && g.kind==2 {let en=b.near_enemy(a.t);if en==3 {sc+=12.;}if en<=2{sc-=18.;}}
    }
    if !st.dash&&!u.attacked {let attack=b.enemy.iter().filter(|&&ei|b.d(u.pos,b.units[ei].pos)<=st.range).map(|&ei|attack_score(b,&tc,ui,u.pos,ei,false)).fold(0.,f64::max);sc-=attack.max(0.);}
    let reveal=(0..b.n).filter(|&p|b.d(a.t,p)<=if st.boat&&u.k==42||b.tiles[a.t].ter==4{2}else{1}&&!b.tiles[p].known).count() as f64;
    sc+=reveal*1.2;
    if b.tiles[a.t].st==1&&b.tiles[a.t].owner!=b.me {sc+=if b.tiles[a.t].owner==0{80.}else{62.};}
    if b.tiles[a.t].st==2 {sc+=55.;}
    if b.tiles[a.t].st==1&&b.tiles[a.t].owner==b.me {sc-=6.;}
    if b.tiles[a.s].st==1&&b.tiles[a.s].owner==b.me&&nu<(nc*4+3) {sc+=8.;}
    let mut after=u.clone();after.pos=a.t;if !st.boat&&newboat{after.k=if u.k==12{45}else{43};}if st.boat&&!newboat{after.k=if u.k==45{12}else{2};if matches!(u.k,41|42|44)&&!matches!(b.tiles[a.t].st,1|2){sc-=14.;}}
    let risk=danger(b,&tc,&after,a.t,None);let oldrisk=danger(b,&tc,u,a.s,None);
    sc-=loss_value(&after,risk)*7.;sc+=loss_value(u,oldrisk)*3.;
    if risk>=u.hp {sc-=15.;if b.tiles[a.t].st==1&&b.tiles[a.t].owner!=b.me {sc-=40.;}}
    if u.hp<u.max*0.65&&b.tiles[a.s].owner==b.me&&oldrisk<3. {sc-=18.+(u.max-u.hp)*0.5;}
    if st.dash&&!u.attacked&&!newboat {let mut ab:f64=0.;for &ei in &b.enemy {if b.d(a.t,b.units[ei].pos)<=st.range {ab=ab.max(attack_score(b,&tc,ui,a.t,ei,true));}}sc+=ab*0.85;}
    if matches!(u.k,8|41) {let mut firing:f64=0.;for &ei in &b.enemy {let e=&b.units[ei];if b.d(a.t,e.pos)<=st.range {let d=b.combat(u,a.t,e,e.pos).0;firing=firing.max(d.min(e.hp)*1.5+e.val()*0.5);}}sc+=firing;
      let firing_now=b.enemy.iter().any(|&ei|b.d(a.s,b.units[ei].pos)<=st.range);if firing_now&&u.attacked&&oldrisk<3.{sc-=30.;}
    }
    if u.k==45 {for &ei in &b.enemy {if b.d(a.t,b.units[ei].pos)==1{sc+=b.combat(u,a.t,&b.units[ei],b.units[ei].pos).0*3.;}}}
    if u.k==43 && b.tiles[a.s].owner==b.me&&b.stars>=5.&&(b.tech[13]||b.tech[44]) {sc-=70.;}
    // Stay between the invaders and a vulnerable town.
    if b.tiles[a.s].st==1&&b.tiles[a.s].owner==b.me&&b.near_enemy(a.s)<=2&&b.friendly(a.s,2)<25. {if b.d(a.s,a.t)>1||risk>oldrisk{sc-=35.;}}
    if b.tiles[a.s].st==1&&b.tiles[a.s].owner==b.me&&b.near_enemy(a.s)<=2&&b.stars<3. {
      let relief=b.own.iter().any(|&j|j!=ui&&!b.units[j].moved&&tc.moves[j].contains(&a.s));
      if !relief {sc-=25.;}
    }
    if mem.lastpos[a.s]==a.t{sc-=9.;}sc-=mem.visits[a.t].min(15) as f64*0.7;
    }
  }},
  3=>{match a.p {
   16=>sc=400.,
   7=>{if let Some(ui)=b.umap[a.s]{let u=&b.units[ui];let heal=(u.max-u.hp).min(if b.tiles[a.s].owner==b.me{4.}else{2.});let risk=danger(b,&tc,u,a.s,None);sc=5.+heal*u.val()/u.max*12.;if risk>=u.hp+heal{sc-=25.;}if u.hp<u.max*0.5{sc+=15.;}if b.tiles[a.s].st==1&&b.tiles[a.s].owner!=b.me{sc=-300.;}}},
   9=>{sc=20.;for &j in &b.own{let u=&b.units[j];if b.d(a.s,u.pos)==1{sc+=(u.max-u.hp).min(4.)*3.;}}},
   2=>{sc=-50.;if b.tiles[a.t].res==0&&b.stars<3.&&b.income<8. {if let Some(ci)=b.town[a.t]{if b.cities[ci].level>=3&&avail[ci]<(b.cities[ci].level+1-b.cities[ci].prog) as f64 {sc=8.;}}}},
   19=>sc=if a.t==0 {2.}else{-100.},18=>sc=5.,_=>{}
  }},
  5|6=>{
   let cost=if a.k==5{2.}else{buildcost(a.p)};let pop=population_gain(b,a);
   if pop>0. {if let Some(ci)=b.town[a.t] {let c=&b.cities[ci];let need=(c.level+1-c.prog).max(1) as f64;sc=grow[ci]+(pop/cost.max(1.)-0.4)*18.;if pop>=need {sc+=13.;}sc-=cost*1.5;
     if cost==0. {sc+=60.;}if a.k==6&&b.tiles[a.t].res>0 {sc-=if b.tiles[a.t].res==1||b.tiles[a.t].res==6||b.tiles[a.t].res==3{18.}else{3.};}
     if sc<best_save&&cost>0. {sc-=20.;}if threatened>0&&nu<ne+2&&c.level>1 {sc-=18.;}
   }}
   if a.k==6&&a.p==8 {
    let portdist=ports.iter().map(|&p|b.d(p,a.t)).min().unwrap_or(50);let landneigh=b.adj[a.t].iter().filter(|&&p|!b.water(p)).count();let nearby=b.own.iter().filter(|&&i|b.d(a.t,b.units[i].pos)<=3&&!stat(b.units[i].k).boat).count();
    let nf=(0..b.n).filter(|&p|b.d(a.t,p)<=3&&!b.tiles[p].known).count() as f64;
    let forward=gs.iter().filter(|g|g.kind==2||g.kind==1).map(|g|b.d(a.t,g.pos)).min().unwrap_or(7);
    sc=-100.;if waterfrac>0.14&&portdist>=4&&nearby>0&&landneigh>0 {sc=if ports.is_empty(){64.}else{28.}+waterfrac*15.+nf*0.2-forward as f64*0.8;}
    if b.near_enemy(a.t)<=2 {sc-=20.;}if b.tiles[a.t].res==3 {sc-=8.;}
   }
   if a.k==6&&a.p==50 {let value=b.adj[a.t].iter().filter(|&&p|b.tiles[p].owner==b.me&&matches!(b.tiles[p].st,6|13|22)).map(|&p|b.tiles[p].sl).sum::<i32>();sc=if value>=2{44.+value as f64*4.}else{-50.};}
   if a.k==6&&a.p==3 {sc=-80.;
    let links:Vec<_>=b.cities.iter().filter(|c|c.owner==b.me&&b.d(c.pos,a.t)==1).collect();if links.len()>=2&&links.iter().any(|c|!c.connected){sc=65.;}
    let roadneigh=b.adj[a.t].iter().filter(|&&p|b.tiles[p].road||b.tiles[p].st==1&&b.tiles[p].owner==b.me).count();
    if roadneigh>=1&&b.stars>10. {let mut traffic=0.;for &i in &b.own {let u=&b.units[i];if b.d(a.t,u.pos)<=1 {if let Some(g)=assignment[i] {if b.d(a.t,gs[g].pos)<b.d(u.pos,gs[g].pos) {traffic+=if u.k==3||u.k==4{12.}else{7.};}}}}if traffic>=12.{sc=sc.max(traffic+12.);}}
   }
   if a.k==6&&a.p==48 {sc=-50.;if b.stars>10.&&b.own.iter().any(|&i|b.d(a.t,b.units[i].pos)<=1){let land=b.adj[a.t].iter().filter(|&&p|!b.water(p)).count();if land>=3 {sc=22.;}}}
  },
  7=>{sc=research[a.p as usize];let cost=(4+nc as i32*tier(a.p)) as f64;
   if b.stars-cost<3.&&threatened>0&&nu<=ne+1&& !matches!(a.p,3|7|24){sc-=30.;}
   if b.turn<8&&nu<4&&a.p!=1&&a.p!=16 {sc-=25.;}
   if b.stars>40. {sc+=8.;}if b.turn<=1&&nu<=1 {sc-=18.;}
  },
  4=>{
   if let Some(ui)=b.umap[a.s] {if b.units[ui].k==43 {sc=match a.p{42=>75.,44=>if ne>0{80.}else{65.},41=>if ne>0&&b.stars>20.{82.}else{40.},_=>-100.};if b.units[ui].hp<5.{sc-=25.;}}}
   else{
    let nearest=b.near_enemy(a.s);let useful=gs.iter().map(|g|g.dist[a.s] as f64/10.).fold(1000.,f64::min);let local=b.own.iter().filter(|&&i|b.d(a.s,b.units[i].pos)<=3).count();let enn=b.enemy.iter().filter(|&&i|b.d(a.s,b.units[i].pos)<=4).count();
    let desired=(nc*2+2).max(3)+ne.min(10);let base=if nu<desired {38.}else{12.};
    sc=base+if nearest<=4{17.}else{0.}-local as f64*2.-useful.min(15.)*1.2;
    if nu<2 {sc+=26.;}if nu<4&&b.turn<7{sc+=22.;}if b.turn<=2&&nu<3{sc+=8.;}if enn>local{sc+=17.;}if b.stars>25.{sc+=10.;}if b.stars>50.{sc+=15.;}
    match a.p {
     2=>{sc+=if b.turn<3{8.}else{0.};if b.tech[1]{sc-=24.;}if b.tech[24]{sc-=15.;}if nearest<=2{sc+=5.;}},
     3=>{sc+=12.;if riders>=nc*2+2{sc-=18.;}if nearest<=2{sc-=3.;}},
     4=>{sc+=if ne>=3{16.}else{1.};if knights>=((ne+2)/3).max(2){sc-=25.;}if b.stars<12.{sc-=7.;}},
     5=>{sc+=if nearest<=2{32.}else{-18.};if b.tech[24]&&b.stars>=7.{sc-=15.;}if local>enn&&nearest>1{sc-=20.;}},
     9=>{sc+=if nearest<=4{5.}else{-12.};if b.own.iter().filter(|&&i|b.units[i].k==9).count()>nu/3+1{sc-=20.;}},
     8=>{sc+=if nearest<=6{13.}else{-12.};if enemies_heavy>0{sc+=10.;}if cats>=nu/4+1||cats>=6{sc-=25.;}if nu<5{sc-=18.;}if nearest<=1{sc-=35.;}if b.stars<10.{sc-=10.;}},
     11=>sc+=if nearest<=5{30.}else{8.},
     10=>sc-=20.,38=>sc-=20.,_=>sc=-200.
    }
    if sc<best_save&&b.stars<15.&&nu>=2&&nearest>3 {sc-=30.;}
    if nu>=nc*5+5 {sc-=35.;}if useful>20.&&nearest>7 {sc-=30.;}
    // recruit before a threatened city is occupied, but do not feed ranged siege without counterplay
    if nearest<=1 {sc+=7.;}
    let newu=Unit{pos:a.s,owner:b.me,k:a.p,hp:stat(a.p).hp,max:stat(a.p).hp,..Default::default()};
    let risk=danger(b,&tc,&newu,a.s,None);if risk>=newu.hp {sc-=20.;}else if risk>0.{sc+=8.-risk*0.3;}
   }
  },_=>{}
  }
  // Stable, geometry-neutral tie breaking.
  sc+=((a.s*17+a.t*7+a.p.max(0) as usize*13)%37) as f64*0.001;
  if sc>bestscore {bestscore=sc;best=ai;}
 }
 best
}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){let Ok(line)=line else{break};let Ok(o)=serde_json::from_str::<Value>(&line)else{continue};let mut b=Board::read(&o);prepare_towns(&mut b,&mut mem);let a=choose(&b,&mem);let act=&b.acts[a];if act.k==1 {mem.lastpos[act.t]=act.s;mem.chosen_goal[act.t]=mem.chosen_goal[act.s];mem.visits[act.t]+=1;}writeln!(out,"{}",json!({"action_id":act.id})).unwrap();out.flush().unwrap();}}
