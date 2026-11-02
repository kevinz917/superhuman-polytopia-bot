use serde_json::{Value,json};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn i(v:&Value,k:&str)->i32 {v[k].as_i64().unwrap_or(0) as i32}
fn b(v:&Value,k:&str)->bool {v[k].as_bool().unwrap_or(false)}
#[derive(Clone,Copy,Default)] struct Tile {known:bool,ter:i32,owner:i32,st:i32,res:i32,road:bool,cap:i32,sl:i32,city:usize}
#[derive(Clone,Copy,Default,Debug)] struct Unit {owner:i32,k:i32,hp:f64,max:f64,moved:bool,attacked:bool,kills:i32,home:i32,pass:i32,veteran:bool}
#[derive(Clone,Default)] struct City {tile:usize,owner:i32,level:i32,progress:i32,prod:i32,rewards:[i32;9],connected:bool}
#[derive(Clone,Copy)] struct Stats {a:f64,d:f64,h:f64,c:f64,m:i32,r:i32,dash:bool,stiff:bool,fort:bool,val:f64}
fn stats(k:i32)->Stats {
 let (a,d,h,c,m,r,dash,stiff,fort,val)=match k {
 2=>(2.,2.,10.,2.,1,1,true,false,true,7.),3=>(2.,1.,10.,3.,2,1,true,false,true,9.),4=>(3.5,1.,10.,8.,3,1,true,false,true,23.),5=>(1.,3.,15.,3.,1,1,false,false,true,10.),
 8=>(4.,0.,10.,8.,1,3,false,true,false,24.),9=>(2.,1.,10.,3.,1,2,true,false,true,10.),10=>(0.,1.,10.,5.,1,1,false,true,false,14.),11=>(3.,3.,15.,5.,1,1,true,false,false,17.),12=>(5.,4.,40.,10.,1,1,false,false,false,48.),
 37=>(2.,2.,10.,2.,1,1,true,false,false,7.),38=>(0.,0.5,5.,8.,2,1,true,true,false,18.),39=>(0.,0.5,5.,2.,2,1,true,true,false,18.),40=>(2.,2.,10.,2.,2,1,true,false,false,10.),
 41=>(3.,2.,10.,15.,2,3,false,true,false,37.),42=>(2.,1.,10.,5.,3,2,true,false,false,19.),43=>(0.,1.,10.,0.,2,2,false,true,false,8.),44=>(3.,3.,10.,5.,3,1,true,false,false,22.),45=>(4.,4.,40.,10.,2,1,false,true,false,52.),
 _=>(2.,2.,10.,3.,1,1,true,false,false,10.)}; Stats{a,d,h,c,m,r,dash,stiff,fort,val}
}
fn sea(k:i32)->bool {(39..=45).contains(&k)}
#[derive(Clone)] struct Act {id:String,k:i32,s:usize,t:usize,p:i32}
struct Board {n:usize,sz:usize,p:i32,turn:i32,stars:f64,tiles:Vec<Tile>,u:Vec<Unit>,cities:Vec<City>,tech:[bool;128],acts:Vec<Act>,adj:Vec<Vec<usize>>,dist:Vec<i32>,own:Vec<usize>,enemy:Vec<usize>,oc:Vec<usize>,ec:Vec<usize>,income:f64}
impl Board {
 fn d(&self,a:usize,b:usize)->i32 {self.dist[a*self.n+b]}
 fn new(v:&Value,mem:&mut Memory)->Self {
 let vs=&v["visible_state"];let sz=i(vs,"board_size") as usize;let n=sz*sz;let p=i(v,"player");
 let mut tiles=vec![Tile::default();n];let mut u=vec![Unit::default();n];let mut cities=Vec::new();let mut tech=[false;128];
 for t in vs["technology"].as_array().unwrap() {let k=t.as_u64().unwrap_or(0) as usize;if k<128 {tech[k]=true;}}
 for t in vs["tiles"].as_array().unwrap() {let x=i(t,"tile") as usize;tiles[x]=Tile{known:b(t,"known"),ter:i(t,"terrain"),owner:i(t,"owner"),st:i(t,"structure"),res:i(t,"resource"),road:i(t,"road")>0,cap:i(t,"capital"),sl:i(t,"structure_level"),city:n};}
 for a in vs["units"].as_array().unwrap() {let x=i(a,"tile") as usize;let k=i(a,"kind");let pass=i(a,"passenger");let hp=a["health"].as_f64().unwrap_or(10.);let mut mx=stats(if pass>0 {pass}else{k}).h+if b(a,"veteran"){5.}else{0.};if sea(k)&&i(a,"owner")!=p {mx=mx.max(hp);if hp>10.&&hp<=15. {mx=15.;}}
 u[x]=Unit{owner:i(a,"owner"),k,hp,max:mx,moved:b(a,"moved"),attacked:b(a,"attacked"),kills:i(a,"kills"),home:i(a,"home"),pass,veteran:b(a,"veteran")};}
 for a in vs["cities"].as_array().unwrap() {let mut rewards=[0;9];if let Some(ar)=a["rewards"].as_array(){for (j,r) in ar.iter().enumerate().take(9){rewards[j]=r.as_i64().unwrap_or(0) as i32;}}
 cities.push(City{tile:i(a,"tile") as usize,owner:i(a,"owner"),level:i(a,"level"),progress:i(a,"progress"),prod:i(a,"production"),rewards,connected:b(a,"connected")});}
 let acts=v["legal_actions"].as_array().unwrap().iter().map(|a|{let c=&a["command"];Act{id:a["action_id"].as_str().unwrap().to_string(),k:i(c,"kind"),s:i(c,"source").max(0) as usize,t:i(c,"target").max(0) as usize,p:i(c,"parameter")}}).collect();
 let mut adj=vec![Vec::new();n];let mut dist=vec![0;n*n];for a in 0..n{for z in 0..n {let d=((a%sz) as i32-(z%sz) as i32).abs().max(((a/sz) as i32-(z/sz) as i32).abs());dist[a*n+z]=d;if d==1{adj[a].push(z);}}}
 if mem.visits.len()!=n {mem.visits=vec![0;n];mem.prev=vec![n;n];mem.city=vec![n;n];mem.owner=vec![0;n];mem.lastturn=-1;}
 for x in 0..n {if tiles[x].owner==p {let old=mem.city[x];if old<n&&cities.iter().any(|c|c.tile==old&&c.owner==p)&&mem.owner[x]==p {tiles[x].city=old;}else {tiles[x].city=cities.iter().filter(|c|c.owner==p).min_by_key(|c|dist[x*n+c.tile]).map(|c|c.tile).unwrap_or(n);}}}
 if let Some((typ,ct))=mem.claim.take(){for x in 0..n{if tiles[x].owner==p&&((typ==8&&mem.owner[x]!=p)||(typ==5&&mem.owner[x]!=p)) {tiles[x].city=ct;}}}
 for x in 0..n {mem.city[x]=tiles[x].city;mem.owner[x]=tiles[x].owner;}
 let own=(0..n).filter(|&x|u[x].owner==p).collect::<Vec<_>>();let enemy=(0..n).filter(|&x|u[x].owner>0&&u[x].owner!=p).collect::<Vec<_>>();let oc=cities.iter().filter(|c|c.owner==p).map(|c|c.tile).collect::<Vec<_>>();let ec=cities.iter().filter(|c|c.owner>0&&c.owner!=p).map(|c|c.tile).collect::<Vec<_>>();
 let income=cities.iter().filter(|c|c.owner==p).map(|c|(c.prod+(tiles[c.tile].cap==p) as i32) as f64).sum::<f64>()+tiles.iter().filter(|t|t.owner==p&&t.st==50).map(|t|t.sl as f64).sum::<f64>();
 Board{n,sz,p,turn:i(vs,"turn"),stars:i(vs,"stars") as f64,tiles,u,cities,tech,acts,adj,dist,own,enemy,oc,ec,income}
 }
 fn city(&self,x:usize)->Option<&City>{self.cities.iter().find(|c|c.tile==x)}
 fn bonus(&self,u:Unit,x:usize)->f64 {
 let t=self.tiles[x];let own=u.owner==self.p;let f=stats(u.k).fort;
 if t.ter==4 {1.5} else if t.ter==5 {if (own&&self.tech[18])||(!own&&self.turn>=10){1.5}else{1.}}
 else if t.ter<=2 {if own&&self.tech[12]{1.5}else{1.}}
 else if t.st==1&&t.owner==u.owner&&f {if own&&self.city(x).map(|c|c.rewards[1]>0).unwrap_or(false){4.}else{1.5}}else{1.}
 }
 fn combat(&self,a:Unit,ap:usize,d:Unit,dp:usize)->(f64,f64) {
 if a.k==10{return(d.hp,0.)}let sa=stats(a.k);let sd=stats(d.k);let af=sa.a*a.hp/a.max.max(1.);let df=sd.d*d.hp/d.max.max(1.)*self.bonus(d,dp);let tot=(af+df).max(0.001);
 let damage=(af/tot*sa.a*4.5+0.500001).floor();let ret=if damage>=d.hp||sd.stiff||self.d(ap,dp)>sd.r||a.k==37||a.k==40 {0.}else{(df/tot*sd.d*4.5+0.500001).floor()};(damage,ret)
 }
 fn reach(&self,s:usize,enemy:bool)->Vec<usize> {
 let unit=self.u[s];let sk=stats(unit.k);let boat=sea(unit.k);let owner=unit.owner;let budget=sk.m*2;let mut dd=vec![999;self.n];dd[s]=0;let mut heap=BinaryHeap::new();heap.push(Reverse((0,s)));
 while let Some(Reverse((cost,x)))=heap.pop(){if dd[x]!=cost||cost>=budget {continue;}let tx=self.tiles[x];
 let roadx=(tx.road||tx.st==1||tx.st==48)&&(tx.owner==owner||tx.owner==0);
 let zoc=self.adj[x].iter().any(|&z|self.u[z].owner>0&&self.u[z].owner!=owner);
 if x!=s&&(zoc||tx.ter==4||(tx.ter==5&&!roadx)||(!boat&&tx.st==8)||(boat&&tx.ter>=3)){continue;}
 for &y in &self.adj[x]{let t=self.tiles[y];let road=(t.road||t.st==1||t.st==48)&&(t.owner==owner||t.owner==0);if !t.known{continue;}if t.ter==4&&!enemy&&!self.tech[20] {continue;}
 if t.ter<=2&&t.st!=48 {if !boat&&!(t.st==8&&t.owner==owner){continue;}if t.ter==2&&!enemy&&!self.tech[13]{continue;}}
 if self.u[y].owner>0&&self.u[y].owner!=owner{continue;}
 let re=roadx&&road;let mut nc=cost+if re{1}else{2};if t.ter==5&&!re {nc=nc.max(budget);}if nc<dd[y] {dd[y]=nc;heap.push(Reverse((nc,y)));}
 }
 }
 (0..self.n).filter(|&x|x!=s&&dd[x]<999&&self.u[x].owner==0).collect()
 }
}
#[derive(Default)] struct Memory {visits:Vec<i32>,prev:Vec<usize>,city:Vec<usize>,owner:Vec<i32>,lastturn:i32,claim:Option<(i32,usize)>}
#[derive(Clone)] struct Goal {x:usize,kind:i32,value:f64,ds:Vec<f64>}
struct Eval<'a> {s:&'a Board,goals:Vec<Goal>,assignment:Vec<Option<usize>>,threat:Vec<Vec<(usize,usize)>>,emove:Vec<Vec<usize>>,explore:Vec<f64>,local:Vec<f64>,near_e:Vec<i32>,want_tech:Option<(i32,f64)>,reserve:f64,naval:bool,force:f64}
impl<'a> Eval<'a> {
 fn paths(s:&Board,target:usize)->Vec<f64> {
 // Reverse shortest paths over land and ship states. Unknown terrain is only a routing hypothesis.
 let n=s.n;let mut dd=vec![100000;n*2];let mut heap=BinaryHeap::new();
 for md in 0..2{dd[md*n+target]=0;heap.push(Reverse((0,md*n+target)));}
 while let Some(Reverse((cost,id)))=heap.pop(){if cost!=dd[id]{continue;}let y=id%n;let mode=id/n;let ty=s.tiles[y];
 for &x in &s.adj[y] {let tx=s.tiles[x];
 for pm in 0..2 {let ok;let edge;
 if pm==0&&mode==0 {ok=(!ty.known||ty.ter>=3||ty.st==48)&&(!tx.known||tx.ter>=3||tx.st==48);edge=if (ty.road||ty.st==1)&&(tx.road||tx.st==1){55}else if !ty.known{125}else{100};}
 else if pm==0&&mode==1 {ok=ty.st==8&&ty.owner==s.p&&tx.ter>=3;edge=145;}
 else if pm==1&&mode==0 {ok=(!tx.known||tx.ter<=2)&&ty.ter>=3;edge=145;}
 else {ok=(!ty.known||ty.ter<=2)&&(!tx.known||tx.ter<=2);edge=if !ty.known{75}else{45};}
 if !ok {continue;}if ty.ter==4&&!s.tech[20] {continue;}if mode==1&&ty.ter==2&&!s.tech[13] {continue;}
 let mut nc=cost+edge;if ty.owner>0&&ty.owner!=s.p&&ty.st!=1{nc+=15;}if s.u[y].owner>0&&s.u[y].owner!=s.p{nc+=120;}else if s.u[y].owner==s.p&&y!=target {nc+=95;}
 let pid=pm*n+x;if nc<dd[pid]{dd[pid]=nc;heap.push(Reverse((nc,pid)));}
 }
 }
 }
 dd.iter().map(|&d|d as f64/100.).collect()
 }
 fn new(s:&'a Board)->Self {
 let n=s.n;let mut threat=vec![Vec::new();n];let mut emove=vec![Vec::new();n];let mut local=vec![0.;n];let mut near_e=vec![99;n];
 for &e in &s.enemy {let st=stats(s.u[e].k);emove[e]=s.reach(e,true);let mut pos=vec![e];if st.dash {pos.extend(emove[e].iter().copied());}
 for x in 0..n {near_e[x]=near_e[x].min(s.d(e,x));if st.a>0.||s.u[e].k==10 {let best=pos.iter().filter(|&&p|s.d(p,x)<=st.r&&p!=x).min_by_key(|&&p|s.d(p,x));if let Some(&ap)=best{threat[x].push((e,ap));}}}
 }
 for x in 0..n {local[x]=s.own.iter().filter(|&&u|s.d(x,u)<=3).map(|&u|stats(s.u[u].k).val*s.u[u].hp/s.u[u].max/(s.d(x,u) as f64+1.)).sum();}
 let mut explore=vec![0.;n];for x in 0..n {if !s.tiles[x].known{continue;}let r=if s.tiles[x].ter==4{2}else{1};for y in 0..n {if s.d(x,y)<=r&&!s.tiles[y].known {let edge=(y%s.sz).min(s.sz-1-y%s.sz).min((y/s.sz).min(s.sz-1-y/s.sz));explore[x]+=if edge==0 {0.65}else{1.};}}
 }
 let water=s.tiles.iter().filter(|t|t.known&&t.ter<=2).count();let known=s.tiles.iter().filter(|t|t.known).count();let naval=water*5>known&&s.turn>=5;
 let mut goals=Vec::new();for c in &s.cities {if c.owner!=s.p&&s.u[c.tile].owner!=s.p {goals.push(Goal{x:c.tile,kind:if c.owner==0{0}else{1},value:if c.owner==0{32.}else{35.},ds:Self::paths(s,c.tile)});}else if c.owner==s.p&&near_e[c.tile]<=2&&(s.u[c.tile].owner!=s.p)&&s.own.iter().filter(|&&u|s.d(u,c.tile)<=2).count()<3 {goals.push(Goal{x:c.tile,kind:3,value:if s.u[c.tile].owner>0&&s.u[c.tile].owner!=s.p{60.}else{26.},ds:Self::paths(s,c.tile)});}}
 for x in 0..n {if s.tiles[x].st==2&&s.u[x].owner!=s.p {goals.push(Goal{x,kind:2,value:19.,ds:Self::paths(s,x)});}}
 let mut front=(0..n).filter(|&x|explore[x]>=1.&&s.tiles[x].known&&(s.tiles[x].ter!=4||s.tech[20])).collect::<Vec<_>>();front.sort_by(|&a,&b|explore[b].partial_cmp(&explore[a]).unwrap());let mut picked=Vec::new();for x in front {if picked.iter().all(|&y|s.d(x,y)>=3){picked.push(x);if picked.len()>=16{break;}}}
 for x in picked {goals.push(Goal{x,kind:4,value:9.+explore[x]*0.85,ds:Self::paths(s,x)});}
 // Enemy soldiers are goals as well, to prosecute a war even in fully explored territory.
 for &e in &s.enemy {if !goals.iter().any(|g|g.kind==5&&s.d(g.x,e)<=2) {goals.push(Goal{x:e,kind:5,value:30.+stats(s.u[e].k).val*0.18,ds:Self::paths(s,e)});}}
 let mut assignment=vec![None;n];let mut load=vec![0.;goals.len()];let mut order=s.own.clone();order.sort_by_key(|&u| if sea(s.u[u].k){2}else if s.u[u].k==12{1}else{0});
 let mut pairs=Vec::new();for &u in &order {if s.tiles[u].st==1&&s.tiles[u].owner!=s.p {continue;}for (j,g) in goals.iter().enumerate(){let d=g.ds[u+if sea(s.u[u].k){n}else{0}];if d>=999.{continue;}let val=g.value/(d+2.).powf(0.7);pairs.push((val,u,j));}}
 pairs.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());
 // Sequential auctions account for already committed units and prevent armies all chasing one village.
 for _ in 0..s.own.len() {let mut best=(-1.,0,0);for &(v,u,j) in &pairs{if assignment[u].is_some(){continue;}let g=&goals[j];let crowd=if g.kind==0||g.kind==2||g.kind==4 {load[j]*2.8}else{load[j]*0.25};let mut v=v/(1.+crowd);if sea(s.u[u].k)&&s.u[u].k!=43&&(g.kind==0||g.kind==2){v*=0.70;}if stats(s.u[u].k).r>=3&&(g.kind==0||g.kind==2){v*=0.45;}if v>best.0{best=(v,u,j);}}if best.0<0.{break;}assignment[best.1]=Some(best.2);load[best.2]+=1.;}
 let force=s.own.iter().map(|&u|stats(s.u[u].k).val*s.u[u].hp/s.u[u].max).sum::<f64>();
 let mut out=Eval{s,goals,assignment,threat,emove,explore,local,near_e,want_tech:None,reserve:0.,naval,force};out.set_tech();out
 }
 fn danger(&self,u:Unit,x:usize,removed:usize)->f64 {
 let s=self.s;let mut damages=Vec::new();for &(e,ap) in &self.threat[x] {if e==removed {continue;}let eu=s.u[e];let dmg=if eu.k==10{u.hp}else{s.combat(eu,ap,u,x).0};if dmg>0.{damages.push(dmg);}}
 damages.sort_by(|a,b|b.partial_cmp(a).unwrap());let mut total=0.;for (j,d) in damages.iter().enumerate(){total+=d*match j{0=>1.,1=>0.85,2=>0.65,_=>0.3};}
 total
 }
 fn exposure(&self,u:Unit,x:usize,removed:usize)->f64 {
 let danger=self.danger(u,x,removed);if danger==0.{return 0.;}let val=stats(u.k).val;let ratio=danger/u.hp.max(1.);let loss=val*0.18*danger.min(u.hp)/u.max;
 loss+if ratio>=1. {val*(0.65+(ratio-1.).min(1.)*0.15)}else if ratio>0.7 {val*0.07}else{0.}
 }
 fn support_damage(&self,e:usize,exclude:usize)->f64 {
 let s=self.s;let mut total=0.;for &a in &s.own {let u=s.u[a];if a==exclude||u.attacked{continue;}let st=stats(u.k);if st.a<=0. {continue;}let d=s.d(a,e);if d<=st.r||(!u.moved&&st.dash&&d<=st.r+st.m) {total+=s.combat(u,a,s.u[e],e).0*if d<=st.r{1.}else{0.6};}}total
 }
 fn attack_value(&self,u:Unit,src:usize,dst:usize)->f64 {
 let s=self.s;let d=s.u[dst];let (hit,ret)=s.combat(u,src,d,dst);let st=stats(u.k);let dt=stats(d.k);if st.a<=0.&&u.k!=10&&u.k!=38&&u.k!=39{return -100.;}if u.k==38||u.k==39{return 30.;}if u.k==10{return dt.val*1.8;}
 let kill=hit>=d.hp;let dead=ret>=u.hp;let mut value=dt.val*(0.65*hit.min(d.hp)/d.max+if kill{0.58}else{0.})-st.val*(0.85*ret/u.max+if dead{0.9}else{0.});
 if !kill&&self.support_damage(dst,src)+hit>=d.hp {value+=dt.val*0.20;}
 if kill&&st.r==1&&s.tiles[dst].st==1 {value+=if s.tiles[dst].owner!=s.p{15.}else{24.};}
 if s.tiles[dst].st==1&&s.tiles[dst].owner==s.p {value+=if kill{22.}else{10.};}
 if !dead {let at=if kill&&st.r==1&&(s.tiles[dst].ter>=3||sea(u.k)){dst}else{src};let mut nu=u;nu.hp-=ret;let after=self.exposure(nu,at,if kill{dst}else{s.n});let before=self.exposure(u,src,s.n);value-= (after-before*0.5)*0.6;
 if s.tiles[src].st==1&&s.tiles[src].owner!=s.p&&at!=src {value-=20.;}
 }
 if u.k==41 {for &x in &s.adj[dst] {if s.u[x].owner>0&&s.u[x].owner!=s.p {let dm=(s.combat(u,src,s.u[x],x).0*0.5).floor();value+=stats(s.u[x].k).val*(0.65*dm.min(s.u[x].hp)/s.u[x].max+if dm>=s.u[x].hp{0.5}else{0.});}}}
 if u.k==4&&kill {let chains=s.adj[dst].iter().filter(|&&x|s.u[x].owner>0&&s.u[x].owner!=s.p&&s.combat(u,dst,s.u[x],x).0>=s.u[x].hp).count();value+=chains as f64*10.;}
 value+0.6
 }
 fn set_tech(&mut self) {
 let s=self.s;let nc=s.oc.len() as f64;let mut counts=[0.0f64;10];let mut forest=0.0f64;let mut mountain=0.0f64;let mut water=0.0f64;let mut ports=0;let mut hut=0.0f64;
 for t in &s.tiles {if t.known&&t.ter==4 {mountain+=1.;}if t.known&&t.ter<=2{water+=1.;}if t.owner==s.p {if t.res>=0&&t.res<10&&t.st==0 {counts[t.res as usize]+=1.;}if t.ter==5&&t.st==0 {forest+=1.;}if t.st==12{hut+=1.;}if t.st==8{ports+=1;}}}
 let enemy_close=s.oc.iter().map(|&x|self.near_e[x]).min().unwrap_or(99)<=5;
 let front=s.enemy.len()>0;let nunit=s.own.len() as f64;let enemystrong=s.enemy.iter().any(|&x|matches!(s.u[x].k,11|12|45|5));let enemyweak=s.enemy.iter().filter(|&&x|s.u[x].max<=10.).count();
 let waterfrac=water/(s.tiles.iter().filter(|t|t.known).count().max(1) as f64);
 let mut desires=Vec::new();
 let mut push=|k:usize,v:f64| {if !s.tech[k]&&v>0. {desires.push((k,v));}};
 push(1,if s.turn>=1 {13.+(nc*1.5).min(6.)}else{0.});
 push(16,if forest>=2. {10.+forest.min(9.)*1.4+if s.turn>=5{3.}else{0.}}else{0.});
 push(6,if counts[6]>=1. {15.+counts[6].min(6.)*3.}else{if s.turn>25{7.}else{0.}});
 push(20,if mountain>=1. {6.+mountain.min(10.)*0.9+if s.turn>7{6.}else{0.}}else{0.});
 push(10,if waterfrac>0.14 {10.+counts[3].min(6.)*2.+if s.turn>8 {9.}else{0.}}else if counts[3]>1.{10.}else{0.});
 push(13,if self.naval||ports>0{20.+if ports>0{8.}else{0.}+if s.turn>=12{10.}else{0.}}else{0.});
 push(17,if hut>=2.||forest>=4.||enemystrong {22.+hut.min(8.)*1.1+if front{8.}else{0.}}else if nc>=3.&&s.turn>12 {19.}else{0.});
 push(18,if enemy_close {22.}else if front&&s.turn>=10 {16.}else{0.});
 push(23,if counts[5]>=2. {9.+counts[5].min(5.)*2.}else if front&&s.turn>20{12.}else{0.});
 push(24,if front {26.}else if s.turn>20 {19.}else{0.});
 push(44,if ports>0&&s.turn>10 {26.}else{0.});
 push(14,if ports>0&&s.tech[13]&&s.turn>12 {37.}else{0.});
 push(7,if self.naval&&ports>0 {19.}else if enemy_close&&nunit<nc*1.5 {17.}else{0.});
 push(4,if nc>=3.&&s.turn>10 {13.+if s.tech[3]{7.}else{0.}}else{0.});
 push(5,if hut>=4.&&s.tech[17]&&s.turn>=15 {24.}else{0.});
 push(2,if s.turn>=18&&front&&s.income>=18. {18.}else{0.});
 push(3,if front&&s.turn>=18&&s.income>=18. {26.+(enemyweak as f64*1.8).min(12.)}else{0.});
 push(8,if counts[2]>=2.{10.+counts[2].min(6.)*2.}else{0.});
 push(9,if counts[2]>=4.{14.}else{0.});
 push(12,if ports>0&&s.own.iter().filter(|&&u|sea(s.u[u].k)).count()>=3{18.}else{0.});
 let prereq=|k:usize|->usize{match k{2|4=>1,3=>2,5=>4,7|8=>6,9=>8,13|44=>10,14=>13,12=>44,17=>16,18=>15,23|21=>20,24=>23,22=>21,38=>7,_=>0}};
 let tier=|k:usize|->f64 {match k{1|6|10|15|20=>1.,2|4|7|8|13|16|18|21|23|44=>2.,_=>3.}};
 let mut vals=[-1000.0f64;128];for (k,v) in desires {let mut root=k;let mut steps=0;while !s.tech[prereq(root)]&&prereq(root)!=0&&steps<5 {root=prereq(root);steps+=1;}let cost=4.+nc*tier(root);let score=v*0.82f64.powi(steps)-cost*if s.stars>cost*3.{0.15}else{0.55};
 let combattech=matches!(root,18|24|17|14|3|7);
 let score=score-if enemy_close&&self.force<(s.enemy.iter().map(|&e|stats(s.u[e].k).val).sum::<f64>())*0.8&&!combattech {8.}else{0.};
 vals[root]=vals[root].max(score);
 }
 let choice=(0..128).filter(|&k|!s.tech[k]).max_by(|&a,&b|vals[a].partial_cmp(&vals[b]).unwrap());
 if let Some(k)=choice {if vals[k]>3. {let cost=4.+nc*tier(k);self.want_tech=Some((k as i32,vals[k]));
 if s.turn>=3&&nunit>=2. {self.reserve=cost;}if enemy_close&&self.force<(s.enemy.len() as f64*12.) {self.reserve*=0.3;}
 }}
 }
 fn growth(&self,ct:usize,pop:f64,cost:f64)->f64 {
 let s=self.s;let Some(c)=s.city(ct) else{return -10.;};if c.owner!=s.p{return -10.;}let need=(c.level+1-c.progress).max(1) as f64;let level=c.level;
 let base=match level {1=>22.,2=>23.,3=>21.,4=>36.,_=>25.};
 let mut value=base*pop/need.min(pop+2.)+if pop>=need {5.}else{0.};
 if level==4&&pop>=need {value+=12.;}if level==1 {value+=6.;}
 let pressure=self.near_e[ct];if pressure<=1&&s.u[ct].owner!=s.p {value-=6.;}
 let price=if s.stars<self.reserve+cost {2.0}else{1.1};value-=cost*price;
 if s.turn<5&&s.own.len()<2 {value-=4.;}
 let deficit=s.enemy.iter().filter(|&&e|s.d(e,ct)<=4).count() as f64 - s.own.iter().filter(|&&u|s.d(u,ct)<=4).count() as f64;
 if deficit>0.&&self.near_e[ct]<=3&&level!=4 {value-=deficit*3.;}
 value
 }
 fn build_value(&self,a:&Act)->f64 {
 let s=self.s;let x=a.t;let t=s.tiles[x];let city=t.city;let p=a.p;
 if p==3 {return self.road_value(x);}
 if p==48 {let adjacent_land=s.adj[x].iter().filter(|&&y|s.tiles[y].ter>=3).count();let un=s.own.iter().filter(|&&u|s.d(u,x)<=2).count();return if adjacent_land>=2&&un>=2&&s.stars>self.reserve+8.{2.5}else{-5.};}
 if p==50 {let val=s.adj[x].iter().filter(|&&y|s.tiles[y].owner==s.p&&matches!(s.tiles[y].st,6|13|22)).map(|&y|s.tiles[y].sl).sum::<i32>().min(8);return val as f64*4.-4.5;}
 if p==8 {
 let nearby=s.tiles.iter().enumerate().filter(|(y,t)|t.st==8&&t.owner==s.p&&s.d(x,*y)<=4).count();let mut val=self.growth(city,1.,7.)-5.;
 let landunit=s.own.iter().filter(|&&u|!sea(s.u[u].k)&&s.d(u,x)<=3).count();
 if self.naval&&nearby==0&&s.turn>=6 {val+=18.+(landunit.min(3) as f64)*2.;}
 if nearby>0 {val-=8.*nearby as f64;}
 let water_adj=s.adj[x].iter().filter(|&&y|s.tiles[y].ter<=2||!s.tiles[y].known).count();val+=water_adj as f64*0.3;
 if !s.tech[13]&&s.stars<self.reserve+7.{val-=7.;}
 if t.res==3{val-=3.;}
 return val;
 }
 let (cost,mut pop)=match p {12=>(3.,1.),5|21=>(5.,2.),6|13=>(5.,0.),22=>(5.,0.),23..=29=>(0.,3.),17=>(20.,1.),18=>(15.,1.),19|20=>(20.,1.),_=>(99.,0.)};
 if matches!(p,6|13|22){let base=match p{6=>5,13=>12,_=>21};pop=s.adj[x].iter().filter(|&&y|s.tiles[y].owner==s.p&&s.tiles[y].st==base).count() as f64*if p==22{2.}else{1.};}
 let mut value=self.growth(city,pop,cost);
 if matches!(p,12|5|21){let mult=match p{12=>13,5=>6,_=>22};for &y in &s.adj[x]{if s.tiles[y].st==mult&&s.tiles[y].owner==s.p{value+=self.growth(s.tiles[y].city,if p==21 {2.}else{1.},0.)*0.65;}}
 }
 if p==12 {let potential=s.adj[x].iter().filter(|&&y|s.tiles[y].owner==s.p&&s.tiles[y].ter==3&&s.tiles[y].st==0).count();value+=potential.min(2) as f64*0.5;}
 if t.res==1||t.res==3||t.res==6 {value-=5.;}if (23..=29).contains(&p){value+=8.;if t.res>0 {value-=1.;}}
 if matches!(p,6|13|22){let maxother=s.acts.iter().filter(|z|z.k==6&&z.p==p&&s.tiles[z.t].city==city).map(|z|s.adj[z.t].iter().filter(|&&y|s.tiles[y].owner==s.p&&s.tiles[y].st==match p{6=>5,13=>12,_=>21}).count()).max().unwrap_or(0);if pop<(maxother as f64)*if p==22{2.}else{1.}{value-=8.;}}
 value
 }
 fn road_value(&self,x:usize)->f64 {
 let s=self.s;if s.stars<self.reserve+6.{return -5.;}let mut score=-5.0f64;let node=|y:usize|s.tiles[y].road||s.tiles[y].st==1||s.tiles[y].st==8||s.tiles[y].st==48;
 for &ct in &s.oc {let Some(c)=s.city(ct)else{continue;};if c.connected||s.tiles[ct].cap==s.p{continue;}if s.d(x,ct)>3{continue;}
 let mut connected=vec![false;s.n];let mut stack=Vec::new();for &cap in &s.oc{if s.tiles[cap].cap==s.p{connected[cap]=true;stack.push(cap);}}
 while let Some(y)=stack.pop(){for &z in &s.adj[y]{if !connected[z]&&(node(z)||z==x)&&(s.tiles[z].owner==s.p||s.tiles[z].owner==0){connected[z]=true;stack.push(z);}}}
 if connected[ct]{score=score.max(10.+self.growth(ct,1.,0.)*0.4);}else if s.adj[x].iter().any(|&y|connected[y])&&s.d(x,ct)<=2 {score=score.max(2.);}
 }
 for &u in &s.own {if s.u[u].moved {continue;}if s.d(u,x)>1{continue;}if let Some(g)=self.assignment[u]{let goal=&self.goals[g];if goal.ds[x]+0.5<goal.ds[u]&&s.adj[x].iter().any(|&y|node(y)&&goal.ds[y]<goal.ds[x]){score+=2.;}}
 }
 score
 }
 fn move_value(&self,a:&Act,mem:&Memory)->f64 {
 let s=self.s;let src=a.s;let dst=a.t;let mut u=s.u[src];let st=stats(u.k);let from=s.tiles[src];let to=s.tiles[dst];
 if from.st==1&&from.owner!=s.p {return -80.;}
 let oldexp=self.exposure(u,src,s.n);let mut value=-0.8;
 if let Some(j)=self.assignment[src] {let g=&self.goals[j];let mode=if sea(u.k){s.n}else{0};let newmode=if (sea(u.k)&&to.ter<=2)||(!sea(u.k)&&to.st==8){s.n}else{0};let d0=g.ds[src+mode];let d1=g.ds[dst+newmode];
 if d1<999. {let improve=(d0-d1).clamp(-5.,5.);value+=improve*if g.kind==0||g.kind==2{5.0}else{3.2};value+=if improve>0.{g.value/(d1+3.)*0.25}else{0.};
 if g.kind==3&&dst==g.x&&s.u[src].hp>6.{value+=5.;}if g.kind==4{value+=self.explore[dst]*0.55;}
 } }
 value+=self.explore[dst]*if s.turn<12{1.15}else{0.65};
 if to.st==1&&to.owner!=s.p {value+=if to.owner==0{32.}else{37.};}
 if to.st==2 {value+=21.;}
 if to.st==1&&to.owner==s.p {value-=2.;}
 if to.st==8&&!sea(u.k) {if !s.tech[13]&&!s.tech[44]{value-=8.;}else{value+=3.;}u.k=if u.k==12{45}else{43};}
 if sea(u.k)&&to.ter>=3 {if matches!(u.k,41|42|44) {value-=if to.st==1&&to.owner!=s.p{if u.k==41{22.}else{8.}}else{if u.k==41{30.}else{15.}};}u.k=if u.pass>0{u.pass}else if u.k==45{12}else{2};}
 let exposure=self.exposure(u,dst,s.n);
 value-=exposure*0.90;
 if s.u[src].hp>=s.u[src].max*0.7 {value+=oldexp*0.65;}
 if s.u[src].k==12&&self.near_e[dst]<=1&&self.near_e[src]>1 {value+=5.;}
 // Wounded troops seek shelter rather than marching to their deaths.
 if s.u[src].hp<s.u[src].max*0.7 {value+=oldexp*0.8;if to.owner==s.p {value+=2.;}if self.near_e[dst]>self.near_e[src]&&oldexp>0.{value+=3.;}}
 let mut best_attack:f64=-100.;if st.dash&&!s.u[src].attacked&&u.k!=43 {for &e in &s.enemy {if s.d(dst,e)<=st.r {best_attack=best_attack.max(self.attack_value(u,dst,e));}}}
 if best_attack>0. {value+=best_attack*0.85+3.;}
 if st.r>=2 {let mut firing=0.;for &e in &s.enemy {let d=s.d(dst,e);if d<=st.r {firing=f64::max(firing,8.+stats(s.u[e].k).val*0.07);}else if d==st.r+1{firing=f64::max(firing,2.);}}
 value+=firing;
 let mut oldfire=false;for &e in &s.enemy {if s.d(src,e)<=st.r {oldfire=true;}}
 if oldfire&&!st.dash&&!s.u[src].attacked {value-=8.;}
 if self.near_e[dst]<=1 {value-=7.;}
 }
 if s.u[src].k==45 {for &e in &s.enemy{if s.d(dst,e)==1{value+=s.combat(s.u[src],dst,s.u[e],e).0*0.35;}}}
 if from.st==1&&from.owner==s.p&&s.stars>=3.&&matches!(s.u[src].k,2|3|9) {value+=10.;}
 if from.st==1&&from.owner==s.p {let risks=self.threat[src].len();if risks>0 {let back=s.own.iter().filter(|&&x|x!=src&&s.d(x,src)<=1&&!s.u[x].moved).count();if back==0 {value-=if s.stars>=3. {4.}else{14.+risks as f64*3.};}}else if s.stars>=2. {value+=4.;}}
 if s.u[src].k==43&&from.owner==s.p&&!sea(s.u[dst].k) {if s.acts.iter().any(|z|z.k==4&&z.s==src) {value-=30.;}}
 if mem.prev[src]==dst {value-=4.;}value-=mem.visits[dst].min(10) as f64*0.12;
 if self.near_e[dst]<3 {value+=(self.local[dst]-stats(u.k).val/(s.d(src,dst) as f64+1.)).min(15.)*0.16;}
 value
 }
 fn summon_value(&self,a:&Act)->f64 {
 let s=self.s;let x=a.s;let k=a.p;let st=stats(k);let near=self.near_e[x];let own=s.own.iter().map(|&z|0.7+0.3*s.u[z].hp/s.u[z].max).sum::<f64>();let cities=s.oc.len() as f64;
 if s.u[x].k==43 {
 let need=self.near_e[x]<=5;let count=s.own.iter().filter(|&&u|s.u[u].k==41).count();
 return match k {41=>if s.tech[14] {26.+if need{8.}else{0.}+if count<2{5.}else{0.}-if s.u[x].max<15.{8.}else{0.}-if count>=3{7.}else{0.}}else{-10.},42=>if !s.tech[14]||s.stars<20. {23.+if need{2.}else{5.}}else{15.},44=>if need {30.}else{19.},_=>-100.};
 }
 let mut val=0.;let unmet=self.goals.iter().filter(|g|(g.kind==0||g.kind==2||g.kind==4)&&g.ds[x]<8.).count() as f64;
 let desired=if s.enemy.is_empty() {(cities*1.4+unmet.min(5.)*0.55).max(2.)}else{cities*3.0+3.}+(s.stars/25.).min(12.);
 let deficit=(desired-own).clamp(-5.,5.);
 let nearby=s.own.iter().filter(|&&u|s.d(u,x)<=3).count();let enemies=s.enemy.iter().filter(|&&u|s.d(u,x)<=5).count();
 val+=8.+deficit*2.4;if own<2. {val+=10.;}if own>=desired+2. {val-=20.;}
 if near<=4 {val+=6.+((enemies as i32-nearby as i32).max(0) as f64)*2.;}
 if near<=2 {val+=8.;}
 if near>7&&nearby>=4 {val-=8.;}
 let fund=if s.stars-st.c<self.reserve{4.5}else{0.};val-=fund;
 let cat=s.own.iter().filter(|&&u|s.d(u,x)<=5&&s.u[u].k==8).count();let knights=s.own.iter().filter(|&&u|s.u[u].k==4).count();let arch=s.own.iter().filter(|&&u|s.d(u,x)<=4&&s.u[u].k==9).count();
 match k {
 2=>{val+=if s.turn<=6{4.}else{0.};if s.tech[24]{val-=14.;}if s.tech[1]&&near>3{val-=12.;}if s.turn>15{val-=5.;}},
 3=>{val+=if near>3{6.}else{1.};if s.turn>9 {val-=6.;}if s.tech[24] {val-=10.;}if s.turn>18 {val-=5.;}if s.tech[3]{val-=5.;}},
 4=>{val+=5.+if s.enemy.iter().any(|&e|s.d(e,x)<=7&&s.u[e].max<=10.){9.}else{0.};val-=knights as f64*1.5;if s.stars<self.reserve+8.{val-=2.;}},
 5=>{val-=5.;if near<=2 {val+=12.;}if self.naval&&s.tiles.iter().enumerate().any(|(y,t)|t.st==8&&t.owner==s.p&&s.d(x,y)<=2)&&near>2 {val+=25.;}if near>4&&!self.naval{val-=15.;}},
 8=>{val+=if near>=3&&near<=7 {10.}else{-3.};val-=cat as f64*4.;if near<=2{val-=17.;}if own<3.{val-=12.;}if s.enemy.iter().any(|&e|s.u[e].k==4&&s.d(e,x)<=5){val-=10.;}},
 9=>{val+=if near<=5{5.}else{-7.};val-=arch as f64*3.;if cat>=1{val-=3.;}if nearby<=1{val-=5.;}},
 11=>{val+=11.;if near<=3{val+=6.;}},
 10=>{val-=20.;},38=>{val-=20.;},_=>{val-=100.;}
 }
 // Training into overwhelming fire is wasteful, except to prevent immediate occupation.
 let nu=Unit{owner:s.p,k,hp:st.h,max:st.h,..Unit::default()};let risk=self.exposure(nu,x,s.n);val-=risk*0.7;
 if k==8&&self.danger(nu,x,s.n)>0. {val-=14.;}
 if s.turn==0 {val-=15.;}if s.stars<=3.&&s.turn<6&&own>=2.{val-=6.;}
 val
 }
 fn score(&self,a:&Act,mem:&Memory)->f64 {
 let s=self.s;
 match a.k {
 11=>-1e6,10=>0.,8=>if s.tiles[a.s].st==1{200.}else{110.},
 9=>match a.p{6=>1000.,3=>1000.,7=>1000.,8=>{let c=s.city(a.t).unwrap();let free=s.tiles.iter().filter(|t|t.owner==s.p&&t.city==a.t&&t.st==0&&(t.ter==5||t.res>0)).count();995.+if free>=2{35.}else{0.}+if c.progress>=2{10.}else{0.}},5=>{let mut resources=0.;for x in 0..s.n{if s.d(x,a.t)==2&&s.tiles[x].owner==0 {resources+=if s.tiles[x].res>0{1.5}else if s.tiles[x].ter==5{1.}else{0.};}}960.+resources*3.+if self.naval&&!s.tiles.iter().any(|t|t.owner==s.p&&t.ter==1) && (0..s.n).any(|x|s.d(x,a.t)==2&&s.tiles[x].ter==1&&s.tiles[x].owner==0){160.}else{0.}},1=>900.,4=>900.,_=>800.},
 2=>{let val=self.attack_value(s.u[a.s],a.s,a.t);if val>0.{val+10.}else{val}},
 1=>self.move_value(a,mem),
 5=>{let t=s.tiles[a.t];let mut v=self.growth(t.city,1.,2.);if s.turn==0{v+=20.;}v+2.},
 6=>self.build_value(a),
 7=>if let Some((k,v))=self.want_tech {if k==a.p {v-if s.oc.iter().any(|&x|s.u[x].owner>0&&s.u[x].owner!=s.p){15.}else{0.}+if s.stars>=self.reserve+4.{3.}else{0.}-if s.oc.iter().any(|&ct|s.city(ct).unwrap().level==4)&&!matches!(k,16|17|10|6) {6.}else{0.}}else{-20.}}else{-20.},
 4=>self.summon_value(a),
 3=>match a.p {16=>90.,7=>{let u=s.u[a.s];let missing=u.max-u.hp;let heal=missing.min(if s.tiles[a.s].owner==s.p{4.}else{2.});let mut v=heal/u.max*stats(u.k).val*0.9+if u.hp<=u.max*0.65{5.}else{-1.};if self.danger(u,a.s,s.n)>=u.hp+heal {v-=10.;}if s.tiles[a.s].st==1&&s.tiles[a.s].owner!=s.p{v=-100.;}v},9=>10.,2=>{let t=s.tiles[a.t];if t.res!=0{-20.}else{-10.}},19=>if a.t==0{1.}else{-100.},18=>1.,_=>-100.},_=>-100.
 }
 }
}
fn choose(obs:&Value,mem:&mut Memory)->String {
 let s=Board::new(obs,mem);if mem.lastturn!=s.turn {for v in &mut mem.visits{*v=(*v-1).max(0);}mem.lastturn=s.turn;}
 let ev=Eval::new(&s);let mut best=-1e20;let mut idx=0;for (j,a) in s.acts.iter().enumerate(){let value=ev.score(a,mem);if value>best {best=value;idx=j;}}

 let a=&s.acts[idx];if a.k==1 {mem.prev[a.t]=a.s;mem.prev[a.s]=s.n;mem.visits[a.t]+=3;}if a.k==8&&s.tiles[a.s].st==1{mem.claim=Some((8,a.s));}if a.k==9&&a.p==5{mem.claim=Some((5,a.t));}
 a.id.clone()
}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){let Ok(line)=line else{break;};if line.trim().is_empty(){continue;}let obs:Value=serde_json::from_str(&line).unwrap();let id=choose(&obs,&mut mem);writeln!(out,"{}",json!({"action_id":id})).unwrap();out.flush().unwrap();}}
