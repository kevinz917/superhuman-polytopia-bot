use serde_json::{Value,json};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn iv(v:&Value,k:&str)->i32 {v[k].as_i64().unwrap_or(0) as i32}
fn bv(v:&Value,k:&str)->bool {v[k].as_bool().unwrap_or(false)}
fn arr<'a>(v:&'a Value,k:&str)-> &'a [Value] {v[k].as_array().map(|x|x.as_slice()).unwrap_or(&[])}
#[derive(Clone,Copy,Default)] struct Stat {hp:f64,atk:f64,def:f64,mv:i32,range:i32,cost:f64,value:f64,dash:bool,stiff:bool,fort:bool}
fn stat(k:i32)->Stat {
 let (hp,atk,def,mv,range,cost,value,dash,stiff,fort)=match k {
 2=>(10.,2.,2.,1,1,2.,5.,true,false,true),3=>(10.,2.,1.,2,1,3.,6.,true,false,true),4=>(10.,3.5,1.,3,1,8.,14.,true,false,true),
 5=>(15.,1.,3.,1,1,3.,7.,false,false,true),8=>(10.,4.,0.,1,3,8.,14.,false,true,false),9=>(10.,2.,1.,1,2,3.,7.,true,false,true),
 10=>(10.,0.,1.,1,1,5.,8.,false,true,false),11=>(15.,3.,3.,1,1,5.,12.,true,false,false),12=>(40.,5.,4.,1,1,10.,32.,false,false,false),
 37=>(10.,2.,2.,1,1,2.,5.,true,false,true),38=>(5.,0.,0.5,2,1,8.,12.,true,true,false),39=>(5.,0.,0.5,2,1,2.,12.,true,true,false),
 40=>(10.,2.,2.,2,1,2.,8.,true,false,false),41=>(10.,3.,2.,2,3,15.,24.,false,true,false),42=>(10.,2.,1.,3,2,5.,11.,true,false,false),
 43=>(10.,0.,1.,2,2,0.,5.,false,true,false),44=>(10.,3.,3.,3,1,5.,13.,true,false,false),45=>(40.,4.,4.,2,1,10.,36.,false,true,false),
 _=>(10.,2.,2.,1,1,3.,6.,true,false,true)};
 Stat{hp,atk,def,mv,range,cost,value,dash,stiff,fort}
}
fn boat(k:i32)->bool {(39..=45).contains(&k)}
#[derive(Clone,Default)] struct Tile {known:bool,ter:i32,owner:i32,st:i32,res:i32,road:bool,capital:i32,sl:i32}
#[derive(Clone,Default)] struct Unit {tile:usize,owner:i32,k:i32,hp:f64,max:f64,moved:bool,attacked:bool,veteran:bool,kills:i32,home:usize,pass:i32}
#[derive(Clone,Default)] struct City {tile:usize,owner:i32,level:i32,progress:i32,prod:i32,connected:bool,rewards:[i32;9]}
struct Board {sz:usize,n:usize,p:i32,turn:i32,stars:f64,tech:[bool;128],t:Vec<Tile>,u:Vec<Option<Unit>>,cities:Vec<City>,cityat:Vec<Option<usize>>,holder:Vec<Option<usize>>,own:Vec<usize>,enemy:Vec<usize>,adj:Vec<Vec<usize>>,income:f64}
impl Board {
 fn parse(o:&Value)->Self {
 let v=&o["visible_state"]; let sz=iv(v,"board_size") as usize; let n=sz*sz; let p=iv(o,"player");
 let mut b=Board{sz,n,p,turn:iv(v,"turn"),stars:iv(v,"stars") as f64,tech:[false;128],t:vec![Tile::default();n],u:vec![None;n],cities:vec![],cityat:vec![None;n],holder:vec![None;n],own:vec![],enemy:vec![],adj:vec![vec![];n],income:0.};
 for a in arr(v,"technology") {if let Some(k)=a.as_u64(){if k<128 {b.tech[k as usize]=true;}}}
 for a in arr(v,"tiles") {let i=iv(a,"tile") as usize;if i<n {b.t[i]=Tile{known:bv(a,"known"),ter:iv(a,"terrain"),owner:iv(a,"owner"),st:iv(a,"structure"),res:iv(a,"resource"),road:iv(a,"road")>0,capital:iv(a,"capital"),sl:iv(a,"structure_level")};}}
 for a in arr(v,"units") {let i=iv(a,"tile") as usize;let k=iv(a,"kind");let pass=iv(a,"passenger"); let mut max=stat(if pass>0{pass}else{k}).hp+if bv(a,"veteran"){5.}else{0.};let hp=a["health"].as_f64().unwrap_or(10.);max=max.max(hp); if boat(k)&&pass==0 && hp>10. {max=15f64.max(hp);}
 b.u[i]=Some(Unit{tile:i,owner:iv(a,"owner"),k,hp,max,moved:bv(a,"moved"),attacked:bv(a,"attacked"),veteran:bv(a,"veteran"),kills:iv(a,"kills"),home:iv(a,"home").max(1) as usize-1,pass});if iv(a,"owner")==p {b.own.push(i);}else{b.enemy.push(i);}}
 for a in arr(v,"cities") {let i=iv(a,"tile") as usize; let mut c=City{tile:i,owner:iv(a,"owner"),level:iv(a,"level").max(1),progress:iv(a,"progress"),prod:iv(a,"production"),connected:bv(a,"connected"),rewards:[0;9]};for (j,r) in arr(a,"rewards").iter().enumerate().take(9){c.rewards[j]=r.as_i64().unwrap_or(0) as i32;}
 if c.owner==p {b.income+=c.prod as f64+if b.t[i].capital==p{1.}else{0.};}b.cityat[i]=Some(b.cities.len());b.cities.push(c);}
 for i in 0..n {let x=i%sz;let y=i/sz;for yy in y.saturating_sub(1)..=(y+1).min(sz-1){for xx in x.saturating_sub(1)..=(x+1).min(sz-1){let j=yy*sz+xx;if j!=i{b.adj[i].push(j);}}}
 if b.t[i].owner==p {b.holder[i]=b.cities.iter().enumerate().filter(|(_,c)|c.owner==p).min_by_key(|(_,c)|b.d(i,c.tile)).map(|(j,_)|j);}
 if b.t[i].st==50&&b.t[i].owner==p{b.income+=b.t[i].sl as f64;}}
 b
 }
 fn d(&self,a:usize,b:usize)->i32 {((a%self.sz) as i32-(b%self.sz) as i32).abs().max(((a/self.sz) as i32-(b/self.sz) as i32).abs())}
 fn nc(&self)->usize {self.cities.iter().filter(|c|c.owner==self.p).count()}
 fn bonus(&self,u:&Unit,pos:usize)->f64 {let t=&self.t[pos];if t.ter==4 {1.5} else if t.ter==5 && (u.owner!=self.p && self.turn>9 || u.owner==self.p&&self.tech[18]) {1.5} else if t.st==1&&t.owner==u.owner&&stat(u.k).fort {if let Some(ci)=self.cityat[pos]{if self.cities[ci].rewards[1]>0{4.}else{1.5}}else{1.5}}else if t.ter<=2&&t.known&&u.owner==self.p&&self.tech[12] {1.5} else {1.}}
 fn damage(&self,a:&Unit,ap:usize,d:&Unit,dp:usize)->(f64,f64) {let sa=stat(a.k);let sd=stat(d.k);let af=sa.atk*a.hp/a.max;let df=sd.def*d.hp/d.max*self.bonus(d,dp);let sum=(af+df).max(0.001); let dmg=(af/sum*sa.atk*4.5+0.500001).floor();let ret=if dmg>=d.hp||sd.stiff||self.d(ap,dp)>sd.range||a.k==37||a.k==40 {0.}else{(df/sum*sd.def*4.5+0.500001).floor()};(dmg,ret)}
 // movement reachable for tactical threat estimation, including terrain and zones of control
 fn reach(&self,u:&Unit)->Vec<usize> {
 let budget=stat(u.k).mv*2;let mut ds=vec![99;self.n];ds[u.tile]=0;let mut heap=BinaryHeap::new();heap.push(Reverse((0,u.tile)));let mut out=vec![u.tile];
 while let Some(Reverse((cost,i)))=heap.pop(){if ds[i]!=cost||cost>=budget {continue;}if i!=u.tile {let t=&self.t[i];let zoc=self.adj[i].iter().any(|&j|self.u[j].as_ref().map_or(false,|e|e.owner!=u.owner));if zoc||t.ter==4||t.ter<=2&&!boat(u.k)||boat(u.k)&&t.ter>2 {continue;}}
 for &j in &self.adj[i] {let t=&self.t[j];if !t.known{continue;}let mountain_ok=u.owner!=self.p||self.tech[20];let ocean_ok=u.owner!=self.p||self.tech[13]; if t.ter==4&&!mountain_ok || t.ter==2&&!ocean_ok {continue;}
 if t.ter<=2&&t.st!=48&&!boat(u.k)&&!(t.st==8&&t.owner==u.owner){continue;}
 if self.u[j].as_ref().map_or(false,|e|e.owner!=u.owner){continue;}
 let r1=(self.t[i].road||self.t[i].st==1||self.t[i].st==48)&&(self.t[i].owner==0||self.t[i].owner==u.owner);
 let r2=(t.road||t.st==1||t.st==48)&&(t.owner==0||t.owner==u.owner);let road=r1&&r2;
 let nc=if t.ter==5&&!road {budget.max(cost+2)}else{cost+if road{1}else{2}};if nc<ds[j]&&nc<=budget {ds[j]=nc;heap.push(Reverse((nc,j)));if self.u[j].is_none(){out.push(j);}}
 }}out
 }
 // weighted strategic distance: transitions through our ports can cross the sea
 fn paths(&self,src:usize,kind:i32)->Vec<f64> {
 let mut ds=vec![1e6;self.n];ds[src]=0.;let mut done=vec![false;self.n];let mv=stat(kind).mv as f64;
 for _ in 0..self.n {let mut i=self.n;let mut best=1e6;for j in 0..self.n {if !done[j]&&ds[j]<best{best=ds[j];i=j;}}if i==self.n {break;}done[i]=true;
 for &j in &self.adj[i] {let t=&self.t[j];if t.known&&t.ter==4&&!self.tech[20]{continue;}let iw=self.t[i].known&&self.t[i].ter<=2&&self.t[i].st!=48;let jw=t.known&&t.ter<=2&&t.st!=48;
 if jw&&!iw&&!(t.st==8&&t.owner==self.p&&self.tech[10]){continue;}if jw&&t.ter==2&&!self.tech[13]{continue;}
 let mut c=if jw {1./if boat(kind){mv}else{2.}} else {1./if boat(kind){1.}else{mv}};
 if !t.known {c=0.9;}else if t.ter==4||t.ter==5 {c=c.max(1.);}if self.t[i].road&&t.road&&!jw{c*=0.55;}
 if iw!=jw {c+=0.5;}if let Some(u)=&self.u[j] {if j!=src {c+=if u.owner==self.p {0.12}else{1.2};}}
 if best+c<ds[j] {ds[j]=best+c;}
 }}ds
 }
}
#[derive(Clone)]struct Action {id:String,k:i32,s:usize,t:usize,p:i32}
#[derive(Default)]struct Memory {prev:Vec<usize>,last:Vec<i32>,goals:Vec<usize>,lastturn:i32}
struct Context<'a> {b:&'a Board,threat:Vec<Vec<usize>>,dist:Vec<Vec<f64>>,goals:Vec<(usize,f64,i32)>,naval_need:f64,front:Vec<i32>,army:f64}
impl<'a> Context<'a> {
 fn new(b:&'a Board)->Self {
 let mut threat=vec![vec![];b.n];
 for &e in &b.enemy {let u=b.u[e].as_ref().unwrap();let st=stat(u.k);let rs=if st.dash{b.reach(u)}else{vec![e]};let mut can=vec![false;b.n];for at in rs {for j in 0..b.n {if b.d(at,j)<=st.range&&at!=j {can[j]=true;}}}for j in 0..b.n {if can[j] {threat[j].push(e);}}}
 let mut dist=vec![vec![];b.n];for &i in &b.own {dist[i]=b.paths(i,b.u[i].as_ref().unwrap().k);}
 let mut goals=vec![];for c in &b.cities {if c.owner==0 {goals.push((c.tile,42.,0));}else if c.owner!=b.p {goals.push((c.tile,45.+if b.t[c.tile].capital>0{8.}else{0.},1));}else {let mut danger=0.;for &e in &b.enemy {let d=b.d(e,c.tile);if d<=5 {danger+=stat(b.u[e].as_ref().unwrap().k).value/(d as f64+1.);}}if danger>1. {goals.push((c.tile,20.+danger.min(20.)*1.8,2));}}}
 for i in 0..b.n {if b.t[i].st==2 {goals.push((i,28.,3));}}
 // frontiers. Each explored tile has a local exploration reward, with central bias.
 for i in 0..b.n {if !b.t[i].known{continue;}let rad=if b.t[i].ter==4 {2}else{1};let mut fog=0.;for j in 0..b.n {if !b.t[j].known&&b.d(i,j)<=rad{let edge=(j%b.sz).min(b.sz-1-j%b.sz).min((j/b.sz).min(b.sz-1-j/b.sz));fog+=if edge>=1{1.}else{0.6};}}
 if fog>0. {let cornerfog=[0,b.sz-1,b.n-b.sz,b.n-1].iter().any(|&j|!b.t[j].known&&b.d(i,j)<=rad); goals.push((i,5.+fog*1.5+if cornerfog{6.}else{0.},4));}}
 let mut front=vec![30;b.n];for i in 0..b.n {for &e in &b.enemy{front[i]=front[i].min(b.d(i,e));}for c in &b.cities {if c.owner>0&&c.owner!=b.p{front[i]=front[i].min(b.d(i,c.tile));}}}
 let water=b.t.iter().filter(|t|t.known&&t.ter<=2).count() as f64/b.t.iter().filter(|t|t.known).count().max(1) as f64;
 let mut landseen=vec![false;b.n];let mut stack:Vec<usize>=b.cities.iter().filter(|c|c.owner==b.p).map(|c|c.tile).collect();for &x in &stack{landseen[x]=true;}while let Some(i)=stack.pop(){for &j in &b.adj[i] {if !landseen[j] && (!b.t[j].known||b.t[j].ter>2||b.t[j].st==48) {landseen[j]=true;stack.push(j);}}}
 let isolated=goals.iter().filter(|&&(t,_,k)|k==0||k==1).any(|&(t,_,_)|!landseen[t]);
 let landfog=goals.iter().filter(|&&(t,_,k)|k==4&&landseen[t]&&b.t[t].ter>2).count();
 let naval_need=if isolated{1.}else if water>0.22&&landfog==0{1.}else{(water*1.8).min(0.85)};
 let army=b.own.iter().map(|&i|{let u=b.u[i].as_ref().unwrap();stat(u.k).value*(0.5+0.5*u.hp/u.max)}).sum();
 Self{b,threat,dist,goals,naval_need,front,army}
 }
 fn incoming(&self,u:&Unit,pos:usize,skip:usize)->f64 {
 let b=self.b;let mut hits=vec![];
 for &e in &self.threat[pos] {if e==skip {continue;}let en=b.u[e].as_ref().unwrap();if en.k==43||en.k==38||en.k==39{continue;}
 let (damage,_)=b.damage(en,e,u,pos);let d=b.d(e,pos);let q=if d<=stat(en.k).range {1.}else{0.85};hits.push(damage*q);}
 hits.sort_by(|a,b|b.partial_cmp(a).unwrap());let mut total=0.;for (i,h) in hits.iter().enumerate(){total+=h*match i{0=>1.,1=>0.85,2=>0.65,_=>0.4};}total
 }
 fn risk(&self,u:&Unit,pos:usize,skip:usize)->f64 {
 let inc=self.incoming(u,pos,skip);let st=stat(u.k);let mut risk=inc.min(u.hp)*st.value/u.max*2.;
 if inc>=u.hp {risk+=st.value*2.8;}else if inc>u.hp*0.65 {risk+=st.value*0.7;}
 if u.k==8||u.k==41||u.k==10 {risk*=1.45;}risk
 }
 fn attack_score(&self,a:&Unit,ap:usize,dp:usize)->f64 {
 let b=self.b;let Some(d)=b.u[dp].as_ref() else{return -1000.};let sa=stat(a.k);let sd=stat(d.k);
 if a.k==10 {return 80.+sd.value*5.;}if a.k==38||a.k==39{return 65.;}
 let (dmg,ret)=b.damage(a,ap,d,dp);let kill=dmg>=d.hp;let dead=ret>=a.hp;
 let gain=(dmg.min(d.hp)/d.max*sd.value*0.9+if kill{sd.value*0.65+2.}else{0.})*5.;
 let loss=ret.min(a.hp)/a.max*sa.value*4.5+if dead{sa.value*3.5}else{0.};
 let mut sc=10.+gain-loss;
 if b.t[dp].st==1 {if b.t[dp].owner==b.p {sc+=if kill{65.}else{28.};}else{sc+=if kill&&sa.range==1{40.}else{8.};}}
 if a.k==4&&kill {let chains=b.adj[dp].iter().filter(|&&x|b.u[x].as_ref().map_or(false,|e|e.owner!=b.p&&e.hp<=10.&&stat(e.k).def<=2.)).count();sc+=chains as f64*25.;}
 if a.k==41 {for &j in &b.adj[dp] {if let Some(e)=&b.u[j]{if e.owner!=b.p{let (dm,_)=b.damage(a,ap,e,j);sc+=dm.min(e.hp)*0.5/e.max*stat(e.k).value*5.;if dm*0.5>=e.hp{sc+=15.;}}}}}
 if !kill && !dead {let mut follow=0.;for &i in &b.own {if i==a.tile{continue;}let u=b.u[i].as_ref().unwrap();if !u.attacked&&b.d(i,dp)<=stat(u.k).range {follow+=b.damage(u,i,d,dp).0;}}
 if follow>=d.hp-dmg {sc+=15.+sd.value*0.6;}}
 if !dead {let at=if kill&&sa.range==1&& (b.t[dp].ter>2||boat(a.k)){dp}else{ap};let mut after=a.clone();after.hp-=ret;let oldrisk=self.risk(a,ap,usize::MAX);let risk=self.risk(&after,at,if kill{dp}else{usize::MAX});sc-=risk*0.6;sc+=oldrisk*0.3;
 if b.t[ap].st==1&&b.t[ap].owner!=b.p&&b.t[ap].owner>0&&at!=ap {sc-=45.;}}
 sc
 }
 fn potential(&self,u:&Unit,pos:usize,paths:&[f64])->(f64,usize) {
 let b=self.b;let mut best=-1.;let mut target=pos;
 for &(g,base,kind) in &self.goals {
 let d=paths[g];if d>500.{continue;}let mut w=base;
 if kind==0||kind==3||kind==4 {
 for &j in &b.own {if j==u.tile{continue;}let dd=self.dist[j].get(g).copied().unwrap_or(1e6);if dd<self.dist[u.tile].get(g).copied().unwrap_or(d)+0.3 {let other=b.u[j].as_ref().unwrap();if other.hp>3. {w*=if kind==0||kind==3 {0.23}else{0.6};if j==g {w*=0.1;} }} }
 if u.k==8||u.k==41 {w*=if kind==4{0.3}else{0.55};}
 } else if kind==1 {
 if u.k==8||u.k==9||u.k==41||u.k==42 {let range=stat(u.k).range;let r=b.d(pos,g);if r<=range {w*=0.7;} }
 }else if kind==2 {
 let defenders=b.own.iter().filter(|&&j|j!=u.tile&&b.d(j,g)<=2).count();w/=1.+defenders as f64*0.35;
 }
 let value=w/(2.5+d);if value>best{best=value;target=g;}
 }
 // pressure enemy army, keeping ranged units at firing distance
 for &g in &b.enemy {let d=(paths[g]-stat(u.k).range as f64*0.7).max(0.);let val=(15.+stat(b.u[g].as_ref().unwrap().k).value*0.3)/(2.5+d);if val>best{best=val;target=g;}}
 (best.max(0.),target)
 }
}
fn buildcost(p:i32)->f64 {match p {3|12=>3.,8=>7.,17=>20.,18=>15.,19|20=>20.,23..=29=>0.,_=>5.}}
fn tier(k:i32)->f64 {match k {1|6|10|15|20=>1.,2|4|7|8|13|16|18|21|23|44=>2.,_=>3.}}
fn growth(b:&Board,c:usize,pop:f64,cost:f64,ctx:&Context)->f64 {
 let city=&b.cities[c];let rem=(city.level+1-city.progress).max(1) as f64;
 let mut v=match city.level {1=>25.,2=>17.,3=>16.,4=>28.,_=>22.};
 let front=ctx.front[city.tile];if city.level>=4 {v*=if front<7{1.25}else{0.85};}
 let completes=pop>=rem;let near=(pop/rem).min(1.);
 let mut s=7.+v*pop/cost.max(1.)+near*13.+if completes {14.}else{0.}-cost*1.3;
 // Avoid stranding cash on impossible city levels: require remaining cheap population potential.
 let mut available=0.;for i in 0..b.n {if b.holder[i]!=Some(c)||b.t[i].st!=0 {continue;}let t=&b.t[i];available+=match t.res {1=>1.,3 if b.tech[10]=>1.,6 if b.tech[6]=>1.,5 if b.tech[23]=>2.,2 if b.tech[8]=>2.,_=>0.};if t.ter==5&&b.tech[16]{available+=1.;}}
 if rem>available+pop+3.5&&city.level>1&&!completes {s-=8.;} if b.stars>cost+12. {s+=8.;}
 if b.stars<cost+3.&&b.own.len()<b.nc()+1 && b.turn>0{s-=16.;}
 if front<=2 && city.level<4{s-=12.;}if b.u[city.tile].as_ref().map_or(false,|u|u.owner!=b.p) {s+=if city.level>=4&&pop>=rem {150.}else if city.level==4 {35.}else{-15.};}
 s
}
fn choose(obs:&Value,mem:&mut Memory)->String {
 let b=Board::parse(obs);let actions:Vec<Action>=arr(obs,"legal_actions").iter().map(|v|{let c=&v["command"];Action{id:v["action_id"].as_str().unwrap_or("").to_string(),k:iv(c,"kind"),s:iv(c,"source").max(0) as usize,t:iv(c,"target").max(0) as usize,p:iv(c,"parameter")}}).collect();
 if actions.is_empty(){return String::new();}
 if mem.prev.len()!=b.n {mem.prev=vec![b.n;b.n];mem.last=vec![-100;b.n];mem.goals=vec![b.n;b.n];}
 let ctx=Context::new(&b);
 let nc=b.nc() as f64;let units=b.own.len() as f64;let n_enemy=b.enemy.len() as f64;
 let mut resources=[0.;10];let mut woods=0f64;let mut huts=0f64;let mut ports=0.;let mut rafts=0.;let mut ships=0.;let mut cats=0.;let mut riders=0.;let mut knights=0.;let mut swords=0.;
 for i in 0..b.n {let t=&b.t[i];if t.owner==b.p {if t.st==0 {if t.res<10{resources[t.res as usize]+=1.;}if t.ter==5 {woods+=1.;}}if t.st==12{huts+=1.;}if t.st==8{ports+=1.;}}}
 for &i in &b.own {match b.u[i].as_ref().unwrap().k {43=>rafts+=1.,39..=45=>ships+=1.,8=>cats+=1.,3=>riders+=1.,4=>knights+=1.,11=>swords+=1.,_=>{}}}
 let contact=ctx.front.iter().enumerate().filter(|(i,_)|b.t[*i].owner==b.p).map(|(_,d)|*d).min().unwrap_or(30);
 let roads=road_plan(&b,&actions);
 let mut best=-1e12;let mut chosen=0;
 let mut pathcache:std::collections::HashMap<(usize,i32),Vec<f64>>=std::collections::HashMap::new();
 let mut srcpot=vec![(0.,0);b.n];for &i in &b.own {srcpot[i]=ctx.potential(b.u[i].as_ref().unwrap(),i,&ctx.dist[i]);}
 let mut bestatt=vec![-1e3f64;b.n];for a in &actions{if a.k==2 {if let Some(u)=&b.u[a.s]{bestatt[a.s]=bestatt[a.s].max(ctx.attack_score(u,a.s,a.t));}}}
 for (ai,a) in actions.iter().enumerate() {
 let s=a.s.min(b.n-1);let t=a.t.min(b.n-1);let p=a.p;let mut score=-10000.;
 match a.k {
 10=>score=0.,11=>{},
 9=>{score=match p {3=>1000.,4=>900.,7=>1000.,1=>850.,6=>1100.,2=>500.,8=>1000.,5=>980.,_=>100.};
 if p==5||p==8 {let mut free=0.;for i in 0..b.n {if b.d(i,t)==2&&b.t[i].owner==0 {free+=if b.t[i].res>0{1.3}else if b.t[i].ter==5{0.9}else{0.3};}}if p==5&&free>4.5 {score+=40.;}}
 },
 8=>{score=if b.t[s].st==1{if b.t[s].owner>0 {2000.}else{250.}}else{180.};},
 2=>{if let Some(u)=&b.u[s] {score=ctx.attack_score(u,s,t);}},
 3=>{match p {
 16=>score=250.,
 7=>{if let Some(u)=&b.u[s] {let missing=u.max-u.hp;let heal=missing.min(if b.t[s].owner==b.p{4.}else{2.});score=heal/u.max*stat(u.k).value*5.+if u.hp/u.max<0.55{18.}else{0.};if ctx.front[s]>5 {score+=if u.hp<0.8*u.max{6.}else{-6.};}if ctx.incoming(u,s,b.n)>=u.hp+heal{score-=35.;}if b.t[s].st==1&&b.t[s].owner!=b.p{score=-100.;}}},
 9=>{score=0.;for &j in &b.adj[s]{if let Some(u)=&b.u[j]{if u.owner==b.p {score+=(u.max-u.hp).min(4.)*stat(u.k).value/u.max*5.;}}}},
 2=>{score=-20.;if b.t[t].res==0&&b.stars<3.&&contact<=3{score=15.;} // preserve population resources
 },
 19=>score=if t==0{10.}else{-100.},18=>score=15.,
 _=>{}}},
 5=>{if let Some(ci)=b.holder[t]{score=growth(&b,ci,1.,2.,&ctx)+5.;}},
 6=>{let cost=buildcost(p);if let Some(ci)=b.holder[t] {
 let mut pop=match p{5|21=>2.,12|8=>1.,23..=29=>3.,_=>0.};
 if p==6||p==13||p==22 {let matchst=if p==6{5}else if p==13{12}else{21};let count=b.adj[t].iter().filter(|&&j|b.t[j].st==matchst&&b.t[j].owner==b.p).count();pop=count as f64*if p==22{2.}else{1.};}
 if p==12 {for &j in &b.adj[t] {if b.t[j].st==13&&b.t[j].owner==b.p{pop+=1.;}}}
 if pop>0. {score=growth(&b,ci,pop,cost,&ctx);}
 if p==12&&b.t[t].res==1 {score-=30.;}if (p==6||p==13||p==22||p>=23&&p<=29)&&b.t[t].res>0{score-=10.;}
 if (23..=29).contains(&p) {score+=100.;}
 if p==50 {let prod:f64=b.adj[t].iter().filter(|&&j|b.t[j].owner==b.p&&[6,13,22].contains(&b.t[j].st)).map(|&j|b.t[j].sl as f64).sum::<f64>().min(8.);score=prod*13.-8.;}
 }
 if p==8 {let nearport=b.t.iter().enumerate().filter(|(_,x)|x.owner==b.p&&x.st==8).map(|(j,_)|b.d(j,t)).min().unwrap_or(30);let nearunit=b.own.iter().filter(|&&i|!boat(b.u[i].as_ref().unwrap().k)).map(|&i|b.d(i,t)).min().unwrap_or(30);
 let sea=b.adj[t].iter().filter(|&&j|b.t[j].ter<=2).count();score=if sea>=2&&nearport>4&&nearunit<5 {12.+ctx.naval_need*27.+if ports==0.{8.}else{0.}-nearunit as f64*2.}else{-20.};if b.stars<10.&&ports>0.{score-=20.;}if b.t[t].res==3{score-=10.;}}
 if p==3||p==48 {score=roads[t];}
 },
 7=>{let cost=4.+nc*tier(p);let reserve=(if contact<=3{4f64}else{2f64}).min(b.income*0.5);let afford=b.stars-cost;
 score=match p {
 1=>if b.turn>=1 {32.+if units<nc+2.{8.}else{0.}}else{-50.},
 6=>if resources[6]>0.{18.+resources[6]*5.}else{-30.},
 10=>if resources[3]>0.||ctx.naval_need>0.35 {8.+resources[3]*5.+ctx.naval_need*18.}else{-20.},
 16=>if woods>=2. {23.+woods.min(8.)*2.8}else{-30.},
 17=>if huts>=2.||contact<6 {24.+huts.min(10.)*2.+if cats==0.&&contact<5{10.}else{0.}}else{-30.},
 18=>if contact<7 {30.+if contact<4{8.}else{0.}}else{if b.turn>12{14.}else{-10.}},
 20=>{let mountains=b.t.iter().filter(|t|t.known&&t.ter==4).count();if mountains>0 {18.+(mountains.min(15) as f64)*0.8+if b.turn>10{8.}else{0.}}else{-50.}},
 23=>if resources[5]>0. {16.+resources[5]*5.}else{if b.income>=14.{15.}else{-30.}},
 24=>if contact<9 {30.+if swords==0.{7.}else{0.}}else{15.},
 13=>if ports>0.||ctx.naval_need>0.65 {35.+rafts*8.}else{-50.},
 44=>if rafts+ships>0.||ctx.naval_need>0.8 {28.+rafts*5.}else{-40.},
 14=>if ships+rafts>=2. {27.+ships*2.}else{-30.},
 4=>if b.income>=12. {18.}else{-30.},
 5=>if huts>=3.&&b.tech[17] {25.}else{-30.},
 2=>if b.income>=12.&&contact<8 {19.}else{-40.},
 3=>if b.income>=12.&&contact<8 {33.}else{-40.},
 7=>if contact<=3&&units<nc*2. {22.}else if b.income>25.{8.}else{-30.},
 8=>if resources[2]>=2.{18.+resources[2]*4.}else{-50.},
 9=>if b.income>30.&&resources[2]>2.{15.}else{-30.},
 12=>if ships>=3.{23.}else{-40.},
 21|22|38=>if b.income>35. {5.}else{-50.},
 _=>-100.};
 score-=cost*0.85;
 if afford<reserve {score-=18.;}if p==1&&b.own.len()<=2&&b.turn>=2{score+=8.;}
 if b.turn==0 {score=-100.;}if units<nc&&contact<=4&& ![7,18,24].contains(&p) {score-=20.;} if b.stars>cost+15. {score+=10.;} if b.stars>50. {score+=25.;} if b.stars>100. {score+=40.;}
 },
 4=>{let st=stat(p);let front=ctx.front[s];let local=b.own.iter().filter(|&&i|b.d(i,s)<=3).count() as f64;
 if b.u[s].as_ref().map_or(false,|u|u.k==43) {score=match p{42=>45.+ctx.naval_need*10.,44=>42.+if ctx.front[s]<5{12.}else{0.},41=>if b.stars>=20.{if (b.own.iter().filter(|&&j|b.u[j].as_ref().unwrap().k==41).count() as f64) <(ships/3.).max(1.) {65.}else{35.}}else{15.},_=>-100.};if let Some(u)=&b.u[s] {if u.hp<6.{score-=25.;}}}
 else {
 let desired=(nc*2.+2.).max(n_enemy*1.2+2.).min(34.);let need=desired-units;
 score=13.+need*3.5+if front<6 {12.}else{0.}-local*2.0;
 if units<2. {score+=25.;}if units>=desired {score-=25.;}
 if front>9 && units>nc+3. {score-=16.;}
 if units>=40. {score=-100.;}
 score+=match p {
 2=>if b.tech[24]||b.tech[18]||b.tech[1] {-12.}else{0.},
 3=>if riders<nc+1. {6.}else{-8.},
 9=>if front<=7 {if b.own.iter().filter(|&&i|b.u[i].as_ref().unwrap().k==9).count() as f64>units*0.45{1.}else{11.}}else{-3.},
 8=>if cats<(units/4.).max(1.)&&front<=10 {29.}else{-20.},
 11=>if front<9 {25.}else{10.},
 4=>if knights<(units/7.).max(1.) {16.}else{-10.},
 5=>if front<3&&local<3. {5.}else if ctx.naval_need>0.7&&ports>0. {3.}else{-20.},
 10=>-35.,38=>-40.,_=>-40.};
 score-=st.cost*1.3; if b.stars>20. {score+=7.;}
 if b.stars<st.cost+2.&&units>=2.&&contact>4{score-=8.;}
 if b.turn==0 {score=-100.;}
 // Defensive recruitment must actually withstand the visible attackers.
 let nu=Unit{tile:s,owner:b.p,k:p,hp:st.hp,max:st.hp,..Unit::default()};let risk=ctx.incoming(&nu,s,b.n);
 if front<=2 {score+=12.;if risk>=st.hp{score-=30.;}if p==8 {score-=35.;}}
 }
 },
 1=>{if let Some(u)=&b.u[s] {
 if b.t[s].st==1&&b.t[s].owner!=b.p&&b.t[s].owner>0&&!u.moved&&!u.attacked{score=-200.;}
 else {
 let paths=pathcache.entry((t,u.k)).or_insert_with(||b.paths(t,u.k));
 let (pot,goal)=ctx.potential(u,t,paths);let oldpot=srcpot[s].0;
 score=5.+(pot-oldpot)*10.;
 let g=srcpot[s].1;let olddist=ctx.dist[s][g];let newdist=paths[g];if olddist<500.&&newdist<500. {score+=(olddist-newdist)*3.;}
 let mut nu=u.clone();nu.tile=t;
 if b.t[t].ter<=2&&!boat(u.k)&&b.t[t].st==8 {nu.k=if u.k==12{45}else{43};}
 if b.t[t].ter>2&&boat(u.k){nu.k=if u.pass>0{u.pass}else if u.k==45{12}else{2};}
 let risk=ctx.risk(&nu,t,b.n);let oldrisk=ctx.risk(u,s,b.n);score-=risk*0.85;score+=oldrisk*0.45;
 if !u.attacked&&stat(nu.k).dash {let mut atk=-1e3f64;for &e in &b.enemy{if b.d(t,e)<=stat(nu.k).range {atk=atk.max(ctx.attack_score(&nu,t,e));}}
 if atk>0. {score+=atk*0.78;}}
 if bestatt[s]>0. {score-=bestatt[s]*0.65;}
 if b.t[t].st==1&&b.t[t].owner!=b.p {score+=if b.t[t].owner==0 {55.}else{68.};if ctx.incoming(&nu,t,b.n)>=nu.hp {score-=25.;}}
 if b.t[t].st==2 {score+=32.;}
 if boat(u.k)&&u.k!=43&&b.t[t].ter>2&&b.t[t].st!=1&&b.t[t].st!=2 {
 score-=if u.k==41 {35.}else if u.k==45 {5.}else{14.};
 }
 if u.k==45 {let splash:f64=b.adj[t].iter().filter_map(|&j|b.u[j].as_ref()).filter(|e|e.owner!=b.p).map(|e|{let (d,_)=b.damage(&nu,t,e,e.tile);d.min(e.hp)*0.5/e.max*stat(e.k).value*5.}).sum();score+=splash;}

 if b.t[s].st==1&&b.t[s].owner==b.p {score+=5.; // freeing recruitment
 let exposed=b.enemy.iter().any(|&e| b.d(e,s)<=stat(b.u[e].as_ref().unwrap().k).mv);if exposed&&b.d(s,t)>1 {score-=25.;}}
 // hurt units prefer recovery unless the move escapes danger or secures a village
 if !u.attacked&&u.hp<u.max*0.7 && b.t[t].st!=1 {score-=if b.t[s].owner==b.p{16.}else{8.};}
 if mem.prev[s]==t&&mem.last[s]>=b.turn-1 {score-=12.;}
 if u.k==8||u.k==41 {if ctx.front[t]<2 {score-=25.;}if bestatt[s]>5. {score-=60.;}}
 if b.t[s].st==1&&b.t[s].owner==b.p&&ctx.front[s]<=2&&b.stars<2.&&local_defenders(&b,s)<2 {score-=25.;}
 // Don't shuffle without accomplishing anything.
 if pot-oldpot<0.02&&newdist>=olddist&&oldrisk<=risk&&b.t[t].st!=1&&b.t[t].st!=2 {score-=8.;}
 let _=goal;
 }
 }},
 _=>{}
 }
 // deterministic positional tie breaker
 score+=( ((s*17+t*7+p.max(0) as usize*3)%23) as f64)*0.0001;
 if score>best {best=score;chosen=ai;}
 }
 if let Some((ai,sc))=combo_plan(&b,&ctx,&actions) {if sc>best {chosen=ai;best=sc;}}
 if let Some((ai,sc))=knight_plan(&b,&ctx,&actions) {if sc>best {chosen=ai;best=sc;}}
 let a=&actions[chosen];if a.k==1 {mem.prev[a.t]=a.s;mem.last[a.t]=b.turn;mem.prev[a.s]=b.n;}
 #[cfg(not(target_arch="wasm32"))] {if std::env::var("BOT_DEBUG").is_ok(){eprintln!("turn {} stars {} best {:.2}: {} {} {} {}",b.turn,b.stars,best,a.k,a.s,a.t,a.p);}}
 a.id.clone()
}
fn local_defenders(b:&Board,s:usize)->usize {b.own.iter().filter(|&&i|b.d(s,i)<=2).count()}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){if let Ok(line)=line{if let Ok(obs)=serde_json::from_str::<Value>(&line){let id=choose(&obs,&mut mem);writeln!(out,"{}",json!({"action_id":id})).unwrap();out.flush().unwrap();}}}}
fn road_plan(b:&Board,acts:&[Action])->Vec<f64> {
 let mut out=vec![-100.;b.n];if !b.tech[4] {return out;}
 let cap=b.cities.iter().find(|c|c.owner==b.p&&b.t[c.tile].capital==b.p).map(|c|c.tile);let Some(cap)=cap else{return out;};
 let mut buildable=vec![false;b.n];for a in acts {if a.k==6&&(a.p==3||a.p==48){buildable[a.t]=true;}}
 let mut ds=vec![1e6f64;b.n];let mut par=vec![b.n;b.n];let mut done=vec![false;b.n];ds[cap]=0.;
 for _ in 0..b.n {let mut i=b.n;let mut best=1e6;for j in 0..b.n {if !done[j]&&ds[j]<best{best=ds[j];i=j;}}if i==b.n{break;}done[i]=true;
 for &j in &b.adj[i] {let t=&b.t[j];if !t.known||t.owner>0&&t.owner!=b.p{continue;}let node=t.road||t.st==1||t.st==48||t.st==8;let cost=if node{0.01}else if buildable[j] {if t.ter<=2 {5.01}else{3.01}}else{continue;};if best+cost<ds[j]{ds[j]=best+cost;par[j]=i;}}
 }
 for c in &b.cities {if c.owner!=b.p||c.tile==cap||c.connected||ds[c.tile]>b.stars.min(18.) {continue;}
 let mut path=vec![];let mut i=c.tile;for _ in 0..b.n{if i==cap||i==b.n{break;}if buildable[i]&&!b.t[i].road {path.push(i);}i=par[i];}
 if path.is_empty(){continue;}
 let value=40./path.len() as f64+if c.progress>=c.level {22.}else{0.}+if b.stars>20.{12.}else{0.};
 for &i in &path {out[i]=out[i].max(value);}
 }
 // Forward military supply roads. Place paired roads only where armies are traveling.
 if b.stars>=12. {
 for i in 0..b.n {if !buildable[i]||b.t[i].ter<=2{continue;}let army=b.own.iter().filter(|&&j|b.d(i,j)<=1).count();let roadadj=b.adj[i].iter().any(|&j|b.t[j].road||b.t[j].st==1);let enemy=b.enemy.iter().map(|&j|b.d(i,j)).min().unwrap_or(30);
 if army>=3&&roadadj&&enemy>=2&&enemy<=7 {out[i]=out[i].max(5.+army as f64*1.5);}}
 }
 out
}
#[derive(Clone)]struct Strike {ai:usize,src:usize,pos:usize,travel:bool,eff:f64}
fn combo_plan(b:&Board,ctx:&Context,acts:&[Action])->Option<(usize,f64)> {
 if b.enemy.is_empty(){return None;}let mut best:Option<(usize,f64)>=None;
 for &target in &b.enemy {
 let defender=b.u[target].as_ref().unwrap();if defender.hp<=5. {continue;}
 let mut options:Vec<Option<Strike>>=vec![None;b.n];
 for (ai,a) in acts.iter().enumerate() {
 if !(a.k==2&&a.t==target||a.k==1){continue;}
 let Some(u)=&b.u[a.s] else {continue;};let st=stat(u.k);if u.attacked||st.atk<=0. {continue;}
 let pos=if a.k==1{a.t}else{a.s};if a.k==1&&(!st.dash||b.d(pos,target)>st.range||b.t[pos].ter<=2&&!boat(u.k)||b.t[pos].ter>2&&boat(u.k)){continue;}
 let (dm,rt)=b.damage(u,pos,defender,target);let eff=dm-rt*st.value/u.max*0.5-ctx.risk(u,pos,target)*0.08-if a.k==1{1.}else{0.};
 if options[a.s].as_ref().map_or(true,|s|eff>s.eff){options[a.s]=Some(Strike{ai,src:a.s,pos,travel:a.k==1,eff});}
 }
 let mut strikes:Vec<Strike>=options.into_iter().flatten().collect();if strikes.len()<2{continue;}
 strikes.sort_by(|x,y|y.eff.partial_cmp(&x.eff).unwrap());strikes.truncate(7);
 let mut solver=Combo{b,ctx,target,strikes,best:-1e6,first:0};
 solver.search(defender.hp,0,0.,0.,0,0);
 if solver.best>0.&&best.map_or(true,|(_,v)|solver.best>v){best=Some((solver.first,solver.best));}
 }best
}
struct Combo<'a,'b> {b:&'a Board,ctx:&'a Context<'b>,target:usize,strikes:Vec<Strike>,best:f64,first:usize}
impl Combo<'_,'_> {
 fn search(&mut self,hp:f64,used:u32,loss:f64,tempo:f64,first:usize,depth:i32) {
 let d=self.b.u[self.target].as_ref().unwrap();let sd=stat(d.k);
 if hp<=0. {
 let mut gain=10.+(d.hp/d.max*sd.value*0.9+sd.value*0.65+2.)*5.;
 if self.b.t[self.target].st==1 {gain+=if self.b.t[self.target].owner==self.b.p {65.}else{30.};}
 let score=gain-loss-tempo;
 if score>self.best{self.best=score;self.first=first;}return;
 }
 if depth>=4 {return;}
 for j in 0..self.strikes.len() {if used&(1<<j)!=0{continue;}let s=self.strikes[j].clone();let u=self.b.u[s.src].as_ref().unwrap();let st=stat(u.k);let mut def=d.clone();def.hp=hp;let(dm,rt)=self.b.damage(u,s.pos,&def,self.target);if dm<=0.{continue;}
 let mut cost=rt.min(u.hp)/u.max*st.value*4.5;if rt>=u.hp {cost+=st.value*3.5;}else{let mut after=u.clone();after.hp-=rt;let at=if dm>=hp&&st.range==1{self.target}else{s.pos};cost+=self.ctx.risk(&after,at,self.target)*0.3;}
 // Spend each attack only once; discourage making a multi-piece exchange for a cheap target.
 let extra=3.+if s.travel{2.}else{0.};
 if loss+cost+tempo+extra>sd.value*9.+65. {continue;}
 self.search(hp-dm,used|(1<<j),loss+cost,tempo+extra,if depth==0{s.ai}else{first},depth+1);
 }
 }
}
fn knight_plan(b:&Board,ctx:&Context,acts:&[Action])->Option<(usize,f64)> {
 let mut best=None;
 for (ai,a) in acts.iter().enumerate() {
 if a.k!=1&&a.k!=2{continue;}let Some(u)=&b.u[a.s] else{continue;};if u.k!=4||u.attacked {continue;}
 let pos=if a.k==1{a.t}else{a.s};
 let mut targets=vec![];if a.k==2 {targets.push(a.t);}else {for &j in &b.enemy{if b.d(pos,j)==1{targets.push(j);}}}
 for t in targets {let d=b.u[t].as_ref().unwrap();if b.damage(u,pos,d,t).0<d.hp {continue;}if b.t[t].ter<=2||b.t[t].ter==4&&!b.tech[20]{continue;}
 let mut killed=vec![false;b.n];killed[t]=true;
 let val=10.+(d.hp/d.max*stat(d.k).value*0.9+stat(d.k).value*0.65+2.)*5.;
 let mut search=Knight{nodes:0,b,ctx,u,best:-1e6};search.dfs(t,&mut killed,val,1);
 let sc=search.best-if a.k==1{5.}else{0.};if best.map_or(true,|(_,v)|sc>v){best=Some((ai,sc));}
 }
 }best
}
struct Knight<'a,'b>{nodes:usize,b:&'a Board,ctx:&'a Context<'b>,u:&'a Unit,best:f64}
impl Knight<'_,'_>{
 fn dfs(&mut self,pos:usize,killed:&mut [bool],value:f64,depth:i32){self.nodes+=1;if self.nodes>2000{return;}
 let mut risk=0.;for &j in &self.ctx.threat[pos]{if killed[j]{continue;}let e=self.b.u[j].as_ref().unwrap();risk+=self.b.damage(e,j,self.u,pos).0*0.75;}
 let score=value-risk*2.-if risk>=self.u.hp{35.}else{0.}+if self.b.t[pos].st==1 {35.}else{0.};self.best=self.best.max(score);
 if depth>=7{return;}for &j in &self.b.adj[pos]{if killed[j]{continue;}let Some(e)=&self.b.u[j] else{continue;};if e.owner==self.u.owner||self.b.t[j].ter<=2||self.b.t[j].ter==4&&!self.b.tech[20]{continue;}
 if self.b.damage(self.u,pos,e,j).0<e.hp{continue;}killed[j]=true;let st=stat(e.k);let gain=(e.hp/e.max*st.value*0.9+st.value*0.65+2.)*5.;self.dfs(j,killed,value+gain,depth+1);killed[j]=false;
 }
 }
}
