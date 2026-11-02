use serde_json::{Value,json};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn iv(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn bv(v:&Value,k:&str)->bool{v[k].as_bool().unwrap_or(false)}
#[derive(Clone,Copy,Default)]
struct Tile{known:bool,terrain:i32,owner:i32,st:i32,res:i32,road:bool,cap:i32,sl:i32}
#[derive(Clone,Copy,Default)]
struct Unit{pos:usize,owner:i32,k:i32,hp:f32,max:f32,moved:bool,attacked:bool,kills:i32,home:i32,pass:i32,poison:bool}
#[derive(Clone,Default)]
struct City{pos:usize,owner:i32,level:i32,progress:i32,prod:i32,rewards:[i32;9],connected:bool}
#[derive(Clone)]
struct Act{id:String,k:i32,s:usize,t:usize,p:i32}
#[derive(Clone,Copy)]
struct Stats{a:f32,d:f32,h:f32,c:f32,m:i32,r:i32,dash:bool,stiff:bool,fort:bool}
fn stats(k:i32)->Stats{
 let (a,d,h,c,m,r,dash,stiff,fort)=match k{
 2=>(2.,2.,10.,2.,1,1,true,false,true),3=>(2.,1.,10.,3.,2,1,true,false,true),4=>(3.5,1.,10.,8.,3,1,true,false,true),5=>(1.,3.,15.,3.,1,1,false,false,true),8=>(4.,0.,10.,8.,1,3,false,true,false),9=>(2.,1.,10.,3.,1,2,true,false,true),10=>(0.,1.,10.,5.,1,1,false,true,false),11=>(3.,3.,15.,5.,1,1,true,false,false),12=>(5.,4.,40.,10.,1,1,false,false,false),37=>(2.,2.,10.,2.,1,1,true,false,false),38=>(0.,0.5,5.,8.,2,1,true,true,false),39=>(0.,0.5,5.,8.,2,1,true,true,false),40=>(2.,2.,10.,2.,2,1,true,false,false),41=>(3.,2.,10.,15.,2,3,false,true,false),42=>(2.,1.,10.,5.,3,2,true,false,false),43=>(0.,1.,10.,0.,2,2,false,true,false),44=>(3.,3.,10.,5.,3,1,true,false,false),45=>(4.,4.,40.,10.,2,1,false,true,false),_=>(2.,2.,10.,3.,1,1,true,false,true)};
 Stats{a,d,h,c,m,r,dash,stiff,fort}
}
fn boat(k:i32)->bool{(39..=45).contains(&k)}
fn value(u:&Unit)->f32{if u.k==12||u.k==45{36.}else{5.+stats(u.k).c*2.+if boat(u.k){stats(u.pass).c.max(2.)}else{0.}}}
struct Board{size:usize,n:usize,p:i32,turn:i32,stars:f32,tiles:Vec<Tile>,units:Vec<Unit>,at:Vec<Option<usize>>,cities:Vec<City>,ci:Vec<Option<usize>>,tech:[bool;128],acts:Vec<Act>,adj:Vec<Vec<usize>>,dist:Vec<i32>,own:Vec<usize>,enemy:Vec<usize>,income:f32,cityof:Vec<usize>,walls:Vec<bool>,enemytech:[bool;128]}
struct Memory{lastcmd:Option<(i32,usize,usize,i32)>,enemytech:[bool;128],lastturn:i32,cityof:Vec<usize>,lastowners:Vec<i32>,walls:Vec<bool>,lastattack:Option<(usize,f32,f32,f32)>,turn:i32,visits:Vec<i32>,lastmove:Option<(usize,usize)>,counter:usize}
impl Default for Memory{fn default()->Self{Self{lastcmd:None,enemytech:[false;128],lastturn:-1,cityof:vec![],lastowners:vec![],walls:vec![],lastattack:None,turn:0,visits:vec![],lastmove:None,counter:0}}}
impl Board{
 fn new(o:&Value,mem:&mut Memory)->Self{
 let v=&o["visible_state"]; let size=iv(v,"board_size") as usize;let n=size*size;let p=iv(o,"player");
 let mut tiles=vec![Tile::default();n];let mut adj=vec![vec![];n];let mut dist=vec![0;n*n];
 for i in 0..n{let (x,y)=((i%size) as i32,(i/size) as i32);for j in 0..n{let d=(x-(j%size) as i32).abs().max((y-(j/size) as i32).abs());dist[i*n+j]=d;if d==1{adj[i].push(j)}}}
 for t in v["tiles"].as_array().unwrap(){let i=iv(t,"tile") as usize;tiles[i]=Tile{known:bv(t,"known"),terrain:iv(t,"terrain"),owner:iv(t,"owner"),st:iv(t,"structure"),res:iv(t,"resource"),road:iv(t,"road")>0,cap:iv(t,"capital"),sl:iv(t,"structure_level")};}
 let mut tech=[false;128];for t in v["technology"].as_array().unwrap(){tech[t.as_u64().unwrap_or(0) as usize]=true;}
 let mut units=vec![];let mut at=vec![None;n];let mut own=vec![];let mut enemy=vec![];
 for u in v["units"].as_array().unwrap(){let pos=iv(u,"tile") as usize;let k=iv(u,"kind");let pass=iv(u,"passenger");let hp=u["health"].as_f64().unwrap_or(10.) as f32;let max=stats(if pass>0{pass}else{k}).h+if bv(u,"veteran"){5.}else{0.};let idx=units.len();at[pos]=Some(idx);let owner=iv(u,"owner");if owner==p{own.push(idx)}else{enemy.push(idx)};units.push(Unit{pos,owner,k,hp,max:if owner!=p&&boat(k)&&hp>10.&&hp<=15.{15.}else{max.max(hp)},moved:bv(u,"moved"),attacked:bv(u,"attacked"),kills:iv(u,"kills"),home:iv(u,"home"),pass,poison:bv(u,"poison")});}
 let mut cities=vec![];let mut ci=vec![None;n];let mut income=0.;for c in v["cities"].as_array().unwrap(){let pos=iv(c,"tile") as usize;let mut rewards=[0;9];if let Some(a)=c["rewards"].as_array(){for i in 0..9{rewards[i]=a[i].as_i64().unwrap_or(0) as i32;}}let city=City{pos,owner:iv(c,"owner"),level:iv(c,"level").max(1),progress:iv(c,"progress"),prod:iv(c,"production"),rewards,connected:bv(c,"connected")};if city.owner==p{income+=city.prod as f32+if tiles[pos].cap==p{1.}else{0.}}ci[pos]=Some(cities.len());cities.push(city);}
 for t in &tiles{if t.owner==p&&t.st==50{income+=t.sl as f32;}if t.owner>0&&t.owner!=p{let tech=match t.st{5=>8,6=>9,8=>10,12=>16,13=>17,21=>23,22=>24,50=>5,_=>0};mem.enemytech[tech]=true;}}
 for &i in &enemy{let u=units[i];let tech=match u.k{3=>1,4=>3,5=>7,8=>17,9=>18,10=>22,11=>24,38|39=>38,41=>14,42=>13,44=>44,_=>0};mem.enemytech[tech]=true;if tiles[u.pos].terrain==4{mem.enemytech[20]=true;}}
 if mem.lastturn!=iv(v,"turn"){for x in &mut mem.visits{*x=(*x-1).max(0);}mem.lastturn=iv(v,"turn");}
 if mem.cityof.len()!=n{mem.cityof=vec![n;n];mem.lastowners=vec![0;n];mem.walls=vec![false;n];mem.visits=vec![0;n];}
 if let Some((t,h,d1,d4))=mem.lastattack.take(){if let Some(ui)=at[t]{let delta=h-units[ui].hp;if delta>0.&&(delta-d4).abs()+0.1<(delta-d1).abs(){mem.walls[t]=true;}}}
 let claim=mem.lastcmd.take().and_then(|(k,s,t,param)|if k==8&&tiles[s].st==1&&tiles[s].owner==p{Some(s)}else if k==9&&param==5{Some(t)}else{None});
 for i in 0..n{if tiles[i].owner==p&&mem.lastowners[i]!=p{if let Some(c)=claim{mem.cityof[i]=c;}}
 if tiles[i].owner==p && (mem.lastowners[i]!=p&&claim.is_none()||mem.cityof[i]>=n||tiles[mem.cityof[i]].owner!=p){let c=cities.iter().filter(|c|c.owner==p).min_by_key(|c|dist[c.pos*n+i]*10+if c.rewards[5]>0{1}else{0});if let Some(c)=c{mem.cityof[i]=c.pos;}}mem.lastowners[i]=tiles[i].owner;}
 let acts=o["legal_actions"].as_array().unwrap().iter().map(|a|{let c=&a["command"];Act{id:a["action_id"].as_str().unwrap().to_owned(),k:iv(c,"kind"),s:iv(c,"source").max(0) as usize,t:iv(c,"target").max(0) as usize,p:iv(c,"parameter")}}).collect();
 Self{size,n,p,turn:iv(v,"turn"),stars:iv(v,"stars") as f32,tiles,units,at,cities,ci,tech,acts,adj,dist,own,enemy,income,cityof:mem.cityof.clone(),walls:mem.walls.clone(),enemytech:mem.enemytech}
 }
 fn d(&self,a:usize,b:usize)->i32{self.dist[a*self.n+b]}
 fn nc(&self)->usize{self.cities.iter().filter(|c|c.owner==self.p).count()}
 fn bonus(&self,u:&Unit,pos:usize)->f32{let t=self.tiles[pos];let own=u.owner==self.p;let mut bonus=1.;if t.terrain==4{bonus=1.5;}if t.terrain==5&&((own&&self.tech[18])||(!own&&self.enemytech[18])){bonus=1.5;}if t.terrain<=2&&((own&&self.tech[12])||(!own&&self.enemytech[12])){bonus=1.5;}if t.st==1&&t.owner==u.owner&&stats(u.k).fort{bonus=if own{if self.ci[pos].map_or(false,|c|self.cities[c].rewards[1]>0){4.}else{1.5}}else if self.walls[pos]{4.}else{1.5};}if u.poison{bonus*=0.5;}bonus}
 fn combat(&self,a:&Unit,apos:usize,d:&Unit,dpos:usize)->(f32,f32){self.combat_bonus(a,apos,d,dpos,self.bonus(d,dpos))}
 fn combat_bonus(&self,a:&Unit,apos:usize,d:&Unit,dpos:usize,bonus:f32)->(f32,f32){let sa=stats(a.k);let sd=stats(d.k);let aa=sa.a*a.hp/a.max;let dd=sd.d*d.hp/d.max*bonus*if d.poison{0.7}else{1.};let sum=(aa+dd).max(0.001);let damage=(aa/sum*sa.a*4.5+0.00001).round().max(0.);let ret=if damage<d.hp&&!sd.stiff&&self.d(apos,dpos)<=sd.r&&a.k!=37&&a.k!=40{(dd/sum*sd.d*4.5+0.00001).round()}else{0.};(damage,ret)}
 // Enemy reachability with visible terrain and zone of control.
 fn reach(&self,u:&Unit)->Vec<usize>{let mut cost=vec![999;self.n];let mut heap=BinaryHeap::new();cost[u.pos]=0;heap.push(Reverse((0,u.pos)));let budget=stats(u.k).m*2;let aquatic=boat(u.k);let creep=u.k==38||u.k==39;let owner=u.owner;
 while let Some(Reverse((c,x)))=heap.pop(){if c!=cost[x]||c>=budget{continue;}if x!=u.pos{let t=self.tiles[x];let zoc=!creep&&self.adj[x].iter().any(|&j|self.at[j].map_or(false,|ui|self.units[ui].owner!=owner));if zoc||t.terrain==4||(t.terrain==5&&!t.road&&!creep)||(t.st==8&&!aquatic)||(aquatic&&t.terrain>2){continue;}}
 for &y in &self.adj[x]{let t=self.tiles[y];if !t.known{continue;}if self.at[y].map_or(false,|i|self.units[i].owner!=owner){continue;}if !aquatic&&t.terrain<=2&&t.st!=48&&!(t.st==8&&t.owner==owner){continue;}if t.terrain==4&&owner==self.p&&!self.tech[20]{continue;}if t.terrain==2&&owner==self.p&&!self.tech[13]{continue;}
 let road=|i:usize|{let t=self.tiles[i];(t.road||t.st==1||t.st==48)&&(t.owner==0||t.owner==owner)};let re=road(x)&&road(y);let mut nc=c+if re{1}else{2};if t.terrain==5&&!re&&!creep{nc=nc.max(budget);}if nc<cost[y]{cost[y]=nc;heap.push(Reverse((nc,y)));}
 }}(0..self.n).filter(|&i|cost[i]<999&&(self.at[i].is_none()||i==u.pos)).collect()}
}
struct Tactics{threat:Vec<Vec<bool>>,ourmoves:Vec<Vec<(usize,usize)>>,direct:Vec<Vec<(usize,usize)>>}
impl Tactics{
 fn new(b:&Board)->Self{let mut enemyreach=vec![vec![];b.units.len()];for &i in &b.enemy{let u=b.units[i];enemyreach[i]=if stats(u.k).dash{b.reach(&u)}else{vec![u.pos]};}
 let mut ourmoves=vec![vec![];b.units.len()];let mut direct=vec![vec![];b.units.len()];for (ai,a) in b.acts.iter().enumerate(){if a.k==1{if let Some(i)=b.at[a.s]{ourmoves[i].push((a.t,ai));}}if a.k==2{if let Some(i)=b.at[a.s]{direct[i].push((a.t,ai));}}}
 let mut threat=vec![vec![false;b.n];b.units.len()];for &ei in &b.enemy{let r=stats(b.units[ei].k).r;for &x in &enemyreach[ei]{for y in 0..b.n{if x!=y&&b.d(x,y)<=r{threat[ei][y]=true;}}}}
 Self{threat,ourmoves,direct}}
 fn incoming(&self,b:&Board,u:&Unit,pos:usize,ignore:usize)->f32{let mut dmg=vec![];for &ei in &b.enemy{if ei==ignore{continue;}let e=&b.units[ei];let es=stats(e.k);if es.a<=0.&&e.k!=10{continue;}let can=self.threat[ei][pos];if can{let damage=if e.k==10{u.hp}else{b.combat(e,e.pos,u,pos).0};let factor=if b.d(e.pos,pos)>es.r{0.88}else{1.};dmg.push(damage*factor);}}dmg.sort_by(|a,c|c.partial_cmp(a).unwrap());dmg.iter().take(4).enumerate().map(|(i,d)|d*if i==0{1.}else{0.85}).sum()}
 fn risk(&self,b:&Board,u:&Unit,pos:usize,ignore:usize)->f32{let damage=self.incoming(b,u,pos,ignore);let val=value(u);let ratio=damage/u.hp.max(1.);val*(0.28*(damage/u.max).min(1.)+0.7*((ratio-0.60)/0.6).clamp(0.,1.))}
}
// Strategic travel costs measured in tenths of a turn; ports change travel mode.
fn travel(b:&Board,u:&Unit,start:usize,reverse:bool)->Vec<i32>{
 let n=b.n;let mut ds=vec![100000;n*2];let mut q=BinaryHeap::new();
 if reverse{for mode in 0..2{ds[start+mode*n]=0;q.push(Reverse((0,start+mode*n)));}}
 else{let mode=if boat(u.k){1}else{0};ds[start+mode*n]=0;q.push(Reverse((0,start+mode*n)));}
 let landk=if boat(u.k){if u.pass>0{u.pass}else if u.k==45{12}else{2}}else{u.k};let lm=stats(landk).m.max(1);let bm=if boat(u.k){stats(u.k).m.max(1)}else{if b.tech[13]{3}else{2}};
 let edge=|x:usize,y:usize,mode:usize|->Option<(usize,i32)>{let a=b.tiles[x];let t=b.tiles[y];if t.known&&t.terrain==4&&!b.tech[20]{return None;}if t.known&&t.terrain==2&&!b.tech[13]{return None;}
 let water=t.known&&t.terrain<=2&&t.st!=48;let mut mode2=mode;let mut cost;
 if mode==0 {if water{if t.st!=8||t.owner!=b.p{return None;}mode2=1;cost=14;}else{cost=10/lm;if t.terrain==4||t.terrain==5{cost=10;}if (a.road||a.st==1)&&(t.road||t.st==1)&&(a.owner==0||a.owner==b.p)&&(t.owner==0||t.owner==b.p){cost=5/lm;}}}
 else {if !water&&t.known {mode2=0;cost=14;}else{cost=10/bm;}}
 if !t.known{cost+=6;}
 if let Some(i)=b.at[y]{if b.units[i].owner!=b.p{cost+=24;}else if b.units[i].pos!=u.pos{cost+=3;}}
 Some((mode2,cost.max(2)))};
 while let Some(Reverse((d,xm)))=q.pop(){if ds[xm]!=d{continue;}let x=xm%n;let mode=xm/n;
 for &y in &b.adj[x]{if !reverse{if let Some((mo,c))=edge(x,y,mode){let j=y+mo*n;if d+c<ds[j]{ds[j]=d+c;q.push(Reverse((d+c,j)));}}}else{for mo in 0..2{if let Some((mt,c))=edge(y,x,mo){if mt==mode{let j=y+mo*n;if d+c<ds[j]{ds[j]=d+c;q.push(Reverse((d+c,j)));}}}}}}
 }
 ds
}
#[derive(Clone)]
struct Mission{goal:usize,kind:i32,importance:f32,path:Vec<i32>}
fn missions(b:&Board,mem:&Memory)->Vec<Mission>{
 let n=b.n;let mut goals:Vec<(usize,i32,f32)>=vec![];
 for c in &b.cities{if c.owner!=b.p{if !b.at[c.pos].map_or(false,|i|b.units[i].owner==b.p){goals.push((c.pos,if c.owner==0{1}else{2},if c.owner==0{47.}else{32.}));}}else{let threat=b.enemy.iter().map(|&i|{let d=b.d(c.pos,b.units[i].pos);if d<4{(4-d) as f32*4.}else{0.}}).sum::<f32>();if threat>0.{goals.push((c.pos,3,18.+threat.min(25.)));}}}
 for i in 0..n{let t=b.tiles[i];if b.tech[14]&&t.res==8&&t.st==0&&!b.at[i].map_or(false,|u|b.units[u].owner==b.p){goals.push((i,6,20.));}if t.st==2&&!b.at[i].map_or(false,|u|b.units[u].owner==b.p){goals.push((i,4,27.));}
 if t.known&&(t.terrain!=4||b.tech[20]){let rad=if t.terrain==4{2}else{1};let unknown=(0..n).filter(|&j|!b.tiles[j].known&&b.d(i,j)<=rad).count();if unknown>0{let edge=(i%b.size).min(i/b.size).min(b.size-1-i%b.size).min(b.size-1-i/b.size);let corner=(0..n).any(|j|!b.tiles[j].known&&b.d(i,j)<=rad&&(j==0||j==b.size-1||j==n-b.size||j==n-1));let unvisited=(0..n).filter(|&j|!b.tiles[j].known&&b.d(i,j)<=rad&&j%b.size>=1&&j/b.size>=1&&j%b.size+1<b.size&&j/b.size+1<b.size&&b.cities.iter().all(|c|b.d(j,c.pos)>=3)).count();
 let imp=3.+unknown as f32*0.65+unvisited as f32*1.5+if corner{3.}else{0.}+if edge>=2{3.}else{0.};goals.push((i,0,imp));}}
 }
 for &e in &b.enemy{let u=b.units[e];if b.tiles[u.pos].st!=1{goals.push((u.pos,5,18.+value(&u)*0.2));}}
 let fallback=b.cities.iter().find(|c|c.owner!=b.p&&c.owner>0).map(|c|c.pos).unwrap_or(n/2+b.size/2);if goals.is_empty(){goals.push((fallback,2,20.));}
 let mut out=vec![Mission{goal:0,kind:0,importance:0.,path:vec![]};b.units.len()];let mut assigned=vec![0.;goals.len()];
 let mut bids=vec![vec![-100000.;goals.len()];b.units.len()];
 for &i in &b.own {let u=&b.units[i];let ds=travel(b,u,u.pos,false);
 for (j,&(g,kind,importance)) in goals.iter().enumerate(){let d=ds[g].min(ds[g+n]);if d>=100000{continue;}let mut imp=importance;
 if kind==6{if !boat(u.k){imp*=0.3;}if u.k==41||u.k==45{imp*=0.3;}}
 if kind==1||kind==4 {if stats(u.k).r>=2&&u.k!=43{imp*=0.65;}if u.hp<5.{imp*=0.7;}}
 if kind==2||kind==3||kind==5{if u.k==12||u.k==45{imp*=1.5;}if u.k==8||u.k==41{imp*=1.2;}if u.hp< u.max*0.4{imp*=0.65;}}
 if kind==0{if u.k==12||u.k==8||u.k==41{imp*=0.5;}if b.turn>20{imp*=0.7;}}
 if kind==3&&g==u.pos{imp*=1.2;}
 bids[i][j]=imp/(1.+d as f32/24.)-mem.visits[g] as f32*if kind==0{0.05}else{0.};
 }}
 let mut taken=vec![false;b.units.len()];
 for _ in 0..b.own.len(){let mut best=-1e10;let mut unit=b.own[0];let mut choice=0;
 for &i in &b.own{if taken[i]{continue;}for (j,&(_,kind,_)) in goals.iter().enumerate(){let penalty=assigned[j]*if kind==1||kind==4||kind==6{24.}else if kind==0{7.}else{3.5};let score=bids[i][j]-penalty;
 if score>best{best=score;choice=j;unit=i;}}}
 let (g,kind,importance)=goals[choice];out[unit]=Mission{goal:g,kind,importance,path:travel(b,&b.units[unit],g,true)};taken[unit]=true;assigned[choice]+=1.;
 if kind==0{for (j,&(g2,k2,_)) in goals.iter().enumerate(){if k2==0&&b.d(g,g2)<=2{assigned[j]+=0.7;}}}
 }

 out
}
fn econ_gain(b:&Board,city:usize,pop:f32)->f32{
 let Some(ci)=b.ci.get(city).copied().flatten() else{return 0.;};let c=&b.cities[ci];if c.owner!=b.p{return 0.;}
 let need=(c.level+1-c.progress).max(1) as f32;let level=c.level;let reward=match level{1=>15.,2=>10.,3=>13.,_=>29.};
 let fraction=(pop/need).min(1.);let mut val=pop*(if level<4{2.5}else{2.0})+reward*fraction;
 if pop>=need{val+=4.;}if level==1{val+=3.;}if b.turn<8{val*=1.25;}
 let danger=b.enemy.iter().map(|&i|b.d(b.units[i].pos,city)).min().unwrap_or(99);if danger<=2&&b.at[city].map_or(false,|i|b.units[i].owner!=b.p){val*=0.5;}val
}
fn buildpop(b:&Board,t:usize,k:i32)->Vec<(usize,f32)>{
 let mut p=vec![];let city=b.cityof[t];let mut base=match k{5|21=>2.,8|12|17|18|19|20=>1.,23..=29=>3.,_=>0.};
 if k==6||k==13||k==22{let req=if k==6{5}else if k==13{12}else{21};base=b.adj[t].iter().filter(|&&j|b.tiles[j].st==req&&b.tiles[j].owner==b.p).count() as f32*if k==22{2.}else{1.};}
 if base>0.{p.push((city,base));}
 if k==12||k==5||k==21{let mul=if k==12{13}else if k==5{6}else{22};for &j in &b.adj[t]{if b.tiles[j].owner==b.p&&b.tiles[j].st==mul{p.push((b.cityof[j],if mul==22{2.}else{1.}));}}}
 p
}
fn costbuild(k:i32)->f32{match k{3|12=>3.,5|6|13|21|22|48|50=>5.,8=>7.,18=>15.,17|19|20=>20.,_=>0.}}
fn econ_score(b:&Board,a:&Act)->f32{
 let t=a.t;let c=b.cityof[t];let cost=if a.k==5{2.}else{costbuild(a.p)};let pops=if a.k==5{vec![(c,1.)]}else{buildpop(b,t,a.p)};
 let val=pops.iter().map(|&(c,p)|econ_gain(b,c,p)).sum::<f32>();let mut score=val*2.-cost*2.0;if b.own.len()>=3&&b.ci.get(c).copied().flatten().map_or(false,|i|b.cities[i].level==1){score+=5.;}
 if a.k==6&&(a.p==17||a.p==18||a.p==19||a.p==20){
 if b.stars>b.income*3.+25.&&b.ci.get(c).copied().flatten().map_or(false,|i|b.cities[i].level+1-b.cities[i].progress<=2){score+=25.;}else{return -100.;}}

 if a.k==6&&a.p==12&&b.tiles[t].res==1{score-=12.;}
 // Save high-adjacency fields for the sawmill, and don't pave over resources.
 if a.k==6&&(a.p==13||a.p==6||a.p==22||a.p==50||a.p>=23&&a.p<=29){if b.tiles[t].res!=0{score-=3.;}let forests=b.adj[t].iter().filter(|&&j|b.tiles[j].terrain==5&&b.tiles[j].owner==b.p).count();if a.p!=13&&forests>=3{score-=4.;}}
 score
}
fn attack_score(b:&Board,tc:&Tactics,ui:usize,apos:usize,di:usize,moving:bool)->f32{
 let u=&b.units[ui];let d=&b.units[di];if u.k==10{return 55.+value(d)*2.;}if u.k==38||u.k==39{return 70.;}
 let (damage,ret)=b.combat(u,apos,d,d.pos);let kill=damage>=d.hp;let die=ret>=u.hp;let mut sc=damage.min(d.hp)/d.max*value(d)*1.8-ret.min(u.hp)/u.max*value(u)*1.35;
 if kill{sc+=value(d)*0.65+7.;if u.kills==2{sc+=6.;}if u.k==4{sc+=12.+b.enemy.iter().filter(|&&j|j!=di&&b.d(b.units[j].pos,d.pos)<=1&&b.units[j].hp<=10.).count() as f32*6.;}}
 if die{sc-=value(u)*0.8+6.;}
 if b.tiles[d.pos].st==1 {sc+=if b.tiles[d.pos].owner==b.p{22.}else if kill{13.}else{5.};}
 let end=if kill&&stats(u.k).r==1&&(! (b.tiles[d.pos].terrain<=2)||boat(u.k)){d.pos}else{apos};
 let mut after=*u;after.hp=(u.hp-ret).max(0.1);let risk=tc.risk(b,&after,end,if kill{di}else{usize::MAX});
 let before=tc.risk(b,u,u.pos,usize::MAX);sc-= (risk-before*0.55).max(0.)*0.85;
 if u.k==3&&!moving{sc+=3.;} // Escape preserves riders.
 if u.k==41{for &ei in &b.enemy{let e=&b.units[ei];if ei!=di&&b.d(e.pos,d.pos)==1{let dmg=(b.combat(u,apos,e,e.pos).0*0.5).floor();sc+=dmg.min(e.hp)/e.max*value(e)*1.6;if dmg>=e.hp{sc+=8.;}}}}
 sc+5.
}
fn tech_tier(k:usize)->f32{match k{1|6|10|15|20=>1.,2|4|7|8|13|16|18|21|23|44=>2.,_=>3.}}
fn tech_cost(b:&Board,k:usize)->f32{let c=4.+b.nc() as f32*tech_tier(k);if b.tech[22]{(c*0.66666).ceil()}else{c}}
fn tech_priority(b:&Board,k:usize)->f32{
 let n=b.n;let mut fruit:f32=0.;let mut fish:f32=0.;let mut forest:f32=0.;let mut metal:f32=0.;let mut crop:f32=0.;let mut huts:f32=0.;let mut water:f32=0.;let mut mountain:f32=0.;let mut ports:f32=0.;
 for t in &b.tiles{if t.known{if t.terrain<=2{water+=1.;}if t.terrain==4{mountain+=1.;}if t.owner==b.p{if t.st==0{match t.res{6=>fruit+=1.,3=>fish+=1.,5=>metal+=1.,2=>crop+=1.,_=>{}}if t.terrain==5{forest+=1.;}}if t.st==12{huts+=1.;}if t.st==8{ports+=1.;}}}}
 let near=b.own.iter().flat_map(|&i|b.enemy.iter().map(move|&j|b.d(b.units[i].pos,b.units[j].pos))).min().unwrap_or(30);
 let nc=b.nc() as f32;let nu=b.own.len() as f32;let wealth=b.stars+b.income*1.5;let navy=b.own.iter().filter(|&&i|boat(b.units[i].k)).count() as f32;
 let mut pr=match k{
 1=>if b.turn>=2&&nu>=2.{26.+if near<6{8.}else{0.}}else{0.},
 6=> if fruit>=2.{20.+fruit.min(6.)*2.}else if b.turn>15&&b.income>12.{8.}else{0.},
 16=>if forest>=3.{25.+forest.min(9.)*1.5+if b.turn>=5{4.}else{0.}}else if forest>=2.&&b.turn>=4{19.}else{0.},
 17=>if huts>=3.{25.+huts.min(10.)*1.8+if near<6{10.}else{0.}}else if b.income>12.&&near<6{20.}else{0.},
 20=>if mountain>=3.&&b.turn>=3{19.+(mountain*0.7).min(9.)}else{0.},
 10=>if fish>=2.{23.+fish.min(6.)*1.8}else if water>=n as f32*0.12&&b.turn>=6{28.}else{0.},
 13=>if ports>0.||navy>0.{37.}else if water>n as f32*0.22&&b.turn>=10{27.}else{0.},
 44=>if navy>=1.&&b.income>=10.{33.}else{0.},
 14=>if navy>=2.&&b.income>=13.{46.}else if navy>=1.&&b.turn>25{30.}else{0.},
 12=>if navy>=4.{20.}else{0.},
 18=>if near<5&&b.income>=9.{21.+b.enemy.iter().filter(|&&i|b.units[i].k==12||b.units[i].k==5).count() as f32*2.}else{0.},
 23=>if metal>=2.{25.+metal*2.}else if near<6&&wealth>38.&&b.income>=13.{27.}else{0.},
 24=>if b.income>=13.&&near<8{42.}else{0.},
 8=>if crop>=3.{22.+crop*1.5}else{0.},
 9=>if b.tiles.iter().filter(|t|t.owner==b.p&&t.st==5).count()>=3{24.}else{0.},
 4=>if nc>=3.&&b.income>=11.{23.}else if nc>=2.&&b.turn>=10&&b.income>=8.{18.}else{0.},
 5=>if b.tiles.iter().filter(|t|t.owner==b.p&&(t.st==13||t.st==22||t.st==6)).count()>=2{27.}else{0.},
 2=>if b.income>=18.&&near<7{23.}else{0.},
 3=>if b.income>=18.&&near<7{35.}else{0.},
 7=>if ports>0.&&b.income>=15.{17.}else{0.},
 21=>if b.income>=30.{9.}else{0.},22=>if b.income>=30.{17.}else{0.},38=>if b.income>=35.{6.}else{0.},_=>0.};
 if b.turn<2{pr=0.;}
 // Do not research more infrastructure while the previous unlock has not paid back.
 if b.turn<8 && matches!(k,1|6|10|20) && b.tech[16] && huts<2. {pr-=12.;}
 if b.turn<6 && matches!(k,20|1) && nu<3. {pr-=4.;}
 if k==4 && b.nc()>=2 {pr+=7.;}
 if nu<2.&&b.turn<6{pr*=0.35;}if near<=4&&nu<nc+2.{pr-=10.;}
 let cost=tech_cost(b,k);pr-=cost*0.45;
 // Keep a few stars to exploit the new technology immediately.
 if b.stars<cost+3.&&near<=3{pr-=12.;}
 pr
}
fn prerequisites(k:usize)->usize{match k{2|4=>1,3=>2,5=>4,7|8=>6,9=>8,13|44=>10,14|12=>44,16|18=>15,17=>16,19=>18,21|23=>20,22=>21,24=>23,38=>7,_=>0}}
fn port_score(b:&Board,t:usize)->f32{
 let mut seen=vec![false;b.n];let mut stack=vec![t];seen[t]=true;let mut water=0;let mut fog=0;
 while let Some(i)=stack.pop(){water+=1;for &j in &b.adj[i]{if !b.tiles[j].known{fog+=1;}if !seen[j]&&b.tiles[j].known&&b.tiles[j].terrain<=2{seen[j]=true;stack.push(j);}}}
 if water<3{return -100.;}
 let existing=b.tiles.iter().enumerate().filter(|&(i,v)|v.st==8&&v.owner==b.p&&seen[i]).map(|(i,_)|b.d(i,t)).min().unwrap_or(99);
 if existing<=4{return -80.;}
 let nearland=b.own.iter().filter(|&&i|!boat(b.units[i].k)).map(|&i|b.d(b.units[i].pos,t)).min().unwrap_or(99);
 let reachable_enemies=b.cities.iter().filter(|c|c.owner>0&&c.owner!=b.p).map(|c|b.d(t,c.pos)).min().unwrap_or(8);
 let wc=b.tiles.iter().filter(|t|t.known&&t.terrain<=2).count();let known=b.tiles.iter().filter(|t|t.known).count();let frac=wc as f32/known.max(1) as f32;
 let mut score=if existing==99{if frac>0.25{31.}else{13.}}else{5.};
 score+= (fog.min(16) as f32)*0.5+(10-reachable_enemies).max(0) as f32*1.2;
 score-=nearland.min(8) as f32*1.8;
 if b.turn<6{score-=30.;}if b.stars<10.&&b.income<10.{score-=12.;}
 if b.tiles[t].res==3{score-=12.;}
 score
}
fn train_score(b:&Board,tc:&Tactics,a:&Act)->f32{
 let pos=a.s;let k=a.p;let nu=b.own.len() as f32;let nc=b.nc() as f32;let near=b.enemy.iter().map(|&i|b.d(pos,b.units[i].pos)).min().unwrap_or(30);
 if b.at[pos].map_or(false,|i|b.units[i].k==43){let u=b.units[b.at[pos].unwrap()];let navy=b.own.iter().filter(|&&i|boat(b.units[i].k)&&b.units[i].k!=43).count();let bombers=b.own.iter().filter(|&&i|b.units[i].k==41).count();let mut val=match k{42=>if navy>=3{26.}else{33.},44=>if navy>=2{38.}else{31.},41=>if bombers*2<=navy{42.}else{25.},_=>-100.};if k==42&&navy<3&&b.tiles.iter().filter(|t|!t.known).count()>b.n/3{val+=10.;}if near<=4{val+=12.;}if k==44&&b.enemy.iter().any(|&i|b.d(pos,b.units[i].pos)<=5&&matches!(b.units[i].k,41|42)){val+=7.;}if u.hp<6.{val-=18.;}return val;}
 let around=b.own.iter().filter(|&&i|b.d(pos,b.units[i].pos)<=4).count() as f32;
 let enemies=b.enemy.iter().filter(|&&i|b.d(pos,b.units[i].pos)<=5).count() as f32;
 let nearbyvillages=b.cities.iter().filter(|c|c.owner==0&&b.d(pos,c.pos)<=5).count();
 let count=b.own.iter().filter(|&&i|b.units[i].k==k).count() as f32;
 let mut desired=nc*1.3+2.;if !b.enemy.is_empty(){desired=nc*1.7+4.+(b.enemy.len() as f32)*0.65;}
 desired=desired.min(if b.enemy.len()>18||b.stars>b.income*2.{40.}else{30.});
 if nu>=desired&&near>4{return -50.;}
 if around>=8.&&near>4{return -50.;}
 let mut val=15.+(desired-nu).clamp(-4.,5.)*1.4+(6-near).max(0) as f32*3.;
 if nu<2.{val+=24.;}if nu<3.&&b.turn>=3&&b.turn<8{val+=8.;}if b.turn==0{val-=30.;}if nearbyvillages>0&&around<2.{val+=10.;}
 val-=stats(k).c*0.7;
 if near>7&&around>3.{val-=16.;}if b.turn>10&&near>6&&nu>nc*1.5{val-=8.;}
 match k{
 2=>{if b.tech[1]&&near>3{val-=6.;}if b.tech[24]{val-=14.;}if near<4{val+=1.;}},
 3=>{val+=if near>3{7.}else{3.};if count>nc*1.2+2.{val-=12.;}if b.tech[24]&&near<5{val-=5.;}},
 4=>{val+=9.;if count>nc*0.5+1.{val-=12.;}if b.enemy.iter().any(|&i|b.d(pos,b.units[i].pos)<7&&matches!(b.units[i].k,8|9|41|42)){val+=8.;}},
 5=>{val-=12.;let ports=b.tiles.iter().enumerate().any(|(i,t)|t.st==8&&t.owner==b.p&&b.d(pos,i)<=2);if ports{val+=24.;}if ports&&count<nc*0.6&&b.tech[13]{val+=6.;}if near<=2{val+=10.;}if near<=3&&b.own.iter().any(|&i|b.units[i].k==8&&b.d(b.units[i].pos,pos)<=3){val+=9.;}},
 8=>{val+=if near<=6{12.}else{-18.};if near<=2{val-=15.;}if count>nc*0.65+1.{val-=25.;}if count>=b.own.len() as f32*0.3{val-=14.;}if b.enemy.iter().any(|&i|b.d(pos,b.units[i].pos)<=6&&matches!(b.units[i].k,12|5|11)){val+=5.;}},
 9=>{val+=if near<=5{7.}else{-7.};if count>nc+1.{val-=17.;}if b.tech[24]{val-=5.;}},
 11=>{val+=13.;if near>9{val-=5.;}},
 10=>{val-=22.;let giants=b.enemy.iter().filter(|&&i|b.d(pos,b.units[i].pos)<=5&&matches!(b.units[i].k,12|45)).count();if giants>0&&count<2.{val+=30.;}if count>=2.{val-=30.;}},38=>{val-=22.;},_=>val-=30.
 }
 if near<=3&&enemies>around{val+=15.;}if near<=4&&nu<nc+2.{val+=9.;}
 let u=Unit{pos,owner:b.p,k,hp:stats(k).h,max:stats(k).h,..Unit::default()};let inc=tc.incoming(b,&u,pos,usize::MAX);
 if inc>=u.hp {val-=15.+stats(k).c; if near<=1&&b.nc()==1{val+=15.;}}
 val
}
fn move_score(b:&Board,tc:&Tactics,ms:&[Mission],mem:&Memory,a:&Act)->f32{
 let Some(ui)=b.at[a.s] else{return -1000.;};let u=&b.units[ui];let t=a.t;let s=a.s;let st=stats(u.k);let tile=b.tiles[t];let src=b.tiles[s];
 if src.st==1&&src.owner!=b.p{return -1000.;}if src.st==2&&!u.attacked&&!u.moved{return -1000.;}
 let m=&ms[ui];let mode=if boat(u.k){1}else{0};let nextmode=if mode==0&&tile.st==8{1}else if mode==1&&tile.terrain>2{0}else{mode};
 let ds=m.path.get(s+mode*b.n).copied().unwrap_or(100000);let dt=m.path.get(t+nextmode*b.n).copied().unwrap_or(100000);
 let incoming=tc.incoming(b,u,t,usize::MAX);
 let survival=if incoming<u.hp*0.55{1.}else if incoming<u.hp{0.65}else if incoming<u.hp*1.4{0.18}else{0.02};
 let mut score=-3.;
 if dt<ds{score+=6.+(ds-dt).min(25) as f32*0.6+ m.importance/(dt as f32/10.+3.)*0.18;}else if dt==ds{score-=2.;}else{score-=7.+((dt-ds) as f32*0.4).min(15.);}
 if m.kind==1||m.kind==4||m.kind==6{if t==m.goal{score+=28.*survival;}}
 if tile.st==1&&tile.owner!=b.p{score+=(if tile.owner==0{32.}else{42.})*survival;}if tile.st==2{score+=24.*survival;}if tile.res==8&&b.tech[14]&&tile.st==0{score+=15.*survival;}
 let radius=if tile.terrain==4||u.k==42||u.k==38||u.k==39{2}else{1};let reveal=(0..b.n).filter(|&j|!b.tiles[j].known&&b.d(t,j)<=radius).count();score+=reveal as f32*if m.kind==0{1.2}else{0.6};
 // Move-attack lookahead: only Dash units can profit immediately.
 let mut best_attack:f32=-100.;if st.dash&&!u.attacked&&!(tile.st==8&&!boat(u.k))&&!(boat(u.k)&&tile.terrain>2){for &ei in &b.enemy{let e=&b.units[ei];if b.d(t,e.pos)<=st.r{best_attack=best_attack.max(attack_score(b,tc,ui,t,ei,true));}}}
 let risk=tc.risk(b,u,t,usize::MAX);let srisk=tc.risk(b,u,s,usize::MAX);
 let support=b.own.iter().filter(|&&i|i!=ui&&b.d(b.units[i].pos,t)<=2&&b.units[i].hp>=b.units[i].max*0.6).count();
 let en=b.enemy.iter().filter(|&&i|b.d(b.units[i].pos,t)<=3).count();
 let multiplier=if support>=en+2&&en>0{1.35}else{1.65};score-=risk*multiplier;score+=srisk*0.8;
 if best_attack>0.{
 let directbest=tc.direct[ui].iter().filter_map(|&(p,_)|b.at[p].map(|ei|attack_score(b,tc,ui,s,ei,false))).fold(-100.0f32,f32::max);
 score+=best_attack*0.80+3.;if directbest>=best_attack-1.{score-=16.;}
 }
 if b.enemy.iter().any(|&i|b.units[i].k==4&&b.d(t,b.units[i].pos)<=5)&&u.hp<=10.{
 let vulnerable=b.own.iter().filter(|&&i|i!=ui&&b.d(t,b.units[i].pos)<=1&&b.units[i].hp<=10.&&stats(b.units[i].k).d<=2.).count();score-=vulnerable as f32*4.;}
 if st.r>=2&&u.k!=43{let near=b.enemy.iter().map(|&i|b.d(t,b.units[i].pos)).min().unwrap_or(99);let oldnear=b.enemy.iter().map(|&i|b.d(s,b.units[i].pos)).min().unwrap_or(99);
 if near<=st.r{score+=8.;}if near<st.r{score-=(st.r-near) as f32*8.;}if oldnear<=st.r&&!u.attacked&&!st.dash{score-=15.;}
 if st.r==3&&near<=3&&near>=2{score+=5.;}
 }
 if u.k==12||u.k==45{let near=b.enemy.iter().map(|&i|b.d(t,b.units[i].pos)).min().unwrap_or(99);if near==1{score+=8.;}if u.hp>25.{score+=risk*0.5;}}
 if u.k==45{for &ei in &b.enemy{let e=&b.units[ei];if b.d(t,e.pos)==1{let damage=(b.combat(u,t,e,e.pos).0*0.5).floor();score+=damage.min(e.hp)/e.max*value(e)*1.8;if damage>=e.hp{score+=value(e)*0.65+7.;}}}}
 if b.enemy.iter().any(|&i|b.units[i].k==41&&b.d(t,b.units[i].pos)<=4){let adj=b.own.iter().filter(|&&i|i!=ui&&b.d(b.units[i].pos,t)==1).count();score-=adj as f32*2.;}
 let missing=u.max-u.hp;
 if missing>=3.&&!u.attacked{score-=missing/u.max*14.+if src.owner==b.p{4.}else{0.};}
 if u.attacked&&u.k==3{score+=6.+srisk*0.8;}
 if src.st==1&&src.owner==b.p{let threat=b.enemy.iter().any(|&i|b.d(b.units[i].pos,s)<=2);if threat{let inc=tc.incoming(b,u,s,usize::MAX);if inc<u.hp+4.{score-=10.;}}else{score+=6.;}}
 if tile.st==1&&tile.owner==b.p{score-=7.;}
 if tile.st==8&&u.k==43{score-=3.;}
 if boat(u.k)&&tile.terrain>2&&u.k!=43 {score-=if tile.st==1&&tile.owner!=b.p{0.}else if u.k==45{4.}else{15.};}
 if u.k==43&&(b.tech[13]||b.tech[44])&&b.stars>=5.&&tile.owner==b.p&&src.owner!=b.p{score+=20.;}
 if u.k==43&&src.owner==b.p&&!b.tech[13]&&!b.tech[44]{score+=0.;}else if u.k==43&&src.owner==b.p&&b.stars>=5.&&(b.tech[13]||b.tech[44]){score-=45.;}
 // Occupancy penalties prevent clogging and cyclic shuffling.
 if mem.lastmove==Some((t,s)){score-=8.;}
 score-=mem.visits[t].min(10) as f32*0.15;
 score
}
fn choose(o:&Value,mem:&mut Memory)->String{
 let b=Board::new(o,mem);mem.counter+=1;if b.turn!=mem.turn{mem.turn=b.turn;mem.lastmove=None;}
 let tc=Tactics::new(&b);let ms=missions(&b,mem);let mut road_scores=roads_plan(&b);tactical_roads(&b,&tc,&ms,&mut road_scores);let eco_scores=economic_plans(&b);let mut combo_scores=vec![-1000.;b.acts.len()];combos(&b,&tc,&mut combo_scores);knight_chains(&b,&tc,&mut combo_scores);
 let mut best=0;let mut bs=-1e9f32;let mut guard_scores=vec![0.;b.acts.len()];city_guards(&b,&tc,&mut guard_scores);
 let mut save=0.;let mut saveprio=0.;for k in [1,6,16,17,20,10,13,44,14,18,23,24,8,4,5,2,3]{if !b.tech[k]&&b.tech[prerequisites(k)]{let pr=tech_priority(&b,k);if pr>saveprio&&pr>24.&&tech_cost(&b,k)>b.stars&&b.own.len()>=b.nc()+2{save=tech_cost(&b,k);saveprio=pr;}}}
 for (i,a) in b.acts.iter().enumerate(){let mut sc=match a.k{
 11=>-100000.,10=>0.,
 9=>match a.p{3=>1000.,4=>900.,7=>1000.,1=>{let danger=b.enemy.iter().filter(|&&i|b.d(b.units[i].pos,a.t)<=3).count();if b.nc()==1&&danger>=3{1040.}else{900.}},6=>1100.,2=>800.,8=>1000.,5=>border_reward(&b,a.t),_=>500.},
 8=>{let t=b.tiles[a.s];if t.st==1{if t.owner==0{140.}else{200.}}else{130.}},
 2=>if let (Some(ui),Some(di))=(b.at[a.s],b.at[a.t]){attack_score(&b,&tc,ui,a.s,di,false)+25.}else{if b.at[a.s].map_or(false,|i|b.units[i].k==38||b.units[i].k==39){100.}else{-100.}},
 1=>move_score(&b,&tc,&ms,mem,a),
 5=>econ_score(&b,a),
 6=>match a.p{8=>port_score(&b,a.t),3=>road_scores[i],48=>road_scores[i].max(bridge_score(&b,a.t)),50=>{let l=b.adj[a.t].iter().filter(|&&j|b.tiles[j].owner==b.p&&matches!(b.tiles[j].st,6|13|22)).map(|&j|b.tiles[j].sl).sum::<i32>().min(8);if l>=2{23.+l as f32*4.}else{-20.}},_=>econ_score(&b,a)},
 7=>{let pr=tech_priority(&b,a.p as usize);let capturing=b.acts.iter().any(|a|a.k==8&&b.tiles[a.s].st==1&&b.tiles[a.s].owner==0);if capturing&&pr>15.&&b.stars>=tech_cost(&b,a.p as usize)+4.{pr+140.}else{pr}},
 4=>train_score(&b,&tc,a),
 3=>match a.p{
 16=>150.,7=>{if let Some(ui)=b.at[a.s]{let u=&b.units[ui];let missing=u.max-u.hp;let gain=missing.min(if b.tiles[a.s].owner==b.p{4.}else{2.});2.+gain/u.max*value(u)*2.+if u.hp<u.max*0.65{10.}else{0.}+if b.tiles[a.s].st==1&&b.tiles[a.s].owner==b.p&&tc.incoming(&b,u,a.s,usize::MAX)>=u.hp{18.}else{0.}}else{-100.}},
 9=>30.,8=>disband_score(&b,a.s),2=>{let city=b.cityof[a.t];if b.tiles[a.t].res==1{-50.}else if b.ci.get(city).copied().flatten().map_or(false,|i|b.cities[i].level>=5)&&b.stars<2.&&b.enemy.iter().any(|&i|b.d(b.units[i].pos,city)<=2){5.}else{-50.}},19=>if a.t==0{2.}else{-50.},18=>5.,_=>-100.},_=>-100.
 };
 if save>0.&&matches!(a.k,4|5|6){let cost=if a.k==4{stats(a.p).c}else if a.k==5{2.}else{costbuild(a.p)};if b.stars-cost<save{sc-=if saveprio>30.{20.}else{12.};}}
 if a.k==2&&sc<30.{sc-=30.;} // do not make actively bad trades just because attack is possible
 sc=sc.max(combo_scores[i]).max(eco_scores[i]);sc+=guard_scores[i];if sc>bs{bs=sc;best=i;}
 }
 let a=&b.acts[best];mem.lastcmd=Some((a.k,a.s,a.t,a.p));if a.k==1{mem.lastmove=Some((a.s,a.t));mem.visits[a.t]+=1;}
 if a.k==2{if let (Some(ui),Some(di))=(b.at[a.s],b.at[a.t]){let u=&b.units[ui];let d=&b.units[di];if b.tiles[a.t].st==1{mem.lastattack=Some((a.t,d.hp,b.combat_bonus(u,a.s,d,a.t,1.5).0,b.combat_bonus(u,a.s,d,a.t,4.).0));}}}
 a.id.clone()
}
fn main(){let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){let Ok(line)=line else{break};let Ok(obs)=serde_json::from_str::<Value>(&line)else{continue};let id=choose(&obs,&mut mem);writeln!(stdout,"{}",json!({"action_id":id})).unwrap();stdout.flush().unwrap();}}

// Connect cities by the cheapest visible route. Connections grant population to both endpoints.
fn roads_plan(b:&Board)->Vec<f32>{
 let mut scores=vec![-100.;b.acts.len()];if !b.tech[4]{return scores;}let Some(cap)=b.cities.iter().find(|c|c.owner==b.p&&b.tiles[c.pos].cap==b.p).map(|c|c.pos)else{return scores;};
 let n=b.n;let mut buildat=vec![None;n];for (ai,a) in b.acts.iter().enumerate(){if a.k==6&&(a.p==3||a.p==48){buildat[a.t]=Some(ai);}}
 let node=|i:usize|{let t=b.tiles[i];(t.road||t.st==1||t.st==48||t.st==8)&&(t.owner==0||t.owner==b.p)};
 let mut ds=vec![9999;n];let mut par=vec![n;n];let mut q=BinaryHeap::new();ds[cap]=0;q.push(Reverse((0,cap)));
 while let Some(Reverse((d,x)))=q.pop(){if d!=ds[x]{continue;}for &y in &b.adj[x]{let t=b.tiles[y];if !t.known||t.owner>0&&t.owner!=b.p||b.at[y].map_or(false,|i|b.units[i].owner!=b.p){continue;}
 let cost=if node(y){0}else if buildat[y].is_some(){if t.terrain<=2{5}else{3}}else if t.terrain==3||t.terrain==5{3}else{continue;};let nd=d+cost;if nd<ds[y]{ds[y]=nd;par[y]=x;q.push(Reverse((nd,y)));}
 }
 }
 for c in &b.cities{if c.owner!=b.p||c.pos==cap||c.connected||ds[c.pos]>15||ds[c.pos]==0{continue;}let cost=ds[c.pos] as f32;let gain=econ_gain(b,cap,1.)+econ_gain(b,c.pos,1.);let mut utility=gain*1.4-cost*0.85+9.;
 if cost>b.stars+b.income*0.7{utility-=15.;}
 let mut x=c.pos;let mut path=vec![];let mut guard=0;while x!=cap&&x<n&&guard<n{path.push(x);x=par[x];guard+=1;}
 if x!=cap{continue;}
 for &x in path.iter().rev(){if let Some(ai)=buildat[x]{utility+=b.own.iter().filter(|&&i|b.d(b.units[i].pos,x)<=1).count() as f32; scores[ai]=scores[ai].max(utility);break;}}
 }
 scores
}
#[derive(Clone)]
struct Combo{hp:f32,used:u128,positions:Vec<usize>,first:usize,cost:f32,steps:usize}
// Small tactical beam search: concentrate fire and order attacks by their true HP-dependent damage.
fn combos(b:&Board,tc:&Tactics,scores:&mut [f32]){
 if b.own.len()>120{return;}
 let mut targets=b.enemy.clone();targets.sort_by_key(|&i|{let u=b.units[i];(if b.tiles[u.pos].st==1{0}else{1},(u.hp*10.) as i32)});targets.truncate(14);
 for &di in &targets{
 let defender=b.units[di];let mut options:Vec<(usize,Vec<(usize,usize,bool)>)>=vec![];
 for &ui in &b.own{let u=b.units[ui];let st=stats(u.k);if u.attacked||st.a<=0.||matches!(u.k,38|39){continue;}
 let mut opts:Vec<(usize,usize,bool)>=tc.direct[ui].iter().filter(|&&(dst,_)|dst==defender.pos).map(|&(_,ai)|(u.pos,ai,false)).collect();
 if st.dash&&!u.moved{let mut mv:Vec<(f32,usize,usize)>=vec![];for &(pos,ai) in &tc.ourmoves[ui]{let tile=b.tiles[pos];if b.d(pos,defender.pos)>st.r||(!boat(u.k)&&tile.st==8)||(boat(u.k)&&tile.terrain>2){continue;}
 let s=tc.risk(b,&u,pos,di)+b.d(pos,u.pos) as f32*0.1;mv.push((s,pos,ai));}
 mv.sort_by(|a,c|a.0.partial_cmp(&c.0).unwrap());for &(_,pos,ai) in mv.iter().take(if opts.is_empty(){3}else{1}){opts.push((pos,ai,true));}}
 if !opts.is_empty(){options.push((ui,opts));}
 }
 if options.len()<2{continue;}
 options.sort_by(|(a,_),(c,_)|{let ua=&b.units[*a];let uc=&b.units[*c];let va=stats(ua.k).a*ua.hp/ua.max;let vc=stats(uc.k).a*uc.hp/uc.max;vc.partial_cmp(&va).unwrap()});options.truncate(9);
 let mut beam=vec![Combo{hp:defender.hp,used:0,positions:vec![],first:usize::MAX,cost:0.,steps:0}];
 for _depth in 0..4{let mut next=vec![];
 for state in &beam{for (oi,(ui,opts)) in options.iter().enumerate(){if state.used>>oi&1!=0{continue;}let u=b.units[*ui];let mut d=defender;d.hp=state.hp;
 for &(pos,ai,moving) in opts{if state.positions.contains(&pos){continue;}let (damage,ret)=b.combat(&u,pos,&d,d.pos);if damage<=0.{continue;}let kill=damage>=d.hp;let mut after=u;after.hp=u.hp-ret;let died=after.hp<=0.;let end=if kill&&stats(u.k).r==1&& (b.tiles[d.pos].terrain>2||boat(u.k)){d.pos}else{pos};
 let inc=if died{0.}else{tc.risk(b,&after,end,if kill{di}else{usize::MAX})};let prior=tc.risk(b,&u,u.pos,usize::MAX);let c=ret/u.max*value(&u)*1.4+if died{value(&u)*0.75}else{(inc-prior*0.65).max(0.)*0.55}+if moving{0.7}else{0.}+1.5;
 let mut ns=state.clone();ns.hp=(state.hp-damage).max(0.);ns.used|=1<<oi;ns.positions.push(end);ns.cost+=c;ns.steps+=1;if ns.first==usize::MAX{ns.first=ai;}
 if kill{
 let city=b.tiles[d.pos].st==1;let gain=value(&defender)*(defender.hp/defender.max*1.8+0.65)+8.+if city{if b.tiles[d.pos].owner==b.p{32.}else{24.}}else{0.};
 let utility=gain-ns.cost;
 if utility>4.{let score=31.+utility/(0.65+ns.steps as f32*0.30);scores[ns.first]=scores[ns.first].max(score);}
 }else if !died||damage>4.{next.push(ns);}
 }
 }}
 if next.is_empty(){break;}next.sort_by(|a,c|{let sa=(defender.hp-a.hp)/defender.max*value(&defender)*1.8-a.cost;let sc=(defender.hp-c.hp)/defender.max*value(&defender)*1.8-c.cost;sc.partial_cmp(&sa).unwrap()});
 next.truncate(if b.enemy.len()>12{16}else{28});beam=next;
 }
 }
}
#[derive(Clone)]
struct EcoOp{tile:usize,kind:i32,param:i32,cost:f32,pop:f32,tech:usize}
fn economic_plans(b:&Board)->Vec<f32>{
 let mut out=vec![-1000.0f32;b.acts.len()];
 let mut research=[None;128];let mut action_at=std::collections::HashMap::new();for (i,a) in b.acts.iter().enumerate(){if a.k==7{research[a.p as usize]=Some(i);}if a.k==5||a.k==6{action_at.insert((a.k,a.t,a.p),i);}}
 let mut costs:Vec<f32>=vec![0.;128];for k in 1..128{if !b.tech[k]{costs[k]=tech_cost(b,k);if !b.tech[prerequisites(k)]{costs[k]+=tech_cost(b,prerequisites(k))+5.;}}}
 let eco_tech=[0usize,6,10,16,17,23,8,9,24];
 for city in &b.cities{if city.owner!=b.p{continue;}
 let land:Vec<usize>=(0..b.n).filter(|&i|b.cityof[i]==city.pos&&b.tiles[i].owner==b.p).collect();
 if land.is_empty(){continue;}
 // Choose the highest-value adjacent multiplier location, retaining candidate variants.
 let mut mills:Vec<(f32,usize,i32,usize)>=vec![];
 for &i in &land{let t=b.tiles[i];if t.st!=0{continue;}for (k,req,tech) in [(13,12,17),(6,5,9),(22,21,24)]{if k!=22&&t.terrain!=3||k==22&&t.terrain!=3&&t.terrain!=5{continue;}let exists=land.iter().any(|&j|b.tiles[j].st==k);if exists{continue;}let cnt=b.adj[i].iter().filter(|&&j|b.tiles[j].owner==b.p&&(b.tiles[j].st==req||b.tiles[j].st==0&&match req{12=>b.tiles[j].terrain==5,5=>b.tiles[j].res==2,_=>b.tiles[j].res==5})).count();if cnt>=2{mills.push((cnt as f32,i,k,tech));}}}
 mills.sort_by(|a,c|c.0.partial_cmp(&a.0).unwrap());mills.truncate(3);
 for variant in 0..=mills.len(){
 let mill=if variant==0{None}else{Some(mills[variant-1])};
 let mut ops=vec![];
 for &i in &land{let t=b.tiles[i];if t.st!=0{continue;}if b.at[i].map_or(false,|ui|b.units[ui].owner!=b.p){continue;}
 if let Some((_,mt,mk,mtech))=mill{if mt==i{let cnt=b.adj[i].iter().filter(|&&j|b.tiles[j].owner==b.p&&b.tiles[j].st==if mk==13{12}else if mk==6{5}else{21}).count() as f32;ops.push(EcoOp{tile:i,kind:6,param:mk,cost:5.,pop:cnt*if mk==22{2.}else{1.},tech:mtech});continue;}}
 match t.res{1=>ops.push(EcoOp{tile:i,kind:5,param:0,cost:2.,pop:1.,tech:15}),3=>ops.push(EcoOp{tile:i,kind:5,param:0,cost:2.,pop:1.,tech:10}),6=>ops.push(EcoOp{tile:i,kind:5,param:0,cost:2.,pop:1.,tech:6}),2=>ops.push(EcoOp{tile:i,kind:6,param:5,cost:5.,pop:2.,tech:8}),5=>ops.push(EcoOp{tile:i,kind:6,param:21,cost:5.,pop:2.,tech:23}),_=>{}}
 if t.terrain==5{let extra=b.adj[i].iter().filter(|&&j|b.tiles[j].st==13&&b.tiles[j].owner==b.p&&b.cityof[j]==city.pos).count() as f32;ops.push(EcoOp{tile:i,kind:6,param:12,cost:3.,pop:1.+extra,tech:16});}
 }
 // Free monuments are finite but very strong population accelerators.
 for a in &b.acts{if a.k==6&&(23..=29).contains(&a.p)&&b.cityof[a.t]==city.pos{ops.push(EcoOp{tile:a.t,kind:6,param:a.p,cost:0.,pop:3.,tech:0});break;}}
 for &extra_tech in &eco_tech{
 if extra_tech>0&&(b.tech[extra_tech]||!b.tech[prerequisites(extra_tech)]){continue;}
 let mut available=b.tech;if extra_tech>0{available[extra_tech]=true;}
 let mut state=b.tiles.clone();let mut chosen=vec![];let mut used=vec![false;ops.len()];let mut totalcost=if extra_tech>0{costs[extra_tech]}else{0.};let mut level=city.level;let mut progress=city.progress as f32;let mut income_gain=0.;let mut giant=false;let mut first:Option<usize>=None;let mut bought_tech=false;
 for _ in 0..18{
 let mut bestop=None;let mut bestvalue=-1e6;
 for (oi,op) in ops.iter().enumerate(){if used[oi]||!available[op.tech]{continue;}let t=state[op.tile];if t.st!=0{continue;}if op.kind==6&&op.param==12&&t.res==1{continue;}if op.kind==5&&t.res==0{continue;}
 let mut pop=op.pop;
 if matches!(op.param,13|6|22)&&op.kind==6{let req=if op.param==13{12}else if op.param==6{5}else{21};pop=b.adj[op.tile].iter().filter(|&&j|state[j].owner==b.p&&state[j].st==req).count() as f32*if op.param==22{2.}else{1.};if pop<=0.{continue;}}
 let mut future=0.;if let Some((_,mt,mk,_))=mill{if op.param==12&&mk==13&&b.d(mt,op.tile)==1||op.param==5&&mk==6&&b.d(mt,op.tile)==1||op.param==21&&mk==22&&b.d(mt,op.tile)==1{future=if state[mt].st==mk{if mk==22{2.}else{1.}}else{0.55};}}
 let merit=(pop+future)/(op.cost+0.5);if merit>bestvalue{bestvalue=merit;bestop=Some((oi,pop));}}
 let Some((oi,mut pop))=bestop else{break;};let op=&ops[oi];used[oi]=true;
 if op.tech==extra_tech&&extra_tech>0{bought_tech=true;}
 if first.is_none(){first=if op.tech==extra_tech&&extra_tech>0{research[extra_tech]}else{action_at.get(&(op.kind,op.tile,op.param)).copied()};if first.is_none(){break;}}
 if let Some((_,mt,mk,_))=mill{if state[mt].st==mk&&b.d(mt,op.tile)==1&&(op.param==12&&mk==13||op.param==5&&mk==6||op.param==21&&mk==22){pop+=if mk==22{2.}else{1.};}}
 if op.kind==5{state[op.tile].res=0;}else{state[op.tile].st=op.param;}
 totalcost+=op.cost;chosen.push(oi);progress+=pop;
 let oldlevel=level;
 while progress>= (level+1) as f32 {progress-=(level+1) as f32;level+=1;income_gain+=1.;if level==2{income_gain+=1.;}if level==3{totalcost-=5.;}if level==4&&border_reward(b,city.pos)<1000.{progress+=3.;}if level>=5{giant=true;}}
 if level==oldlevel{continue;}
 if extra_tech>0&&!bought_tech{continue;}
 let horizon=if b.turn<12{6.5}else if b.turn<25{5.}else{4.};let reward=income_gain*horizon+if giant{25.}else{0.};
 let net=reward-totalcost*0.85;let turns=(totalcost-b.stars).max(0.)/b.income.max(2.);
 let mut priority=16.+net*1.8/(1.+turns*0.6)+if giant{6.}else{0.};
 if b.turn<3&&b.own.len()<2{priority-=15.;}
 if totalcost>b.stars+b.income*2.{priority-=15.;}
 if b.enemy.iter().any(|&i|b.d(b.units[i].pos,city.pos)<=2){if b.at[city.pos].map_or(false,|i|b.units[i].owner!=b.p){priority-=25.;}else if giant{priority+=8.;}}
 if priority>0.{if let Some(ai)=first{out[ai]=out[ai].max(priority);}}
 if level>=5||level>=city.level+2{break;}
 }
 }
 }
 }
 out
}
#[derive(Clone)]
struct Chain{pos:usize,dead:u128,gain:f32,first:usize,steps:i32}
fn knight_chains(b:&Board,tc:&Tactics,scores:&mut [f32]){
 if b.enemy.len()>120{return;}
 for &ui in &b.own{let u=&b.units[ui];if u.k!=4||u.attacked{continue;}
 let mut initial=vec![(u.pos,usize::MAX)];if !u.moved{initial.extend(tc.ourmoves[ui].iter().map(|&(t,a)|(t,a)));}
 let mut bests=vec![];
 for (pos,ma) in initial{if b.tiles[pos].st==8{continue;}let mut stack=vec![Chain{pos,dead:0,gain:0.,first:ma,steps:0}];let mut expanded=0;
 while let Some(st)=stack.pop(){expanded+=1;if expanded>400{break;}for (j,&di) in b.enemy.iter().enumerate(){if st.dead>>j&1!=0{continue;}let d=&b.units[di];if b.d(st.pos,d.pos)!=1{continue;}let (dmg,_)=b.combat(u,st.pos,d,d.pos);if dmg<d.hp{continue;}
 let mut ns=st.clone();ns.dead|=1<<j;ns.steps+=1;ns.gain+=value(d)*(0.65+1.8*d.hp/d.max)+7.+if b.tiles[d.pos].st==1{16.}else{0.};
 if ns.first==usize::MAX{if let Some((_,ai))=tc.direct[ui].iter().find(|&&(t,_)|t==d.pos){ns.first=*ai;}else{continue;}}
 if b.tiles[d.pos].terrain>2&&(b.tiles[d.pos].terrain!=4||b.tech[20]){ns.pos=d.pos;}
 let risk=tc.risk(b,u,ns.pos,di);let utility=ns.gain-risk*0.85+25.;bests.push((ns.first,utility));
 if ns.steps<8{stack.push(ns);}
 }}
 }
 for (a,s) in bests{scores[a]=scores[a].max(s);}
 }
}
fn city_guards(b:&Board,tc:&Tactics,actscores:&mut [f32]){
 for c in &b.cities{if c.owner!=b.p{continue;}let pos=c.pos;let intruder=b.at[pos].filter(|&i|b.units[i].owner!=b.p);
 let nearby:Vec<usize>=b.enemy.iter().copied().filter(|&i|b.d(pos,b.units[i].pos)<=stats(b.units[i].k).m+1).collect();
 if nearby.is_empty(){continue;}
 if let Some(di)=intruder{ // A siege must be lifted before the opponent's next action phase.
 for (ai,a) in b.acts.iter().enumerate(){if a.k==2&&a.t==pos{actscores[ai]+=25.;}if a.k==1{if let Some(ui)=b.at[a.s]{let u=b.units[ui];if stats(u.k).dash&&!u.attacked&&b.d(a.t,pos)<=stats(u.k).r{actscores[ai]+=15.;}}}}
 let _=di;continue;
 }
 let defender=b.at[pos].filter(|&i|b.units[i].owner==b.p);
 let units_home=b.own.iter().filter(|&&i|b.units[i].home==pos as i32+1).count();let can_train=units_home<=c.level as usize&&b.stars>=2.;
 for (ai,a) in b.acts.iter().enumerate(){if a.k==1&&a.s==pos {let exposed=nearby.iter().any(|&i|b.d(b.units[i].pos,pos)<=stats(b.units[i].k).m+if b.tiles[pos].road{1}else{0});if exposed&&!can_train{actscores[ai]-=23.;}else if exposed{actscores[ai]-=7.;}}
 if a.k==4&&a.s==pos{let u=Unit{pos,k:a.p,owner:b.p,hp:stats(a.p).h,max:stats(a.p).h,..Unit::default()};let inc=tc.incoming(b,&u,pos,usize::MAX);let ownnear=b.own.iter().filter(|&&i|b.d(b.units[i].pos,pos)<=2).count();actscores[ai]+=if defender.is_none(){if inc<u.hp{14.}else if ownnear<2{7.}else{1.}}else{0.};}
 }
 }
}
fn border_reward(b:&Board,pos:usize)->f32{
 let mut pop=0.;let mut shallow=0;let mut newland=0;let mut fog=0;
 for j in 0..b.n{if b.d(pos,j)!=2{continue;}let t=b.tiles[j];if !t.known{fog+=1;}if t.owner!=0{continue;}if t.terrain==1&&t.st==0{shallow+=1;}if t.terrain>=3{newland+=1;}if t.res==1||t.res==3||t.res==6{pop+=1.;}if t.terrain==5{pop+=1.5;}if t.res==5||t.res==2{pop+=2.;}}
 let ownedshallow=b.tiles.iter().filter(|t|t.owner==b.p&&t.terrain==1).count();let water=b.tiles.iter().filter(|t|t.known&&t.terrain<=2).count();let known=b.tiles.iter().filter(|t|t.known).count();
 if shallow>0&&ownedshallow==0&&water*5>known&&b.turn<30{return 1050.;}
 if pop>=8.&&b.turn<25{return 1010.+pop;}
 if pop>=5.&&b.income>15.&&newland>=8&&b.turn<30{return 1005.;}
 if shallow>=3&&water*2>known&&b.nc()<=2{return 1030.;}
 925.+pop+fog as f32
}
fn bridge_score(b:&Board,t:usize)->f32{
 let adjacent_land:Vec<usize>=b.adj[t].iter().copied().filter(|&i|b.tiles[i].known&&b.tiles[i].terrain>2).collect();if adjacent_land.len()<2{return -100.;}
 let near=b.own.iter().filter(|&&i|!boat(b.units[i].k)).map(|&i|b.d(t,b.units[i].pos)).min().unwrap_or(99);if near>4{return -30.;}
 let mut component=vec![usize::MAX;b.n];let mut cn=0;
 for i in 0..b.n{if !b.tiles[i].known||b.tiles[i].terrain<=2&&b.tiles[i].st!=48||component[i]!=usize::MAX{continue;}component[i]=cn;let mut stack=vec![i];while let Some(x)=stack.pop(){for &y in &b.adj[x]{if component[y]==usize::MAX&&b.tiles[y].known&&(b.tiles[y].terrain>2||b.tiles[y].st==48){component[y]=cn;stack.push(y);}}}cn+=1;}
 let mut ids=vec![];for i in adjacent_land{if !ids.contains(&component[i]){ids.push(component[i]);}}if ids.len()<2{return -20.;}
 let mut own_comps=vec![false;cn];for &i in &b.own{let x=b.units[i].pos;if component[x]<cn{own_comps[component[x]]=true;}}
 if !ids.iter().any(|&i|own_comps[i]){return -20.;}
 let new_land=ids.iter().any(|&i|!own_comps[i]);let mut value=if new_land{30.}else{18.};
 let goals=b.cities.iter().filter(|c|c.owner!=b.p&&ids.contains(&component[c.pos])&&!own_comps[component[c.pos]]).count();value+=goals as f32*6.;value-=(near-1).max(0) as f32*2.;
 value
}
fn tactical_roads(b:&Board,tc:&Tactics,missions:&[Mission],scores:&mut [f32]){
 if !b.tech[4]||b.stars<5.||b.income<10.{return;}
 let node=|i:usize|{let t=b.tiles[i];(t.road||t.st==1||t.st==48)&&(t.owner==0||t.owner==b.p)};
 for (ai,a) in b.acts.iter().enumerate(){if a.k!=6||a.p!=3{continue;}let t=a.t;let n=b.adj[t].iter().filter(|&&j|node(j)).count();if n==0{continue;}
 let mut gain=0.;for &ui in &b.own{let u=b.units[ui];if u.moved||boat(u.k){continue;}if b.d(u.pos,t)>3{continue;}let m=&missions[ui];if m.path.is_empty(){continue;}
 if m.path[t]+5>=m.path[u.pos]{continue;}
 if node(u.pos)&&b.d(u.pos,t)==1{let beyond=b.adj[t].iter().any(|&j|node(j)&&m.path[j]+5<m.path[t]&&b.at[j].is_none());if beyond{gain+=if u.k==12{3.5}else if u.k==8{2.0}else{2.5};}}
 if u.pos==t&&b.adj[t].iter().any(|&j|node(j)&&m.path[j]+5<m.path[t]){gain+=1.5;}
 }
 if gain>=2.5{let mut val=10.+gain*3.;if b.stars>b.income*2.{val+=8.;}scores[ai]=scores[ai].max(val);}
 }
 let _=tc;
}
fn disband_score(b:&Board,s:usize)->f32{
 let Some(ui)=b.at[s]else{return -100.;};let u=b.units[ui];if !matches!(u.k,2|3|9|43){return -100.;}
 let near=b.enemy.iter().map(|&i|b.d(s,b.units[i].pos)).min().unwrap_or(99);if near<5{return -100.;}
 if u.home<=0{return -100.;}let home=(u.home-1) as usize;if home>=b.n{return -100.;}
 let Some(ci)=b.ci[home]else{return -100.;};let c=&b.cities[ci];if c.owner!=b.p{return -100.;}
 let count=b.own.iter().filter(|&&i|b.units[i].home==u.home).count();let capacity=c.level as usize+1;
 if count<capacity||b.at[home].is_some()&&home!=s{return -100.;}
 if b.stars<5.||!b.tech[24]&&!b.tech[3]{return -100.;}
 let homefront=b.enemy.iter().map(|&i|b.d(home,b.units[i].pos)).min().unwrap_or(99);
 if homefront>6{return -100.;}
 if u.hp<=u.max*0.5&&b.d(s,home)<=4{return 14.;}
 if u.k==2&&b.tech[24]&&b.d(s,home)>=5&&homefront<4{return 12.;}
 -100.
}
