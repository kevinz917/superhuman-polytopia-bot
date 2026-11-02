use serde_json::{json,Value};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
const INF:f64=1e6;
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn flag(v:&Value,k:&str)->bool{v[k].as_bool().unwrap_or(false)}
#[derive(Clone,Copy,Default,Debug)]
struct Tile{known:bool,ter:i32,owner:i32,res:i32,st:i32,lv:i32,road:bool,cap:i32}
#[derive(Clone,Copy,Default,Debug)]
struct Unit{owner:i32,k:i32,hp:f64,vet:bool,moved:bool,attacked:bool,kills:i32,home:i32,pass:i32,poison:bool}
#[derive(Clone,Default,Debug)]
struct City{owner:i32,lv:i32,prog:i32,prod:i32,rewards:[i32;9],connected:bool}
#[derive(Clone,Copy,Debug)]
struct Stat{hp:f64,at:f64,de:f64,mv:i32,range:i32,cost:f64,dash:bool,escape:bool,stiff:bool,fort:bool}
fn stat(k:i32)->Stat{
 let (hp,at,de,mv,range,cost,dash,escape,stiff,fort)=match k{
 2=>(10.,2.,2.,1,1,2.,true,false,false,true),3=>(10.,2.,1.,2,1,3.,true,true,false,true),
 4=>(10.,3.5,1.,3,1,8.,true,false,false,true),5=>(15.,1.,3.,1,1,3.,false,false,false,true),
 8=>(10.,4.,0.,1,3,8.,false,false,true,false),9=>(10.,2.,1.,1,2,3.,true,false,false,true),
 10=>(10.,0.,1.,1,1,5.,false,false,true,false),11=>(15.,3.,3.,1,1,5.,true,false,false,false),
 12=>(40.,5.,4.,1,1,15.,false,false,false,false),37=>(10.,2.,2.,1,1,2.,true,false,false,true),
 38=>(5.,0.,0.5,2,1,8.,true,false,true,false),39=>(5.,0.,0.5,2,1,10.,true,false,true,false),
 40=>(10.,2.,2.,2,1,4.,true,false,false,false),41=>(10.,3.,2.,2,3,17.,false,false,true,false),
 42=>(10.,2.,1.,3,2,7.,true,false,false,false),43=>(10.,0.,1.,2,2,2.,false,false,true,false),
 44=>(10.,3.,3.,3,1,7.,true,false,false,false),45=>(40.,4.,4.,2,1,20.,false,false,true,false),
 _=>(10.,2.,2.,1,1,3.,true,false,false,false)};
 Stat{hp,at,de,mv,range,cost,dash,escape,stiff,fort}
}
fn naval(k:i32)->bool{(39..=45).contains(&k)}
fn maxhp(u:Unit)->f64{
 let mut h=stat(if u.pass>0 {u.pass}else{u.k}).hp+if u.vet{5.}else{0.};
 if naval(u.k)&&u.pass==0&&u.hp>h{h=if u.hp>20.{40.}else if u.hp>15.{20.}else{15.};}h
}
fn value(u:Unit)->f64{let s=stat(u.k);s.cost+if u.pass>0 {stat(u.pass).cost-2.}else{0.}}
#[derive(Clone,Debug)]
struct Act{id:String,k:i32,s:usize,t:usize,p:i32}
struct Board{size:usize,n:usize,me:i32,turn:i32,stars:f64,tiles:Vec<Tile>,units:Vec<Unit>,cities:Vec<City>,tech:[bool;128],adj:Vec<Vec<usize>>,dist:Vec<i32>,acts:Vec<Act>,own:Vec<usize>,enemy:Vec<usize>,mycities:Vec<usize>,territory:Vec<usize>,income:f64,water:f64,walls:Vec<bool>}
impl Board{
 fn parse(o:&Value)->Self{
 let v=&o["visible_state"];let size=num(v,"board_size") as usize; let n=size*size;let me=num(o,"player");
 let mut b=Board{size,n,me,turn:num(v,"turn"),stars:num(v,"stars") as f64,tiles:vec![Tile::default();n],units:vec![Unit::default();n],cities:vec![City::default();n],tech:[false;128],adj:vec![vec![];n],dist:vec![0;n*n],acts:vec![],own:vec![],enemy:vec![],mycities:vec![],territory:vec![n;n],income:0.,water:0.,walls:vec![false;n]};
 for x in v["technology"].as_array().unwrap(){b.tech[x.as_u64().unwrap() as usize]=true;}
 for t in v["tiles"].as_array().unwrap(){let i=num(t,"tile") as usize;b.tiles[i]=Tile{known:flag(t,"known"),ter:num(t,"terrain"),owner:num(t,"owner"),res:num(t,"resource"),st:num(t,"structure"),lv:num(t,"structure_level"),road:num(t,"road")>0,cap:num(t,"capital")};}
 for u in v["units"].as_array().unwrap(){let i=num(u,"tile") as usize; b.units[i]=Unit{owner:num(u,"owner"),k:num(u,"kind"),hp:u["health"].as_f64().unwrap_or(10.),vet:flag(u,"veteran"),moved:flag(u,"moved"),attacked:flag(u,"attacked"),kills:num(u,"kills"),home:num(u,"home"),pass:num(u,"passenger"),poison:flag(u,"poison")};if b.units[i].owner==me{b.own.push(i)}else{b.enemy.push(i)}}
 for c in v["cities"].as_array().unwrap(){let i=num(c,"tile") as usize;let mut rewards=[0;9];if let Some(a)=c["rewards"].as_array(){for j in 0..9{rewards[j]=a[j].as_i64().unwrap_or(0) as i32;}}b.cities[i]=City{owner:num(c,"owner"),lv:num(c,"level"),prog:num(c,"progress"),prod:num(c,"production"),rewards,connected:flag(c,"connected")};if num(c,"owner")==me{b.mycities.push(i);b.income+=num(c,"production") as f64+if b.tiles[i].cap==me{1.}else{0.};}}
 for i in 0..n{for j in 0..n{let d=((i%size) as i32-(j%size) as i32).abs().max(((i/size) as i32-(j/size) as i32).abs());b.dist[i*n+j]=d;if d==1{b.adj[i].push(j)}}}
 for i in 0..n{if b.tiles[i].owner==me{let mut best=100;for &c in &b.mycities{let radius=if b.cities[c].rewards[5]>0{2}else{1};let d=b.d(i,c);if d<=radius&&d<best{b.territory[i]=c;best=d;}}}}
 for a in o["legal_actions"].as_array().unwrap(){let c=&a["command"]; b.acts.push(Act{id:a["action_id"].as_str().unwrap().into(),k:num(c,"kind"),s:num(c,"source").max(0) as usize,t:num(c,"target").max(0) as usize,p:num(c,"parameter")});}
 let known=b.tiles.iter().filter(|t|t.known).count().max(1) as f64;b.water=b.tiles.iter().filter(|t|t.known&&t.ter<=2).count() as f64/known;
 b
 }
 fn d(&self,a:usize,b:usize)->i32{self.dist[a*self.n+b]}
 fn db(&self,u:Unit,pos:usize)->f64{
 let t=self.tiles[pos];let own=u.owner==self.me;let mut bonus=1.;
 if t.ter==4 {bonus=1.5;} else if t.ter==5 {if !own||self.tech[18]{bonus=1.5;}} else if t.ter<=2{if own&&self.tech[12]{bonus=1.5;}}else if t.st==1&&t.owner==u.owner&&stat(u.k).fort{bonus=if (own&&self.cities[pos].rewards[1]>0)||(!own&&self.walls[pos]) {4.}else{1.5};}
 bonus*if u.poison{0.5}else{1.}
 }
 fn damage(&self,a:Unit,d:Unit,apos:usize,dpos:usize)->(f64,f64){
 let aa=stat(a.k);let dd=stat(d.k);let af=aa.at*a.hp/maxhp(a);let df=dd.de*d.hp/maxhp(d)*self.db(d,dpos)*if d.poison{0.7}else{1.};let total=(af+df).max(0.01);
 let hit=(af/total*aa.at*4.5+0.00001).round();let ret=if d.hp>hit&&!dd.stiff&&self.d(apos,dpos)<=dd.range&&a.k!=37&&a.k!=40{(df/total*dd.de*4.5+0.00001).round()}else{0.};(hit,ret)
 }
 // Approximate the engine's movement exactly on explored tiles; used to predict threat and plan combos.
 fn reach(&self,src:usize,u:Unit,us:&[Unit],enemytech:bool)->Vec<usize>{
 let mut costs=vec![1000;self.n];costs[src]=0;let budget=stat(u.k).mv*2;let mut heap=BinaryHeap::new();heap.push(Reverse((0,src)));let mut out=vec![];
 let tech=|k:usize| if enemytech{true}else{self.tech[k]};
 while let Some(Reverse((cost,i)))=heap.pop(){if cost!=costs[i]{continue}if i!=src&&us[i].owner==0{out.push(i)}
 let t=self.tiles[i]; let road=t.road||t.st==1||t.st==48;
 let zoc=self.adj[i].iter().any(|&j|us[j].owner>0&&us[j].owner!=u.owner);
 if cost>=budget||(i!=src&&(t.ter==4||(t.ter==5&&!road)||zoc||(naval(u.k)&&t.ter>2)||(!naval(u.k)&&t.st==8))){continue}
 for &j in &self.adj[i]{let t=self.tiles[j];if !t.known{continue}if us[j].owner>0&&us[j].owner!=u.owner{continue}if t.ter==4&&!tech(20){continue}if t.ter==2&&!tech(13){continue}if t.ter==1&&!tech(10){continue}if !naval(u.k)&&t.ter<=2&&t.st!=48&&!(t.st==8&&t.owner==u.owner){continue}
 let roadj=(t.road||t.st==1||t.st==48)&&(t.owner==0||t.owner==u.owner);let edge=road&&roadj;let mut nc=cost+if edge{1}else{2};if t.ter==5&&!edge{nc=nc.max(budget)}if nc<costs[j]{costs[j]=nc;heap.push(Reverse((nc,j)));}
 }
 }out
 }
}
#[derive(Default)]
struct Memory{visits:Vec<i32>,last_turn:i32,last_src:usize,last_dst:usize,round_actions:i32,walls:Vec<bool>,lastatk:Option<(usize,f64,f64)>}
#[derive(Clone)]struct Goal{pos:usize,weight:f64,kind:i32}
struct Eval<'a>{b:&'a Board,goals:Vec<Goal>,paths:Vec<Vec<f64>>,assignment:Vec<usize>,threats:Vec<Vec<usize>>,enemy_range:Vec<Vec<usize>>,citydanger:Vec<f64>,roadscore:Vec<f64>,navneed:bool}
impl<'a> Eval<'a>{
 fn new(b:&'a Board)->Self{
 let n=b.n;let mut e=Eval{b,goals:vec![],paths:vec![],assignment:vec![usize::MAX;n],threats:vec![vec![];n],enemy_range:vec![vec![];n],citydanger:vec![0.;n],roadscore:vec![-10.;n],navneed:false};
 for &en in &b.enemy{let u=b.units[en];let s=stat(u.k);let mut origins=vec![en];if s.dash{origins.extend(b.reach(en,u,&b.units,true));}let mut covered=vec![false;n];for &p in &origins{for t in 0..n{if b.d(p,t)<=s.range{covered[t]=true;}}}for t in 0..n{if covered[t]{e.threats[t].push(en);}}e.enemy_range[en]=origins;}
 for &c in &b.mycities{for &en in &b.enemy{let d=b.d(c,en);if d<=4{e.citydanger[c]+=value(b.units[en])*b.units[en].hp/maxhp(b.units[en])*(5-d) as f64/4.;}}}
 // villages and ruins have much higher value than arbitrary exploration.
 for i in 0..n{let t=b.tiles[i];if t.st==1&&t.owner!=b.me{let mut w=if t.owner==0{36.}else{35.};if b.units[i].owner==b.me{w=0.;}e.goals.push(Goal{pos:i,weight:w,kind:if t.owner==0{1}else{2}});}else if t.st==2{e.goals.push(Goal{pos:i,weight:19.,kind:3});}}
 for &en in &b.enemy{let u=b.units[en];if stat(u.k).range>=2||u.k==12{e.goals.push(Goal{pos:en,weight:15.+value(u)*0.5,kind:6});}}
 for &c in &b.mycities{if e.citydanger[c]>0. {
 let urgent=b.enemy.iter().any(|&en|b.d(c,en)<=2)||(b.units[c].owner>0&&b.units[c].owner!=b.me);
 let localpower:f64=b.own.iter().filter(|&&fr|b.d(c,fr)<=2).map(|&fr|value(b.units[fr])*b.units[fr].hp/maxhp(b.units[fr])).sum();
 let weight=if urgent {14.+(e.citydanger[c]-localpower*0.7).max(0.)*0.7} else {5.+(e.citydanger[c]-localpower).max(0.)*0.4};
 e.goals.push(Goal{pos:c,weight,kind:4});}}
 // Select useful frontier vantage points, with spacing to distribute scouts.
 let mut front:Vec<(f64,usize)>=vec![];
 for i in 0..n{let t=b.tiles[i];if !t.known||(t.ter==4&&!b.tech[20])||(t.ter<=2&&!b.tech[10]){continue}let radius=if t.ter==4{2}else{1};let count=(0..n).filter(|&j|!b.tiles[j].known&&b.d(i,j)<=radius).count();if count==0{continue}let center=(b.size as f64-1.)/2.;let cent=((i%b.size) as f64-center).abs().max(((i/b.size) as f64-center).abs());let w=10.+count as f64*0.8-cent*0.8;front.push((w,i));}
 front.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());let mut selected=vec![];for (w,i) in front{if selected.iter().any(|&j|b.d(i,j)<3){continue}e.goals.push(Goal{pos:i,weight:w,kind:5});selected.push(i);}
 // A distant undiscovered corner is still an objective after other frontiers are occupied.
 if e.goals.is_empty(){let pos=b.mycities.first().copied().unwrap_or(n/2);e.goals.push(Goal{pos,weight:1.,kind:5});}
 // Path fields for two modes (land and embarked), physical movement cost incl impassable water.
 for mode in 0..2{for g in &e.goals{e.paths.push(e.path(g.pos,mode==1,false));}}
 // Route inland troops to existing embarkation points when the goal is across water.
 let ports:Vec<usize>=(0..n).filter(|&i|b.tiles[i].st==8&&b.tiles[i].owner==b.me).collect();
 for &port in &ports{
 let toport=e.path(port,false,true);
 for j in 0..e.goals.len(){let sea=e.paths[e.goals.len()+j][port];if sea>=INF/2.{continue}for src in 0..n{if toport[src]>=INF/2.{continue}let via=toport[src]+sea+2.;if via<e.paths[j][src]{e.paths[j][src]=via;}}}
 }
 let landtargets=e.goals.iter().filter(|g|g.kind==1||g.kind==2).any(|g| b.mycities.iter().any(|&c|e.paths[e.goal_index(g)][c]<INF/2.));
 let ports=b.tiles.iter().filter(|t|t.st==8&&t.owner==b.me).count();
 e.navneed=b.water>0.23&&(!landtargets||b.turn>10||b.mycities.len()<=2)&&ports< b.mycities.len().max(1).min(3);
 // Greedy unit-to-objective matching with soft capacity.
 let mut order=b.own.clone();order.sort_by(|&a,&bb|{let va=if b.units[a].k==12{0}else if naval(b.units[a].k){1}else{2};let vb=if b.units[bb].k==12{0}else if naval(b.units[bb].k){1}else{2};va.cmp(&vb).then(a.cmp(&bb))});
 let mut load=vec![0.;e.goals.len()];
 for src in order{let u=b.units[src];let mut best=-INF;let mut target=0;
 for (j,g) in e.goals.iter().enumerate(){let mode=naval(u.k) as usize;let mut d=e.paths[mode*e.goals.len()+j][src];if d>=INF/2. {continue}d/=stat(u.k).mv as f64;
 let capacity=match g.kind{1|3|5=>1.,2=>3.,4=>2.,_=>1.};let w=g.weight/(1.+load[j]/capacity*1.3);
 let mut score=w/(d+2.0);if g.kind==4&&u.k==12{score*=0.65;}if g.kind==4&&u.k==5{score*=1.4;}if g.kind==4&&naval(u.k){score*=0.5;}if g.kind==4&&u.k==8{score*=0.8;}if g.kind==5&&u.k==12{score*=0.35;}if g.kind==1&&u.k==8{score*=0.5;}if u.hp<maxhp(u)*0.4&&g.kind==2{score*=0.5;}if score>best{best=score;target=j;}
 }
 e.assignment[src]=target;load[target]+=1.;
 }
 e.plan_roads();
 e
 }
 fn plan_roads(&mut self){
 let b=self.b;if !b.tech[4]||b.mycities.len()<2{return}let n=b.n;
 let cap=b.mycities.iter().copied().find(|&c|b.tiles[c].cap==b.me).unwrap_or(b.mycities[0]);
 let roadok=|i:usize|{let t=b.tiles[i];t.known&&(t.owner==0||t.owner==b.me)&&(t.ter==3||t.ter==5||t.st==48||t.st==1)};
 let node=|i:usize|{let t=b.tiles[i];(t.road||t.st==1||t.st==48||t.st==8)&&(t.owner==0||t.owner==b.me)};
 let mut reached=vec![false;n];let mut stack=vec![cap];reached[cap]=true;while let Some(i)=stack.pop(){for &j in &b.adj[i]{if !reached[j]&&node(j){reached[j]=true;stack.push(j);}}}
 let mut d=vec![10000;n];let mut prev=vec![n;n];let mut heap=BinaryHeap::new();for i in 0..n{if reached[i]{d[i]=0;heap.push(Reverse((0,i)));}}
 while let Some(Reverse((cost,i)))=heap.pop(){if cost!=d[i]{continue}for &j in &b.adj[i]{if !roadok(j){continue}let c=cost+if node(j){1}else{100};if c<d[j]{d[j]=c;prev[j]=i;heap.push(Reverse((c,j)));}}}
 let mut best=-INF;let mut path=vec![];
 for &c in &b.mycities{if reached[c]||d[c]>=10000{continue}let count=(d[c]/100) as f64;if count==0.{continue}let budget=count*3.;let wait=((budget-b.stars).max(0.)/b.income.max(1.)).ceil();let mut score=15.-count*1.2-wait*4.;if b.cities[c].lv==1||b.cities[cap].prog==b.cities[cap].lv {score+=5.;}if self.citydanger[c]>10.{score-=5.;}if score>best{best=score;path.clear();let mut i=c;for _ in 0..n{if reached[i]||i>=n{break}if !node(i){path.push(i);}i=prev[i];}}}
 if best>0.{let count=path.len();for (j,&i) in path.iter().enumerate(){self.roadscore[i]=best+if count==1{6.}else{0.}+(j as f64)*0.01;}}
 // Short road links from a city help riders get out of dense forest, but do not over-invest.
 if b.stars>b.income+8. {for i in 0..n{if b.tiles[i].ter!=5||!roadok(i)||node(i){continue}let connected=b.adj[i].iter().any(|&j|node(j));if !connected{continue}let users=b.own.iter().filter(|&&fr|!b.units[fr].moved&&stat(b.units[fr].k).mv>1&&b.d(fr,i)<=1).count();if users>=2 {self.roadscore[i]=self.roadscore[i].max(3.+users as f64);}}}
 }
 fn goal_index(&self,g:&Goal)->usize{self.goals.iter().position(|x|x.pos==g.pos&&x.kind==g.kind).unwrap_or(0)}
 fn pass(&self,i:usize,sea:bool,allowport:bool)->bool{let t=self.b.tiles[i];if !t.known{return true}if t.ter==4&&!self.b.tech[20]{return false}if t.ter<=2 {return sea&&(t.ter!=2||self.b.tech[13])||t.st==48||(allowport&&t.st==8&&t.owner==self.b.me)} true}
 fn path(&self,target:usize,sea:bool,allowport:bool)->Vec<f64>{
 let b=self.b;let mut d=vec![INF;b.n];d[target]=0.;let mut heap=BinaryHeap::new();heap.push(Reverse((0i32,target)));
 while let Some(Reverse((cost,i)))=heap.pop(){if cost as f64/100.>d[i]+0.001{continue}for &j in &b.adj[i]{if !self.pass(j,sea,allowport)||!self.pass(i,sea,allowport){continue}
 let t=b.tiles[i];let tj=b.tiles[j];let r1=t.road||t.st==1||t.st==48;let r2=tj.road||tj.st==1||tj.st==48;
 let edge=if r1&&r2{0.5}else if t.ter==4||t.ter==5{1.5}else{1.};let mut nc=cost as f64/100.+edge;if !t.known{nc+=0.25;}if b.units[i].owner>0&&b.units[i].owner!=b.me{nc+=0.7;}if sea&&t.ter>2&&tj.ter>2{nc+=1.5;}
 if nc+0.001<d[j]{d[j]=nc;heap.push(Reverse(((nc*100.).round() as i32,j)));}
 }}d
 }
 fn threat_loss(&self,src:usize,pos:usize,u:Unit,ignore:usize)->f64{
 let b=self.b;let mut hp=u.hp;let mut attackers:Vec<(f64,usize)>=vec![];
 for &en in &self.threats[pos]{if en==ignore{continue}let eu=b.units[en];if stat(eu.k).at==0.&&eu.k!=10{continue}let ds=b.d(en,pos);let mut hit=b.damage(eu,u,en,pos).0;if eu.k==10{hit=hp;}let support=b.own.iter().filter(|&&a|a!=src&&b.d(a,en)<=stat(b.units[a].k).range+1&&!b.units[a].attacked).count() as f64;
 // Enemies cannot all abandon exposed cities and attack into concentrated forces for free.
 let discount=if ds>stat(eu.k).range{0.78}else{1.};hit*=discount*(1.-(support*0.05).min(0.25));attackers.push((hit,en));}
 attackers.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());let mut damage=0.;for (hit,_) in attackers.into_iter().take(5){damage+=hit;hp-=hit;if hp<=0.{break}}
 let val=value(u);let loss=damage.min(u.hp)/maxhp(u)*(val+2.)+if hp<=0.{val*0.9+3.}else{0.};loss
 }
 fn attack_score(&self,src:usize,pos:usize,target:usize,u:Unit)->f64{
 let b=self.b;let d=b.units[target];if d.owner==0||d.owner==b.me{return -INF}if u.k==10{return 35.+value(d)*3.;}if u.k==38||u.k==39{return 12.+b.tiles[target].lv as f64*4.;}
 let (hit,ret)=b.damage(u,d,pos,target);let kill=hit>=d.hp;let val=value(d);let ownval=value(u);
 let gain=hit.min(d.hp)/maxhp(d)*(val+3.)*2.8 + if kill{6.+val*1.7}else{0.};
 let loss=ret.min(u.hp)/maxhp(u)*(ownval+3.)*2.0+if ret>=u.hp{ownval*2.7+7.}else{0.};
 let mut score=gain-loss;
 if kill{
  let overkill=(hit-d.hp).max(0.);score-=overkill*0.22;
  let other_targets=b.enemy.iter().filter(|&&en|en!=target&&b.d(pos,en)<=stat(u.k).range).count();
  if other_targets>0 {let cheap=b.own.iter().any(|&fr|fr!=src&&!b.units[fr].attacked&&b.d(fr,target)<=stat(b.units[fr].k).range&&value(b.units[fr])<ownval&&b.damage(b.units[fr],d,fr,target).0>=d.hp);if cheap{score-=4.;}}
 }
 if !kill&&b.units[src].hp<=ret{score-=5.;}
 if d.k==10{score+=if kill{12.}else{3.};}if b.tiles[target].st==1{if b.tiles[target].owner==b.me{score+=if kill{30.}else{hit*1.5};}else{score+=if kill&&stat(u.k).range==1{13.}else{hit*0.5};}}
 if kill&&u.k==4{let chain=b.enemy.iter().filter(|&&en|en!=target&&b.d(en,target)==1&&b.units[en].hp<=b.damage(u,b.units[en],target,en).0).count();score+=chain as f64*15.;}
 if u.k==41{for &en in &b.adj[target]{if b.units[en].owner>0&&b.units[en].owner!=b.me{score+=b.damage(u,b.units[en],pos,en).0.min(b.units[en].hp)*0.65;}}}
 if ret<u.hp{let mut after=u;after.hp-=ret;let at=if kill&&stat(u.k).range==1{target}else{pos};let danger=self.threat_loss(src,at,after,if kill{target}else{b.n});let current=self.threat_loss(src,src,u,b.n);score-=(danger-current*0.5).max(0.)*if u.k==3{0.22}else{0.7};}
 if !kill&&hit<d.hp&&ret>0.{// Encourage a finishing team attack, not feeding piecemeal to giants.
 let mut potential=0.;for &fr in &b.own{if fr==src||b.units[fr].attacked{continue}let fu=b.units[fr];let fs=stat(fu.k);let distance=b.d(fr,target);if distance<=fs.range+(if fs.dash&&!fu.moved{fs.mv}else{0}){potential+=b.damage(fu,d,fr,target).0*0.75;}}
 if potential+hit>=d.hp{score+=8.;}else if d.k==12||d.k==45{score-=4.;}
 }
 if u.k==3&&!u.moved{score+=2.;}if kill&&u.kills==2&&!u.vet{score+=7.;}if u.k==8||u.k==41{score+=4.;}score
 }
 fn move_score(&self,a:&Act,m:&Memory)->f64{
 let b=self.b;let src=a.s;let dst=a.t;let u=b.units[src];let s=stat(u.k);if u.owner!=b.me{return -INF}
 let j=self.assignment[src];if j==usize::MAX{return -INF}let g=&self.goals[j];let path=&self.paths[(naval(u.k) as usize)*self.goals.len()+j];let mut progress=path[src]-path[dst];if s.range>=2&&(g.kind==2||g.kind==6){progress=(path[src]-(s.range as f64-0.5)).max(0.)-(path[dst]-(s.range as f64-0.5)).max(0.);}if progress.is_nan()||progress.abs()>INF/2.{progress=0.}
 let mut score=progress*(3.5+g.weight/(path[src]/s.mv as f64+3.)*0.7);
 if g.kind==4&&b.d(dst,g.pos)<=2 {
 progress=0.25;
 if b.tiles[dst].st==1&&b.units[g.pos].owner!=b.me{progress+=0.7;}
 }
 if progress<=0.{score-=2.8;}let radi=if b.tiles[dst].ter==4||u.k==42||u.k==38||u.k==39{2}else{1};let fresh=(0..b.n).filter(|&i|!b.tiles[i].known&&b.d(dst,i)<=radi).count() as f64;score+=fresh*0.75;
 let t=b.tiles[dst];if t.st==1&&t.owner!=b.me{score+=if t.owner==0{29.}else{33.};}if t.st==2{score+=18.;}
 if u.attacked&&s.escape{score=progress*1.5;}
 // Moving onto a port is an intentional transition to naval strategy.
 if !naval(u.k)&&t.st==8&&t.owner==b.me{
 let sea_path=&self.paths[self.goals.len()+j];let nearport=self.goals.iter().enumerate().filter(|(_,g)|g.kind==1||g.kind==2||g.kind==5).map(|(k,g)|g.weight/(self.paths[self.goals.len()+k][dst]+2.)).fold(0.,f64::max);
 score=if self.navneed||path[src]>20.||b.water>0.3{6.+nearport*2.}else{-3.};if u.k==12{score+=8.;}if sea_path[dst]<path[src]{score+=4.;}
 }
 let danger=self.threat_loss(src,dst,u,b.n);let before=self.threat_loss(src,src,u,b.n);
 let mut bestattack:f64=-INF; if s.dash&&!u.attacked&&!(t.st==8&&!naval(u.k)) {for &en in &b.enemy{if b.d(dst,en)<=s.range{bestattack=bestattack.max(self.attack_score(src,dst,en,u));}}}
 if bestattack>0.{score+=bestattack*0.95+3.;}
 if u.k==45{let mut stomp=0.;for &en in &b.adj[dst]{if b.units[en].owner>0&&b.units[en].owner!=b.me{let damage=(b.damage(u,b.units[en],dst,en).0*0.5).floor();stomp+=damage.min(b.units[en].hp)*(value(b.units[en])+2.)/maxhp(b.units[en])*2.;if damage>=b.units[en].hp{stomp+=5.;}}}score+=stomp;}
 if u.k==12||u.k==45{for &en in &b.enemy{if b.units[en].k==10&&b.d(en,dst)==1 {score-=50.;}}}
 score-=danger*if bestattack>8.{0.55}else{1.65};score+=before*0.85;
 if u.hp<maxhp(u)*0.5&&before<2.&&fresh==0.&&t.st!=1&&t.st!=2{score-=6.;}
 if b.tiles[src].st==1&&b.tiles[src].owner!=b.me{return -INF} // Never walk away from a capturable city or ruin.
 if b.tiles[src].st==2&&!u.moved&&!u.attacked{return -INF}
 if b.tiles[src].st==1&&b.tiles[src].owner==b.me{
 let danger=self.citydanger[src];if danger>0.{let near=b.own.iter().filter(|&&f|f!=src&&b.d(f,src)<=1&&!b.units[f].moved).count();let homecount=b.own.iter().filter(|&&f|b.units[f].home==src as i32+1).count();let cantrain=b.stars>=2.&&homecount<=b.cities[src].lv as usize;if near==0&&!cantrain{score-=danger*0.9+if self.threats[src].len()>0{8.}else{0.};}else if cantrain{score+=6.;}}
 score+=if b.stars>=2. {5.5}else{0.};
 }
 // Ranged units seek standoff positions; giants close in, but don't waste a stationary attack.
 if u.k==8||u.k==41{
  let min_enemy=b.enemy.iter().map(|&en|b.d(dst,en)).min().unwrap_or(20);
  let min_before=b.enemy.iter().map(|&en|b.d(src,en)).min().unwrap_or(20);
  if min_enemy==s.range&&min_before>s.range{score+=5.;}
  if min_enemy<s.range{score-=(s.range-min_enemy) as f64*3.;}
 }
 if (u.k==8||u.k==41)&&!u.attacked{let attackable=b.enemy.iter().any(|&en|b.d(src,en)<=s.range);if attackable{score-=30.;}}
 if naval(u.k)&&t.ter>2&&t.st!=1&&t.st!=2&&g.kind!=4{score-=5.;}
 if m.visits.len()==b.n&&m.visits[dst]>=b.turn-2{score-=2.5;}
 if dst==m.last_src&&src==m.last_dst{score-=4.;}
 if u.attacked&&s.escape{score+=self.threat_loss(src,src,u,b.n)*0.4;if danger<before{score+=3.;}}
 // Avoid covering cities unless actually defending them or capturing.
 if t.st==1&&t.owner==b.me&&self.citydanger[dst]<1.{score-=5.;}
 if t.st==1&&t.owner==b.me&&self.citydanger[dst]>1.&&b.units[dst].owner==0{
 let vulnerable=b.enemy.iter().any(|&en|b.d(en,dst)<=stat(b.units[en].k).mv+1);
 if vulnerable&&danger<value(u)+3.{score+=5.;}
 }
 score
 }
 fn bridge_score(&self,pos:usize)->f64{
 let b=self.b;if b.stars<8.{return -5.}let adjacentown=b.own.iter().filter(|&&fr|b.d(fr,pos)<=1&&!naval(b.units[fr].k)&&!b.units[fr].moved).count();if adjacentown==0{return -5.}
 let adjacentland=b.adj[pos].iter().filter(|&&i|b.tiles[i].known&&b.tiles[i].ter>2&&b.tiles[i].ter!=4).count();if adjacentland<2{return -5.}let mut score:f64=-5.;for &src in &b.own{if b.d(src,pos)>1||naval(b.units[src].k){continue}let j=self.assignment[src];let path=&self.paths[j];let g=&self.goals[j];if b.d(src,g.pos)>b.d(pos,g.pos)&&path[src]>b.d(src,g.pos) as f64+3.{score=score.max(9.+adjacentown as f64*2.);}}score
 }
 fn pop_gain(&self,a:&Act)->f64{
 let b=self.b;if a.k==5{return 1.}if a.k!=6{return 0.}let mut gain=match a.p{5|21=>2.,12|8=>1.,6|13=>0.,22=>0.,23..=29=>3.,_=>0.};
 let matches=|p:i32,k:i32|{(p==6&&k==5)||(p==13&&k==12)||(p==22&&k==21)};
 if a.p==6||a.p==13||a.p==22{for &nb in &b.adj[a.t]{if b.tiles[nb].owner==b.me&&matches(a.p,b.tiles[nb].st){gain+=if a.p==22{2.}else{1.};}}}
 if a.p==12||a.p==5||a.p==21{for &nb in &b.adj[a.t]{if b.tiles[nb].owner==b.me&&matches(b.tiles[nb].st,a.p){gain+=if a.p==21{2.}else{1.};}}}
 gain
 }
 fn build_cost(p:i32)->f64{match p{3|12=>3.,5|6|13|21|22|50|48=>5.,8=>7.,17|19|20=>20.,18=>15.,23..=29=>0.,_=>10.}}
 fn economic_score(&self,a:&Act)->f64{
 let b=self.b;let pos=a.t;let c=b.territory[pos];let gain=self.pop_gain(a);if c>=b.n||gain<=0.{return -INF}let city=&b.cities[c];let cost=if a.k==5{2.}else{Self::build_cost(a.p)};
 let need=(city.lv+1-city.prog) as f64;let complete=gain>=need;let danger=self.citydanger[c];let threatened=b.units[c].owner>0&&b.units[c].owner!=b.me;
 let mut score=2.0+gain*1.7-cost*0.6;
 let reward=match city.lv{1=>19.,2=>16.,3=>15.,_=>38.};
 let mut opts:Vec<(f64,f64)>=vec![];
 for ac in &b.acts{if (ac.k==5||ac.k==6)&&ac.t!=pos&&b.territory[ac.t]==c{let g=self.pop_gain(ac);if g>0.{opts.push((if ac.k==5{2.}else{Self::build_cost(ac.p)},g));}}}
 opts.sort_by(|a,b|(a.0/a.1).partial_cmp(&(b.0/b.1)).unwrap());let mut remain=(need-gain).max(0.);let mut totalcost=cost;for (co,ga) in opts{if remain<=0.{break}totalcost+=co;remain-=ga;}
 let waits=((totalcost-b.stars).max(0.)/b.income.max(1.)).ceil();
 if remain<=0.{score+=reward/(1.+waits*0.6)*gain.min(need)/need.max(1.);if complete{score+=reward*0.45;}}
 else{score-=3.;}
 if city.lv>=5{score-=city.lv as f64*0.25;}
 if b.stars-cost<2.&&b.own.len()<=b.mycities.len(){score-=5.;}
 // Economy pays for itself early; in a direct attack reserve a military budget.
 if danger>0.&&!(complete&&city.lv>=4){score-=danger.min(12.)*0.5;}
 if threatened&&!(complete&&city.lv>=4){score-=20.;}
 if a.k==6&&a.p==8{score-=6.;}
 if a.k==6&&(a.p==6||a.p==13||a.p==22){score+=gain*0.9;}
 // Prefer the most cost efficient actions, preserve multiplier locations and resources.
 score+=gain/cost.max(1.)*3.;
 if a.k==6&&a.p==12&&b.tiles[pos].res==1{score-=2.5;}
 if a.k==6&&a.p!=12&&b.tiles[pos].res==6{score-=2.;}
 if b.turn>150{score-=3.;}if !complete{score-=self.reserve_penalty(cost);}
 score
 }
 fn research_score(&self,p:i32)->f64{
 let b=self.b;let nc=b.mycities.len() as f64;let tier=match p{1|6|10|15|20=>1.,2|4|7|8|13|16|18|21|23|44=>2.,_=>3.};let cost=4.+nc*tier;let spare=b.stars-cost;let pressure=self.citydanger.iter().copied().fold(0.,f64::max);let count=|res:i32|b.tiles.iter().filter(|t|t.owner==b.me&&t.res==res&&t.st==0).count() as f64;
 let woods=b.tiles.iter().filter(|t|t.owner==b.me&&t.ter==5&&t.st==0).count() as f64;
 let mut score=match p{
 1=>if b.turn>=2&&b.own.len()<6 {11.}else{1.},
 16=>if woods>=3.{16.+woods.min(10.)*0.8}else{1.},
 17=>{let huts=b.tiles.iter().filter(|t|t.owner==b.me&&t.st==12).count() as f64; if huts>=2. {19.+huts.min(12.)*0.8+if !b.enemy.is_empty(){4.}else{0.}}else if woods>=4.&&b.tech[16]{8.}else{0.}},
 6=>if count(6)>=2.{7.+count(6)*1.2}else if b.turn>15{2.}else{-5.},
 10=>if self.navneed{17.}else if count(3)>=2.{8.+count(3)}else if b.water>0.2&&b.turn>=4{5.}else{-5.},
 20=>{let mountains=b.tiles.iter().filter(|t|t.known&&t.ter==4).count();let stuck=b.own.iter().filter(|&&src|{let j=self.assignment[src];j!=usize::MAX&&self.paths[j][src]>=INF/2.}).count();if stuck>0||mountains>=5&&b.turn>=2{10.+(stuck as f64)*3.}else if mountains>=3&&b.turn>=9{7.}else{-4.}},
 13=>if b.water>0.25&&b.turn>=7 {if self.navneed||b.units.iter().any(|u|naval(u.k)){19.}else{10.}}else{-2.},
 44=>if b.units.iter().any(|u|u.owner==b.me&&u.k==43)||b.water>0.3&&b.turn>15{14.}else{-5.},
 14=>if b.water>0.25&&b.turn>16 {17.+if b.units.iter().any(|u|u.owner==b.me&&u.k==43){5.}else{0.}}else{-8.},
 18=>if pressure>3.&&!b.tech[17]{10.}else if b.turn>15&&b.enemy.len()>2{9.}else{0.},
 4=>if b.mycities.len()>=2&&b.turn>8{7.}else{-5.},
 8=>if count(2)>=2.{7.+count(2)}else{-5.},
 9=>if count(2)>=4.{8.}else{-10.},
 23=>if count(5)>0.{9.+count(5)*1.5}else if b.turn>22&&pressure>0. {7.}else{-4.},
 24=>if b.turn>14&&pressure>2.{16.}else if b.turn>25{9.}else{-5.},
 2=>if b.turn>16&&b.stars>cost+15.{8.}else{-7.},
 3=>if b.turn>16&&b.stars>cost+8.{17.}else{-5.},
 7=>if pressure>6.&&b.stars>=cost+3.{9.}else{-8.},
 5=>if b.tech[17]&&b.mycities.len()>=3&&b.stars>=cost+8.{10.}else{-6.},
 21=>if b.turn>25&&b.stars>cost+15.{3.}else{-10.},
 22=>if b.turn>30&&b.stars>cost+10.{10.}else{-10.},
 _=>-15.};
 score-=cost*0.35; if spare<2.{score-=7.;}if spare<3.&&b.own.len()<2{score-=10.;}if pressure>5.&&spare<5.{score-=8.;}
 if b.stars>cost+b.income {score+=4.;}if b.turn<=1{score-=20.;}
 score
 }
 fn reserve_penalty(&self,cost:f64)->f64{
 let b=self.b;if b.turn<4||b.own.len()<3{return 0.}
 let mut reserve=0.;let nc=b.mycities.len() as f64;
 let woods=b.tiles.iter().filter(|t|t.owner==b.me&&t.ter==5&&t.st==0).count();
 let huts=b.tiles.iter().filter(|t|t.owner==b.me&&t.st==12).count();
 if !b.tech[16]&&woods>=3{reserve=4.+nc*2.;}
 else if !b.tech[17]&&huts>=2{reserve=4.+nc*3.;}
 if b.water>0.3&&b.units.iter().any(|u|u.owner==b.me&&u.k==43)&&!b.tech[13]&&!b.tech[44]{reserve=4.+nc*2.;}
 if reserve==0.||b.stars>=reserve+cost{return 0.}let needed=(reserve-b.stars).max(0.);if needed>b.income*1.5{return 0.}
 if b.stars>=reserve {4.}else{3.}
 }
 fn train_score(&self,a:&Act)->f64{
 let b=self.b;let src=a.s;let k=a.p;let cost=if (41..=44).contains(&k){match k{41=>15.,_=>5.}}else{stat(k).cost};let danger=self.citydanger[src];let mut nearest=30;let mut _enemies=0.;for &en in &b.enemy{nearest=nearest.min(b.d(en,src));if b.d(en,src)<7{_enemies+=value(b.units[en]);}}
 let ownnear=b.own.iter().filter(|&&fr|b.d(fr,src)<=3).count() as f64;let total=b.own.len() as f64;let nc=b.mycities.len() as f64;let mut bestgoal:f64=0.;let mut dist:f64=99.;
 for (j,g) in self.goals.iter().enumerate(){let d=self.paths[j][src];let v=g.weight/(d+2.);if v>bestgoal{bestgoal=v;dist=d;}}
 let mut score=6.+bestgoal*0.7+danger.min(15.)*0.5;
 match k{
 2=>{if total<2.{score+=14.;}else if total<nc+1.{score+=7.;}if b.tech[1]{score-=3.;}if nearest<=3{score+=4.;}if b.tech[24]||b.tech[17]{score-=5.;}if b.turn>20{score-=5.;}},
 3=>{score+=5.;if dist>3.{score+=3.;}if total<nc*1.5+1.{score+=5.;}if nearest<=2{score-=7.;}if b.turn>20&&total>nc*1.5{score-=5.;}},
 4=>{score+=6.;let squishy=b.enemy.iter().filter(|&&en|b.d(src,en)<=7&&b.units[en].hp<=10.).count();score+=squishy.min(5) as f64*2.;if nearest<=2 {score-=6.;}},
 5=>{let defenders=b.own.iter().filter(|&&fr|b.units[fr].k==5&&b.d(fr,src)<=3).count();score+=if danger>5.&&defenders==0 {17.}else{-10.};if b.water>0.3&&self.navneed{score+=3.;}},
 8=>{let cats=b.own.iter().filter(|&&f|b.units[f].k==8).count();let fighters=total-cats as f64;score+=if nearest<=6{9.}else{1.};if (cats as f64)< fighters*0.55{score+=6.;}else{score-=8.;}if nearest<=2&&ownnear<3.{score-=25.;}},
 9=>{score+=if nearest<=4{7.}else{0.};if b.tech[17]{score-=4.;}},
 10=>{let big=b.enemy.iter().any(|&en|b.d(src,en)<=5&&(b.units[en].k==12||b.units[en].k==45));let num=b.own.iter().filter(|&&fr|b.units[fr].k==10).count();score+=if big&&num<2{12.}else{-12.};},
 11=>{score+=9.;if nearest<=4{score+=5.;}},
 38=>{score-=12.;},
 41|42|44=>{score+=12.;if b.units[src].k==43{score+=12.;if b.units[src].hp<5.{score-=9.;}}if k==41 {score+=if b.turn>16{9.}else{-2.};if nearest<3{score-=7.;}}if k==44{score+=if nearest<=4{4.}else{-1.};}if k==42{score+=if b.turn<20{4.}else{-3.};}},
 _=>score-=15.,
 }
 if ownnear>2.{score-=(ownnear-2.)*3.;}if total>nc*3.+2.{score-=(total-nc*3.-2.)*2.;}
 if dist>=INF/2.&&b.tiles.iter().all(|t|t.st!=8||t.owner!=b.me){score-=20.;}
 if total>=nc+1.&&nearest>=6{score-=4.;}
 score-=cost*0.65;
 if b.stars-cost<2.&&danger<5.&&total>nc{score-=4.;}
 if b.stars>cost+b.income*1.5 {score+=5.;}if b.stars>cost+b.income*3.{score+=6.;}
 if b.turn==0{score-=25.;}if nearest>3{score-=self.reserve_penalty(cost);}
 score
 }
 fn reward_score(&self,a:&Act)->f64{let b=self.b;match a.p{
 3=>1000.,4=>920.,7=>1000.,1=>if self.citydanger[a.t]>15. {1010.}else{930.},6=>1100.,2=>850.,
 8=>{let mut pop=0.;for i in 0..b.n{let t=b.tiles[i];if b.territory[i]==a.t&&t.st==0{if t.res==1||t.res==6||t.res==3{pop+=1.;}if t.ter==5{pop+=1.;}}}if pop<3.{950.}else{1010.}},
 5=>{let mut newpop=0.;for i in 0..b.n{let t=b.tiles[i];if b.d(a.t,i)==2&&t.owner==0&&t.known{if t.res==1||t.res==3||t.res==6{newpop+=1.;}if t.ter==5{newpop+=1.;}}}950.+newpop*5.},_=>900.}}
 fn score(&self,a:&Act,m:&Memory)->f64{
 let b=self.b;
 match a.k{
 9=>self.reward_score(a),8=>if b.tiles[a.s].st==1{500.}else{150.},
 2=>self.attack_score(a.s,a.s,a.t,b.units[a.s])+5.,
 1=>self.move_score(a,m),
 5=>self.economic_score(a),
 6=>{match a.p{3=>self.roadscore[a.t],48=>self.bridge_score(a.t),8=>{let ports=b.tiles.iter().filter(|t|t.st==8&&t.owner==b.me).count();let near=b.own.iter().filter(|&&fr|!naval(b.units[fr].k)&&b.d(fr,a.t)<=3).count();let mut wateraround=0.;for i in 0..b.n{if b.d(i,a.t)<=3&&b.tiles[i].known&&b.tiles[i].ter<=2{wateraround+=1.;}}
 self.economic_score(a).max(if self.navneed&&near>0&&ports<3{13.+wateraround*0.25+if ports==0{8.}else{0.}-ports as f64*5.}else{-8.})},
 50=>{let mut stars=0.;for &i in &b.adj[a.t]{if b.tiles[i].owner==b.me&&(b.tiles[i].st==6||b.tiles[i].st==13||b.tiles[i].st==22){stars+=b.tiles[i].lv as f64;}}if stars>=2.{10.+stars*2.}else{-5.}},
 _=>self.economic_score(a)}},
 7=>self.research_score(a.p),4=>self.train_score(a),
 3=>match a.p{16=>{let u=b.units[a.s];if u.hp<maxhp(u)-1.||u.attacked{300.}else{2.}},
 7=>{let u=b.units[a.s];let missing=maxhp(u)-u.hp;let danger=self.threat_loss(a.s,a.s,u,b.n);let mut score=if missing>=4.{3.+missing*0.25}else{-2.};if u.k==12||u.k==45{score+=if u.hp<24.{6.}else if u.hp<33.{2.}else{-3.};}if danger>5.{score-=danger*0.45;}if b.tiles[a.s].owner==b.me{score+=2.;}score},
 9=>{let mut heal=0.;for &i in &b.adj[a.s]{if b.units[i].owner==b.me{heal+=(maxhp(b.units[i])-b.units[i].hp).min(4.);}}heal*2.},
 2=>{let t=b.tiles[a.t];if t.res==1{-30.}else{-8.}},
 19=>if a.t==0{1.}else{-50.},18=>if b.turn>30{1.}else{-20.},_=>-100.},
 10=>0.,_=>-INF,
 }
 }
}
fn choose(o:&Value,mem:&mut Memory)->String{
 let mut b=Board::parse(o);if mem.walls.len()!=b.n{mem.walls=vec![false;b.n];}if let Some((target,oldhp,expected))=mem.lastatk.take(){let u=b.units[target];if u.owner>0&&u.owner!=b.me&&oldhp-u.hp+1.2<expected{mem.walls[target]=true;}}b.walls=mem.walls.clone();if mem.visits.len()!=b.n{mem.visits=vec![-100;b.n];mem.last_turn=-1;}if b.turn!=mem.last_turn{mem.round_actions=0;mem.last_turn=b.turn;}mem.round_actions+=1;
 let e=Eval::new(&b);let mut best=-INF;let mut chosen=0;
 for (i,a) in b.acts.iter().enumerate(){let mut s=e.score(a,mem);if mem.round_actions>150&&a.k!=9&&a.k!=8&&a.k!=2&&a.k!=10{s-=100.;}s+=((a.s*17+a.t*31+a.p.max(0) as usize*7)%97) as f64*0.00001;
 if s>best{best=s;chosen=i;}}

 let a=&b.acts[chosen];if a.k==2&&b.tiles[a.t].st==1&&b.tiles[a.t].owner!=b.me&&b.units[a.s].k!=10{mem.lastatk=Some((a.t,b.units[a.t].hp,b.damage(b.units[a.s],b.units[a.t],a.s,a.t).0));}if a.k==1{mem.visits[a.s]=b.turn;mem.last_src=a.s;mem.last_dst=a.t;}
 a.id.clone()
}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){if let Ok(line)=line{if let Ok(o)=serde_json::from_str::<Value>(&line){let id=choose(&o,&mut mem);writeln!(out,"{}",json!({"action_id":id})).unwrap();out.flush().unwrap();}}}}
