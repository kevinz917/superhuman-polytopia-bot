use serde_json::{json,Value};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn num(v:&Value,k:&str)->i32 {v[k].as_i64().unwrap_or(0) as i32}
fn flag(v:&Value,k:&str)->bool{v[k].as_bool().unwrap_or(false)}
#[derive(Clone,Copy,Default,Debug)]
struct Stats {hp:f64,a:f64,d:f64,m:i32,r:i32,c:f64,dash:bool,stiff:bool,fort:bool,boat:bool}
fn stats(k:i32)->Stats {let (hp,a,d,m,r,c,dash,stiff,fort,boat)=match k{
2=>(10.,2.,2.,1,1,2.,true,false,true,false),3=>(10.,2.,1.,2,1,3.,true,false,true,false),4=>(10.,3.5,1.,3,1,8.,true,false,true,false),5=>(15.,1.,3.,1,1,3.,false,false,true,false),8=>(10.,4.,0.,1,3,8.,false,true,false,false),9=>(10.,2.,1.,1,2,3.,true,false,true,false),10=>(10.,0.,1.,1,1,5.,false,true,false,false),11=>(15.,3.,3.,1,1,5.,true,false,false,false),12=>(40.,5.,4.,1,1,15.,false,false,false,false),37=>(10.,2.,2.,1,1,2.,true,false,true,false),38=>(5.,0.,0.5,2,1,8.,true,true,false,false),39=>(5.,0.,0.5,2,1,8.,true,true,false,true),40=>(10.,2.,2.,2,1,4.,true,false,false,true),41=>(10.,3.,2.,2,3,15.,false,true,false,true),42=>(10.,2.,1.,3,2,5.,true,false,false,true),43=>(10.,0.,1.,2,2,0.,false,true,false,true),44=>(10.,3.,3.,3,1,5.,true,false,false,true),45=>(40.,4.,4.,2,1,15.,false,true,false,true),_=>(10.,2.,2.,1,1,3.,true,false,false,false)}; Stats{hp,a,d,m,r,c,dash,stiff,fort,boat}}
#[derive(Clone,Default,Debug)]
struct Tile{known:bool,ter:i32,owner:i32,st:i32,res:i32,road:bool,cap:i32,level:i32}
#[derive(Clone,Default,Debug)]
struct Unit{pos:usize,owner:i32,k:i32,hp:f64,max:f64,moved:bool,attacked:bool,veteran:bool,kills:i32,home:i32,pass:i32}
#[derive(Clone,Default,Debug)]
struct City{pos:usize,owner:i32,level:i32,progress:i32,prod:i32,connected:bool,rewards:Vec<i32>}
#[derive(Clone,Default)]
struct Act{id:String,k:i32,s:usize,t:usize,p:i32}
struct Board{size:usize,n:usize,me:i32,turn:i32,stars:f64,tiles:Vec<Tile>,units:Vec<Unit>,at:Vec<Option<usize>>,cities:Vec<City>,cityat:Vec<Option<usize>>,tc:Vec<Option<usize>>,tech:[bool;128],adj:Vec<Vec<usize>>,acts:Vec<Act>,income:f64}
impl Board{
fn dist(&self,a:usize,b:usize)->i32 {((a%self.size) as i32-(b%self.size) as i32).abs().max(((a/self.size) as i32-(b/self.size) as i32).abs())}
fn parse(o:&Value)->Self{let v=&o["visible_state"];let size=num(v,"board_size") as usize; let n=size*size;let me=num(o,"player");let mut b=Board{size,n,me,turn:num(v,"turn"),stars:num(v,"stars") as f64,tiles:vec![Tile::default();n],units:vec![],at:vec![None;n],cities:vec![],cityat:vec![None;n],tc:vec![None;n],tech:[false;128],adj:vec![vec![];n],acts:vec![],income:0.};
for x in v["technology"].as_array().unwrap(){let i=x.as_u64().unwrap_or(0) as usize;if i<128{b.tech[i]=true;}}
for t in v["tiles"].as_array().unwrap(){let i=num(t,"tile") as usize;if i<n{b.tiles[i]=Tile{known:flag(t,"known"),ter:num(t,"terrain"),owner:num(t,"owner"),st:num(t,"structure"),res:num(t,"resource"),road:num(t,"road")>0,cap:num(t,"capital"),level:num(t,"structure_level")};}}
for u in v["units"].as_array().unwrap(){let pos=num(u,"tile") as usize;let k=num(u,"kind");let pass=num(u,"passenger");let hp=u["health"].as_f64().unwrap_or(10.);let max=(stats(if pass>0{pass}else{k}).hp+if flag(u,"veteran"){5.}else{0.}).max(hp);b.at[pos]=Some(b.units.len());b.units.push(Unit{pos,owner:num(u,"owner"),k,hp,max,moved:flag(u,"moved"),attacked:flag(u,"attacked"),veteran:flag(u,"veteran"),kills:num(u,"kills"),home:num(u,"home"),pass});}
for c in v["cities"].as_array().unwrap(){let pos=num(c,"tile") as usize;let city=City{pos,owner:num(c,"owner"),level:num(c,"level"),progress:num(c,"progress"),prod:num(c,"production"),connected:flag(c,"connected"),rewards:c["rewards"].as_array().map(|a|a.iter().map(|x|x.as_i64().unwrap_or(0) as i32).collect()).unwrap_or(vec![0;9])};if city.owner==me{b.income+=city.prod as f64+if b.tiles[pos].cap==me{1.}else{0.};}b.cityat[pos]=Some(b.cities.len());b.cities.push(city);}
for i in 0..n{let (x,y)=((i%size) as i32,(i/size) as i32);for dy in -1..=1{for dx in -1..=1{let (xx,yy)=(x+dx,y+dy);if (dx!=0||dy!=0)&&xx>=0&&yy>=0&&xx<size as i32&&yy<size as i32{b.adj[i].push(yy as usize*size+xx as usize);}}}if b.tiles[i].owner==me{b.tc[i]=b.cities.iter().enumerate().filter(|(_,c)|c.owner==me&&b.dist(c.pos,i)<=if c.rewards[5]>0{2}else{1}).min_by_key(|(_,c)|b.dist(c.pos,i)).map(|(j,_)|j);}}
for a in o["legal_actions"].as_array().unwrap(){let c=&a["command"];b.acts.push(Act{id:a["action_id"].as_str().unwrap().to_string(),k:num(c,"kind"),s:num(c,"source").max(0) as usize,t:num(c,"target").max(0) as usize,p:num(c,"parameter")});}b}
fn bonus(&self,u:&Unit,pos:usize)->f64{let t=&self.tiles[pos];let own=u.owner==self.me; if t.ter==4{1.5}else if t.ter==5&&(!own||self.tech[18]){1.5}else if (t.ter==1||t.ter==2)&&own&&self.tech[12]{1.5}else if t.st==1&&t.owner==u.owner&&stats(u.k).fort{if let Some(c)=self.cityat[pos]{if self.cities[c].rewards[1]>0{return 4.;}}1.5}else{1.}}
fn damage(&self,u:&Unit,pos:usize,d:&Unit,dp:usize)->(f64,f64){if u.k==10{return(d.hp,0.);}let a=stats(u.k);let ds=stats(d.k);let af=a.a*u.hp/u.max;let df=ds.d*d.hp/d.max*self.bonus(d,dp);let sum=(af+df).max(0.01);let hit=(af/sum*a.a*4.5+0.500001).floor();let ret=if hit>=d.hp||ds.stiff||self.dist(pos,dp)>ds.r||u.k==37||u.k==40{0.}else{(df/sum*ds.d*4.5+0.500001).floor()};(hit,ret)}
fn val(&self,u:&Unit)->f64{let s=stats(u.k);let cost=s.c+if s.boat{stats(u.pass).c.max(2.)}else{0.};(cost+3.)*(if u.veteran{1.2}else{1.})}
fn reach(&self,u:&Unit,enemy:bool)->Vec<usize>{let st=stats(u.k);let mut d=vec![1000;self.n];let mut heap=BinaryHeap::new();d[u.pos]=0;heap.push(Reverse((0,u.pos)));let mut out=vec![u.pos];let budget=2*st.m;while let Some(Reverse((di,p)))=heap.pop(){if di!=d[p]||di>=budget{continue;}let tp=&self.tiles[p];let roadp=tp.road||tp.st==1||tp.st==48;let terminal=p!=u.pos&&(tp.ter==4||tp.ter==5&&!roadp||!st.boat&&tp.st==8||st.boat&&tp.ter>2||self.adj[p].iter().any(|&q|self.at[q].map(|j|self.units[j].owner!=u.owner).unwrap_or(false)));if terminal{continue;}
for &q in &self.adj[p]{let t=&self.tiles[q];if !t.known{continue;}if self.at[q].map(|j|self.units[j].owner!=u.owner).unwrap_or(false){continue;}let water=t.ter==1||t.ter==2;if t.ter==4&&!enemy&&!self.tech[20]||water&&!st.boat&&t.st!=48&&!(t.st==8&&t.owner==u.owner)||t.ter==2&&!enemy&&!self.tech[13]{continue;}let roadq=(t.road||t.st==1||t.st==48)&&(t.owner==0||t.owner==u.owner);let nd=if t.ter==5&&!(roadp&&roadq){budget}else{di+if roadp&&roadq{1}else{2}};if nd<=budget&&nd<d[q]{d[q]=nd;heap.push(Reverse((nd,q)));if self.at[q].is_none(){out.push(q);}}}}
out}
}
#[derive(Default)]struct Memory{turn:i32,visits:Vec<i32>,prev:Vec<usize>}
struct Eval<'a>{b:&'a Board,enemy:Vec<usize>,own:Vec<usize>,reach:Vec<Vec<usize>>,threat:Vec<Vec<usize>>,support:Vec<f64>,obj:Vec<(usize,f64,i32)>,paths:Vec<Vec<f64>>,assigned:Vec<Option<usize>>,growth:Vec<f64>,water:f64,navneed:bool,reserve:f64,roads:Vec<f64>}
impl<'a> Eval<'a>{
fn new(b:&'a Board)->Self{let own:Vec<_>=b.units.iter().enumerate().filter(|(_,u)|u.owner==b.me).map(|(i,_)|i).collect();let enemy:Vec<_>=b.units.iter().enumerate().filter(|(_,u)|u.owner!=b.me).map(|(i,_)|i).collect();let mut reach=vec![vec![];b.units.len()];let mut threat=vec![vec![];b.n];let mut support=vec![0.;b.n];for (j,u) in b.units.iter().enumerate(){reach[j]=b.reach(u,u.owner!=b.me);let st=stats(u.k);if st.a<=0.&&u.k!=10{continue;}let from=if st.dash{reach[j].clone()}else{vec![u.pos]};for t in 0..b.n{if from.iter().any(|&p|p!=t&&b.dist(p,t)<=st.r){if u.owner!=b.me{threat[t].push(j);}else{support[t]+=st.a*u.hp/u.max;}}}}
let water=b.tiles.iter().filter(|t|t.known&&t.ter<=2).count() as f64/(b.tiles.iter().filter(|t|t.known).count().max(1) as f64);
let mut obj=vec![];for c in &b.cities{if c.owner!=b.me{obj.push((c.pos,if c.owner==0{32.}else{27.},if c.owner==0{0}else{1}));}else if !threat[c.pos].is_empty(){obj.push((c.pos,24.,2));}}
for i in 0..b.n{let t=&b.tiles[i];if t.st==2{obj.push((i,22.,3));}if t.known&&t.ter>2&&b.adj[i].iter().any(|&j|!b.tiles[j].known){let n=(0..b.n).filter(|&q|!b.tiles[q].known&&b.dist(i,q)<=if t.ter==4{2}else{1}).count();obj.push((i,5.+n as f64*0.55,4));}}
for &j in &enemy{if !obj.iter().any(|(p,_,_)|b.dist(*p,b.units[j].pos)<=1){obj.push((b.units[j].pos,12.,5));}}
// Water exploration is important when all reachable land is settled.
for i in 0..b.n{if b.tiles[i].known&&b.tiles[i].ter<=2&&b.adj[i].iter().any(|&j|!b.tiles[j].known){obj.push((i,9.,4));}}
let mut ev=Eval{b,own,enemy,reach,threat,support,obj,paths:vec![],assigned:vec![None;b.units.len()],growth:vec![0.;b.cities.len()],water,navneed:false,reserve:0.,roads:vec![-100.;b.n]};
for (p,_,_) in &ev.obj{ev.paths.push(ev.path(*p,false));}
// allocate nearest capturers, then remaining soldiers spread over the front
let mut loads=vec![0.;ev.obj.len()];let mut pairs=vec![];for &j in &ev.own{let u=&b.units[j];for (oi,&(_,w,kind)) in ev.obj.iter().enumerate(){let d=ev.paths[oi][u.pos];if d<900.{let bonus=if kind==0||kind==3{1.3}else{1.};pairs.push((w*bonus/(d+2.),j,oi));}}}pairs.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());for (_,j,_) in &pairs{if ev.assigned[*j].is_some(){continue;}let u=&b.units[*j];let mut best=0.;let mut goal=None;for (oi,&(p,w,kind)) in ev.obj.iter().enumerate(){let d=ev.paths[oi][u.pos];let discount=if kind==0||kind==3||kind==4{1.+loads[oi]*3.0}else{1.+loads[oi]*0.22};let score=w/(d+2.)/discount;if score>best&&!(kind==0&&b.at[p].map(|x|b.units[x].owner==b.me&&x!=*j).unwrap_or(false)){best=score;goal=Some(oi);}}if let Some(oi)=goal{ev.assigned[*j]=Some(oi);loads[oi]+=1.;}}
let mut landseen=vec![false;b.n];let mut stack=vec![];for c in &b.cities{if c.owner==b.me{landseen[c.pos]=true;stack.push(c.pos);}}while let Some(p)=stack.pop(){for &q in &b.adj[p]{let t=&b.tiles[q];if !landseen[q]&&t.known&&(t.ter>=3&&(t.ter!=4||b.tech[20])||t.st==48){landseen[q]=true;stack.push(q);}}}
let landgoals=ev.obj.iter().any(|(p,_,k)|(*k==0||*k==1||*k==3)&&landseen[*p]);
let reachable_targets=ev.obj.iter().enumerate().any(|(oi,(_,_,k))|*k<2&&ev.own.iter().any(|&j|ev.paths[oi][b.units[j].pos]<900.));ev.navneed=water>0.14&&(!landgoals||!reachable_targets||b.turn>18);
let cities=b.cities.iter().filter(|c|c.owner==b.me).count() as f64;
if b.income>=9.&&ev.enemy.len()>=3&&ev.own.len()>=4 {
let target=if b.tech[2]&&!b.tech[3]{3}else if b.tech[23]&&!b.tech[24]{24}else if !b.tech[2]&&b.tech[1] {2}else{0};
if target>0{let cost=4.+cities*if target==2{2.}else{3.};if b.stars<cost+8.{ev.reserve=cost+8.;}}
}
ev}
fn path(&self,goal:usize,boat:bool)->Vec<f64>{let b=self.b;let mut d=vec![10000.;b.n];let mut heap=BinaryHeap::new();d[goal]=0.;heap.push(Reverse((0,goal)));while let Some(Reverse((cost,p)))=heap.pop(){if cost as f64>d[p]*10.+0.1{continue;}for &q in &b.adj[p]{let t=&b.tiles[q];let tp=&b.tiles[p];let pw=tp.ter==1||tp.ter==2;let qw=t.ter==1||t.ter==2;if !t.known{continue;}if t.ter==4&&!b.tech[20]{continue;}
let mut step=10;if pw&&!qw&&tp.st!=48&&!(tp.st==8&&tp.owner==b.me){continue;}if pw&&(!b.tech[10]||tp.ter==2&&!b.tech[13]){continue;}if qw&&(!b.tech[10]||t.ter==2&&!b.tech[13]){continue;}
if tp.ter==5||tp.ter==4{step=14;}if t.road&&tp.road{step=6;}if self.b.at[p].map(|j|b.units[j].owner!=b.me).unwrap_or(false){step+=12;}let nd=cost+step;if (nd as f64)<d[q]*10.{d[q]=nd as f64/10.;heap.push(Reverse((nd,q)));}}}d}
fn risk(&self,u:&Unit,pos:usize,hp:f64,exclude:Option<usize>)->f64{let b=self.b;let mut hits=vec![];let mut v=u.clone();v.hp=hp;for &j in &self.threat[pos]{if Some(j)==exclude{continue;}let e=&b.units[j];let st=stats(e.k);let range=b.dist(e.pos,pos);let mut q=e.pos; if range>st.r{if let Some(&p)=self.reach[j].iter().filter(|&&p|b.dist(p,pos)<=st.r&&p!=pos).min_by_key(|&&p|b.dist(p,pos)){q=p;}}
let (dam,ret)=b.damage(e,q,&v,pos);let cover=self.support[e.pos];let factor=if dam>=hp{1.}else if cover>st.a+3.&&ret>=e.hp{0.35}else{1.};hits.push(dam*factor);}
hits.sort_by(|a,b|b.partial_cmp(a).unwrap());let mut harm=0.;for (i,hit) in hits.iter().enumerate(){harm+=hit*if i==0{1.}else if i==1{0.85}else{0.65};}if stats(u.k).d<=1.&&self.enemy.iter().any(|&j|b.units[j].k==4&&b.dist(b.units[j].pos,pos)<=4){
 let clustered=b.adj[pos].iter().filter(|&&p|b.at[p].map(|j|b.units[j].owner==b.me&&stats(b.units[j].k).d<=1.).unwrap_or(false)).count();harm+=clustered as f64*1.0;
}
let val=b.val(u);let fraction=(harm/hp.max(1.)).min(1.); let loss=val*(harm/u.max).min(1.)*0.36+if harm>=hp{val*0.9}else{val*0.25*fraction*fraction};loss}
fn knight_chain(&self,u:&Unit,pos:usize,taken:&mut [bool],depth:usize)->f64{
 if depth>=6{return 0.;}let b=self.b;let mut best:f64=0.;let mut options=0;
 for &p in &b.adj[pos]{if let Some(j)=b.at[p]{let e=&b.units[j];if e.owner==b.me||taken[j]{continue;}let (d,_)=b.damage(u,pos,e,p);if d<e.hp{continue;}
 taken[j]=true;let val=b.val(e)*(e.hp/e.max*1.1+0.8)+2.+self.knight_chain(u,p,taken,depth+1);taken[j]=false;best=best.max(val);options+=1;if options>=5{break;}}}best}
fn attack_score(&self,j:usize,pos:usize,target:usize)->f64{let b=self.b;let u=&b.units[j];let Some(ei)=b.at[target] else{return -100.;};let e=&b.units[ei];if e.owner==u.owner{return -100.;}let (dam,ret)=b.damage(u,pos,e,target);let kill=dam>=e.hp;if dam<=0. {return -100.;}let val=b.val(e);let mut score=dam.min(e.hp)/e.max*val*1.1+if kill{val*0.8+2.}else{0.};score-=ret.min(u.hp)/u.max*b.val(u)*0.95;if ret>=u.hp{score-=b.val(u)*0.8;}
let at=if kill&&stats(u.k).r==1{target}else{pos};let afterrisk=if ret<u.hp{self.risk(u,at,u.hp-ret,if kill{Some(ei)}else{None})}else{0.};let nowrisk=self.risk(u,u.pos,u.hp,None);score-=afterrisk*0.40;score+=nowrisk*0.25;
if b.tiles[target].st==1{if b.tiles[target].owner==b.me{score+=if kill{55.}else{25.};}else if kill&&stats(u.k).r==1{score+=10.;}}
if !kill&&dam>0. {
let mut hurt=e.clone();hurt.hp=(e.hp-dam).max(1.);let mut follow=0.;
for &aj in &self.own {if aj==j{continue;}let au=&b.units[aj];if au.attacked{continue;}let ast=stats(au.k);let from=if !au.moved&&ast.dash{&self.reach[aj][..]}else{std::slice::from_ref(&au.pos)};
for &ap in from{if ap!=target&&b.dist(ap,target)<=ast.r{let (d2,r2)=b.damage(au,ap,&hurt,target);let (d0,_)=b.damage(au,ap,e,target);if d2>=hurt.hp&&d0<e.hp&&r2<au.hp{let f=val*0.30+2.;if f>follow{follow=f;}}}}
}score+=follow;
}
if u.k==10{score+=val*1.5;}if u.k==4&&kill{let mut taken=vec![false;b.units.len()];taken[ei]=true;score+=self.knight_chain(u,target,&mut taken,0)*0.85;}
if u.k==41{for &p in &b.adj[target]{if let Some(x)=b.at[p]{let e=&b.units[x];if e.owner!=b.me{let (d,_)=b.damage(u,pos,e,p);score+=((d*0.5).floor().min(e.hp)/e.max)*b.val(e)*1.5;}}}}
score*4.+12.}
}
impl<'a> Eval<'a>{
fn pop_gain(&self,a:&Act)->f64 {let b=self.b;match a.k{5=>1.,6=>match a.p{5|21=>2.+b.adj[a.t].iter().filter(|&&p|b.tiles[p].owner==b.me&&b.tiles[p].st==if a.p==5{6}else{22}).count() as f64*if a.p==21{2.}else{1.},12=>1.+b.adj[a.t].iter().filter(|&&p|b.tiles[p].owner==b.me&&b.tiles[p].st==13).count() as f64,8=>1.,6|13|22=>{let st=if a.p==6{5}else if a.p==13{12}else{21};let n=b.adj[a.t].iter().filter(|&&p|b.tiles[p].owner==b.me&&b.tiles[p].st==st).count() as f64;n*if a.p==22{2.}else{1.}},23..=29=>3.,_=>0.},_=>0.}}
fn cost(&self,a:&Act)->f64{match a.k{5=>2.,4=>stats(a.p).c,6=>match a.p{3|12=>3.,8=>7.,23..=29=>0.,17=>20.,18=>15.,19|20=>20.,_=>5.},7=>{let tier=match a.p{1|6|10|15|20=>1,2|4|7|8|13|16|18|21|23|44=>2,_=>3};let x=4.+self.b.cities.iter().filter(|c|c.owner==self.b.me).count() as f64*tier as f64;if self.b.tech[22]{(x*0.66666).ceil()}else{x}},_=>0.}}
fn econ_setup(&mut self){let b=self.b;
if b.tech[4]{
 if let Some(cap)=b.cities.iter().find(|c|c.owner==b.me&&b.tiles[c.pos].cap==b.me){
 let mut dist=vec![9999;b.n];let mut prev=vec![b.n;b.n];let mut h=BinaryHeap::new();dist[cap.pos]=0;h.push(Reverse((0,cap.pos)));
 while let Some(Reverse((d,p)))=h.pop(){if d!=dist[p]{continue;}for &q in &b.adj[p]{let t=&b.tiles[q];if !t.known||t.owner>0&&t.owner!=b.me||t.ter==4||t.ter<=2&&t.st!=48{continue;}let cost=if t.road||t.st==1||t.st==48{0}else{30};let nd=d+cost+1;if nd<dist[q]{dist[q]=nd;prev[q]=p;h.push(Reverse((nd,q)));}}}
 for c in &b.cities{if c.owner!=b.me||c.connected||c.pos==cap.pos||dist[c.pos]>=9999{continue;}let roads=dist[c.pos]/30;let score=if roads<=1{45.}else if roads==2{28.}else if roads==3{15.}else{-10.};if score<0.{continue;}let mut p=c.pos;for _ in 0..b.n {let t=&b.tiles[p];if !t.road&&t.st!=1&&t.st!=48{self.roads[p]=self.roads[p].max(score+if b.adj[p].iter().any(|&q|b.tiles[q].road||b.tiles[q].st==1){2.}else{0.});}if p==cap.pos||prev[p]>=b.n{break;}p=prev[p];}}
 }
 // Extend established roads for units that can use the extra movement now or next turn.
 for i in 0..b.n{if b.tiles[i].road||b.tiles[i].st==1||b.tiles[i].ter<=2{continue;}
 let linked=b.adj[i].iter().any(|&q|b.tiles[q].road||b.tiles[q].st==1&&b.tiles[q].owner==b.me);
 if linked {let mut usage:f64=0.;for &j in &self.own{let u=&b.units[j];if b.dist(i,u.pos)<=1&&matches!(u.k,3|4|11|12){if let Some(oi)=self.assigned[j]{if self.paths[oi][i]<self.paths[oi][u.pos] {usage+=if u.k==4{12.}else{6.};}}}}
 if usage>=12.&&b.stars>10.{self.roads[i]=self.roads[i].max(usage.min(30.));}}
 }
}
for (ci,c) in b.cities.iter().enumerate(){if c.owner!=b.me{continue;}let mut options=vec![];for i in 0..b.n{if b.tc[i]!=Some(ci)||b.tiles[i].st!=0{continue;}let t=&b.tiles[i];if t.res==1||t.res==6&&b.tech[6]||t.res==3&&b.tech[10]{options.push(2.);}if t.ter==5&&b.tech[16]{options.push(3.);}if t.res==5&&b.tech[23]||t.res==2&&b.tech[8]{options.push(2.5);options.push(2.5);}}
options.sort_by(|a,b|a.partial_cmp(b).unwrap());let need=(c.level+1-c.progress).max(1) as usize;let avail=options.len();let price: f64=options.iter().take(need).sum();let benefit=match c.level{1=>22.,2=>18.,3=>20.,4=>48.,_=>40.};let mut pv=benefit/(price.max(need as f64*2.)+2.);if avail<need{pv*=0.4;}self.growth[ci]=pv;}}
fn econ_score(&self,a:&Act)->f64{let b=self.b;let Some(ci)=b.tc[a.t] else{return -100.;};let c=&b.cities[ci];let pop=self.pop_gain(a);let cost=self.cost(a);if a.k==6&&matches!(a.p,17|18|19|20){return -100.;}if pop<=0.{return -100.;}let need=(c.level+1-c.progress).max(1) as f64;let imminent=need<=pop;let mut score=12.+self.growth[ci]*8.*pop/cost.max(1.);if imminent{score+=if c.level==1{32.}else if c.level==2{28.}else if c.level==4{45.}else{20.};}else{score+=c.progress.max(0) as f64/(c.level+1) as f64*10.;}if cost==0.{score+=45.;}
if a.k==6&&a.p==12&&b.tiles[a.t].res==1{score-=20.;}if a.k==6&&b.tiles[a.t].res==6&&!matches!(a.p,5|21){score-=8.;}if !self.threat[c.pos].is_empty(){score-=8.;if c.level==4&&imminent{score+=25.;}}
if b.stars<cost+3.&&!self.enemy.is_empty()&&!imminent{score-=15.;}if self.own.len()<2&&b.turn>0&&!imminent{score-=18.;}if self.growth[ci]<0.8&&!imminent{score-=18.;}if c.level>=5&&!imminent{score-=8.;}
let home=self.own.iter().filter(|&&j|b.units[j].home==c.pos as i32+1).count();if home>=c.level as usize+1&&c.level<4{score+=10.;}
score}
fn tech_score(&self,a:&Act)->f64{let b=self.b;let cost=self.cost(a);let free=b.stars-cost;let cities=b.cities.iter().filter(|c|c.owner==b.me).count();let count_res=|r|b.tiles.iter().filter(|t|t.owner==b.me&&t.st==0&&t.res==r).count() as f64;let forests=b.tiles.iter().filter(|t|t.owner==b.me&&t.st==0&&t.ter==5).count() as f64;let mountains=b.tiles.iter().filter(|t|t.known&&t.ter==4).count() as f64;let enemies=self.enemy.len();let army=self.own.len();let mut s=match a.p{
1=>if army<4&&b.turn>=1{26.}else{12.},
6=>if count_res(6)>=2.{24.+count_res(6)*2.}else{-30.},
16=>if forests>=2.{28.+forests.min(10.)}else{-25.},
17=>{let huts=b.tiles.iter().filter(|t|t.owner==b.me&&t.st==12).count();if huts>=3{28.+huts.min(8) as f64}else if enemies>0&&b.income>=10.{25.}else{-25.}},
18=>if enemies>0{27.+(enemies.min(5) as f64)*2.}else if army>=4&&b.turn>8{16.}else{-35.},
20=>if mountains>=3.&&b.turn>3{22.+mountains.min(15.)*0.4}else{-25.},
23=>if count_res(5)>=2.{27.+count_res(5)*2.}else if b.turn>12&&b.income>=10.{23.}else{-25.},
24=>if enemies>0&&b.income>=8.{42.}else if b.income>18.{28.}else{-25.},
10=>if count_res(3)>=2.{25.+count_res(3)*2.}else if self.navneed{32.}else{-25.},
13=>if self.navneed||self.own.iter().any(|&j|stats(b.units[j].k).boat){35.}else{-25.},
44=>if self.own.iter().any(|&j|stats(b.units[j].k).boat)&&b.income>9.{23.}else{-25.},
14=>if self.own.iter().any(|&j|stats(b.units[j].k).boat)&&b.income>12.{30.}else{-25.},
7=>if b.income>10.&&(self.water>0.2||enemies>army){17.}else{-35.},
2=>if b.income>=8.&&enemies>2{34.}else{-35.},
3=>if b.income>=8.&&enemies>1{48.}else{-25.},
4=>if cities>=3&&b.income>=10.{18.}else{-35.},
5=>if b.tiles.iter().filter(|t|t.owner==b.me&&matches!(t.st,6|13|22)).count()>=2{26.}else{-35.},
8=>if count_res(2)>=3.{22.}else{-35.},
9=>if b.tiles.iter().filter(|t|t.owner==b.me&&t.st==5).count()>=4{18.}else{-35.},
_=>-100.};
let enemyknights=self.enemy.iter().filter(|&&j|b.units[j].k==4).count();
if enemyknights>0&&matches!(a.p,20|23|24){s+=15.;}
if free<2.{s-=22.;}else if free<3.&&a.p!=1{s-=12.;}if a.p==1&&free>=3.{s+=6.;}if cost>b.income*1.7&&army>1{s-=12.;}if b.turn<2&&a.p!=1{s-=30.;}if enemies>0&&army<3&&a.p!=3&&a.p!=24{s-=20.;} if a.p==24&&b.income>=8.{s+=10.;}s}
fn recruit_score(&self,a:&Act)->f64{let b=self.b;let st=stats(a.p);let cost=st.c;let near=self.enemy.iter().map(|&j|b.dist(a.s,b.units[j].pos)).min().unwrap_or(30);let around=self.own.iter().filter(|&&j|b.dist(a.s,b.units[j].pos)<=3).count();let count=self.own.len();let cities=b.cities.iter().filter(|c|c.owner==b.me).count();let mut s;
if b.at[a.s].map(|j|b.units[j].k==43).unwrap_or(false){s=match a.p{42=>46.,44=>if near<5{51.}else{32.},41=>if b.income>=15.&&near<8{55.}else{22.},_=>-100.};if let Some(j)=b.at[a.s]{let u=&b.units[j];if u.hp<6.{s-=15.;}if u.max>=15.{s+=8.;if a.p==41{s+=12.;}}}return s;}
let explorers=self.obj.iter().filter(|(_,_,k)|*k==0||*k==3).count();s=if near<=3{42.}else if count<2{46.}else if count<cities+2&&explorers>0{30.}else if near<=6{26.}else if count<cities+1{24.}else{8.};
if count>cities*3+3{s-=22.;}if around>=4{s-=(around-3) as f64*9.;}if near>6&&count>cities+2{s-=18.;}if self.navneed&&around<3&&count<cities*2+3{s+=8.;}
s+=match a.p{2=>if near<5{-6.}else if b.tech[1]{-28.}else{3.},3=>if near<4{-2.}else{10.},9=>if near<6{13.}else{0.},11=>if near<7{23.}else{10.},8=>if near<=6{17.}else{0.},4=>if near<=7{22.}else{8.},5=>if self.navneed&&near>4{8.}else if near<3{0.}else{-18.},10=>-40.,38=>-40.,_=>-100.};
if a.p==4&&self.enemy.iter().filter(|&&j|b.units[j].hp<=10.).count()>=3{s+=10.;}
let heavy=self.enemy.iter().filter(|&&j|matches!(b.units[j].k,5|11|12|45)).count();
if a.p==11{s+=heavy.min(3) as f64*4.;if self.enemy.iter().any(|&j|b.units[j].k==4){s+=15.;}}
if a.p==8{s+=heavy.min(3) as f64*4.;}
if a.p==5&&self.enemy.iter().any(|&j|b.units[j].k==4&&b.dist(a.s,b.units[j].pos)<=4){s+=20.;}
if a.p==9&&heavy>1{s-=10.;}
if a.p==4{let squishy=self.enemy.iter().filter(|&&j|stats(b.units[j].k).d<=1.&&b.dist(a.s,b.units[j].pos)<=6).count();s+=squishy.min(3) as f64*4.;}
if a.p==8{let cats=self.own.iter().filter(|&&j|b.units[j].k==8).count();if cats*3>=count{s-=25.;}if near<=2{s-=45.;}}
if a.p==9{let arch=self.own.iter().filter(|&&j|b.units[j].k==9).count();if arch*2>count{s-=10.;}}
if a.p==4{let knights=self.own.iter().filter(|&&j|b.units[j].k==4).count();if knights*3>count{s-=18.;}}
if !self.threat[a.s].is_empty(){let u=Unit{owner:b.me,k:a.p,hp:st.hp,max:st.hp,..Unit::default()};let risk=self.risk(&u,a.s,st.hp,None);s-=risk*2.;if self.threat[a.s].len()==1&&st.hp>=15.{s+=8.;}}
if b.stars<cost+2.&&near>4&&count>=2{s-=9.;}
if near<=3&&around<=2{s+=16.;}if a.p==5&&near<=3{s+=20.;}s}
fn move_score(&self,a:&Act,mem:&Memory)->f64{let b=self.b;let Some(j)=b.at[a.s] else{return -100.;};let u=&b.units[j];let st=stats(u.k);let src=&b.tiles[a.s];let dst=&b.tiles[a.t];let mut score=2.;let cap=dst.st==1&&dst.owner!=b.me;let ruin=dst.st==2;
if src.st==1&&src.owner!=b.me||src.st==2{return -100.;}
let path=self.assigned[j];if let Some(oi)=path{let old=self.paths[oi][a.s];let new=self.paths[oi][a.t];if old<900.&&new<900.{let (p,w,kind)=self.obj[oi];score+=((old-new)*5.).max(-16.);score+=w/(new+2.)*0.35;if kind==2&&b.dist(p,a.s)<=1&&new>old{score-=8.;}}}
if st.boat&&path.is_none(){let mut best=-100.;for (oi,&(p,w,kind)) in self.obj.iter().enumerate(){let _=oi;let old=b.dist(a.s,p) as f64;let new=b.dist(a.t,p) as f64;let mut gain=(old-new)*4.+w/(new+2.)*0.5;if kind==4&&dst.ter>2{gain-=5.;}if kind==1&&dst.ter>2{gain+=5.;}if gain>best{best=gain;}}score+=best;}
let radius=if dst.ter==4||u.k==42{2}else{1};let unseen=(0..b.n).filter(|&p|!b.tiles[p].known&&b.dist(p,a.t)<=radius).count();score+=unseen as f64*1.5;if unseen>0{score+=4.;let center=(b.size/2)*b.size+b.size/2;score+=(b.dist(a.s,center)-b.dist(a.t,center)) as f64*3.;}if cap{score+=if dst.owner==0{48.}else{55.};}if ruin{score+=36.;}
if st.dash&&!u.attacked&&!(dst.st==8&&!st.boat)&&!(st.boat&&dst.ter>2){let attack=self.enemy.iter().filter(|&&x|b.dist(a.t,b.units[x].pos)<=st.r).map(|&x|self.attack_score(j,a.t,b.units[x].pos)).fold(-100.,f64::max);if attack>0.{score+=attack*0.65+4.;}}
let mut risk=self.risk(u,a.t,u.hp,None);
if st.dash&&!u.attacked&&!(st.boat&&dst.ter>2)&&!(dst.st==8&&!st.boat){
 let mut bestrisk=risk;let mut bestattack=-100.;
 for &ei in &self.enemy {let e=&b.units[ei];if b.dist(a.t,e.pos)>st.r{continue;}let (hit,ret)=b.damage(u,a.t,e,e.pos);let ascore=self.attack_score(j,a.t,e.pos);if hit>=e.hp&&ascore>bestattack{let after=if st.r==1{e.pos}else{a.t};let r=self.risk(u,after,u.hp-ret,Some(ei));bestattack=ascore;bestrisk=r;}}
 if bestattack>10.{risk=risk*0.3+bestrisk*0.7;}
}
let oldrisk=self.risk(u,a.s,u.hp,None);score-=risk*3.;score+=oldrisk*1.5;
if st.r>1{let nearest=self.enemy.iter().map(|&x|b.dist(a.t,b.units[x].pos)).min().unwrap_or(20);if nearest==st.r{score+=9.;}if nearest<st.r{score-=9.*(st.r-nearest) as f64;}if !st.dash&&nearest<=st.r{score+=5.;}}
if u.hp<u.max*0.55&&!cap&&!ruin{score-=8.;if dst.owner==b.me&&src.owner!=b.me{score+=8.;}}
if src.st==1&&src.owner==b.me{let danger=self.threat[a.s].len();if danger>0{let allies=b.adj[a.s].iter().filter(|&&p|b.at[p].map(|q|b.units[q].owner==b.me&&q!=j).unwrap_or(false)).count();score-=if allies>0{8.}else{28.};}else{score+=4.;}}
if a.t<mem.visits.len(){score-=(mem.visits[a.t].min(12) as f64)*0.3;}
if a.s<mem.prev.len()&&mem.prev[a.s]==a.t&&!cap{score-=5.;}
// Units on the way to another settlement should not all follow the same scout.
let crowd=self.own.iter().filter(|&&x|x!=j&&b.dist(b.units[x].pos,a.t)<=1).count();if self.enemy.is_empty()&&!cap&&!ruin{score-=crowd as f64*2.;}
if dst.st==8&&!st.boat{score+=if self.navneed{10.}else{-20.};}
if st.boat&&dst.ter>2&&!cap&&!ruin {
let closecity=b.cities.iter().filter(|c|c.owner!=b.me).map(|c|b.dist(c.pos,a.t)).min().unwrap_or(30);
if closecity>2{score-=if u.k==41{30.}else if u.k==43{3.}else{13.};}
}
if u.k==41&&dst.ter>2&&!cap{score-=15.;}
score}
fn choose(&mut self,mem:&mut Memory)->usize{let b=self.b;self.econ_setup();let mut best=0;let mut bs=-1e9;for (i,a) in b.acts.iter().enumerate(){let mut score=match a.k{
10=>0.,11=>-1e8,8=>if b.tiles[a.s].st==1{500.}else{250.},
9=>match a.p{1=>{let ci=b.cityat[a.t];let defender=b.at[a.t].map(|j|stats(b.units[j].k).fort).unwrap_or(false);if defender&&self.threat[a.t].len()>=2&&ci.map(|i|b.cities[i].level<=3).unwrap_or(false){1020.}else{900.}},3=>1000.,7=>1000.,6=>1000.,8=>{let cnt=(0..b.n).filter(|&p|b.dist(a.t,p)==2&&b.tiles[p].owner==0&&(b.tiles[p].res>0||b.tiles[p].ter==5)).count();if cnt>=6{980.}else{1000.}},5=>{let coast=(0..b.n).filter(|&p|b.dist(p,a.t)==2&&b.tiles[p].owner==0&&b.tiles[p].ter==1).count();let owned_coast=b.tiles.iter().any(|t|t.owner==b.me&&t.ter==1);if coast>0&&!owned_coast&&self.water>0.12{1040.}else{990.}},_=>900.},
2=>if let Some(j)=b.at[a.s]{self.attack_score(j,a.s,a.t)}else{-100.},
1=>self.move_score(a,mem),
4=>self.recruit_score(a),
5=>self.econ_score(a),
7=>self.tech_score(a),
6=>match a.p{3=>self.roads[a.t],48=>{let mut benefit:f64=-30.;if b.stars>=8.{for &j in &self.own{let u=&b.units[j];if stats(u.k).boat||b.dist(u.pos,a.t)>2{continue;}
for &(goal,w,k) in &self.obj{if k>3{continue;}let d=b.dist(u.pos,goal);if b.dist(a.t,goal)<d&&d<=6{let beyond=b.adj[a.t].iter().any(|&q|b.tiles[q].ter>=3&&b.dist(q,goal)<b.dist(a.t,goal));if beyond{benefit=benefit.max(15.+w*0.2);}}}}}benefit},8=>{let mut s=self.econ_score(a)-30.;let ports=b.tiles.iter().enumerate().filter(|(p,t)|t.owner==b.me&&t.st==8&&b.dist(*p,a.t)<=5).count();if self.navneed&&ports==0{let units=self.own.iter().filter(|&&j|b.dist(b.units[j].pos,a.t)<=3).count();s=40.+units.min(3) as f64*3.;let unknown=(0..b.n).filter(|&p|!b.tiles[p].known&&b.dist(p,a.t)<=3).count();s+=unknown.min(15) as f64*0.2;
let center=(b.size/2)*b.size+b.size/2; s+=(b.size as i32-b.dist(a.t,center)) as f64*0.35;
let coastal_exposure=b.adj[a.t].iter().filter(|&&p|b.tiles[p].ter<=2).count();s+=coastal_exposure as f64*0.4;
let cities=b.cities.iter().filter(|c|c.owner!=b.me).map(|c|b.dist(c.pos,a.t)).min().unwrap_or(20);s-=cities as f64*0.6;
}if b.tiles[a.t].res==3{s-=15.;}s},50=>{let inc=b.adj[a.t].iter().filter(|&&p|b.tiles[p].owner==b.me&&matches!(b.tiles[p].st,6|13|22)).map(|&p|b.tiles[p].level).sum::<i32>();if inc>=2{35.+inc as f64*3.}else{-20.}},_=>self.econ_score(a)},
3=>match a.p{16=>400.,7=>{if let Some(j)=b.at[a.s]{let u=&b.units[j];let heal=(u.max-u.hp).min(if b.tiles[a.s].owner==b.me{4.}else{2.});let danger=self.risk(u,a.s,u.hp,None);if b.tiles[a.s].st==1&&b.tiles[a.s].owner!=b.me{-100.}else{4.+heal/u.max*b.val(u)*8.-danger*1.5}}else{-100.}},9=>30.,2=>{let t=&b.tiles[a.t];if t.res==0&&b.stars<3.&&self.enemy.iter().any(|&j|b.cities.iter().any(|c|c.owner==b.me&&b.dist(c.pos,b.units[j].pos)<=3)) {4.}else{-100.}},19=>if a.t==0{20.}else{-100.},18=>10.,_=>-100.},_=>-100.};
if self.reserve>b.stars&&matches!(a.k,4|5|6)&&self.cost(a)>0. {
let critical=a.k==4&&self.threat[a.s].len()>0;
let imminent=(a.k==5||a.k==6)&&b.tc[a.t].map(|ci|b.cities[ci].level<=2&&(b.cities[ci].level+1-b.cities[ci].progress) as f64<=self.pop_gain(a)).unwrap_or(false);
if !critical&&!imminent {score-=45.;}
}
if self.reserve>b.stars&&a.k==7&&!matches!(a.p,2|3|24){score-=30.;}
let tie=((a.s*17+a.t*37+a.p as usize*7+i)%31) as f64*0.0001; if score+tie>bs{bs=score+tie;best=i;}}
let a=&b.acts[best];if a.k==1{if mem.visits.len()!=b.n{mem.visits=vec![0;b.n];mem.prev=vec![b.n;b.n];}mem.visits[a.t]+=1;mem.prev[a.t]=a.s;mem.prev[a.s]=b.n;}best}
}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){let Ok(line)=line else{break;};let Ok(obs)=serde_json::from_str::<Value>(&line)else{continue;};let b=Board::parse(&obs);let mut ev=Eval::new(&b);let idx=ev.choose(&mut mem);writeln!(out,"{}",json!({"action_id":b.acts[idx].id})).unwrap();out.flush().unwrap();}}
