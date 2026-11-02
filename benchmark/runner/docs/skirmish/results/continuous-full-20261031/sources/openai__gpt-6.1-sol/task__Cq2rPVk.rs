use serde_json::{Value,json};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn iv(v:&Value,k:&str)->i32 { v[k].as_i64().unwrap_or(0) as i32 }
fn bv(v:&Value,k:&str)->bool { v[k].as_bool().unwrap_or(false) }
#[derive(Clone,Copy,Default)]
struct Tile {known:bool,ter:i32,owner:i32,res:i32,st:i32,road:bool,cap:i32,sl:i32,city:usize}
#[derive(Clone,Copy,Default)]
struct Unit {owner:i32,k:i32,h:f32,vet:bool,moved:bool,attacked:bool,kills:i32,home:i32,pass:i32,poison:bool,boost:bool}
#[derive(Clone,Default)]
struct City {tile:usize,owner:i32,level:i32,progress:i32,prod:i32,rewards:[i32;9],connected:bool}
#[derive(Clone,Copy)]
struct Stat {cost:f32,h:f32,a:f32,d:f32,m:i32,r:i32,dash:bool,escape:bool,fort:bool,stiff:bool,sea:bool}
fn stat(k:i32)->Stat {
 let (cost,h,a,d,m,r,dash,escape,fort,stiff,sea)=match k {
 2=>(2.,10.,2.,2.,1,1,true,false,true,false,false),
 3=>(3.,10.,2.,1.,2,1,true,true,true,false,false),
 4=>(8.,10.,3.5,1.,3,1,true,false,true,false,false),
 5=>(3.,15.,1.,3.,1,1,false,false,true,false,false),
 8=>(8.,10.,4.,0.,1,3,false,false,false,true,false),
 9=>(3.,10.,2.,1.,1,2,true,false,true,false,false),
 10=>(5.,10.,0.,1.,1,1,false,false,false,true,false),
 11=>(5.,15.,3.,3.,1,1,true,false,false,false,false),
 12=>(15.,40.,5.,4.,1,1,false,false,false,false,false),
 37=>(2.,10.,2.,2.,1,1,true,false,false,false,false),
 38=>(8.,5.,0.,0.5,2,1,true,false,false,true,false),
 39=>(8.,5.,0.,0.5,2,1,true,false,false,true,true),
 40=>(2.,10.,2.,2.,2,1,true,false,false,false,true),
 41=>(15.,10.,3.,2.,2,3,false,false,false,true,true),
 42=>(5.,10.,2.,1.,3,2,true,false,false,false,true),
 43=>(0.,10.,0.,1.,2,2,false,false,false,true,true),
 44=>(5.,10.,3.,3.,3,1,true,false,false,false,true),
 45=>(18.,40.,4.,4.,2,1,false,false,false,true,true),
 _=>(3.,10.,2.,2.,1,1,true,false,false,false,false)};
 Stat{cost,h,a,d,m,r,dash,escape,fort,stiff,sea}
}
fn mh(u:Unit)->f32 {
 let mut hp=stat(if u.pass>0 {u.pass}else{u.k}).h+if u.vet {5.}else{0.};
 if stat(u.k).sea && u.pass==0 && u.h>hp {hp=if u.h>20. {40.}else if u.h>15. {20.}else{15.};}hp
}
fn value(u:Unit)->f32 { let s=stat(u.k); (s.cost+if u.pass>0 {stat(u.pass).cost}else{0.}+2.5) * (0.45+0.55*u.h/mh(u)) }
struct Board {size:usize,n:usize,p:i32,turn:i32,stars:i32,t:Vec<Tile>,u:Vec<Unit>,cities:Vec<City>,tech:[bool;128],adj:Vec<Vec<usize>>,dist:Vec<i32>,own:Vec<usize>,enemy:Vec<usize>,ct:Vec<usize>,income:i32,peace:bool}
impl Board {
 fn parse(obs:&Value)->Self {
 let v=&obs["visible_state"];let size=iv(v,"board_size") as usize;let n=size*size;let p=iv(obs,"player");
 let mut b=Board{size,n,p,turn:iv(v,"turn"),stars:iv(v,"stars"),t:vec![Tile{city:usize::MAX,..Tile::default()};n],u:vec![Unit::default();n],cities:vec![],tech:[false;128],adj:vec![vec![];n],dist:vec![0;n*n],own:vec![],enemy:vec![],ct:vec![],income:0,peace:false};
 for t in v["tiles"].as_array().unwrap() {let i=iv(t,"tile") as usize;b.t[i]=Tile{known:bv(t,"known"),ter:iv(t,"terrain"),owner:iv(t,"owner"),res:iv(t,"resource"),st:iv(t,"structure"),road:iv(t,"road")>0,cap:iv(t,"capital"),sl:iv(t,"structure_level"),city:usize::MAX};}
 for t in v["units"].as_array().unwrap() {let i=iv(t,"tile") as usize;b.u[i]=Unit{owner:iv(t,"owner"),k:iv(t,"kind"),h:t["health"].as_f64().unwrap_or(10.) as f32,vet:bv(t,"veteran"),moved:bv(t,"moved"),attacked:bv(t,"attacked"),kills:iv(t,"kills"),home:iv(t,"home"),pass:iv(t,"passenger"),poison:bv(t,"poison"),boost:bv(t,"boosted")};if b.u[i].owner==p {b.own.push(i)} else {b.enemy.push(i)}}
 for c in v["cities"].as_array().unwrap() {let i=iv(c,"tile") as usize;let mut r=[0;9];if let Some(a)=c["rewards"].as_array(){for j in 0..9 {r[j]=a[j].as_i64().unwrap_or(0) as i32;}}
 let cc=City{tile:i,owner:iv(c,"owner"),level:iv(c,"level").max(1),progress:iv(c,"progress"),prod:iv(c,"production"),rewards:r,connected:bv(c,"connected")};if cc.owner==p {b.ct.push(b.cities.len());b.income+=cc.prod+(b.t[i].cap==p) as i32;} b.cities.push(cc);}
 for t in v["technology"].as_array().unwrap() {if let Some(i)=t.as_u64() {if i<128{b.tech[i as usize]=true}}}
 b.peace=v["relations"]["peace"][(2-p) as usize].as_bool().unwrap_or(false);
 for i in 0..n { for j in 0..n {let d=((i%size) as i32-(j%size) as i32).abs().max(((i/size) as i32-(j/size) as i32).abs());b.dist[i*n+j]=d;if d==1 {b.adj[i].push(j);}}}
 // Map owned tiles to city territory using reward radii, nearest wins. Territory is not directly in the observation.
 for i in 0..n {if b.t[i].owner==p {let mut best=100;for &ci in &b.ct {let c=&b.cities[ci];let d=b.d(i,c.tile);if d<=if c.rewards[5]>0{2}else{1} && d<best {best=d;b.t[i].city=ci;}}}}
 b
 }
 fn d(&self,a:usize,b:usize)->i32 {self.dist[a*self.n+b]}
 fn bonus(&self,pos:usize,u:Unit)->f32 {let t=self.t[pos];let own=u.owner==self.p;let s=stat(u.k);
 let x=if t.ter==4 {1.5} else if t.ter==5 {if (own&&self.tech[18])||(!own&&self.turn>10){1.5}else{1.}} else if t.ter<=2 {if own&&self.tech[12]{1.5}else{1.}} else if t.st==1 && t.owner==u.owner && s.fort {
 let wall=if own {self.cities.iter().find(|c|c.tile==pos).map(|c|c.rewards[1]>0).unwrap_or(false)}else{false};if wall {4.}else{1.5}
 } else {1.};x*if u.poison{0.5}else{1.}
 }
 fn combat(&self,src:usize,dst:usize,a:Unit,d:Unit)->(f32,f32) {
 if a.k==10 {return(d.h,0.)}let sa=stat(a.k);let sd=stat(d.k);let af=sa.a*a.h/mh(a);let dd=sd.d*if d.poison{0.7}else{1.};let df=dd*d.h/mh(d)*self.bonus(dst,d);let total=(af+df).max(0.001);let hit=(af/total*sa.a*4.5+0.500001).floor();let ret=if d.h>hit && !sd.stiff && self.d(src,dst)<=sd.r && a.k!=37 && a.k!=40 {(df/total*dd*4.5+0.500001).floor()}else{0.};(hit,ret)
 }
 fn road(&self,i:usize,owner:i32)->bool {let t=self.t[i];(t.road||t.st==1||t.st==48)&&(t.owner==0||t.owner==owner)}
 fn terrain_ok(&self,i:usize,u:Unit,unknown:bool)->bool {let t=self.t[i];let own=u.owner==self.p;let s=stat(u.k);
 if !t.known {return unknown}if t.ter==4 && own&&!self.tech[20]{return false}if t.ter==2 && own&&!self.tech[13]{return false}if t.ter==1 && own&&!self.tech[10] && t.st!=48{return false}
 if t.ter<=2 && !s.sea && t.st!=48 && !(t.st==8&&t.owner==u.owner) {return false}true}
 // Reachability approximation for opponent threat and positional attacks. Flags/forests match real movement.
 fn reach(&self,src:usize,u:Unit,ignore_units:bool)->Vec<usize> {
 let budget=(stat(u.k).m+u.boost as i32)*2;let mut ds=vec![999;self.n];let mut heap=BinaryHeap::new();ds[src]=0;heap.push(Reverse((0,src)));let mut out=vec![];
 while let Some(Reverse((cost,i)))=heap.pop(){if cost!=ds[i]{continue}if i!=src {if self.u[i].owner==0||ignore_units {out.push(i);}let t=self.t[i];let zoc=self.adj[i].iter().any(|&j|self.u[j].owner>0 && self.u[j].owner!=u.owner);
 if cost>=budget||t.ter==4||zoc||(t.ter==5&&!self.road(i,u.owner))||(!stat(u.k).sea&&t.st==8)||(stat(u.k).sea&&t.ter>2){continue}}
 for &j in &self.adj[i]{if !self.terrain_ok(j,u,false){continue}let o=self.u[j].owner;if !ignore_units&&o>0&&o!=u.owner{continue}let edge=self.road(i,u.owner)&&self.road(j,u.owner);let mut nc=cost+if edge{1}else{2};if self.t[j].ter==5&&!edge {nc=nc.max(budget);}
 if nc<=budget&&nc<ds[j]{ds[j]=nc;heap.push(Reverse((nc,j)));}}
 } out
 }
 fn paths(&self,goal:usize,sea:bool,climb:bool,unknown:bool)->Vec<i32> {
 let mut ds=vec![9999;self.n];
 if self.t[goal].ter==4&&!climb {return ds}if !sea&&self.t[goal].ter<=2&&self.t[goal].st!=48&&!(self.t[goal].st==8&&self.t[goal].owner==self.p) {return ds}
 ds[goal]=0;let mut heap=BinaryHeap::new();heap.push(Reverse((0,goal)));
 while let Some(Reverse((d,i)))=heap.pop(){if ds[i]!=d {continue}for &j in &self.adj[i]{let ti=self.t[i];let tj=self.t[j];if !unknown&&(!ti.known||!tj.known){continue}if tj.ter==4&&!climb{continue}
 let pass=if sea {true}else{tj.ter==0||tj.ter>2||tj.st==48||tj.st==8&&tj.owner==self.p};if !pass{continue}
 let mut c=if !ti.known{4}else if ti.ter==4 {4}else if ti.ter==5 {3}else{2};if self.road(i,self.p)&&self.road(j,self.p){c=1;}
 if sea && ((ti.ter>2)!=(tj.ter>2)) {c+=4;}
 let nd=d+c;if nd<ds[j]{ds[j]=nd;heap.push(Reverse((nd,j)));}}
 } ds
 }
}
#[derive(Clone)]
struct Action {id:String,k:i32,s:usize,t:usize,p:i32}
impl Action {fn parse(a:&Value)->Self {let c=&a["command"];Self{id:a["action_id"].as_str().unwrap().to_string(),k:iv(c,"kind"),s:iv(c,"source").max(0) as usize,t:iv(c,"target").max(0) as usize,p:iv(c,"parameter")}}}
#[derive(Default)]
struct Memory {turn:i32,visits:Vec<i32>,prev:Vec<usize>,actions:i32,naval_since:i32}
struct Goal {tile:usize,v:f32,land:Vec<i32>,sea:Vec<i32>,kind:i32}
struct Tactics {reach:Vec<Vec<usize>>,danger:Vec<Vec<usize>>,support:Vec<f32>}
impl Tactics {
 fn new(b:&Board)->Self {
 let mut r=vec![vec![];b.n];let mut danger=vec![vec![];b.n];let mut support=vec![0.;b.n];
 for &i in b.enemy.iter().chain(b.own.iter()) {r[i]=b.reach(i,b.u[i],false);r[i].push(i);let s=stat(b.u[i].k);let mut marked=vec![false;b.n];for &j in &r[i] {if j!=i&&!s.dash {continue}for t in 0..b.n{if b.d(j,t)<=s.r&&b.d(j,t)>0{marked[t]=true;}}}if b.u[i].owner!=b.p {for t in 0..b.n{if marked[t] {danger[t].push(i)}}} else {for t in 0..b.n{if marked[t] {support[t]+=s.a*b.u[i].h/mh(b.u[i]);}}}
 }
 Self{reach:r,danger,support}
 }
 fn risk(&self,b:&Board,pos:usize,u:Unit,exclude:usize)->f32 {
 if b.peace{return 0.}let mut dmgs=vec![];for &e in &self.danger[pos]{if e==exclude {continue}let eu=b.u[e];if stat(eu.k).a<=0.{if eu.k==10 {dmgs.push((u.h,0.,value(eu)));}continue}
 let src=if b.d(e,pos)<=stat(eu.k).r {e}else{ self.reach[e].iter().copied().filter(|&j|b.d(j,pos)<=stat(eu.k).r && j!=pos).min_by_key(|&j|b.d(j,pos)).unwrap_or(e)};
 let (d,r)=b.combat(src,pos,eu,u);dmgs.push((d,r,value(eu)));
 }
 dmgs.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());let mut total=0.;for (j,&(d,r,_)) in dmgs.iter().enumerate(){let fac=if j<2{1.}else{0.75};total+=d*fac;if r>=7. {total-=1.;}}
 total.max(0.)
 }
}
fn health_loss(u:Unit,dmg:f32)->f32 {let h=u.h; let v=value(u);if dmg>=h {v+4.}else {v*0.7*dmg/mh(u)}}
fn attack_score(b:&Board,tac:&Tactics,src:usize,dst:usize,a:Unit)->f32 {
 let d=b.u[dst];if d.owner==0{return -1000.}let s=stat(a.k);if a.k==38||a.k==39{return 70.}
 let (hit,ret)=b.combat(src,dst,a,d);let kill=hit>=d.h;let mut score=health_loss(d,hit)*5.-health_loss(a,ret)*5.;
 if kill {score+=16.; if d.k==8||d.k==10||d.k==41 {score+=8.;}if a.k==4{score+=12.;}if a.k!=12&&a.k<38&&a.k!=8 {if a.kills==2{score+=12.}else{score+=2.}}}
 if !kill {
 let mut available=0.;for &i in &b.own {if i==src||b.u[i].attacked{continue}let si=stat(b.u[i].k);if b.d(i,dst)<=si.r || !b.u[i].moved&&si.dash&&tac.reach[i].iter().any(|&j|b.d(j,dst)<=si.r){let (h,_)=b.combat(i,dst,b.u[i],d);available+=h;}}
 if available+hit>=d.h {score+=10.;}else if ret>=a.h {score-=20.;}
 }
 if a.k==41 {for &e in &b.enemy {if e!=dst&&b.d(e,dst)==1{let (h,_)=b.combat(src,e,a,b.u[e]);score+=health_loss(b.u[e],(h*0.5).floor())*4.5;}}}
 let city=b.t[dst].st==1;
 if city {if b.t[dst].owner==b.p {score+=30.+if kill{45.}else{0.};} else {score+=6.;if kill&&s.r==1 {score+=28.;}}}
 if ret>=a.h {score-=20.;}
 if ret<a.h {let mut au=a;au.h-=ret;let end=if kill&&s.r==1{dst}else{src};let incoming=tac.risk(b,end,au,if kill {dst}else{usize::MAX});let before=tac.risk(b,src,a,usize::MAX);
 if !s.escape {let loss=(health_loss(au,incoming)-health_loss(a,before)*0.6).max(0.);score-=loss*2.5;}
 }
 if s.r>1&&ret==0. {score+=5.;}
 score
}
fn goals(b:&Board)->Vec<Goal> {
 let mut out=vec![];let known_enemy=b.cities.iter().any(|c|c.owner>0&&c.owner!=b.p);
 for c in &b.cities {
 if c.owner==b.p {let threat=b.enemy.iter().filter(|&&e|b.d(e,c.tile)<=stat(b.u[e].k).m+2).count();if threat>0 {out.push(Goal{tile:c.tile,v:70.+threat as f32*5.,land:vec![],sea:vec![],kind:3});}}
 else {out.push(Goal{tile:c.tile,v:if c.owner==0 {120.}else{110.},land:vec![],sea:vec![],kind:if c.owner==0{1}else{2}});}
 }
 for i in 0..b.n {
 if b.t[i].st==2 && (b.t[i].ter!=4||b.tech[20]) {out.push(Goal{tile:i,v:74.,land:vec![],sea:vec![],kind:4});}
 if !b.t[i].known {continue}if b.u[i].owner>0&&b.u[i].owner!=b.p{continue}
 if b.t[i].ter==4&&!b.tech[20]{continue}if b.t[i].ter==2&&!b.tech[13]{continue}if b.t[i].ter==1&&!b.tech[10]{continue}
 let rad=if b.t[i].ter==4 {2}else{1};let reveal=(0..b.n).filter(|&j|!b.t[j].known&&b.d(i,j)<=rad).count();
 if reveal>0 {
 let nearcity=b.ct.iter().map(|&ci|b.d(i,b.cities[ci].tile)).min().unwrap_or(1);
 let centrality=(b.size as f32-1.)*0.5-((i%b.size) as f32-(b.size as f32-1.)*0.5).abs().max(((i/b.size) as f32-(b.size as f32-1.)*0.5).abs());
 let promising=(0..b.n).filter(|&j| !b.t[j].known && b.d(i,j)<=rad && j%b.size>0 && j%b.size+1<b.size && j/b.size>0 && j/b.size+1<b.size && b.cities.iter().all(|c|b.d(c.tile,j)>=3)).count();
 let v=(if known_enemy{15.}else{26.})+reveal as f32*1.0+nearcity.min(5) as f32+centrality*2.3+promising.min(6) as f32*1.8;
 out.push(Goal{tile:i,v,land:vec![],sea:vec![],kind:0});
 }
 }
 // enemy units are military objectives if not close to an enemy city
 for &i in &b.enemy {if !b.cities.iter().any(|c|c.owner!=b.p&&c.owner>0&&b.d(i,c.tile)<=2) {
 out.push(Goal{tile:i,v:if b.t[i].owner==b.p {80.}else{48.},land:vec![],sea:vec![],kind:5});}}
 for g in &mut out {g.land=b.paths(g.tile,false,b.tech[20],false);g.sea=transport_paths(b,g.tile);}
 out
}
fn transport_paths(b:&Board,goal:usize)->Vec<i32> {
 let mut ds=vec![9999;b.n];ds[goal]=0;let mut heap=BinaryHeap::new();heap.push(Reverse((0,goal)));
 while let Some(Reverse((d,i)))=heap.pop(){if ds[i]!=d{continue}for &j in &b.adj[i]{let ti=b.t[i];let tj=b.t[j];if !ti.known||!tj.known {continue}if tj.ter==4&&!b.tech[20]{continue}
 let wi=ti.ter<=2&&ti.st!=48;let wj=tj.ter<=2&&tj.st!=48;
 if wi && (!b.tech[10]||ti.ter==2&&!b.tech[13]) {continue}if wj&&(!b.tech[10]||tj.ter==2&&!b.tech[13]){continue}
 if wi&&!wj && !(ti.st==8&&ti.owner==b.p){continue}
 let mut c=if ti.ter==4 {4}else if ti.ter==5 {3}else{2};if wi{c=1;}if wi&&!wj{c+=2;}if wj&&!wi{c+=3;}if !wi&&!wj&&b.road(i,b.p)&&b.road(j,b.p){c=1;}
 let nd=d+c;if nd<ds[j]{ds[j]=nd;heap.push(Reverse((nd,j)));}}
 }ds
}
fn goal_value(b:&Board,gs:&[Goal],src:usize,pos:usize,u:Unit)->f32 {
 let mut best=0f32;let s=stat(u.k);
 for g in gs {
 let path=if s.sea||b.tech[10] {&g.sea}else{&g.land};let cost=path[pos];if cost>=9999 {continue}
 let mut val=g.v;
 if g.kind==3 {
 let myd=path[src]; let mut guards=0.;
 for &j in &b.own {if j==src{continue}let dj=path[j];if dj<myd || dj==myd&&j<src {guards+=if b.u[j].h<5. {0.4}else{1.};}}
 val/=1.+guards*1.6;
 if pos==g.tile&&b.t[pos].owner==b.p {val*=0.9;}
 if u.k==8||u.k==9 {val*=0.7;}
 }
 if g.kind==0||g.kind==1||g.kind==4 {let ours=b.u[g.tile].owner==b.p&&g.tile!=src;
 if ours {val*=0.10;}else{
 let myd=path[src];let mut competitors=0.;for &j in &b.own{if j==src{continue}let other=b.u[j];let dj=path[j];if dj<myd || dj==myd&&j<src {competitors+=if other.moved {0.65}else{1.};}}
 val/=1.+competitors*if g.kind==0{0.9}else{1.8};}
 if g.kind==0 && u.k==12 {val*=0.6;}
 }
 if g.kind==2||g.kind==5 {
 // stage artillery at firing range, but melee units want to occupy the objective
 let want=if s.r>=2 {(s.r-1)*2}else{0};let eff=(cost-want).max(0);
 let proximity=val/(1.+eff as f32*0.17);best=best.max(proximity);continue
 }
 best=best.max(val/(1.+cost as f32*0.22));
 }best
}
fn move_score(b:&Board,tac:&Tactics,gs:&[Goal],mem:&Memory,src:usize,dst:usize)->f32 {
 let u=b.u[src];let s=stat(u.k);let t=b.t[dst];let bef=goal_value(b,gs,src,src,u);let aft=goal_value(b,gs,src,dst,u);
 let mut score=(aft-bef)*2.5;
 let cap=t.st==1&&t.owner!=b.p;let ruin=t.st==2;
 if cap {score+=if t.owner==0{65.}else{90.};}if ruin {score+=45.;}
 if u.k==45 {for &e in &b.enemy {if b.d(dst,e)==1 {let (hit,_)=b.combat(dst,e,u,b.u[e]);score+=health_loss(b.u[e],(hit*0.5).floor())*5.;}}}
 let rad=if t.ter==4||u.k==42 {2}else{1};let reveal=(0..b.n).filter(|&i|!b.t[i].known&&b.d(i,dst)<=rad).count() as f32;
 score+=reveal*if b.turn<12 {1.9}else{0.8};
 let mut nu=u;if !s.sea&&t.st==8 {nu.k=if u.k==12 {45}else{43};nu.pass=if u.k==12{0}else{u.k};}
 if s.sea&&t.ter>2&&t.st!=8 {nu.k=if u.pass>0{u.pass}else if u.k==45{12}else{2};nu.pass=0;}
 let risk=tac.risk(b,dst,nu,usize::MAX);let oldrisk=tac.risk(b,src,u,usize::MAX);
 let loss=health_loss(nu,risk);let oldloss=health_loss(u,oldrisk);
 score-=loss*4.5;score+=oldloss*2.8;
 if risk>=nu.h {score-=20.;if cap&&t.owner==0{score+=5.;}}
 if b.t[src].st==1&&b.t[src].owner==b.p {
 let invade=b.enemy.iter().any(|&e|tac.reach[e].contains(&src));if invade {
 let home=b.cities.iter().find(|c|c.tile==src);let capacity=home.map(|c|c.level+1).unwrap_or(1);
 let used=b.own.iter().filter(|&&j|b.u[j].home==src as i32+1).count() as i32;
 if b.stars>=3&&used<capacity {score-=8.;score+=6.;} else {score-=65.;}
 }
 // freeing cities for production is useful, but not sufficient reason for a bad move
 if b.stars>=3 {
 let used=b.own.iter().filter(|&&j|b.u[j].home==src as i32+1).count() as i32;
 let capacity=b.cities.iter().find(|c|c.tile==src).map(|c|c.level+1).unwrap_or(0);
 if used<capacity {score+=if u.k==5 {4.}else if b.enemy.iter().any(|&e|b.d(e,src)<=4) {23.}else{10.};}
 }
 }
 // Cannot attack after moving non-dash, or after embarking/disembarking.
 let can_attack=!u.attacked&&s.dash&&nu.k==u.k;
 if can_attack {
 let mut best=-1000f32;for &e in &b.enemy {if b.d(dst,e)<=s.r {best=best.max(attack_score(b,tac,dst,e,u));}}
 if best>0. {score+=best*0.85+8.;
 let mut relief=0f32;
 for &e in &b.enemy {if b.d(dst,e)>s.r{continue}let (hit,ret)=b.combat(dst,e,u,b.u[e]);if hit>=b.u[e].h && ret<u.h {
 let rr=tac.risk(b,dst,nu,e);relief=relief.max((health_loss(nu,risk)-health_loss(nu,rr))*3.8);
 }}score+=relief;
 }else if s.r>1 {let near=b.enemy.iter().any(|&e|b.d(src,e)<=s.r);if near {score-=25.;}}
 }
 if u.k==8||u.k==41 {
 let targets=b.enemy.iter().filter(|&&e|b.d(dst,e)<=s.r).count();if targets>0&&risk==0. {score+=8.;}
 let shooting=b.enemy.iter().any(|&e|b.d(src,e)<=s.r);if shooting&&!u.attacked {score-=50.;}
 }
 let friendly=b.own.iter().filter(|&&j|j!=src&&b.d(j,dst)<=1).count();if risk>0. {score+=friendly.min(3) as f32*2.;}
 if b.t[src].owner==b.p && u.h<mh(u)-3. && !cap && !ruin && risk>0. {score-=10.;}
 if u.attacked&&s.escape {score+=2.;}
 if b.t[src].st==1&&b.t[src].owner>0&&b.t[src].owner!=b.p {score-=110.;}
 if b.t[src].st==1&&b.t[src].owner==0 {score-=80.;}
 if let Some(&vis)=mem.visits.get(dst){score-=vis.min(15) as f32*0.25;}
 // Always avoid wandering with no useful progress.
 if aft<=bef+0.01&&reveal==0.&&!cap&&!ruin&&risk>=oldrisk {score-=9.;}
 score-1.
}
fn pop_value(b:&Board,ci:usize,amount:i32,cost:f32)->f32 {
 if ci>=b.cities.len(){return -100.}let c=&b.cities[ci];if b.u[c.tile].owner>0&&b.u[c.tile].owner!=b.p {return -100.}
 let need=c.level+1-c.progress;let danger=b.enemy.iter().any(|&e|b.d(e,c.tile)<=2);let mut val=match c.level{1=>15.,2=>12.,3=>15.,_=>18.};
 if amount>=need {val+=match c.level {1=>36.,2=>24.,3=>24.,_=>58.};}else{val+=match c.level{1=>18.,2=>10.,3=>3.,_=>12.}*amount as f32/need as f32;}
 val+=amount as f32*2.;val-=cost*3.2;
 if c.level==1 && b.turn<5 {val+=27.;}
 if b.turn==0&&c.level==1 {val+=28.;}
 if danger&&c.level>=4&&amount>=need {val+=35.;}
 if danger&&c.level<4 {val-=if c.level==1{0.}else{8.};}
 if b.stars>30 {val+=(b.stars-30).min(80) as f32*0.08;}
 // Large investments require a believable path to the next reward.
 if need>amount {let mut avail=0;for i in 0..b.n {let t=b.t[i];if t.city!=ci||t.st!=0{continue}if t.res==1||t.res==6&&b.tech[6]||t.res==3&&b.tech[10]{avail+=1;}else if t.ter==5&&b.tech[16]{avail+=1;}else if t.res==2&&b.tech[8]||t.res==5&&b.tech[23]{avail+=2;}}
 if avail<need-amount&&c.level>=3 {val-=10.;}
 }
 val
}
fn tech_tier(k:i32)->i32 {match k{1|6|10|15|20=>1,2|4|7|8|13|16|18|21|23|44=>2,_=>3}}
fn research_score(b:&Board,k:i32,naval_need:bool)->f32 {
 let cost=4+b.ct.len() as i32*tech_tier(k);let spare=b.stars-cost;let enemies=b.enemy.len() as f32;
 let mut forests=0f32;let mut fruits=0f32;let mut fish=0f32;let mut farms=0f32;let mut mines=0f32;let mut mountains=0f32;let mut sea=0f32;let mut can_upgrade=0f32;
 for i in 0..b.n {let t=b.t[i];if !t.known {continue}if t.ter<=2 {sea+=1.;}if t.ter==4 {mountains+=1.;}if t.owner!=b.p||t.st!=0{continue}if t.ter==5 {forests+=1.;}if t.res==6 {fruits+=1.;}if t.res==3 {fish+=1.;}if t.res==2 {farms+=1.;}if t.res==5 {mines+=1.;}}
 for &ci in &b.ct {if b.cities[ci].level<=2 {can_upgrade+=1.;}}
 let water_ratio=sea/(b.t.iter().filter(|t|t.known).count() as f32).max(1.);
 let navy=if naval_need {1.}else{0.};
 let mut v=match k {
 1=> (if b.turn<2 {6.}else{23.})+ if b.own.len()<4{17.}else{0.},
 6=>fruits.min(5.)*5.+if fruits>=2.&&can_upgrade>0. {12.}else{0.},
 10=>fish.min(5.)*5.+navy*34.+water_ratio*20.,
 16=>forests.min(9.)*4.+if forests>=3. {18.}else{0.},
 17=>(if b.tech[16]{30.}else{0.})+b.ct.len() as f32*3.+enemies.min(4.)*4.,
 18=>(if enemies>0. {30.}else{12.})+forests*0.6,
 20=>mountains.min(12.)*1.7+if b.turn>=4{10.}else{0.},
 23=>mines.min(5.)*6.+if enemies>0. {28.}else{4.},
 24=>(if enemies>0. {55.}else{20.})+if b.stars>cost+5 {12.}else{0.},
 13=>navy*48.+water_ratio*24.+if b.own.iter().any(|&i|b.u[i].k==43){30.}else{0.},
 44=>navy*30.+if b.own.iter().any(|&i|b.u[i].k==43){28.}else{0.}+water_ratio*10.,
 14=>navy*48.+if b.own.iter().any(|&i|b.u[i].k==43){12.}else{0.},
 8=>farms.min(6.)*5.+if farms>=3. {12.}else{0.},
 9=>farms.min(8.)*2.+if farms>=4. {14.}else{0.},
 2=>if b.turn>12&&enemies>=3. {25.}else{-20.},
 3=>if enemies>=3. {48.}else{12.},
 4=>if b.turn>8 {20.}else{0.},
 5=>if b.ct.len()>=3&&b.tech[17] {25.}else{-20.},
 7=>if enemies>0. {18.}else{-30.},
 21=>-25.,22=>if b.ct.len()>=4&&b.stars>35 {20.}else{-30.},
 38=>if b.turn>25 {12.}else{-30.},
 _=>-100.};
 // Value economy research by the actual level-ups it immediately enables, not just the resource count.
 if k==6 || k==10 || k==16 || k==8 || k==23 {
 let mut uplift=0f32;
 for &ci in &b.ct {let c=&b.cities[ci];let mut count=0;
 for i in 0..b.n {let t=b.t[i];if t.city!=ci||t.st!=0 {continue}count+=match k {6=>(t.res==6) as i32,10=>(t.res==3) as i32,16=>(t.ter==5) as i32,8=>2*(t.res==2) as i32,23=>2*(t.res==5) as i32,_=>0};}
 if count>=c.level+1-c.progress&&count>0 {uplift=uplift.max(match c.level {1=>24.,2=>16.,3=>18.,_=>30.});}
 }
 v+=uplift;
 }
 v-=cost as f32*1.5;
 if b.stars>70 && [3,17,24,14,23,2,44].contains(&k) {v+=12.;}
 if spare<2 {v-=15.;}if b.turn==0{v-=65.;}
 if b.own.len()<2&&spare<2 {v-=20.;}
 if b.ct.iter().any(|&ci|b.cities[ci].level==1)&&spare<4 {v-=10.;}
 // Don't unlock technology we cannot exploit until later, except movement.
 if spare<3&&k!=20&&k!=1 {v-=8.;}
 if b.enemy.len()>b.own.len()+1 && spare<6 && k!=18 && k!=24 {v-=30.;}
 if k==2||k==3 {if b.stars<cost+8 {v-=25.;}}
 if b.own.len()<4&&b.enemy.len()>2&&spare<3 {v-=20.;}
 if b.own.len()<3 && b.enemy.len()>=3 && spare<5 && k!=18 && k!=24 {v-=20.;}
 if k==4&&b.ct.len()<3&&b.own.len()<6 {v-=18.;}
 if k==18&&b.tech[24] {v-=12.;}
 v
}
fn build_score(b:&Board,k:i32,dst:usize,naval_need:bool)->f32 {
 let ci=b.t[dst].city;let adj=&b.adj[dst];let mut pop=match k {5|21=>2,8|12=>1,23..=29=>3,_=>0};let cost=match k{3|12=>3.,8=>7.,5|6|13|21|22|48|50=>5.,23..=29=>0.,_=>20.};
 if k==6||k==13||k==22 {let base=if k==6{5}else if k==13{12}else{21};pop=adj.iter().filter(|&&i|b.t[i].owner==b.p&&b.t[i].st==base).count() as i32*if k==22{2}else{1};}
 let mut score=if pop>0 {pop_value(b,ci,pop,cost)}else{-100.};
 if k==12 {if b.t[dst].res==1 {score-=22.;}for &j in adj {if b.t[j].st==13&&b.t[j].owner==b.p{score+=8.;}}}
 if k==5 {for &j in adj {if b.t[j].st==6&&b.t[j].owner==b.p{score+=7.;}}}
 if k==21 {for &j in adj {if b.t[j].st==22&&b.t[j].owner==b.p{score+=12.;}}}
 if k==8 {
 let ports=b.t.iter().filter(|t|t.st==8&&t.owner==b.p).count();
 let mut proximity=999;for &u in &b.own {if !stat(b.u[u].k).sea {proximity=proximity.min(b.d(u,dst));}}
 if naval_need&&ports==0 {score=45.+(5-proximity.min(5)) as f32*4.;} else if naval_need&&ports<3 {score=score.max(12.+(4-proximity.min(4)) as f32*2.);}
 if b.t[dst].res==3 {score-=8.;}
 // choose a port on a connected open sea, preferably towards unexplored/enemy territory
 let water=adj.iter().filter(|&&j|b.t[j].ter<=2).count();score+=water as f32;
 if ports>=3 {score-=18.;}
 }
 if k==3||k==48 {return -50.;}
 if k==50 {let mut yield_=0.;for &j in adj {if b.t[j].owner==b.p&&[6,13,22].contains(&b.t[j].st){yield_+=b.t[j].sl as f32;}}return yield_.min(8.)*9.-12.;}
 if (17..=20).contains(&k){return -100.;}
 // preserve ideal future multiplier sites when adding ordinary improvements
 if k==12&&b.tech[17]{let local=adj.iter().filter(|&&i|b.t[i].ter==5&&b.t[i].owner==b.p).count();if local>=3{score-=2.;}}
 score
}
fn train_score(b:&Board,tac:&Tactics,gs:&[Goal],src:usize,k:i32,naval_need:bool)->f32 {
 let s=stat(k);let nu=b.own.len() as f32;let upgrading=b.u[src].k==43;
 if upgrading {let near=b.enemy.iter().map(|&e|b.d(src,e)).min().unwrap_or(20);return match k {44=>36.+if near<=4{23.}else{0.},42=>30.+if b.turn<16 {10.}else{0.},41=>if b.stars>=18 {55.+if near<=5{20.}else{0.}}else{12.},_=>-100.};}
 if k==38||k==10 {return -80.;}
 let nearest=b.enemy.iter().map(|&e|b.d(src,e)).min().unwrap_or(99);let front=b.cities.iter().filter(|c|c.owner>0&&c.owner!=b.p).map(|c|b.d(src,c.tile)).min().unwrap_or(99);
 let vacant=b.cities.iter().filter(|c|c.owner==0).map(|c|b.d(src,c.tile)).min().unwrap_or(99);
 let mut u=Unit{owner:b.p,k,h:s.h,..Unit::default()};let potential=goal_value(b,gs,src,src,u);
 let mut score=10.+potential*0.18-nu*2.2;
 if nearest<=4 {score+=23.;}if nearest<=2 {score+=20.;}
 if front<=5 {score+=12.;}
 if b.own.len()<2 {score+=26.;}else if b.own.len()<3&&b.turn>=3 {score+=12.;}
 if nu>=b.income as f32*1.5+2. {score-=30.;}
 if naval_need&&nearest>5&&nu>=5. {score-=18.;}
 if nearest>7&&vacant>6&&nu>=6. {score-=20.;}
 match k {
 2=>{score+=5.;if b.tech[24]||b.tech[18]&&nearest<6 {score-=12.;}if b.tech[1]&&nearest>4 {score-=12.;}},
 3=>{score+=if nearest>3{14.}else{0.};if nearest<=2{score-=6.;}},
 4=>{score+=if nearest<=5{24.}else{10.};let soft=b.enemy.iter().filter(|&&e|b.d(src,e)<7&&[2,3,8,9,10].contains(&b.u[e].k)).count();score+=soft.min(4) as f32*4.;},
 5=>{score+=if nearest<=3 {12.}else{-25.};},
 8=>{let count=b.own.iter().filter(|&&i|b.u[i].k==8).count();score+=if nearest<=5||front<=6{25.}else{0.};score-=count as f32*10.;if b.own.len()<3 {score-=25.;}if nearest<=2{score-=35.;}},
 9=>{score+=if nearest<=5 {20.}else{0.};let count=b.own.iter().filter(|&&i|b.u[i].k==9).count();score-=count as f32*3.;if nearest<=1{score-=12.;}},
 11=>{score+=if nearest<=5||front<=6 {28.}else{12.};},
 _=>{score-=40.;}
 }
 score-=s.cost*1.8;
 if b.stars>30 {score+=(b.stars-30).min(100) as f32*0.16;}
 let risk=tac.risk(b,src,u,usize::MAX);if risk>=u.h {score-=health_loss(u,risk)*3.;}else if risk>0. {score-=health_loss(u,risk);}
 // Keep enough economic capital when safe to buy the next income upgrade or research.
 if nearest>4&&b.stars<s.cost as i32+4&&nu>=3. {score-=12.;}
 u.h=0.;score
}
fn choose(obs:&Value,mem:&mut Memory)->String {
 let b=Board::parse(obs);let legal:Vec<Action>=obs["legal_actions"].as_array().unwrap().iter().map(Action::parse).collect();
 if mem.visits.len()!=b.n {mem.visits=vec![0;b.n];mem.prev=vec![usize::MAX;b.n];mem.turn=-1;}
 if mem.turn!=b.turn {mem.turn=b.turn;mem.actions=0;for &i in &b.own {mem.visits[i]+=1;}}
 mem.actions+=1;
 // Resolve compulsory city choices before doing expensive military calculations.
 if legal.iter().any(|a|a.k==9) {let mut best=(&legal[0],-1e9f32);for a in &legal {if a.k!=9{continue}let ci=b.cities.iter().position(|c|c.tile==a.t).unwrap_or(0);let c=&b.cities[ci];let val=match a.p{
 3=>100.,4=>15.,7=>100.,1=>{let adj=b.enemy.iter().filter(|&&e|b.d(e,a.t)<=2).count();if adj>=2&&b.ct.len()<=2 {125.}else if adj>=3 {105.}else{0.}},
 5=>{let mut val=0.;for i in 0..b.n{if b.d(i,a.t)==2&&b.t[i].owner==0{val+=if b.t[i].res==1||b.t[i].res==6||b.t[i].res==3{3.}else if b.t[i].ter==5{1.8}else{0.8};}}45.+val*3.},
 8=>66.+if c.progress>=2{20.}else{0.},6=>120.,2=>0.,_=>-100.};if val>best.1{best=(a,val)}}return best.0.id.clone();}
 let tac=Tactics::new(&b);let gs=goals(&b);let roads=road_scores(&b);
 // A sea route is useful when substantial water blocks known or potential expansion.
 let water=b.t.iter().filter(|t|t.known&&t.ter<=2).count();let known=b.t.iter().filter(|t|t.known).count();
 let disconnected=gs.iter().filter(|g|g.kind==1||g.kind==2).any(|g|b.own.iter().all(|&i|g.land[i]>=9999));
 let exploration_land=gs.iter().filter(|g|g.kind==0&&b.t[g.tile].ter>2).any(|g|b.own.iter().any(|&i|g.land[i]<9999));
 let naval_need=disconnected||(!exploration_land&&water>6)||water as f32/known.max(1) as f32>0.23&&b.turn>8;
 if naval_need&&mem.naval_since==0 {mem.naval_since=b.turn.max(1);}
 let mut best=-1e9f32;let mut which=0;
 for (idx,a) in legal.iter().enumerate(){let score=match a.k {
 11=>-1e8,
 10=>0.,
 8=>{let st=b.t[a.s].st;if st==1{if b.t[a.s].owner>0{1500.}else{1000.}}else{850.}},
 2=>attack_score(&b,&tac,a.s,a.t,b.u[a.s]),
 1=>move_score(&b,&tac,&gs,mem,a.s,a.t),
 5=>pop_value(&b,b.t[a.t].city,1,2.),
 6=>if a.p==3||a.p==48 {roads[a.t]}else{build_score(&b,a.p,a.t,naval_need)},
 7=>research_score(&b,a.p,naval_need),
 4=>train_score(&b,&tac,&gs,a.s,a.p,naval_need),
 3=> match a.p {
 16=>500.,
 7=>{let u=b.u[a.s];let healing=(mh(u)-u.h).min(if b.t[a.s].owner==b.p {4.}else{2.});let danger=tac.risk(&b,a.s,u,usize::MAX);let mut sc=healing*3.;if u.h<=mh(u)*0.6{sc+=8.;}if danger>=u.h+healing {sc-=30.;}else if danger>=u.h {sc+=10.;}sc},
 9=>20.,
 2=>{let ci=b.t[a.t].city;let need=if ci<b.cities.len(){b.cities[ci].level+1-b.cities[ci].progress}else{0};if b.t[a.t].res==1{-80.}else if b.stars<2&&need>4&&ci<b.cities.len()&&b.cities[ci].level>=6 {1.}else{-60.}},
 19=>if a.t==0 {2.}else{-100.},
 _=>-1000.},
 _=>-1000.};
 
 let tie=(a.s.wrapping_mul(31)+a.t.wrapping_mul(17)+a.p.max(0) as usize) as f32*0.00001;
 if score+tie>best {best=score+tie;which=idx;}}
 let a=&legal[which];if a.k==1 {mem.prev[a.t]=a.s;mem.visits[a.t]+=1;}
 a.id.clone()
}
fn main(){let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){let line=match line{Ok(l)=>l,Err(_)=>break};if line.trim().is_empty(){continue}let obs:Value=serde_json::from_str(&line).unwrap();let act=choose(&obs,&mut mem);writeln!(stdout,"{}",json!({"action_id":act})).unwrap();stdout.flush().unwrap();}}
// Plan short economic road connections, also improving land army mobility.
fn road_scores(b:&Board)->Vec<f32> {
 let mut sc=vec![-100.;b.n];if !b.tech[4] {return sc}let cap=b.ct.iter().map(|&ci|b.cities[ci].tile).find(|&i|b.t[i].cap==b.p);let cap=match cap{Some(x)=>x,None=>return sc};
 let mut ds=vec![9999;b.n];let mut parent=vec![usize::MAX;b.n];let mut hp=BinaryHeap::new();ds[cap]=0;hp.push(Reverse((0,cap)));
 while let Some(Reverse((d,i)))=hp.pop(){if d!=ds[i]{continue}for &j in &b.adj[i]{let t=b.t[j];if !t.known||t.owner>0&&t.owner!=b.p||t.ter==4 {continue}if b.u[j].owner>0&&b.u[j].owner!=b.p{continue}let cost=if b.road(j,b.p){0}else if t.ter>2 {3}else if t.st==0&&t.ter==1 {6}else{999};let nd=d+cost;if nd<ds[j]{ds[j]=nd;parent[j]=i;hp.push(Reverse((nd,j)));}}}
 for &ci in &b.ct {let c=&b.cities[ci];if c.tile==cap||c.connected {continue}let distance=ds[c.tile];if distance>15||distance<=0{continue}let pop1=pop_value(b,ci,1,0.);let capci=b.t[cap].city;let pop2=pop_value(b,capci,1,0.);let val=(pop1+pop2)*0.55+15.;
 let roads=distance as f32/3.;let mut at=c.tile;let mut steps=0;while at!=cap&&at!=usize::MAX&&steps<b.n {if !b.road(at,b.p){let mut v=val/roads-4.;if b.stars<distance&&distance>6{v-=12.;}if b.turn>15 {v+=3.;}sc[at]=sc[at].max(v);}at=parent[at];steps+=1;}
 }
 sc
}
