// Bardur skirmish policy. All commands are selected from the current legal catalog.
// Geometry is derived from board_size. Public information only; no map seeds.
// Strategy: efficient growth and expansion; multi-modal land/naval routing;
// terrain-aware danger, coordinated volleys, unit composition, and road links.
// The process keeps only movement history to discourage reversible wandering.
#![allow(dead_code)]
use serde_json::{Value,json};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
fn num(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn flag(v:&Value,k:&str)->bool{v[k].as_bool().unwrap_or(false)}
#[derive(Clone,Default)] struct Tile{known:bool,ter:i32,owner:i32,st:i32,res:i32,road:bool,cap:i32,sl:i32}
#[derive(Clone,Default)] struct Unit{pos:usize,owner:i32,k:i32,hp:f64,mh:f64,moved:bool,attacked:bool,kills:i32,home:i32,vet:bool}
#[derive(Clone,Default)] struct City{pos:usize,owner:i32,level:i32,progress:i32,prod:i32,rewards:[i32;9]}
#[derive(Clone,Copy)] struct Stats{a:f64,d:f64,h:f64,m:i32,r:i32,c:f64,dash:bool,fort:bool,stiff:bool}
fn stats(k:i32)->Stats{let (a,d,h,m,r,c,dash,fort,stiff)=match k{
2=>(2.,2.,10.,1,1,2.,true,true,false),3=>(2.,1.,10.,2,1,3.,true,true,false),4=>(3.5,1.,10.,3,1,8.,true,true,false),5=>(1.,3.,15.,1,1,3.,false,true,false),8=>(4.,0.,10.,1,3,8.,false,false,true),9=>(2.,1.,10.,1,2,3.,true,true,false),10=>(0.,1.,10.,1,1,5.,false,false,true),11=>(3.,3.,15.,1,1,5.,true,false,false),12=>(5.,4.,40.,1,1,15.,false,false,false),38=>(0.,0.5,5.,2,1,8.,true,false,true),41=>(3.,2.,10.,2,3,15.,false,false,true),42=>(2.,1.,10.,3,2,5.,true,false,false),43=>(0.,1.,10.,2,2,2.,false,false,true),44=>(3.,3.,10.,3,1,5.,true,false,false),45=>(4.,4.,40.,2,1,15.,false,false,true),37|40=>(2.,2.,10.,2,1,3.,true,true,false),_=>(2.,2.,10.,1,1,3.,true,false,false)}; Stats{a,d,h,m,r,c,dash,fort,stiff}}
fn boat(k:i32)->bool{(39..=45).contains(&k)}
#[derive(Clone)] struct Action{id:String,k:i32,s:usize,t:usize,p:i32}
struct Game{size:usize,n:usize,me:i32,turn:i32,stars:f64,tiles:Vec<Tile>,units:Vec<Unit>,at:Vec<Option<usize>>,cities:Vec<City>,ct:Vec<Option<usize>>,tech:[bool;128],actions:Vec<Action>,adj:Vec<Vec<usize>>,own:Vec<usize>,enemy:Vec<usize>,mycities:Vec<usize>,territory:Vec<usize>,income:f64,water:f64}
impl Game{
fn read(o:&Value)->Self{let v=&o["visible_state"];let size=num(v,"board_size") as usize;let n=size*size;let me=num(o,"player");let mut tiles=vec![Tile::default();n];for t in v["tiles"].as_array().unwrap(){let i=num(t,"tile") as usize;tiles[i]=Tile{known:flag(t,"known"),ter:num(t,"terrain"),owner:num(t,"owner"),st:num(t,"structure"),res:num(t,"resource"),road:num(t,"road")>0,cap:num(t,"capital"),sl:num(t,"structure_level")};}
let mut tech=[false;128];for t in v["technology"].as_array().unwrap(){tech[t.as_u64().unwrap() as usize]=true;}
let mut units=Vec::new();let mut at=vec![None;n];let(mut own,mut enemy)=(vec![],vec![]);for u in v["units"].as_array().unwrap(){let pos=num(u,"tile") as usize;let k=num(u,"kind");let pass=num(u,"passenger");let mh=(stats(if pass>0{pass}else{k}).h+if flag(u,"veteran"){5.}else{0.}).max(u["health"].as_f64().unwrap_or(10.));let idx=units.len();at[pos]=Some(idx);if num(u,"owner")==me{own.push(idx)}else{enemy.push(idx)};units.push(Unit{pos,owner:num(u,"owner"),k,hp:u["health"].as_f64().unwrap_or(10.),mh,moved:flag(u,"moved"),attacked:flag(u,"attacked"),kills:num(u,"kills"),home:num(u,"home"),vet:flag(u,"veteran")});}
let mut cities=vec![];let mut ct=vec![None;n];let mut mycities=vec![];let mut income=0.;for c in v["cities"].as_array().unwrap(){let pos=num(c,"tile")as usize;let mut rewards=[0;9];if let Some(a)=c["rewards"].as_array(){for i in 0..9{rewards[i]=a[i].as_i64().unwrap_or(0)as i32;}}let owner=num(c,"owner");if owner==me{mycities.push(cities.len());income+=(num(c,"production")+if tiles[pos].cap==me{1}else{0})as f64;}ct[pos]=Some(cities.len());cities.push(City{pos,owner,level:num(c,"level"),progress:num(c,"progress"),prod:num(c,"production"),rewards});}
let actions=o["legal_actions"].as_array().unwrap().iter().map(|a|{let c=&a["command"];Action{id:a["action_id"].as_str().unwrap().to_string(),k:num(c,"kind"),s:num(c,"source").max(0)as usize,t:num(c,"target").max(0)as usize,p:num(c,"parameter")}}).collect();
let mut adj=vec![vec![];n];for i in 0..n{for dy in -1i32..=1{for dx in -1i32..=1{let x=(i%size)as i32+dx;let y=(i/size)as i32+dy;if (dx!=0||dy!=0)&&x>=0&&y>=0&&x<size as i32&&y<size as i32{adj[i].push(y as usize*size+x as usize)}}}}
let water=tiles.iter().filter(|t|t.known&&t.ter<=2).count()as f64/(tiles.iter().filter(|t|t.known).count().max(1)as f64);
let mut g=Self{size,n,me,turn:num(v,"turn"),stars:num(v,"stars")as f64,tiles,units,at,cities,ct,tech,actions,adj,own,enemy,mycities,territory:vec![usize::MAX;n],income,water};
for i in 0..n{if g.tiles[i].owner==me{let mut best=99;for &ci in &g.mycities{let c=&g.cities[ci];let d=g.dist(i,c.pos);if d<best&&d<=if c.rewards[5]>0{2}else{1}{best=d;g.territory[i]=ci}}}}
g}
fn dist(&self,a:usize,b:usize)->i32{((a%self.size)as i32-(b%self.size)as i32).abs().max(((a/self.size)as i32-(b/self.size)as i32).abs())}
fn bonus(&self,u:&Unit,pos:usize)->f64{let t=&self.tiles[pos];if t.ter==4{return 1.5}if t.ter==5&&(u.owner!=self.me||self.tech[18]){return 1.5}if t.ter<=2&&self.tech[12]&&u.owner==self.me{return 1.5}if t.st==1&&t.owner==u.owner&&stats(u.k).fort{if let Some(ci)=self.ct[pos]{if self.cities[ci].rewards[1]>0{return 4.}}return 1.5}1.}
fn damage(&self,a:&Unit,apos:usize,d:&Unit,dpos:usize)->(f64,f64){if a.k==10{return(d.hp,0.)}let sa=stats(a.k);let sd=stats(d.k);let af=sa.a*a.hp/a.mh;let df=sd.d*d.hp/d.mh*self.bonus(d,dpos);let tot=(af+df).max(0.001);let dmg=(af/tot*sa.a*4.5+0.00001).round();let ret=if dmg>=d.hp||sd.stiff||self.dist(apos,dpos)>sd.r||a.k==37||a.k==40{0.}else{(df/tot*sd.d*4.5+0.00001).round()};(dmg,ret)}
fn value(&self,u:&Unit)->f64{let s=stats(u.k);(s.c+3.)*(0.45+0.55*u.hp/u.mh)+if u.k==12||u.k==45{7.}else{0.}}
// Potential attack origins next turn, including terrain, roads and zones of control.
fn reach(&self,u:&Unit)->Vec<usize>{let st=stats(u.k);let mut d=vec![999;self.n];d[u.pos]=0;let mut heap=BinaryHeap::new();heap.push(Reverse((0,u.pos)));let budget=2*st.m;let mut out=vec![u.pos];while let Some(Reverse((cost,x)))=heap.pop(){if cost!=d[x]||cost>=budget{continue}if x!=u.pos{let t=&self.tiles[x];if t.ter==4||(!boat(u.k)&&t.st==8)|| (boat(u.k)&&t.ter>2)||self.adj[x].iter().any(|&p|self.at[p].map_or(false,|i|self.units[i].owner!=u.owner)){continue}}
for &y in &self.adj[x]{let t=&self.tiles[y];if !t.known{continue}if !boat(u.k)&&t.ter<=2&&t.st!=48&&!(t.st==8&&t.owner==u.owner){continue}if t.ter==4&&u.owner==self.me&&!self.tech[20]{continue}if let Some(j)=self.at[y]{if self.units[j].owner!=u.owner{continue}}
let road=(self.tiles[x].road||self.tiles[x].st==1)&&(t.road||t.st==1)&&(t.owner==0||t.owner==u.owner);let mut c=cost+if road{1}else{2};if t.ter==5&&!road{c=c.max(budget)}if c<=budget&&c<d[y]{d[y]=c;heap.push(Reverse((c,y)));if self.at[y].is_none(){out.push(y)}}}}
out}
}
#[derive(Default)]struct Memory{turn:i32,visits:Vec<i32>,prev:Vec<usize>,last_turn:Vec<i32>}
struct Context{threat:Vec<Vec<usize>>,support:Vec<f64>,explore:Vec<f64>,goals:Vec<(usize,f64,i32)>,paths:Vec<Vec<Vec<f64>>>,assign:Vec<usize>,nearest:Vec<i32>,port_need:bool,roads:Vec<f64>}
// Multi-mode shortest paths backwards: land to ships through friendly ports, ships to land anywhere.
fn path(g:&Game,goal:usize,movement:i32,start_boat:bool)->Vec<f64>{let n=g.n;let mut ds=vec![1e6f64;n*2];let mut heap=BinaryHeap::new();for mode in 0..2{ds[mode*n+goal]=0.;heap.push(Reverse((0i32,mode*n+goal)));}while let Some(Reverse((cost,node)))=heap.pop(){if (ds[node]*100.).round()as i32!=cost{continue}let y=node%n;let mode=node/n;for &x in &g.adj[y]{for pm in 0..2{let t=&g.tiles[y];let xt=&g.tiles[x];let yw=t.known&&t.ter<=2&&t.st!=48;let xw=xt.known&&xt.ter<=2&&xt.st!=48;
if (pm==0&&xw)||(pm==1&&!xw){continue}let allowed=if pm==0{if yw{mode==1&&t.st==8&&t.owner==g.me&&g.tech[10]}else{mode==0}}else{if yw{mode==1&&(t.ter!=2||g.tech[13])}else{mode==0}};if !allowed{continue}if t.ter==4&&!g.tech[20]{continue}
let speed=if pm==1{if start_boat{movement}else if g.tech[13]{3}else{2}}else{if start_boat{1}else{movement}};
let road=(xt.road||xt.st==1)&&(t.road||t.st==1)&&(xt.owner==0||xt.owner==g.me)&&(t.owner==0||t.owner==g.me);
let mut edge=if road&&!yw{50./speed as f64}else{100./speed as f64};if t.ter==4||t.ter==5{edge=edge.max(100.)}if pm!=mode{edge=125.}if !t.known{edge+=45.}if let Some(u)=g.at[y]{if g.units[u].owner!=g.me{edge+=120.}else{edge+=10.}}
let nc=cost+edge as i32;let pn=pm*n+x;if (nc as f64)/100.+0.001<ds[pn]{ds[pn]=(nc as f64)/100.;heap.push(Reverse((nc,pn)));}}}}
ds}
fn class(k:i32)->usize{if boat(k){if stats(k).m==3{4}else{3}}else{(stats(k).m-1).clamp(0,2)as usize}}

fn road_plan(g:&Game)->Vec<f64>{
let n=g.n;let mut value=vec![-30.;n];if !g.tech[4]||g.mycities.len()<2{return value}
let cap=match g.mycities.iter().find(|&&i|g.tiles[g.cities[i].pos].cap==g.me){Some(&i)=>g.cities[i].pos,None=>return value};
let mut d=vec![999i32;n];let mut par=vec![usize::MAX;n];let mut heap=BinaryHeap::new();d[cap]=0;heap.push(Reverse((0,cap)));
while let Some(Reverse((cost,x)))=heap.pop(){if d[x]!=cost{continue}for &y in &g.adj[x]{let t=&g.tiles[y];if !t.known||t.owner==3-g.me||t.ter<=2&&t.st!=48&&t.st!=8||t.ter==4{continue}if g.at[y].map_or(false,|i|g.units[i].owner!=g.me){continue}let node=t.road||t.st==1||t.st==48||t.st==8;let dc=if node{0}else{100};if cost+dc<d[y]{d[y]=cost+dc;par[y]=x;heap.push(Reverse((cost+dc,y)));}}}
for &ci in &g.mycities{let c=&g.cities[ci];let missing=d[c.pos]/100;if c.pos==cap||missing<=0||missing>4{continue}let mut route=vec![];let mut x=c.pos;let mut steps=0;while x!=cap&&x!=usize::MAX&&steps<n{if !g.tiles[x].road&&g.tiles[x].st!=1&&g.tiles[x].st!=8&&g.tiles[x].st!=48{route.push(x);}x=par[x];steps+=1;}
let need=(c.level+1-c.progress).max(1);let capc=&g.cities[g.ct[cap].unwrap()];let cn=(capc.level+1-capc.progress).max(1);let benefit=14./need as f64+14./cn as f64+10.;let score=benefit/missing as f64-3.;for &t in &route{value[t]=value[t].max(score+if route.len()==1{7.}else{0.});}}
value}
fn make_context(g:&Game)->Context{let n=g.n;let mut threat=vec![vec![];n];let mut support=vec![0.;n];let mut nearest=vec![99;n];for &ei in &g.enemy{let u=&g.units[ei];let st=stats(u.k);let origins=if st.dash{g.reach(u)}else{vec![u.pos]};for p in 0..n{nearest[p]=nearest[p].min(g.dist(p,u.pos));if st.a>0.||u.k==10{if origins.iter().any(|&x|g.dist(x,p)<=st.r){threat[p].push(ei)}}}}
for &ui in &g.own{let u=&g.units[ui];for p in 0..n{let d=g.dist(p,u.pos);if d<=3{support[p]+=(stats(u.k).a+stats(u.k).d)*u.hp/u.mh/(d as f64+1.);}}}
let mut explore=vec![0.;n];for p in 0..n{if !g.tiles[p].known{continue}let rad=if g.tiles[p].ter==4{2}else{1};for t in 0..n{if !g.tiles[t].known&&g.dist(p,t)<=rad{let x=t%g.size;let y=t/g.size;explore[p]+=if x==0||y==0||x==g.size-1||y==g.size-1{0.65}else{1.};}}}
let mut goals=vec![];for c in &g.cities{if c.owner!=g.me{let val=if c.owner==0{45.}else{35.};goals.push((c.pos,val,if c.owner==0{0}else{1}));}else if nearest[c.pos]<=3&&g.at[c.pos].map_or(true,|i|g.units[i].owner!=g.me){let siege=g.at[c.pos].map_or(false,|i|g.units[i].owner!=g.me);goals.push((c.pos,if siege{85.}else{26.},2));}}
for p in 0..n{if g.tiles[p].st==2{goals.push((p,28.,3))}}
// Select spatially spread exploration destinations.
let mut order:Vec<usize>=(0..n).filter(|&p|explore[p]>0.&&g.tiles[p].known).collect();order.sort_by(|&a,&b|explore[b].partial_cmp(&explore[a]).unwrap());let mut front=vec![];for p in order{if front.iter().all(|&q|g.dist(p,q)>2){front.push(p);goals.push((p,10.+explore[p]*1.1,4));}}
for &e in &g.enemy{let ep=g.units[e].pos;if goals.iter().all(|&(p,_,ty)|ty!=1||g.dist(p,ep)>2){goals.push((ep,29.,1));}}
let classes=[(1,false),(2,false),(3,false),(2,true),(3,true)];let paths:Vec<Vec<Vec<f64>>>=goals.iter().map(|&(p,_,_)|classes.iter().map(|&(m,b)|path(g,p,m,b)).collect()).collect();
let mut assign=vec![usize::MAX;g.units.len()];let mut loads=vec![0.;goals.len()];let mut remaining=g.own.clone();while !remaining.is_empty()&&!goals.is_empty(){let mut best=(-1f64,0,usize::MAX);for (ri,&ui) in remaining.iter().enumerate(){let u=&g.units[ui];let cl=class(u.k);let idx=u.pos+if boat(u.k){n}else{0};for (j,&(p,v,ty))in goals.iter().enumerate(){let d=paths[j][cl][idx];if d>1e5{continue}let penalty=if ty==0||ty==3||ty==4{loads[j]*3.5}else{loads[j]*0.3};let mut value=v/(d+2.)/(1.+penalty);if ty==4&&(u.k==8||u.k==12){value*=0.5}if ty==0&&u.hp<5.{value*=0.75}if ty==2&&g.dist(u.pos,p)>5{value*=0.6}if value>best.0{best=(value,ri,j)}}}if best.2==usize::MAX{break}let ui=remaining.remove(best.1);assign[ui]=best.2;loads[best.2]+=1.;}
let port_need=g.water>0.18&&(g.turn>5||g.mycities.len()>=2);
let roads=road_plan(g);Context{threat,support,explore,goals,paths,assign,nearest,port_need,roads}}
fn danger(g:&Game,ctx:&Context,u:&Unit,p:usize,ignore:usize)->f64{let mut damages=vec![];for &ei in &ctx.threat[p]{if ei==ignore{continue}let e=&g.units[ei];let mut ee=e.clone();let d=g.dist(e.pos,p);let st=stats(e.k);let prob=if d<=st.r{1.}else{0.8};ee.hp=e.hp;let dmg=g.damage(&ee,e.pos,u,p).0;damages.push(dmg*prob);}damages.sort_by(|a,b|b.partial_cmp(a).unwrap());let total=damages.iter().take(4).sum::<f64>();let val=g.value(u);let mut loss=val*(total/u.hp.max(1.)).min(1.4);if total>=u.hp{loss+=val*0.45;}loss}
fn attack_score(g:&Game,ctx:&Context,ui:usize,target:usize,apos:usize)->f64{let u=&g.units[ui];let di=match g.at[target]{Some(i)=>i,None=>return -1e4};let d=&g.units[di];let (dmg,ret)=g.damage(u,apos,d,target);let killed=dmg>=d.hp;let dv=g.value(d);let uv=g.value(u);let mut score=dv*(dmg/d.hp).min(1.)*(if killed{1.3}else{0.8})-uv*(ret/u.hp).min(1.)*1.05;if ret>=u.hp{score-=uv*0.4}if u.k==10{score=dv*2.}
let mut after=u.clone();after.hp-=ret;let end=if killed&&stats(u.k).r==1&&g.tiles[target].ter>2{target}else{apos};if after.hp>0.{let dn=danger(g,ctx,&after,end,if killed{di}else{usize::MAX});let before=danger(g,ctx,u,apos,usize::MAX);score-=0.35*(dn-before).max(0.);}
if killed{score+=2.;if u.k==4{let chain=g.adj[target].iter().filter(|&&p|g.at[p].map_or(false,|i|g.units[i].owner!=g.me&&g.damage(u,target,&g.units[i],p).0>=g.units[i].hp)).count();score+=chain as f64*5.;}if u.kills==2&&!u.vet{score+=4.}}
if g.tiles[target].st==1{if g.tiles[target].owner==g.me{score+=if killed{18.}else{7.};}else if killed{score+=10.}}
if d.k==8||d.k==41||d.k==10{score+=if killed{3.}else{1.}}
// Splash damage.
if u.k==41{for &p in &g.adj[target]{if let Some(i)=g.at[p]{if g.units[i].owner!=g.me{let dd=g.damage(u,apos,&g.units[i],p).0*0.5;score+=g.value(&g.units[i])*(dd/g.units[i].hp).min(1.)}}}}
score*4.+8.}
fn pop_score(g:&Game,ctx:&Context,t:usize,pop:f64,cost:f64)->f64{let ci=g.territory[t];if ci==usize::MAX{return -100.}let c=&g.cities[ci];let need=(c.level+1-c.progress).max(1)as f64;let mut reward=match c.level{1=>18.,2=>17.,3=>20.,_=>30.};if c.level>=4&&ctx.nearest[c.pos]>7{reward=24.}if c.level>=7{reward*=0.75;}
let amount=pop.min(need);let mut val=reward*amount/need.max(pop);if pop>=need{val+=8.;}else{val+=2.*pop;}
// Economical partial development is useful if the city can complete its level.
let scarcity=if g.stars<8.&&g.income<8.{2.1}else{1.65};val-=cost*scarcity;
if c.level==1{val+=6.}if c.level==2{val+=4.;}if g.stars-cost<2.&&g.own.len()<2&&g.turn>0{val-=12.;}
if g.at[c.pos].map_or(false,|i|g.units[i].owner!=g.me){if c.level>=4&&pop>=need{val+=30.}else{val-=18.}}
if ctx.nearest[c.pos]<=2&&g.own.len()<g.enemy.len()+2{val-=5.;}
val*2.}
fn research_score(g:&Game,ctx:&Context,p:i32)->f64{let nc=g.mycities.len()as f64;let tier=match p{1|6|10|15|20=>1.,2|4|7|8|13|16|18|21|23|44=>2.,_=>3.};let cost=(4.+nc*tier)*if g.tech[22]{0.667}else{1.};let remaining=g.stars-cost;let mut fruit=0.0f64;let mut fish=0.0f64;let mut forest=0.0f64;let mut metal=0.0f64;let mut farms=0.0f64;let mut huts=0.0f64;let mut coast=0.0f64;let mut mountains=0.0f64;let mut saw_potential=0.0f64;
for t in &g.tiles{if t.known&&t.ter==4{mountains+=1.;}if t.owner==g.me{if t.st==0{match t.res{6=>fruit+=1.,3=>fish+=1.,5=>metal+=1.,2=>farms+=1.,_=>{}}if t.ter==5{forest+=1.;}if t.ter==1{coast+=1.;}}if t.st==12{huts+=1.;}}}
for i in 0..g.n{if g.tiles[i].owner==g.me&&g.tiles[i].ter==3&&g.tiles[i].st==0{let adj=g.adj[i].iter().filter(|&&j|g.tiles[j].owner==g.me&&(g.tiles[j].st==12||g.tiles[j].ter==5)).count()as f64;saw_potential+=adj.max(0.);}}
let military=g.enemy.len()>0;let mut score=match p{
1=>if g.turn>=1&&g.own.len()>=2{24.}else{8.},
6=>if fruit>=2.{20.+fruit.min(5.)*3.}else if fruit>0.{10.}else{-20.},
10=>if fish>=2.{20.+fish.min(5.)*3.}else if ctx.port_need&&coast>0.{39.}else{-15.},
16=>if forest>=2.{25.+forest.min(10.)*1.6}else{-15.},
17=>if huts>=3.||saw_potential>=7.{29.+huts.min(8.)*2.}else if military&&(g.income>=10.||g.stars>=20.){25.}else{-20.},
18=>if military&&(g.income>=6.||g.stars>=10.){31.}else if g.income>=12.{18.}else{-15.},
20=>if mountains>=4.&&(g.turn>=4||g.own.len()>=3){25.+mountains.min(15.)*0.5}else{-10.},
23=>if metal>=1.{25.+metal.min(5.)*3.}else if military&&(g.income>=14.||g.stars>=22.){22.}else{-15.},
24=>if military&&(g.income>=10.||g.stars>=20.){42.}else if g.income>=20.{25.}else{-15.},
13=>if ctx.port_need&&(coast>0.||g.tiles.iter().any(|t|t.st==8&&t.owner==g.me)){37.}else{-20.},
44=>if g.own.iter().any(|&i|boat(g.units[i].k))||g.enemy.iter().any(|&i|boat(g.units[i].k)){35.}else{-25.},
14=>if g.own.iter().filter(|&&i|boat(g.units[i].k)).count()>=2&&g.income>=12.{30.}else{-20.},
7=>if military&&g.mycities.iter().any(|&i|ctx.nearest[g.cities[i].pos]<=3)&&g.stars>=10.{25.}else if ctx.port_need&&g.tech[44]&&g.income>=12.{23.}else{-20.},
4=>if g.income>=12.&&nc>=3.{25.}else{-25.},
5=>if g.tiles.iter().any(|t|t.owner==g.me&&t.st==13)&&g.income>=15.{27.}else{-30.},
8=>if farms>=2.{22.+farms.min(4.)*3.}else{-25.},
9=>if farms>=3.&&g.income>=20.{12.}else{-30.},
2=>if military&&g.income>=18.&&g.tech[17]{16.}else{-30.},
3=>if military&&g.income>=18.{30.}else{-30.},
12=>if g.own.iter().filter(|&&i|boat(g.units[i].k)).count()>=4{24.}else{-30.},
_=>-40.};
if remaining<3.{score-=12.;}if remaining<5.&&military{score-=8.;}score-=cost*0.65;
if g.own.len()<2&&g.turn<4&&g.enemy.is_empty(){score-=25.}
if g.turn==0{score-=60.}
if g.stars>2.*g.income+10.{score+=12.}if g.stars>40.{score+=10.}
score}
fn recruit_score(g:&Game,ctx:&Context,a:&Action)->f64{let k=a.p;let s=stats(k);let st=&g.tiles[a.s];let nd=ctx.nearest[a.s];let total=g.own.len()as f64;let nc=g.mycities.len()as f64;let nearby=g.own.iter().filter(|&&i|g.dist(g.units[i].pos,a.s)<=4).count()as f64;let enear=g.enemy.iter().filter(|&&i|g.dist(g.units[i].pos,a.s)<=5).count()as f64;let myranged=g.own.iter().filter(|&&i|matches!(g.units[i].k,8|9|41|42)&&g.dist(g.units[i].pos,a.s)<=5).count()as f64;let ranged=g.own.iter().filter(|&&i|matches!(g.units[i].k,8|9|41|42)).count()as f64;let melee=g.own.iter().filter(|&&i|matches!(g.units[i].k,2|3|4|11|12|44|45)).count()as f64;
if g.at[a.s].is_some(){ // Upgrade raft without exhausting it.
let ui=g.at[a.s].unwrap();let u=&g.units[ui];let mut score=if k==42{27.}else if k==44{32.}else if k==41{if g.income>=16.{38.+if g.own.iter().filter(|&&i|g.units[i].k==41).count()*3<g.own.iter().filter(|&&i|boat(g.units[i].k)).count(){7.}else{-12.}}else{12.}}else{-100.};if nd<=4{score+=14.;}if k==42&&g.turn<15{score+=7.}if u.hp<6.{score-=18.}return score;}
let mut score=25.+(nc*2.0-total)*3.0;if total<2.{score+=20.;}if nd<=4{score+=15.+enear*3.-nearby*2.5;}else{score-=nearby*3.;}if total>nc*3.+2.{score-=20.;}
if g.enemy.is_empty()&&total>nc*2.+1.{score-=15.;}let frontier=g.cities.iter().filter(|c|c.owner==0&&g.dist(c.pos,a.s)<6).count();if nd>5&&total>=nc*2.&&frontier==0{score-=20.;}
score+=match k{2=>if nd<=3{if g.tech[24]{-15.}else{2.}}else{0.},3=>if nd<=3{-4.}else{8.},4=>if nd<=6&&g.enemy.iter().any(|&i|matches!(g.units[i].k,8|9|3)){17.}else{-8.},5=>if nd<=2&&myranged>0.{9.}else if ctx.port_need&&g.tech[44]&&nd>4{7.}else{-20.},8=>if nd<=6&&nd>=2{20.-(myranged*3.-nearby).max(0.)+if g.enemy.iter().any(|&i|matches!(g.units[i].k,5|11|12|44|45)){8.}else{0.}}else if nd<=9{7.}else{-16.},9=>if nd<=4{9.-(myranged*2.5-nearby).max(0.)}else{-6.},11=>if nd<=6{13.}else{2.},10=>-30.,38=>-40.,_=>-50.};
if k==8&&ranged>melee+1.{score-=20.}if k==9&&ranged>melee+2.{score-=15.}
if k==3&&g.tech[24]&&nd<=5{score-=7.}
if ctx.port_need&&nd>5&&matches!(k,8|9){score-=14.}
score-=s.c*1.3;if g.stars-s.c<3.&&g.income<8.&&total>=2.{score-=7.;}
if g.turn==0{score-=40.}
// Do not recruit units into certain immediate death, except to block a city capture.
let fake=Unit{pos:a.s,owner:g.me,k,hp:s.h,mh:s.h,..Default::default()};let loss=danger(g,ctx,&fake,a.s,usize::MAX);if loss>g.value(&fake){score-=8.;}if nd==1{score+=10.;}
let _=st;score}
fn move_score(g:&Game,ctx:&Context,mem:&Memory,a:&Action)->f64{let ui=match g.at[a.s]{Some(i)=>i,None=>return -1e4};let u=&g.units[ui];let st=stats(u.k);let p=a.t;let t=&g.tiles[p];let source=&g.tiles[a.s];let before=danger(g,ctx,u,a.s,usize::MAX);let risk=danger(g,ctx,u,p,usize::MAX);let mut score=-3.;let goal=ctx.assign[ui];if goal<ctx.goals.len(){let (_,val,ty)=ctx.goals[goal];let cl=class(u.k);let oldidx=a.s+if boat(u.k){g.n}else{0};let newboat=if boat(u.k){t.ter<=2}else{t.st==8};let idx=p+if newboat{g.n}else{0};let old=ctx.paths[goal][cl][oldidx];let new=ctx.paths[goal][cl][idx];if old<1e5&&new<1e5{let progress=old-new;score+=progress*if ty==0{12.}else if ty==1||ty==2{8.}else{9.};score+=val/(new+2.)*0.25;}}
score+=ctx.explore[p]*if g.turn<12{1.5}else{0.7};
if t.st==1&&t.owner!=g.me{score+=if t.owner==0{46.}else{58.};}
if t.st==2{score+=30.;}
if source.st==1&&source.owner==g.me{score+=if g.stars>=2.{14.}else{5.};if ctx.nearest[a.s]<=2&&g.stars<2.{score-=8.;}if g.adj[a.s].iter().any(|&x|g.at[x].map_or(false,|i|g.units[i].owner!=g.me))&&g.mycities.len()==1{score-=15.;}}
let mut tactical=-1e4f64;let mut current_attack=-1e4f64;if !u.attacked{for &ei in &g.enemy{let e=&g.units[ei];if g.dist(a.s,e.pos)<=st.r{current_attack=current_attack.max(attack_score(g,ctx,ui,e.pos,a.s));}if st.dash&&g.dist(p,e.pos)<=st.r&&!(boat(u.k)&&t.ter>2)&&!(t.st==8&&!boat(u.k)){tactical=tactical.max(attack_score(g,ctx,ui,e.pos,p));}}}
if tactical>0.{score+=tactical*0.8+5.;}if current_attack>0.&&tactical<current_attack{score-=current_attack*0.65;}
let mut riskweight=2.6;if u.k==12||u.k==45{riskweight=1.8;}if t.st==1&&t.owner!=g.me{riskweight=1.8;}
if tactical>20.{riskweight*=0.5;}
score-=risk*riskweight;score+=before*if u.attacked||u.hp<u.mh*0.6{2.5}else{0.7};
if !u.attacked&&u.hp<u.mh-2.{let heal=(u.mh-u.hp).min(if source.owner==g.me{4.}else{2.});score-=heal*g.value(u)/u.mh*3.;}
if st.r>1&&u.k!=43{let nd=ctx.nearest[p];if nd<=1{score-=18.}else if nd<=st.r{score+=if st.dash{6.}else{10.};}if !st.dash&&current_attack>0.{score-=25.;}if boat(u.k)&&t.ter>2{score-=22.;}}
if u.k==43{if t.owner!=g.me&&ctx.nearest[p]<5{score-=15.}if source.owner==g.me&&!u.moved&&g.stars>=5.&&(g.tech[13]||g.tech[44]){score-=25.;}}
if t.st==8&&!boat(u.k){if ctx.port_need{score+=8.}else{score-=15.}}
if u.k==45{for &ei in &g.enemy{let e=&g.units[ei];if g.dist(p,e.pos)==1{score+=g.value(e)*(g.damage(u,p,e,e.pos).0*0.5/e.hp).min(1.)*4.}}}
if u.attacked&&u.k==3{score+=before*2.;if source.st==1&&source.owner!=g.me{score-=25.;}}
if mem.prev.len()==g.n&&mem.prev[a.s]==p&&mem.last_turn[a.s]>=g.turn-1{score-=9.;}
if mem.visits.len()==g.n{score-=mem.visits[p].min(6)as f64*0.7;}
score-=g.own.iter().filter(|&&i|i!=ui&&g.units[i].pos==p).count()as f64*100.;
score}
fn build_score(g:&Game,ctx:&Context,a:&Action)->f64{let t=a.t;let tile=&g.tiles[t];match a.p{
12=>{if tile.res==1{return -40.}let mut pop=1.;for &j in &g.adj[t]{if g.tiles[j].st==13&&g.tiles[j].owner==g.me{pop+=1.;}}pop_score(g,ctx,t,pop,3.)},
5|21=>pop_score(g,ctx,t,2.,5.),
6|13|22=>{let kind=if a.p==6{5}else if a.p==13{12}else{21};let ct=g.adj[t].iter().filter(|&&j|g.tiles[j].owner==g.me&&g.tiles[j].st==kind).count()as f64;if ct<2.&&g.stars<25.{return -15.}pop_score(g,ctx,t,ct*if a.p==22{2.}else{1.},5.)+ct*2.},
23..=29=>{pop_score(g,ctx,t,3.,0.)+18.-if tile.res>0{10.}else{0.}},
8=>{let near_port=(0..g.n).any(|p|g.tiles[p].st==8&&g.tiles[p].owner==g.me&&g.dist(t,p)<=3);if near_port{return -30.}let nearby=g.own.iter().filter(|&&i|!boat(g.units[i].k)&&g.dist(t,g.units[i].pos)<=3).count();let open=g.adj[t].iter().filter(|&&p|g.tiles[p].ter<=2).count();let mut score=if ctx.port_need{28.}else{-20.};score+=nearby.min(3)as f64*3.+open as f64;
let objective=g.cities.iter().filter(|c|c.owner!=g.me).map(|c|g.dist(t,c.pos)).min().unwrap_or(8);score+=(10-objective).max(0)as f64*1.5;
if g.adj[t].iter().any(|&p|g.tiles[p].ter==2){score+=2.;}let ports=g.tiles.iter().filter(|x|x.st==8&&x.owner==g.me).count();score-=ports as f64*9.;if g.tech[13]{score+=6.;}if tile.res==3{score-=12.;}if g.stars<10.&&ports>0{score-=15.;}score},
50=>{let mut sum=0;for &j in &g.adj[t]{if g.tiles[j].owner==g.me&&matches!(g.tiles[j].st,6|13|22){sum+=g.tiles[j].sl;}}if sum>=2{20.+sum as f64*5.}else{-15.}},
3=>{let mut score=ctx.roads[t];if g.stars<6.{score-=12.;}score},
48=>{let land=g.adj[t].iter().filter(|&&j|g.tiles[j].known&&g.tiles[j].ter>2).count();if land>=3&&g.own.iter().any(|&i|g.dist(g.units[i].pos,t)==1)&&g.stars>15.{10.}else{-30.}},_=>-100.}}

// Search coordinated volleys. All candidates come from the current legal catalog;
// only the first action of the chosen plan is executed before replanning.
#[derive(Clone)]struct Shot{ui:usize,pos:usize,action:usize,movefirst:bool,cost:f64}
#[derive(Clone)]struct Volley{mask:u32,hp:f64,loss:f64,first:usize,positions:Vec<usize>,steps:usize}
fn volley(g:&Game,ctx:&Context)->Option<(usize,f64)>{
let mut best:Option<(usize,f64)>=None;
for &ei in &g.enemy{
 let def=&g.units[ei];if def.hp<7.{continue}
 let mut shots=Vec::new();
 for &ui in &g.own{
  let u=&g.units[ui];let st=stats(u.k);if u.attacked||st.a<=0.||u.hp<=1.{continue}
  let mut chosen:Option<Shot>=None;
  for (ai,a) in g.actions.iter().enumerate(){
   if a.s!=u.pos{continue}
   let (pos,mv)=if a.k==2&&a.t==def.pos{(u.pos,false)}else if a.k==1&&st.dash&&g.dist(a.t,def.pos)<=st.r&&!(boat(u.k)&&g.tiles[a.t].ter>2)&&!(!boat(u.k)&&g.tiles[a.t].st==8){(a.t,true)}else{continue};
   let risk=danger(g,ctx,u,pos,ei);let (dmg,ret)=g.damage(u,pos,def,def.pos);if dmg<1.{continue}
   let cost=if mv{1.0+0.35*risk}else{0.15*risk}+if ret>=u.hp{1.}else{0.};
   if chosen.as_ref().map_or(true,|x|cost<x.cost){chosen=Some(Shot{ui,pos,action:ai,movefirst:mv,cost});}
  }
  if let Some(x)=chosen{shots.push(x)}
 }
 if shots.len()<2{continue}
 shots.sort_by(|a,b|{let ua=&g.units[a.ui];let ub=&g.units[b.ui];let (da,ra)=g.damage(ua,a.pos,def,def.pos);let (db,rb)=g.damage(ub,b.pos,def,def.pos);(db-rb*0.65-b.cost).partial_cmp(&(da-ra*0.65-a.cost)).unwrap()});shots.truncate(9);
 let mut beam=vec![Volley{mask:0,hp:def.hp,loss:0.,first:0,positions:vec![],steps:0}];
 let targetval=g.value(def);let citybonus=if g.tiles[def.pos].st==1{if g.tiles[def.pos].owner==g.me{14.}else{8.}}else{0.};
 for depth in 0..shots.len().min(6){
  let mut next=Vec::new();
  for node in &beam{for (j,sh)in shots.iter().enumerate(){if node.mask&(1<<j)!=0||node.positions.contains(&sh.pos){continue}
    let u=&g.units[sh.ui];let mut du=def.clone();du.hp=node.hp;let (dmg,ret)=g.damage(u,sh.pos,&du,def.pos);let loss=g.value(u)*(ret/u.hp).min(1.)*(if ret>=u.hp{1.4}else{0.85})+sh.cost+0.8;
    let hp=node.hp-dmg;let first=if depth==0{sh.action}else{node.first};let mut pos=node.positions.clone();pos.push(sh.pos);
    if hp<=0.{let net=targetval*1.35+citybonus-node.loss-loss;
      // A plan's value is spread over its actions. Avoid expensive sacrifices
      // when an already available single shot or capture is better.
      let val=12.+net*4./(1.+depth as f64*0.18);
      if net>2.&&best.map_or(true,|(_,v)|val>v){best=Some((first,val));}
    }else{next.push(Volley{mask:node.mask|(1<<j),hp,loss:node.loss+loss,first,positions:pos,steps:depth+1});}
  }}
  next.sort_by(|a,b|{let va=targetval*(1.-a.hp/def.hp)-a.loss;let vb=targetval*(1.-b.hp/def.hp)-b.loss;vb.partial_cmp(&va).unwrap()});next.truncate(24);beam=next;if beam.is_empty(){break}
 }
}
best}
fn choose(g:&Game,mem:&Memory)->usize{let ctx=make_context(g);let plan=volley(g,&ctx);let mut best=0;let mut bs=-1e20;for (i,a)in g.actions.iter().enumerate(){let mut score=match a.k{
10=>0.,11=>-1e6,
9=>match a.p{3=>1000.,4=>900.,7=>1000.,1=>900.,6=>1100.,2=>800.,5=>{let mut resources=0.;for t in 0..g.n{if g.dist(t,a.t)==2&&g.tiles[t].owner==0{resources+=if g.tiles[t].res==1||g.tiles[t].res==6||g.tiles[t].res==3{1.}else{0.};if g.tiles[t].ter==5{resources+=0.7;}if g.tiles[t].ter==1&&ctx.port_need&&!g.tiles.iter().any(|x|x.st==8&&x.owner==g.me){resources+=2.;}}}if resources>=3.5{1030.}else{980.}},8=>1000.,_=>500.},
8=>if g.tiles[a.s].st==1{600.}else{400.},
2=>g.at[a.s].map_or(-1e4,|u|attack_score(g,&ctx,u,a.t,a.s)),
1=>move_score(g,&ctx,mem,a),
4=>recruit_score(g,&ctx,a),
5=>pop_score(g,&ctx,a.t,1.,2.),
6=>build_score(g,&ctx,a),
7=>research_score(g,&ctx,a.p),
3=>match a.p{16=>200.,7=>{let u=&g.units[g.at[a.s].unwrap()];let heal=(u.mh-u.hp).min(if g.tiles[a.s].owner==g.me{4.}else{2.});let risk=danger(g,&ctx,u,a.s,usize::MAX);heal*g.value(u)/u.mh*3.+if risk<2.{1.}else{0.}},9=>{g.adj[a.s].iter().filter_map(|&p|g.at[p]).filter(|&i|g.units[i].owner==g.me).map(|i|{let u=&g.units[i];(u.mh-u.hp).min(4.)*g.value(u)/u.mh*4.}).sum::<f64>()},19=>if a.t==0{2.}else{-30.},18=>1.,2=>-20.,_=>-1000.},_=>-1e5};
if let Some((ai,v))=plan{if ai==i{score=score.max(v);}}
// Stable small tie break.
score+=((a.s*13+a.t*7+a.p.max(0)as usize*3)%29)as f64*0.0001;
if score>bs{bs=score;best=i;}}
best}
fn main(){let stdin=io::stdin();let mut out=io::BufWriter::new(io::stdout());let mut mem=Memory::default();for line in stdin.lock().lines(){let line=match line{Ok(l)=>l,Err(_)=>break};let o:Value=match serde_json::from_str(&line){Ok(v)=>v,Err(_)=>continue};let g=Game::read(&o);if mem.visits.len()!=g.n{mem.visits=vec![0;g.n];mem.prev=vec![usize::MAX;g.n];mem.last_turn=vec![-99;g.n];}if mem.turn!=g.turn{for x in &mut mem.visits{*x=(*x-1).max(0)}mem.turn=g.turn;}let idx=choose(&g,&mem);let a=&g.actions[idx];if a.k==1{mem.visits[a.t]+=2;mem.prev[a.t]=a.s;mem.last_turn[a.t]=g.turn;}writeln!(out,"{}",json!({"action_id":a.id})).unwrap();out.flush().unwrap();}}
