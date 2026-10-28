use std::{env,fs,path::PathBuf};
fn main(){
    let path="../../vendor/Polyfish/polyfish-rs/src/mapgen.rs";
    println!("cargo:rerun-if-changed={path}");
    let source=fs::read_to_string(path).unwrap();
    let old="use crate::dotnet_rng::DotNetRandom as StdRng;";
    assert_eq!(source.matches(old).count(),1,"pinned generator import changed");
    let source=source.replace("//!","//")
        .replace(old,"use crate::NativeRandom as StdRng;")
        .replace("use rand::{Rng, SeedableRng};","");
    let mut source=source;
    for (marker,label) in [("    // Biomes","land"),("    for i in 0..tile_count {\n        if !is_land","biomes"),("    // Post-terrain Villages","terrain")] {
        source=source.replace(marker,&format!("    if std::env::var_os(\"MAPGEN_DEBUG\").is_some() {{ eprintln!(\"{label}: {{}}\",rng.0.clone().internal_sample()); }}\n{marker}"));
    }
    source=source.replace("            let cell = active[i][idx];", "            let cell = active[i][idx];\n            if std::env::var_os(\"MAPGEN_DEBUG\").is_some() { eprintln!(\"choice {} {} {} {}\",i,idx,cell,active[i].len()); }");
    fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("mapgen.rs"),source).unwrap();
}
