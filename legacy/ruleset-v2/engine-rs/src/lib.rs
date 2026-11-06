//! Engine modules; the PyO3 bindings (feature "python", on by default) live in `python`.

pub mod capture;
pub mod core;
pub mod cov;
pub mod diplomacy;
pub mod discovery;
pub mod economy;
pub mod geom;
pub mod hash;
pub mod infiltration;
pub mod kernels;
pub mod mapgen;
pub mod masks;
pub mod military;
pub mod pathfinding;
pub mod prof;
pub mod rules;
pub mod state;
pub mod structures;

#[cfg(feature = "python")]
mod python;
