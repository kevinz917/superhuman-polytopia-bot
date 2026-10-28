The JAX rule ports in `src/polytopia_bot/gpu/` are derived from Polyfish's Rust
engine at commit `a9f90ab54607f5633556d02422d7457f548cc58f`:
https://github.com/HenBOMB/Polyfish/tree/a9f90ab54607f5633556d02422d7457f548cc58f/polyfish-rs

Polyfish's Rust engine is licensed under Apache-2.0. Its license is retained in
`licenses/Polyfish-Apache-2.0.txt`. No UI code is imported. The validation oracle
links the pinned Rust engine; it is not a runtime dependency of the GPU kernels.
