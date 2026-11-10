Claude session A (multinode2): hero-1v1-classic-nopeace-warm150 with 2x batch (524,288) and 2x minibatch (32,768),
so a 16-GPU run gives each device the rows an 8-GPU run gives it (2,048 per step) and pays the per-update fixed costs
half as often per sample. Adam LR scaled by sqrt(2) (3e-4 -> 4.243e-4; end 1e-4 -> 1.414e-4); warmup, historical-pool
and snapshot cadences halved in updates so they are unchanged in samples. Needs its own learning check.
