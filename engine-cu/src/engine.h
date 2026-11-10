/* engine.h - the one translation unit of the C/CUDA Polytopia engine: every module header in dependency order.
 * Every function is PC_FN (static inline), so a module may call a later one through the forward prototypes at the
 * top of its header; structs must come before their first use, which fixes this order:
 *   kernels.h    PcMoveScratch        (used by masks.h, core.h)
 *   pathfinding.h PcPathScratch       (used by core.h)
 *   structures.h PcBuildCtx, PcStructScratch  (used by diplomacy.h's and economy.h's forward prototypes of
 *                update_routes / reward, by capture.h, masks.h, core.h; structures.h itself needs only pc.h, rules.h)
 *   military.h   PcSummonCtx          (used by masks.h, core.h)
 * Include this file (and nothing else) from pc_cpu.c / pc_cuda.cu. */
#ifndef PC_ENGINE_H
#define PC_ENGINE_H

#include "pc.h"
#include "hash.h"
#include "rules.h"
#include "kernels.h"
#include "history.h"
#include "pathfinding.h"
#include "discovery.h"
#include "structures.h"
#include "diplomacy.h"
#include "economy.h"
#include "capture.h"
#include "infiltration.h"
#include "military.h"
#include "masks.h"
#include "core.h"

#endif /* PC_ENGINE_H */
