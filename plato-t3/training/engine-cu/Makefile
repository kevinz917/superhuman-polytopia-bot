# engine-cu: one source compiled twice. `make cpu` builds the CPU oracle driver (libpc_cpu.so), `make cuda` the CUDA
# driver (libpc_cuda.so; nvcc needs the toolkit, not a GPU). Numeric discipline: no FMA contraction, no fast math,
# wrapping int32 (docs/gpu-engine-design-2026-11-06.md).
#
#   make cpu
#   make cuda SM=sm_89                                   # one architecture
#   make cuda ARCHFLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89"   # fatbin
#
# Variants (2026-11-06), same source, same results (the gate runs each):
#   SIZE=14    -DPC_MAXSIZE=14: every per-game array sized for 196 tiles (boards 11 and 14 only) -> lib*_14.so
#   SIZE=30    -DPC_MAXSIZE=30: 900 tiles, boards 20 and 30                                    -> lib*_30.so
#   PROFILE=1  -DPC_PROFILE: per-phase clocks in the tick (pc_profile_read)                    -> lib*_prof.so
#   make cpu SIZE=14 PROFILE=1   -> build/libpc_cpu_14_prof.so
#   make -j4 variants            -> all four of each driver (cpu-variants / cuda-variants: one driver)
# rl/gpu_engine.py picks lib*_14.so for boards <= 14 when it exists (POLYTOPIA_PC_VARIANT=full keeps the full build)
# and lib*_prof.so with POLYTOPIA_PC_PROFILE=1.
CC ?= cc
NVCC ?= nvcc
SM ?= sm_89
ARCHFLAGS ?= -arch=$(SM)
NVEXTRA ?=
SIZE ?=
PROFILE ?=
BUILD = build
HEADERS = $(wildcard src/*.h)

SUFFIX = $(if $(SIZE),_$(SIZE),)$(if $(filter 1,$(PROFILE)),_prof,)
DEFS = $(if $(SIZE),-DPC_MAXSIZE=$(SIZE),) $(if $(filter 1,$(PROFILE)),-DPC_PROFILE,)

.PHONY: cpu cuda clean variants cpu-variants cuda-variants cpu-v-full cpu-v-14 cpu-v-prof cpu-v-14-prof \
        cuda-v-full cuda-v-14 cuda-v-prof cuda-v-14-prof

cpu: $(BUILD)/libpc_cpu$(SUFFIX).so
cuda: $(BUILD)/libpc_cuda$(SUFFIX).so

$(BUILD)/libpc_cpu$(SUFFIX).so: src/pc_cpu.c $(HEADERS) | $(BUILD)
	$(CC) -O2 -std=c11 -ffp-contract=off -fwrapv -fPIC -shared $(DEFS) -o $@ src/pc_cpu.c -lm

$(BUILD)/libpc_cuda$(SUFFIX).so: src/pc_cuda.cu $(HEADERS) | $(BUILD)
	$(NVCC) -O2 --fmad=false -Xcompiler -fPIC -shared $(ARCHFLAGS) $(NVEXTRA) $(DEFS) -o $@ src/pc_cuda.cu

# one sub-make per variant, so `make -j4 cuda-variants` builds the four in parallel
cpu-variants: cpu-v-full cpu-v-14 cpu-v-prof cpu-v-14-prof cpu-v-30
cuda-variants: cuda-v-full cuda-v-14 cuda-v-prof cuda-v-14-prof cuda-v-30
cpu-v-full: ; $(MAKE) cpu SIZE= PROFILE=
cpu-v-14: ; $(MAKE) cpu SIZE=14 PROFILE=
cpu-v-prof: ; $(MAKE) cpu SIZE= PROFILE=1
cpu-v-14-prof: ; $(MAKE) cpu SIZE=14 PROFILE=1
cpu-v-30: ; $(MAKE) cpu SIZE=30 PROFILE=
cuda-v-full: ; $(MAKE) cuda SIZE= PROFILE=
cuda-v-14: ; $(MAKE) cuda SIZE=14 PROFILE=
cuda-v-prof: ; $(MAKE) cuda SIZE= PROFILE=1
cuda-v-14-prof: ; $(MAKE) cuda SIZE=14 PROFILE=1
cuda-v-30: ; $(MAKE) cuda SIZE=30 PROFILE=

variants: cpu-variants cuda-variants

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
