/* pc_state_io.h - host-side export / import of one PcState in PC_EXPORT_FIELDS order (pc_api.h), shared by the CPU
 * and CUDA drivers (the CUDA driver copies the game's PcState to / from the device around it). Host code only. */
#ifndef PC_STATE_IO_H
#define PC_STATE_IO_H
#include <string.h>
#include "engine.h"

/* ---- export / import: one routine walks the fields in PC_EXPORT_FIELDS order in either direction, so the two
 * cannot drift apart. `p` is the caller's buffer of the current field. ---- */
typedef struct { void **ptrs; int k; int import; } PcIo;

static void *next_buf(PcIo *io) { return io->ptrs[io->k++]; }

static void io_i32(PcIo *io, int32_t *field, int count) {
    int32_t *buf = (int32_t *)next_buf(io);
    if (io->import) memcpy(field, buf, (size_t)count * sizeof(int32_t)); else memcpy(buf, field, (size_t)count * sizeof(int32_t));
}
static void io_u8(PcIo *io, uint8_t *field, int count) {
    uint8_t *buf = (uint8_t *)next_buf(io);
    if (io->import) memcpy(field, buf, (size_t)count); else memcpy(buf, field, (size_t)count);
}
static void io_f32(PcIo *io, float *field, int count) {
    float *buf = (float *)next_buf(io);
    if (io->import) memcpy(field, buf, (size_t)count * sizeof(float)); else memcpy(buf, field, (size_t)count * sizeof(float));
}
static void io_u32(PcIo *io, uint32_t *field) {
    uint32_t *buf = (uint32_t *)next_buf(io);
    if (io->import) *field = *buf; else *buf = *field;
}
/* [np, n] from rows of PC_MAXN (tiles.explored, history.*) */
static void io_seat_tiles_u8(PcIo *io, uint8_t *rows, int row_stride, int np, int n) {
    uint8_t *buf = (uint8_t *)next_buf(io);
    for (int k = 0; k < np; k++) {
        if (io->import) memcpy(rows + (size_t)k * row_stride, buf + (size_t)k * n, (size_t)n);
        else memcpy(buf + (size_t)k * n, rows + (size_t)k * row_stride, (size_t)n);
    }
}
/* [np, np] from rows of PC_MAXP (players.known, peace, ...) */
static void io_pair_u8(PcIo *io, uint8_t (*rows)[PC_MAXP], int np) {
    uint8_t *buf = (uint8_t *)next_buf(io);
    for (int k = 0; k < np; k++)
        for (int j = 0; j < np; j++) {
            if (io->import) rows[k][j] = buf[k * np + j]; else buf[k * np + j] = rows[k][j];
        }
}
static void io_pair_i32(PcIo *io, int32_t (*rows)[PC_MAXP], int np) {
    int32_t *buf = (int32_t *)next_buf(io);
    for (int k = 0; k < np; k++)
        for (int j = 0; j < np; j++) {
            if (io->import) rows[k][j] = buf[k * np + j]; else buf[k * np + j] = rows[k][j];
        }
}
/* cities.territory dense [n, n]: territory[city].bit(tile) == Rust territory[city * n + tile] */
static void io_territory(PcIo *io, PcState *s, int n) {
    uint8_t *buf = (uint8_t *)next_buf(io);
    for (int c = 0; c < n; c++)
        for (int t = 0; t < n; t++) {
            if (io->import) pc_set_territory(s, c, t, buf[c * n + t] != 0);
            else buf[c * n + t] = (uint8_t)pc_territory(s, c, t);
        }
}
/* history.enemy_tech [np, 128]: bit i of the u128 */
static void io_enemy_tech(PcIo *io, PcState *s, int np) {
    uint8_t *buf = (uint8_t *)next_buf(io);
    for (int k = 0; k < np; k++) {
        if (io->import) {
            pc_u128 b = pc_u128_zero();
            for (int i = 0; i < 128; i++) if (buf[k * 128 + i]) pc_u128_set(&b, i);
            s->history[k].enemy_tech = b;
        } else {
            for (int i = 0; i < 128; i++) buf[k * 128 + i] = (uint8_t)pc_u128_test(s->history[k].enemy_tech, i);
        }
    }
}

static void state_io(PcIo *io, PcState *s, int n, int np) {
    PcTiles *t = &s->tiles; PcUnits *u = &s->units; PcCities *c = &s->cities; PcPlayers *p = &s->players;
    io_i32(io, t->terrain, n); io_i32(io, t->owner, n); io_i32(io, t->capital, n); io_i32(io, t->city, n);
    io_u8(io, t->road, n); io_i32(io, t->structure, n); io_i32(io, t->resource, n);
    io_seat_tiles_u8(io, &t->explored[0][0], PC_MAXN, np, n);
    io_i32(io, t->structure_level, n); io_i32(io, t->structure_turn, n); io_u8(io, t->route, n);
    io_i32(io, u->owner, n); io_i32(io, u->kind, n); io_f32(io, u->health, n); io_u8(io, u->veteran, n);
    io_u8(io, u->moved, n); io_u8(io, u->attacked, n); io_i32(io, u->kills, n); io_u8(io, u->poison, n);
    io_u8(io, u->frozen, n); io_u8(io, u->boosted, n); io_i32(io, u->home, n); io_i32(io, u->created, n);
    io_u8(io, u->converted, n); io_i32(io, u->attacks_performed, n); io_i32(io, u->prev_x, n); io_i32(io, u->prev_y, n);
    io_i32(io, u->last_attack, n); io_i32(io, u->passenger, n); io_u8(io, u->invisible, n); io_i32(io, u->direction, n);
    io_i32(io, c->owner, n); io_i32(io, c->level, n); io_i32(io, c->population, n); io_i32(io, c->progress, n);
    io_i32(io, c->production, n); io_i32(io, c->border, n); io_i32(io, c->rewards, n * 9);
    io_territory(io, s, n);
    io_u8(io, c->connected, n); io_i32(io, c->order, n); io_i32(io, c->infiltrated, n); io_u8(io, c->riot, n);
    io_i32(io, p->stars, np); io_i32(io, p->score, np);
    io_u8(io, &p->tech[0][0], np * 128); io_i32(io, &p->tech_turn[0][0], np * 128);
    io_pair_u8(io, p->known, np);
    io_i32(io, p->kills, np); io_i32(io, p->casualties, np); io_i32(io, p->pacifist, np); io_u8(io, p->attacked, np);
    io_u8(io, &p->built[0][0], np * 128);
    io_i32(io, p->killed_turn, np); io_i32(io, p->resigned_turn, np); io_i32(io, p->killer, np);
    io_pair_u8(io, p->peace, np); io_pair_u8(io, p->embassy, np); io_pair_u8(io, p->peace_offer, np);
    io_pair_i32(io, p->truce_breaker, np); io_pair_i32(io, p->hostile_until, np); io_pair_i32(io, p->peace_broken, np);
    io_u8(io, &p->monuments[0][0], np * 128);
    if (np > 2) io_i32(io, p->out_order, np);
    /* history: [np, HIST_TURNS, n] from history[k].seen_unit[t][PC_MAXN] */
    {
        uint8_t *buf = (uint8_t *)next_buf(io);
        for (int k = 0; k < np; k++)
            for (int j = 0; j < PC_HIST_TURNS; j++) {
                uint8_t *row = s->history[k].seen_unit[j];
                uint8_t *dst = buf + ((size_t)k * PC_HIST_TURNS + j) * n;
                if (io->import) memcpy(row, dst, (size_t)n); else memcpy(dst, row, (size_t)n);
            }
    }
    {
        int32_t *buf = (int32_t *)next_buf(io);
        for (int k = 0; k < np; k++) {
            if (io->import) memcpy(s->history[k].prev_code, buf + (size_t)k * n, (size_t)n * sizeof(int32_t));
            else memcpy(buf + (size_t)k * n, s->history[k].prev_code, (size_t)n * sizeof(int32_t));
        }
    }
    {
        uint8_t *buf = (uint8_t *)next_buf(io);
        for (int k = 0; k < np; k++) {
            if (io->import) memcpy(s->history[k].built, buf + (size_t)k * n, (size_t)n); else memcpy(buf + (size_t)k * n, s->history[k].built, (size_t)n);
        }
    }
    {
        uint8_t *buf = (uint8_t *)next_buf(io);
        for (int k = 0; k < np; k++) {
            if (io->import) memcpy(s->history[k].trained, buf + (size_t)k * n, (size_t)n); else memcpy(buf + (size_t)k * n, s->history[k].trained, (size_t)n);
        }
    }
    io_enemy_tech(io, s, np);
    io_i32(io, &s->current_player, 1); io_i32(io, &s->turn, 1); io_i32(io, &s->version, 1); io_i32(io, &s->state_version, 1);
    io_i32(io, &s->actions, 1); io_u8(io, &s->terminated, 1); io_u8(io, &s->truncated, 1); io_u8(io, &s->fog, 1);
    io_i32(io, &s->reference_max_turns, 1); io_u8(io, &s->core_supported, 1); io_u32(io, &s->seed); io_u32(io, &s->initial_seed);
}

#endif /* PC_STATE_IO_H */
