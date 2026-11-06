"""Export a Studio replay as the compact per-action JSON the blog's replay viewer plays.

Usage: python scripts/export_blog_replay.py REPLAY.json.gz OUT.json [--values OLD_EXPORT.json]

The viewer animates one action at a time, so the export keeps the god-view state before the first action and,
for every action, what it changed: units (by tile), tiles and cities. --values carries over the orange player's
per-half-turn win estimate from an older half-turn export.
"""

import argparse
import gzip
import json


def god_state(frame):
    vs = frame["views"]["all"]["visible_state"]
    tiles = [
        [t["terrain"], t["owner"], t["structure"], t["resource"], t["road"], t["capital"]]
        for t in sorted(vs["tiles"], key=lambda t: t["tile"])
    ]
    units = {}
    for u in vs["units"]:
        assert u["tile"] not in units, f"two units on tile {u['tile']}"
        units[u["tile"]] = [u["owner"], u["kind"], round(u["health"], 1), int(u["veteran"])]
    cities = {c["tile"]: [c["owner"], c["level"]] for c in vs["cities"]}
    score = [frame["views"][p]["hud"]["score"] for p in ("1", "2")]
    return tiles, units, cities, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("out")
    ap.add_argument("--values", help="older half-turn export with snaps[].value")
    args = ap.parse_args()

    replay = json.load(gzip.open(args.replay))
    frames = replay["frames"]
    values = {}
    names = {"1": "Blue", "2": "Orange"}
    unit_names = {}
    if args.values:
        old = json.load(open(args.values))
        names, unit_names = old["names"], old["units"]
        values = {(s["turn"], s["player"]): s["value"][1] for s in old["snaps"]}

    tiles, units, cities, score = god_state(frames[0])
    size = frames[0]["views"]["all"]["visible_state"]["board_size"]
    labels, label_index = [], {}
    actions = []
    value = None
    for frame in frames[1:]:
        event = frame["event"]
        command, label = event["action"]["command"], event["action"]["label"]
        if label not in label_index:
            label_index[label] = len(labels)
            labels.append(label)
        if values.get((event["turn"], event["player"])) is not None:
            value = values[(event["turn"], event["player"])]
        t2, u2, c2, s2 = god_state(frame)
        unit_diff = [[tile] for tile in units if tile not in u2]
        unit_diff += [[tile, *u] for tile, u in u2.items() if units.get(tile) != u]
        tile_diff = [[k, *t] for k, (a, t) in enumerate(zip(tiles, t2)) if a != t]
        city_diff = [[tile, *c] for tile, c in c2.items() if cities.get(tile) != c]
        actions.append([
            event["player"], event["turn"], command["kind"], command["source"], command["target"],
            label_index[label], unit_diff, tile_diff, city_diff, s2, None if value is None else round(value, 3),
        ])
        tiles, units, cities, score = t2, u2, c2, s2

    first_tiles, first_units, first_cities, _ = god_state(frames[0])
    out = {
        "title": replay.get("title", ""),
        "size": size,
        "names": names,
        "units": unit_names,
        "labels": labels,
        "init": {
            "tiles": first_tiles,
            "units": [[tile, *u] for tile, u in first_units.items()],
            "cities": [[tile, *c] for tile, c in first_cities.items()],
        },
        # [player, turn, command kind, source, target, label, unit diff, tile diff, city diff, [score1, score2], value]
        "actions": actions,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(actions)} actions -> {args.out}")


if __name__ == "__main__":
    main()
