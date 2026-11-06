// Games and replays of the in-browser backend, kept in IndexedDB (workers have no localStorage). A record is the
// seed plus the actions taken: the engine is deterministic, so every frame can be rebuilt from it.
import type { GameEvent } from "../api/types.ts";

export interface FrameInfo {
  current_player: number;
  turn: number;
  terminated: boolean;
  truncated: boolean;
}

export interface GameRecord {
  id: string;
  kind: "game" | "replay";
  title: string;
  mode: "play" | "watch";
  seat: number;
  seed: number;
  search: boolean;
  created: number;
  actions: string[]; // catalog ids, one per frame after the first
  events: (GameEvent | null)[]; // per frame; frame 0 has none
  frames: FrameInfo[];
  // Imported Studio replays (bardur-replay-v1) keep their frames as recorded.
  imported?: Record<string, unknown>;
}

const DB = "bardur-local";
// Records are keyed "<kind>:<id>": a game and the replay saved from it share their id, as in the Studio.
const STORE = "records";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "key" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

let db: Promise<IDBDatabase> | null = null;
function database() {
  db ||= open();
  return db;
}

function run<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return database().then(
    (d) =>
      new Promise<T>((resolve, reject) => {
        const request = fn(d.transaction(STORE, mode).objectStore(STORE));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      }),
  );
}

// Storage is a convenience: a private window or blocked site data leaves games in memory only.
export const store = {
  async all(): Promise<GameRecord[]> {
    try {
      return await run("readonly", (s) => s.getAll() as IDBRequest<GameRecord[]>);
    } catch {
      return [];
    }
  },
  async put(record: GameRecord) {
    try {
      await run("readwrite", (s) => s.put({ ...record, key: `${record.kind}:${record.id}` }));
    } catch {
      // In memory only.
    }
  },
  async remove(kind: GameRecord["kind"], id: string) {
    try {
      await run("readwrite", (s) => s.delete(`${kind}:${id}`));
    } catch {
      // In memory only.
    }
  },
};
