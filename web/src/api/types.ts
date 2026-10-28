// Wire types for the studio HTTP API (see src/polytopia_bot/studio/service.py).

export interface Command {
  kind: number;
  source: number;
  target: number;
  parameter: number;
}

export interface LegalAction {
  action_id: string;
  command: Command;
}

export interface Tile {
  tile: number;
  known: boolean;
  terrain?: number;
  owner?: number;
  structure?: number;
  resource?: number;
  road?: number;
  capital?: number;
  structure_level?: number;
}

export interface Unit {
  tile: number;
  owner: number;
  kind: number;
  health: number;
  veteran: boolean;
  moved?: boolean;
  attacked?: boolean;
  kills?: number;
  home?: number;
}

export interface City {
  tile: number;
  owner: number;
  level?: number;
  progress?: number;
  population?: number;
  production?: number;
  connected?: boolean;
  riot?: boolean;
  rewards?: number[];
}

export interface VisibleState {
  board_size: number;
  turn: number;
  stars: number;
  technology: number[];
  tiles: Tile[];
  units: Unit[];
  cities: City[];
  relations?: Record<string, unknown>;
}

export interface Observation {
  player: number;
  state_version: number;
  visible_state: VisibleState;
  hud?: {
    score: number;
    income?: number;
    territory?: Record<string, number[]>;
    players?: { player: number; score: number; cities: number }[];
  };
}

export interface GameEvent {
  player: number;
  turn?: number;
  source?: string;
  action: { label: string; command: Partial<Command> };
  animation?: { kind: number; source: number; target: number };
  rewards?: Record<string, number>;
  reason?: string;
}

export interface FrameData {
  id: string;
  title: string;
  mode: "play" | "watch" | "agent" | "replay";
  seat?: number;
  frames: number;
  frame_index: number;
  current_player: number;
  terminated: boolean;
  truncated: boolean;
  observation: Observation;
  perspective: string;
  available_views: string[];
  event: GameEvent | null;
  bot?: { name: string; training?: string; checkpoint_sha256?: string };
  result?: { reward: number } | null;
  note?: string;
  job_id?: string;
  // Live games: the opponent plays by turn search instead of the raw policy.
  search?: boolean;
  // Watch games: a named rival checkpoint (raw policy) occupies opponent_seat instead of the served bot.
  opponent?: string | null;
  opponent_seat?: number;
}

export interface Rules {
  units: { id: number; name: string; cost: number; attack: number; defense: number; health: number; movement: number; range: number; skills: string[] }[];
  structures: { id: number; name: string; cost: number | null; tribe?: number | null }[];
  technologies: { id: number; name: string; tier: number; units: number[]; structures: number[]; next: number[] }[];
}

export interface Meta {
  bot: { name: string; search?: { lanes: number; replies: number; workers: number }; opponents?: string[] };
  ready: boolean;
  search?: { available: boolean; ready: boolean };
  warmup_job: string | null;
  rules: Rules;
  terrain: Record<string, string>;
  resources: Record<string, string>;
  abilities: Record<string, string>;
  rewards: Record<string, string>;
}

export interface Library {
  games: { id: string; title: string; frames: number; current_player: number }[];
  replays: { id: string; title: string; frames: number; complete: boolean; created: number; source?: string; available_views: string[]; featured?: boolean }[];
}

// One player's state at the start of a turn (GET /api/replays/<id>/series; service.replay_series).
export interface SeriesPoint {
  frame: number;
  turn: number;
  stars: number;
  income: number;
  score: number;
  cities: number;
  population: number;
  units: number;
  army: number; // star cost of the player's units, scaled by remaining health
  final?: boolean;
}

export interface Series {
  available: boolean;
  players?: Record<string, SeriesPoint[]>;
}

export type Layer = "unit" | "city" | "tile";
export interface Current {
  kind: "games" | "replays";
  id: string;
}
