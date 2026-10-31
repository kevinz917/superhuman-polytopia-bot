//! One game's state: the same fields as gpu/state.py::State, struct-of-arrays, runtime board size and seat count.

/// Most seats a game can have.
pub const MAXP: usize = 4;

#[derive(Clone)]
pub struct Tiles {
    pub terrain: Vec<i32>,
    pub owner: Vec<i32>,
    pub capital: Vec<i32>,
    pub city: Vec<i32>,
    pub road: Vec<bool>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    pub explored: [Vec<bool>; MAXP],
    pub structure_level: Vec<i32>,
    pub structure_turn: Vec<i32>,
}

#[derive(Clone)]
pub struct Units {
    pub owner: Vec<i32>,
    pub kind: Vec<i32>,
    pub health: Vec<f32>,
    pub veteran: Vec<bool>,
    pub moved: Vec<bool>,
    pub attacked: Vec<bool>,
    pub kills: Vec<i32>,
    pub poison: Vec<bool>,
    pub frozen: Vec<bool>,
    pub boosted: Vec<bool>,
    pub home: Vec<i32>,
    pub created: Vec<i32>,
    pub converted: Vec<bool>,
    pub attacks_performed: Vec<i32>,
    pub prev_x: Vec<i32>,
    pub prev_y: Vec<i32>,
    pub last_attack: Vec<i32>,
    pub passenger: Vec<i32>,
    pub invisible: Vec<bool>,
    /// Facing (grid direction 0..7 = SW, W, NW, N, NE, E, SE, S; 8 none): toward the map centre at
    /// creation, then the last step of every move and the line of every attack. A push starts from it.
    pub direction: Vec<i32>,
}

impl Units {
    /// jax.tree.map(lambda x: x.at[i].set(0), units)
    pub fn clear(&mut self, i: usize) {
        self.owner[i] = 0;
        self.kind[i] = 0;
        self.health[i] = 0.0;
        self.veteran[i] = false;
        self.moved[i] = false;
        self.attacked[i] = false;
        self.kills[i] = 0;
        self.poison[i] = false;
        self.frozen[i] = false;
        self.boosted[i] = false;
        self.home[i] = 0;
        self.created[i] = 0;
        self.converted[i] = false;
        self.attacks_performed[i] = 0;
        self.prev_x[i] = 0;
        self.prev_y[i] = 0;
        self.last_attack[i] = 0;
        self.passenger[i] = 0;
        self.invisible[i] = false;
        self.direction[i] = 0;
    }
    /// jax.tree.map(lambda x: x.at[t].set(x[s]).at[s].set(0), units)
    pub fn move_unit(&mut self, s: usize, t: usize) {
        macro_rules! mv {
            ($($f:ident),*) => { $( { let v = self.$f[s]; self.$f[t] = v; } )* };
        }
        mv!(owner, kind, health, veteran, moved, attacked, kills, poison, frozen, boosted, home, created, converted,
            attacks_performed, prev_x, prev_y, last_attack, passenger, invisible, direction);
        self.clear(s);
    }
}

#[derive(Clone)]
pub struct Cities {
    pub owner: Vec<i32>,
    pub level: Vec<i32>,
    pub population: Vec<i32>,
    pub progress: Vec<i32>,
    pub production: Vec<i32>,
    pub border: Vec<i32>,
    pub rewards: Vec<i32>,     // [n*9]
    pub territory: Vec<bool>,  // [n*n], row = city tile
    pub connected: Vec<bool>,
    pub order: Vec<i32>,
    pub infiltrated: Vec<i32>,
    pub riot: Vec<bool>,
}

#[derive(Clone)]
pub struct Players {
    pub stars: [i32; MAXP],
    pub score: [i32; MAXP],
    pub tech: [[bool; 128]; MAXP],
    pub tech_turn: [[i32; 128]; MAXP],
    pub known: [[bool; MAXP]; MAXP],
    pub kills: [i32; MAXP],
    pub casualties: [i32; MAXP],
    pub pacifist: [i32; MAXP],
    pub attacked: [bool; MAXP],
    pub built: [[bool; 128]; MAXP],
    pub killed_turn: [i32; MAXP],
    pub resigned_turn: [i32; MAXP],
    pub killer: [i32; MAXP],
    pub peace: [[bool; MAXP]; MAXP],
    pub embassy: [[bool; MAXP]; MAXP],
    pub peace_offer: [[bool; MAXP]; MAXP],
    pub truce_breaker: [[i32; MAXP]; MAXP],
    pub hostile_until: [i32; MAXP],
    pub monuments: [[bool; 128]; MAXP],
    /// Engine-internal (4-player games): 1-based order in which seats left the game (killed or resigned), 0 while
    /// in. Two-player games never set it.
    pub out_order: [i32; MAXP],
}

#[derive(Clone)]
pub struct State {
    pub size: usize,
    pub n: usize,
    /// Seats in this game (2 or 4). Per-player arrays hold MAXP rows; rows >= np stay zero. Two-player games are
    /// bitwise the reference engine's; more seats follow the multi-player rules in docs/four-player-*.md.
    pub np: usize,
    pub tiles: Tiles,
    pub units: Units,
    pub cities: Cities,
    pub players: Players,
    pub current_player: i32,
    pub turn: i32,
    pub version: i32,
    pub state_version: i32,
    pub actions: i32,
    pub terminated: bool,
    pub truncated: bool,
    pub fog: bool,
    pub reference_max_turns: i32,
    pub core_supported: bool,
    pub seed: u32,
    pub initial_seed: u32,
    /// Engine-internal (not a reference State field): structures::remember_monuments has already been applied to
    /// exactly this state, so applying it again is a no-op. Set by core::step; any other way of changing a state
    /// (reset, import) leaves or sets it false.
    pub monuments_settled: bool,
}

impl State {
    pub fn empty(size: usize) -> State {
        State::empty_players(size, 2)
    }

    pub fn empty_players(size: usize, np: usize) -> State {
        assert!((2..=MAXP).contains(&np));
        let n = size * size;
        let z = || vec![0i32; n];
        let b = || vec![false; n];
        State {
            size,
            n,
            np,
            tiles: Tiles {
                terrain: z(), owner: z(), capital: z(), city: vec![-1; n], road: b(), structure: z(), resource: z(),
                explored: [b(), b(), b(), b()], structure_level: z(), structure_turn: z(),
            },
            units: Units {
                owner: z(), kind: z(), health: vec![0.0; n], veteran: b(), moved: b(), attacked: b(), kills: z(),
                poison: b(), frozen: b(), boosted: b(), home: z(), created: z(), converted: b(), attacks_performed: z(),
                prev_x: z(), prev_y: z(), last_attack: z(), passenger: z(), invisible: b(), direction: z(),
            },
            cities: Cities {
                owner: z(), level: z(), population: z(), progress: z(), production: z(), border: z(),
                rewards: vec![0; n * 9], territory: vec![false; n * n], connected: b(), order: z(), infiltrated: z(), riot: b(),
            },
            players: Players {
                stars: [0; MAXP], score: [0; MAXP], tech: [[false; 128]; MAXP], tech_turn: [[0; 128]; MAXP],
                known: [[false; MAXP]; MAXP], kills: [0; MAXP], casualties: [0; MAXP], pacifist: [0; MAXP],
                attacked: [false; MAXP], built: [[false; 128]; MAXP], killed_turn: [0; MAXP], resigned_turn: [0; MAXP],
                killer: [0; MAXP], peace: [[false; MAXP]; MAXP], embassy: [[false; MAXP]; MAXP],
                peace_offer: [[false; MAXP]; MAXP], truce_breaker: [[0; MAXP]; MAXP], hostile_until: [0; MAXP],
                monuments: [[false; 128]; MAXP], out_order: [0; MAXP],
            },
            current_player: 1,
            turn: 0,
            version: 115,
            state_version: 0,
            actions: 0,
            terminated: false,
            truncated: false,
            fog: true,
            reference_max_turns: 45,
            core_supported: true,
            seed: 0,
            initial_seed: 0,
            monuments_settled: false,
        }
    }

    /// Seat index of the player to move (0..np).
    #[inline]
    pub fn p(&self) -> usize {
        (self.current_player - 1) as usize
    }
    /// jnp.clip(owner - 1, 0, np - 1)
    #[inline]
    pub fn pclip(&self, owner: i32) -> usize {
        (owner - 1).clamp(0, self.np as i32 - 1) as usize
    }
    /// JAX gather semantics for `tech[owner - 1]`: a negative index wraps to the last row.
    #[inline]
    pub fn pwrap(&self, owner: i32) -> usize {
        let np = self.np as i32;
        let i = owner - 1;
        (if i < 0 { i + np } else { i }).clamp(0, np - 1) as usize
    }
    /// Seat k (0-based) is still in the game: not killed and not resigned.
    #[inline]
    pub fn alive(&self, k: usize) -> bool {
        self.players.killed_turn[k] == 0 && self.players.resigned_turn[k] == 0
    }
    /// Seats still in the game.
    pub fn alive_count(&self) -> usize {
        (0..self.np).filter(|&k| self.alive(k)).count()
    }
    /// `owner` is another seat's id (not neutral, not the mover's).
    #[inline]
    pub fn foreign(owner: i32, me: i32) -> bool {
        owner > 0 && owner != me
    }
    /// The unit on `i` is hidden from `viewer`: invisible, another player's, and not a peace
    /// partner's (peace partners see each other's hidden units).
    #[inline]
    pub fn hidden_from(&self, i: usize, viewer: i32) -> bool {
        let o = self.units.owner[i];
        self.units.invisible[i] && Self::foreign(o, viewer) && !self.players.peace[self.pclip(viewer)][self.pclip(o)]
    }
    #[inline]
    pub fn territory(&self, city: usize, tile: usize) -> bool {
        self.cities.territory[city * self.n + tile]
    }
    #[inline]
    pub fn reward(&self, city: usize, k: usize) -> i32 {
        self.cities.rewards[city * 9 + k]
    }
    #[inline]
    pub fn reward_count(&self, city: usize) -> i32 {
        self.cities.rewards[city * 9..city * 9 + 9].iter().sum()
    }
    pub fn pending_reward(&self) -> bool {
        let cp = self.current_player;
        (0..self.n).any(|i| self.cities.owner[i] == cp && self.reward_count(i) < self.cities.level[i] - 1)
    }
}
