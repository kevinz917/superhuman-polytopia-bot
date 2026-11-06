//! One game's state: the same fields as gpu/state.py::State, struct-of-arrays, runtime board size.

#[derive(Clone)]
pub struct Tiles {
    pub terrain: Vec<i32>,
    pub owner: Vec<i32>,
    pub capital: Vec<i32>,
    pub city: Vec<i32>,
    pub road: Vec<bool>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    pub explored: [Vec<bool>; 2],
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
    }
    /// jax.tree.map(lambda x: x.at[t].set(x[s]).at[s].set(0), units)
    pub fn move_unit(&mut self, s: usize, t: usize) {
        macro_rules! mv {
            ($($f:ident),*) => { $( { let v = self.$f[s]; self.$f[t] = v; } )* };
        }
        mv!(owner, kind, health, veteran, moved, attacked, kills, poison, frozen, boosted, home, created, converted,
            attacks_performed, prev_x, prev_y, last_attack, passenger, invisible);
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
    pub stars: [i32; 2],
    pub score: [i32; 2],
    pub tech: [[bool; 128]; 2],
    pub tech_turn: [[i32; 128]; 2],
    pub known: [[bool; 2]; 2],
    pub kills: [i32; 2],
    pub casualties: [i32; 2],
    pub pacifist: [i32; 2],
    pub attacked: [bool; 2],
    pub built: [[bool; 128]; 2],
    pub killed_turn: [i32; 2],
    pub resigned_turn: [i32; 2],
    pub killer: [i32; 2],
    pub peace: [[bool; 2]; 2],
    pub embassy: [[bool; 2]; 2],
    pub peace_offer: [[bool; 2]; 2],
    pub truce_breaker: [[i32; 2]; 2],
    pub hostile_until: [i32; 2],
    pub monuments: [[bool; 128]; 2],
}

#[derive(Clone)]
pub struct State {
    pub size: usize,
    pub n: usize,
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
        let n = size * size;
        let z = || vec![0i32; n];
        let b = || vec![false; n];
        State {
            size,
            n,
            tiles: Tiles {
                terrain: z(), owner: z(), capital: z(), city: vec![-1; n], road: b(), structure: z(), resource: z(),
                explored: [b(), b()], structure_level: z(), structure_turn: z(),
            },
            units: Units {
                owner: z(), kind: z(), health: vec![0.0; n], veteran: b(), moved: b(), attacked: b(), kills: z(),
                poison: b(), frozen: b(), boosted: b(), home: z(), created: z(), converted: b(), attacks_performed: z(),
                prev_x: z(), prev_y: z(), last_attack: z(), passenger: z(), invisible: b(),
            },
            cities: Cities {
                owner: z(), level: z(), population: z(), progress: z(), production: z(), border: z(),
                rewards: vec![0; n * 9], territory: vec![false; n * n], connected: b(), order: z(), infiltrated: z(), riot: b(),
            },
            players: Players {
                stars: [0; 2], score: [0; 2], tech: [[false; 128]; 2], tech_turn: [[0; 128]; 2], known: [[false; 2]; 2],
                kills: [0; 2], casualties: [0; 2], pacifist: [0; 2], attacked: [false; 2], built: [[false; 128]; 2],
                killed_turn: [0; 2], resigned_turn: [0; 2], killer: [0; 2], peace: [[false; 2]; 2], embassy: [[false; 2]; 2],
                peace_offer: [[false; 2]; 2], truce_breaker: [[0; 2]; 2], hostile_until: [0; 2], monuments: [[false; 128]; 2],
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

    /// Seat index of the player to move (0/1).
    #[inline]
    pub fn p(&self) -> usize {
        (self.current_player - 1) as usize
    }
    /// jnp.clip(owner - 1, 0, 1)
    #[inline]
    pub fn pclip(owner: i32) -> usize {
        (owner - 1).clamp(0, 1) as usize
    }
    /// JAX gather semantics for `tech[owner - 1]`: a negative index wraps to the last row.
    #[inline]
    pub fn pwrap(owner: i32) -> usize {
        let i = owner - 1;
        (if i < 0 { i + 2 } else { i }).clamp(0, 1) as usize
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
