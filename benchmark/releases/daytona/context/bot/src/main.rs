// One JSON observation per line; reply {"action_id":"..."} and flush.
// Replace choose() with your strategy. Process memory persists within a game.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn choose(obs: &Value) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    // Mining-only skeleton: end turn, or resolve a mandatory city reward.
    let action = legal.iter().find(|a| a["command"]["kind"] == 10).unwrap_or(&legal[0]);
    action["action_id"].as_str().unwrap().to_owned()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let obs: Value = serde_json::from_str(&line.unwrap()).unwrap();
        writeln!(stdout, "{}", json!({"action_id": choose(&obs)})).unwrap();
        stdout.flush().unwrap();
    }
}
