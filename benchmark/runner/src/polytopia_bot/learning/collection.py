"""Episode collection against the public Environment protocol.

No engine internals are imported. A certified GPU adapter can be plugged in
later; fixture collection is explicitly separated from training data.
"""
from .data import demonstrate


def collect_episode(env, *, settings, seed, episode_id, map_hash, teacher, opponent,
                    learner_player=1, max_decisions=10000, full_game_ready=False, fixture=False):
    if not fixture and not full_game_ready:
        raise ValueError("real demonstration collection requires the complete-game readiness gate")
    if learner_player not in (1, 2) or max_decisions < 1:
        raise ValueError("invalid collection limits")
    records, trace = [], []
    obs = env.reset(seed=seed, settings=settings)
    terminated, truncated, rewards, reason = False, False, {}, "collector_action_limit"
    try:
        for decision in range(max_decisions):
            policy = teacher if obs.player == learner_player else opponent
            selected = policy.select(obs)
            if obs.player == learner_player:
                row = demonstrate(obs, teacher, episode_id=episode_id, map_hash=map_hash,
                                  decision_index=decision, source="fixture" if fixture else "teacher")
                if row["selected_command"] != selected.action.command:
                    raise ValueError("teacher changed its choice for the same public observation")
                records.append(row)
            transition = env.step(action_id=selected.action.action_id, expected_state_version=obs.state_version)
            if transition.terminated and transition.truncated:
                raise ValueError("terminal and truncation flags cannot both be set")
            if transition.actor_observation.state_version <= obs.state_version:
                raise ValueError("successful transition must advance the action revision")
            trace.append({"decision_index": decision, "player": obs.player,
                          "state_version": obs.state_version, "command": dict(selected.action.command),
                          "policy_hash": selected.teacher_hash})
            terminated, truncated, rewards, reason = (transition.terminated, transition.truncated,
                                                       dict(transition.rewards), transition.reason)
            if terminated or truncated:
                break
            if transition.next_player not in (1, 2):
                raise ValueError("active transition requires a next player")
            # actor_observation is explicitly NOT necessarily the next player's.
            obs = env.observe(transition.next_player)
            if obs.player != transition.next_player or obs.state_version != transition.actor_observation.state_version:
                raise ValueError("next-player observation has wrong seat or version")
        else:
            truncated, reason = True, "collector_action_limit"
        return records, {"episode_id": episode_id, "map_hash": map_hash, "source": "fixture" if fixture else "teacher",
                         "learner_player": learner_player, "terminated": terminated, "truncated": truncated,
                         "rewards": rewards, "reason": reason, "trace": trace}
    finally:
        env.close()
