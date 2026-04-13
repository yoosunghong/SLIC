"""
Synthetic observation builder for SMACv2 Track 2 analysis.

Constructs fixed 65-dim entity-centric obs vectors directly — no live
StarCraft II process required.  The obs layout is identical to the
MiniGrid PoC, so analyze_attention.py imports this and needs no other
changes.

Coordinate system: normalised [0, 1] × [0, 1]

Scenarios
---------
clustered : All ally rel_pos within radius ~0.06 of (0, 0).
            Enemies placed at (+0.5, +0.5) from self.
spread    : Agents spread to three zone landmarks.
            Enemies near map centre.

Role → slot mapping (same as MiniGrid):
  agent 0 = Strike   (Stalker)
  agent 1 = Vanguard (Zealot)
  agent 2 = Support  (Colossus)

For Condition 4 (No-RoleLabel), zero out:
  self_tok  role dims  obs[4:7]
  ally_tok  role dims  obs[8+4 : 8+7] and obs[16+4 : 16+7]
"""
import numpy as np

OBS_DIM = 65
N_AGENTS = 3
N_ENEMIES = 2
N_ZONES = 3

ROLES = ['strike', 'vanguard', 'support']

# Unit-type one-hots: [1,0,0]=Stalker/Strike, [0,1,0]=Zealot/Vanguard, [0,0,1]=Colossus/Support
ROLE_ONEHOT = np.eye(3, dtype=np.float32)

# Fixed landmark positions (normalised) matching smac_wrapper.ZONE_POSITIONS_NORM
ZONE_POSITIONS = np.array([
    [0.15, 0.15],  # top-left
    [0.85, 0.15],  # top-right
    [0.50, 0.50],  # centre
], dtype=np.float32)


def _build_obs(agent_idx: int, agent_positions: np.ndarray,
               enemy_positions: np.ndarray,
               hp: np.ndarray = None,
               shield: np.ndarray = None) -> np.ndarray:
    """
    Construct a single 65-dim SMACv2 entity-centric obs.

    Parameters
    ----------
    agent_idx : int  (0=Strike, 1=Vanguard, 2=Support)
    agent_positions : (3, 2) normalised [0,1] positions
    enemy_positions : (2, 2) normalised positions
    hp : (3,) health ratios [0,1]; defaults to 1.0
    shield : (3,) shield ratios [0,1]; defaults to 1.0
    """
    if hp is None:
        hp = np.ones(N_AGENTS, dtype=np.float32)
    if shield is None:
        shield = np.ones(N_AGENTS, dtype=np.float32)

    o = np.zeros(OBS_DIM, dtype=np.float32)
    pos = agent_positions[agent_idx]

    # self_tok [0:8]  pos(2)+health(1)+shield(1)+unit_type_onehot(3)+pad(1)
    o[0] = pos[0]
    o[1] = pos[1]
    o[2] = hp[agent_idx]
    o[3] = shield[agent_idx]
    o[4:7] = ROLE_ONEHOT[agent_idx]
    o[7] = 0.0  # padding

    # ally_tok [8:24]  2 allies × 8: rel_pos(2)+health(1)+shield(1)+unit_type_onehot(3)+alive(1)
    ally_indices = [j for j in range(N_AGENTS) if j != agent_idx]
    for slot, j in enumerate(ally_indices):
        base = 8 + slot * 8
        rel = agent_positions[j] - pos
        o[base + 0] = rel[0]
        o[base + 1] = rel[1]
        o[base + 2] = hp[j]
        o[base + 3] = shield[j]
        o[base + 4:base + 7] = ROLE_ONEHOT[j]
        o[base + 7] = 1.0  # alive

    # enemy_tok [24:40]  2 enemies × 8: rel_pos(2)+health(1)+shield(1)+type_conf(3)+alive(1)
    for e in range(N_ENEMIES):
        base = 24 + e * 8
        rel = enemy_positions[e] - pos
        o[base + 0] = rel[0]
        o[base + 1] = rel[1]
        o[base + 2] = 1.0   # full health
        o[base + 3] = 1.0   # full shield
        o[base + 4:base + 7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        o[base + 7] = 1.0   # alive

    # zone_tok [40:58]  3 × 6: rel_pos(2)+owned(1)+contest(1)+value(1)+pad(1)
    for z in range(N_ZONES):
        base = 40 + z * 6
        rel = ZONE_POSITIONS[z] - pos
        o[base + 0] = rel[0]
        o[base + 1] = rel[1]
        o[base + 2] = 0.0   # owned
        o[base + 3] = 0.0   # contest
        o[base + 4] = 1.0   # value
        o[base + 5] = 0.0   # padding

    # masks [58:65]  0=valid, 1=padding
    o[58:65] = 0.0   # all slots valid in synthetic scenarios

    return o


def build_scenario_obs(scenario: str, zero_role_labels: bool = False) -> dict:
    """
    Build per-role observation dicts for Clustered or Spread scenarios.

    Parameters
    ----------
    scenario : 'clustered' | 'spread'
    zero_role_labels : bool
        If True, zero out role one-hot in self_tok and ally_tok slots.
        Used for Condition 4 (No-RoleLabel) analysis.

    Returns
    -------
    dict with keys 'strike', 'vanguard', 'support' → np.ndarray (65,)
    """
    if scenario == 'clustered':
        # All agents clustered near zone 0 (top-left landmark, ~0.15, 0.15)
        # Allies within radius ≈ 0.06 of each other
        agent_positions = np.array([
            [0.15, 0.15],   # Strike at landmark 0
            [0.18, 0.15],   # Vanguard slightly right
            [0.15, 0.18],   # Support slightly below
        ], dtype=np.float32)
        # Enemies far away (bottom-right corner)
        enemy_positions = np.array([
            [0.85, 0.85],
            [0.80, 0.85],
        ], dtype=np.float32)

    elif scenario == 'spread':
        # Agents spread across three zone landmarks
        agent_positions = np.array([
            [0.15, 0.15],   # Strike at landmark 0 (top-left)
            [0.85, 0.15],   # Vanguard at landmark 1 (top-right)
            [0.50, 0.50],   # Support at landmark 2 (centre)
        ], dtype=np.float32)
        # Enemies near map centre
        enemy_positions = np.array([
            [0.50, 0.40],
            [0.50, 0.60],
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown scenario: '{scenario}'. Use 'clustered' or 'spread'.")

    obs = {}
    for i, role in enumerate(ROLES):
        o = _build_obs(i, agent_positions, enemy_positions)
        if zero_role_labels:
            o = _zero_role_dims(o)
        obs[role] = o

    return obs


def _zero_role_dims(obs: np.ndarray) -> np.ndarray:
    """
    Zero out role one-hot dims for Condition 4 (No-RoleLabel).
    self_tok role dims: [4:7]
    ally slot 0 role dims: [8+4 : 8+7]  = [12:15]
    ally slot 1 role dims: [16+4 : 16+7] = [20:23]
    """
    o = obs.copy()
    o[4:7] = 0.0
    o[12:15] = 0.0
    o[20:23] = 0.0
    return o


# Slot-to-role mapping for cross-attention analysis (same as MiniGrid)
ALLY_SLOT_ROLES = {
    0: [1, 2],  # Strike's 2 ally slots: Vanguard(1), Support(2)
    1: [0, 2],  # Vanguard's 2 ally slots: Strike(0), Support(2)
    2: [0, 1],  # Support's 2 ally slots: Strike(0), Vanguard(1)
}
