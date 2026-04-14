"""Build fixed observation vectors for Clustered and Spread scenarios."""
import numpy as np

GRID_SIZE = 32
ROLES = ['strike', 'vanguard', 'support']
ROLE_ONEHOT = np.eye(3, dtype=np.float32)
ZONE_POSITIONS = np.array([(4, 4), (4, 27), (15, 15)], dtype=np.float32)  # top-left, top-right, center

OBS_DIM = 65
N_AGENTS = 3
N_ENEMIES = 2
N_ZONES = 3


def _build_obs(agent_idx: int, agent_positions: np.ndarray, enemy_positions: np.ndarray,
               hp: np.ndarray = None) -> np.ndarray:
    """Construct a 65-dim observation for agent_idx given scenario state."""
    if hp is None:
        hp = np.ones(N_AGENTS, dtype=np.float32)

    o = np.zeros(OBS_DIM, dtype=np.float32)
    pos = agent_positions[agent_idx]

    # self_tok [0:8]
    o[0:2] = pos / GRID_SIZE
    o[2:4] = 0.0  # no velocity in static scenario
    o[4] = hp[agent_idx]
    o[5:8] = ROLE_ONEHOT[agent_idx]

    # ally_tok [8:24]
    ally_indices = [j for j in range(N_AGENTS) if j != agent_idx]
    for slot, j in enumerate(ally_indices):
        base = 8 + slot * 8
        rel_pos = (agent_positions[j] - pos) / GRID_SIZE
        o[base:base+2] = rel_pos
        o[base+2] = hp[j]
        o[base+3] = 1.0  # alive
        o[base+4:base+7] = ROLE_ONEHOT[j]
        o[base+7] = 0.0

    # enemy_tok [24:40]
    for e in range(N_ENEMIES):
        base = 24 + e * 8
        rel_pos = (enemy_positions[e] - pos) / GRID_SIZE
        o[base:base+2] = rel_pos
        o[base+2] = 1.0
        o[base+3] = 1.0
        o[base+4:base+7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        o[base+7] = 0.0

    # zone_tok [40:58]
    for z in range(N_ZONES):
        base = 40 + z * 6
        rel_pos = (ZONE_POSITIONS[z] - pos) / GRID_SIZE
        o[base:base+2] = rel_pos
        o[base+2] = 0.0   # occupancy
        o[base+3] = 0.0   # progress
        o[base+4] = float(agent_idx == z)  # assigned
        o[base+5] = 1.0   # value

    # masks [58:65]: all 0 (valid)
    o[58:65] = 0.0
    return o


def build_scenario_obs(scenario: str) -> dict:
    """
    Returns dict with keys 'strike', 'vanguard', 'support' → np.ndarray (65,)

    Clustered: all allies near zone 0 (top-left), enemies far away.
    Spread: agents spread across zones, enemies near center.
    """
    if scenario == 'clustered':
        agent_positions = np.array([
            [4.0,  4.0],  # Strike at zone 0 (top-left)
            [5.0,  4.0],  # Vanguard near zone 0
            [4.0,  5.0],  # Support near zone 0
        ], dtype=np.float32)
        enemy_positions = np.array([
            [28.0, 28.0],
            [27.0, 28.0],
        ], dtype=np.float32)

    elif scenario == 'spread':
        agent_positions = np.array([
            [ 4.0,  4.0],  # Strike at zone 0 (top-left)
            [ 4.0, 27.0],  # Vanguard at zone 1 (top-right)
            [15.0, 15.0],  # Support at zone 2 (center)
        ], dtype=np.float32)
        enemy_positions = np.array([
            [15.0, 20.0],
            [16.0, 20.0],
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    obs = {}
    for i, role in enumerate(ROLES):
        obs[role] = _build_obs(i, agent_positions, enemy_positions)
    return obs
