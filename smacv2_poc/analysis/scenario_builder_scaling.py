"""
Synthetic 182-dim padded observation builder for the scaling study.

Mirrors smac_wrapper_scaling.py obs layout:
  self_tok   [0:8]      pos(2)+health(1)+pad(1)+role_onehot(3)+pad(1)
  ally_tok   [8:72]     8 slots × 8: rel_pos(2)+health(1)+pad(1)+role_onehot(3)+alive(1)
  enemy_tok  [72:144]   9 slots × 8
  zone_tok   [144:162]  3 slots × 6
  ally_mask  [162:170]  8  (1 = padded/dead)
  enemy_mask [170:179]  9
  zone_mask  [179:182]  3

Used by analyze_attention_scaling.py and occlusion_routing.py for both
3v3 and 9v9 (and 6v6) checkpoints — same obs_dim=182, only the
composition / agent count changes.
"""
import numpy as np

OBS_DIM = 182
MAX_ALLY_SLOTS = 8
MAX_ENEMY_SLOTS = 9
N_ZONE_SLOTS = 3

SELF_DIM, ALLY_DIM, ENEMY_DIM, ZONE_DIM = 8, 8, 8, 6

_SELF_END = SELF_DIM
_ALLY_END = _SELF_END + MAX_ALLY_SLOTS * ALLY_DIM           # 72
_ENEMY_END = _ALLY_END + MAX_ENEMY_SLOTS * ENEMY_DIM        # 144
_ZONE_END = _ENEMY_END + N_ZONE_SLOTS * ZONE_DIM            # 162
_AMASK_END = _ZONE_END + MAX_ALLY_SLOTS                     # 170
_EMASK_END = _AMASK_END + MAX_ENEMY_SLOTS                   # 179

ROLES = ['strike', 'vanguard', 'support']
ROLE_ONEHOT = np.eye(3, dtype=np.float32)

ZONE_POSITIONS = np.array([
    [0.15, 0.15],
    [0.85, 0.15],
    [0.50, 0.50],
], dtype=np.float32)


def composition_to_roles(composition: dict) -> list:
    """{'Strike': k0, 'Vanguard': k1, 'Support': k2} → [0]*k0 + [1]*k1 + [2]*k2."""
    out = []
    for ridx, name in enumerate(['Strike', 'Vanguard', 'Support']):
        out += [ridx] * int(composition.get(name, 0))
    return out


def _build_obs(agent_idx: int,
               agent_roles: list,
               agent_positions: np.ndarray,
               enemy_positions: np.ndarray,
               enemy_roles: list = None,
               hp: np.ndarray = None) -> np.ndarray:
    n_agents = len(agent_roles)
    n_enemies = len(enemy_positions)
    if hp is None:
        hp = np.ones(n_agents, dtype=np.float32)

    o = np.zeros(OBS_DIM, dtype=np.float32)
    # All ally/enemy mask slots default to padded (1); flip valid slots to 0.
    o[_ZONE_END:_AMASK_END] = 1.0
    o[_AMASK_END:_EMASK_END] = 1.0

    pos = agent_positions[agent_idx]
    role = agent_roles[agent_idx]

    # self_tok
    o[0], o[1] = pos[0], pos[1]
    o[2] = hp[agent_idx]
    o[3] = 0.0
    o[4:7] = ROLE_ONEHOT[role]
    o[7] = 0.0

    # ally_tok — every other agent in agent_idx order
    slot = 0
    for j in range(n_agents):
        if j == agent_idx:
            continue
        if slot >= MAX_ALLY_SLOTS:
            break
        base = _SELF_END + slot * ALLY_DIM
        rel = agent_positions[j] - pos
        o[base + 0] = rel[0]
        o[base + 1] = rel[1]
        o[base + 2] = hp[j]
        o[base + 3] = 0.0
        o[base + 4:base + 7] = ROLE_ONEHOT[agent_roles[j]]
        o[base + 7] = 1.0  # alive
        o[_ZONE_END + slot] = 0.0  # unmask
        slot += 1

    # enemy_tok
    for e in range(min(n_enemies, MAX_ENEMY_SLOTS)):
        base = _ALLY_END + e * ENEMY_DIM
        rel = enemy_positions[e] - pos
        o[base + 0] = rel[0]
        o[base + 1] = rel[1]
        o[base + 2] = 1.0
        o[base + 3] = 0.0
        o[base + 4:base + 7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        o[base + 7] = 1.0
        o[_AMASK_END + e] = 0.0  # unmask

    # zone_tok
    for z in range(N_ZONE_SLOTS):
        base = _ENEMY_END + z * ZONE_DIM
        rel = ZONE_POSITIONS[z] - pos
        o[base + 0] = rel[0]
        o[base + 1] = rel[1]
        o[base + 2] = 0.0
        o[base + 3] = 0.0
        o[base + 4] = 1.0
        o[base + 5] = 0.0

    return o


def _ally_slot_roles(agent_idx: int, agent_roles: list) -> list:
    """Role index of each ally slot (variable length up to MAX_ALLY_SLOTS)."""
    return [agent_roles[j] for j in range(len(agent_roles)) if j != agent_idx][:MAX_ALLY_SLOTS]


def build_scenario_obs_scaling(scenario: str, composition: dict,
                               n_enemies: int = None,
                               representative_only: bool = True) -> dict:
    """
    Build per-role observations for a scaling-study composition.

    Parameters
    ----------
    scenario : 'clustered' | 'spread'
    composition : {'Strike': k0, 'Vanguard': k1, 'Support': k2}
    n_enemies : enemy count (defaults to sum of composition)
    representative_only : if True, return one obs per role (using the first
        agent index of each role group). If False, returns obs for every
        agent index.

    Returns
    -------
    dict with keys:
      'strike', 'vanguard', 'support'  → np.ndarray (182,) for the
        representative agent of each role.
      'ally_slot_roles' → dict[role_name] = list of slot→ally_role_idx
      'agent_roles' → list of role indices for all agents
      'composition' → echoed composition dict
    """
    agent_roles = composition_to_roles(composition)
    n_agents = len(agent_roles)
    if n_agents == 0:
        raise ValueError("Empty composition")
    if n_enemies is None:
        n_enemies = n_agents

    if scenario == 'clustered':
        # All agents within ~0.06 of zone 0 (top-left corner area)
        center = np.array([0.15, 0.15], dtype=np.float32)
        offsets = _ring_offsets(n_agents, radius=0.04)
        agent_positions = center[None, :] + offsets
        enemy_center = np.array([0.85, 0.85], dtype=np.float32)
        enemy_offsets = _ring_offsets(n_enemies, radius=0.05)
        enemy_positions = enemy_center[None, :] + enemy_offsets
    elif scenario == 'spread':
        # Distribute agents across the three zone landmarks deterministically.
        agent_positions = _spread_positions(n_agents)
        # Enemies near map centre, fan
        enemy_center = np.array([0.50, 0.50], dtype=np.float32)
        enemy_offsets = _ring_offsets(n_enemies, radius=0.10)
        enemy_positions = enemy_center[None, :] + enemy_offsets
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    out = {'agent_roles': agent_roles, 'composition': dict(composition)}

    # Pick representative agent index per role (first index for that role).
    rep_idx = {}
    for role_idx, name in enumerate(ROLES):
        for i, r in enumerate(agent_roles):
            if r == role_idx:
                rep_idx[name] = i
                break

    slot_roles = {}
    for name, idx in rep_idx.items():
        out[name] = _build_obs(idx, agent_roles, agent_positions, enemy_positions)
        slot_roles[name] = _ally_slot_roles(idx, agent_roles)
    out['ally_slot_roles'] = slot_roles
    out['rep_idx'] = rep_idx
    out['agent_positions'] = agent_positions
    out['enemy_positions'] = enemy_positions
    return out


def _ring_offsets(n: int, radius: float) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def build_random_scenarios(n_scenarios: int, composition: dict,
                           n_enemies: int = None, seed: int = 0) -> list:
    """
    Generate `n_scenarios` random scenarios for the given composition.

    Each scenario places agents and enemies at random positions on the
    [0.05, 0.95]^2 unit map (avoiding edges). Used to stabilize the
    occlusion-routing matrix (especially the Strike row) by averaging over
    many configurations rather than relying solely on the two
    Clustered/Spread anchors.

    Returns
    -------
    list of dicts (same schema as build_scenario_obs_scaling), one per scenario.
    """
    agent_roles = composition_to_roles(composition)
    n_agents = len(agent_roles)
    if n_enemies is None:
        n_enemies = n_agents

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_scenarios):
        agent_positions = rng.uniform(0.05, 0.95, size=(n_agents, 2)).astype(np.float32)
        enemy_positions = rng.uniform(0.05, 0.95, size=(n_enemies, 2)).astype(np.float32)

        scenario = {'agent_roles': agent_roles, 'composition': dict(composition)}
        rep_idx = {}
        for role_idx, name in enumerate(ROLES):
            for i, r in enumerate(agent_roles):
                if r == role_idx:
                    rep_idx[name] = i
                    break
        slot_roles = {}
        for name, idx in rep_idx.items():
            scenario[name] = _build_obs(idx, agent_roles, agent_positions, enemy_positions)
            slot_roles[name] = _ally_slot_roles(idx, agent_roles)
        scenario['ally_slot_roles'] = slot_roles
        scenario['rep_idx'] = rep_idx
        scenario['agent_positions'] = agent_positions
        scenario['enemy_positions'] = enemy_positions
        out.append(scenario)
    return out


def _spread_positions(n_agents: int) -> np.ndarray:
    """
    Place agents at the 3 zone landmarks with small jitter for n>3.
    Ensures genuine spatial dispersion across the map.
    """
    pos = np.zeros((n_agents, 2), dtype=np.float32)
    for i in range(n_agents):
        zone = ZONE_POSITIONS[i % N_ZONE_SLOTS]
        # Small ring jitter for repeated assignments
        layer = i // N_ZONE_SLOTS
        if layer == 0:
            offset = np.zeros(2, dtype=np.float32)
        else:
            angle = (i * 1.7) % (2 * np.pi)
            r = 0.03 * layer
            offset = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        pos[i] = zone + offset
    return pos
