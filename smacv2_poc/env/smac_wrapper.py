"""
SMACv2 environment wrapper.

Adapts StarCraft2Env to the same interface as CooperativeGridEnv:
  reset() → (obs_list, info)
  step(actions) → (obs_list, rewards, done, truncated, info)

Observation layout (65-dim, identical slot positions to MiniGrid PoC):
  self_tok   [0:8]   pos(2) + health(1) + shield(1) + unit_type_onehot(3) + pad(1)
  ally_tok   [8:24]  2 × 8: rel_pos(2)+health(1)+shield(1)+unit_type_onehot(3)+alive(1)
  enemy_tok  [24:40] 2 × 8: rel_pos(2)+health(1)+shield(1)+type_conf(3)+alive(1)
  zone_tok   [40:58] 3 × 6: rel_pos(2)+owned(1)+contest(1)+value(1)+pad(1)
  ally_mask  [58:60] 1=padding
  enemy_mask [60:62] 1=padding
  zone_mask  [62:65] 1=padding
  TOTAL: 65

Role mapping (Protoss unit types):
  Strike   → Stalker  (SC2 unit_type 74, ranged DPS)
  Vanguard → Zealot   (SC2 unit_type 73, melee tank)
  Support  → Colossus (SC2 unit_type  4, area/positioning)

Action mapping (our 6 → SMACv2 discrete):
  0 → no-op (0)
  1 → move N (2)
  2 → move S (3)
  3 → move E (4)
  4 → move W (5)
  5 → attack nearest enemy (first available attack action ≥ 6)
"""
import numpy as np

try:
    from smacv2.env import StarCraft2Env
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    from smacv2.env.starcraft2.distributions import get_distribution
    _SMACV2_AVAILABLE = True
except ImportError:
    _SMACV2_AVAILABLE = False
    StarCraft2Env = None
    StarCraftCapabilityEnvWrapper = None
    get_distribution = None

# SC2 unit type IDs for Protoss roles
UNIT_TYPE_STALKER = 74   # Strike
UNIT_TYPE_ZEALOT = 73    # Vanguard
UNIT_TYPE_COLOSSUS = 4   # Support

ROLE_UNIT_TYPES = [UNIT_TYPE_STALKER, UNIT_TYPE_ZEALOT, UNIT_TYPE_COLOSSUS]
ROLE_NAMES = ['Strike', 'Vanguard', 'Support']
FIXED_ALLY_TEAM = ['stalker', 'zealot', 'colossus']
FIXED_ENEMY_TEAM = ['stalker', 'zealot']

# SMACv2 base action indices
ACTION_NOOP = 0
ACTION_STOP = 1
ACTION_MOVE_N = 2
ACTION_MOVE_S = 3
ACTION_MOVE_E = 4
ACTION_MOVE_W = 5
ATTACK_BASE = 6  # attack_enemy_i = ATTACK_BASE + enemy_index

# Our "idle" action maps to SMAC "stop" for live agents. True no-op is only
# valid for dead agents in SMACv2.
OUR_ACTION_TO_SMAC = [ACTION_STOP, ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_E, ACTION_MOVE_W]

OBS_DIM = 65
N_ALLY_SLOTS = 2
N_ENEMY_SLOTS = 2
N_ZONE_SLOTS = 3

# Fixed landmark positions (normalized 0-1) used as zone_tok proxies.
# These approximate top-left, top-right, center of a typical 32×32 SMACv2 map.
ZONE_POSITIONS_NORM = np.array([
    [0.15, 0.15],  # top-left
    [0.85, 0.15],  # top-right
    [0.50, 0.50],  # center
], dtype=np.float32)

# Reward shaping weights
APPROACH_BONUS = 0.005
ATTACK_BONUS = 0.05
MELEE_BONUS = 0.05
TOO_CLOSE_PENALTY = 0.02
SUPPORT_PROXIMITY_BONUS = 0.01
SPLASH_BONUS = 0.02
TEAM_ALPHA = 0.2
DEFAULT_MAP_SIZE = 32
MAX_RESET_RETRIES = 20


def _safe_unit_type_onehot(unit_type: int) -> np.ndarray:
    """Return [1,0,0]=Stalker, [0,1,0]=Zealot, [0,0,1]=Colossus, [0,0,0]=unknown."""
    oh = np.zeros(3, dtype=np.float32)
    if unit_type == UNIT_TYPE_STALKER:
        oh[0] = 1.0
    elif unit_type == UNIT_TYPE_ZEALOT:
        oh[1] = 1.0
    elif unit_type == UNIT_TYPE_COLOSSUS:
        oh[2] = 1.0
    return oh


class SMACWrapper:
    """
    Wraps StarCraft2Env into the CooperativeGridEnv interface.

    Parameters
    ----------
    cfg : dict
        Keys: map_name, n_agents (3), n_enemies (2), n_zones (3),
              max_steps, obs_dim (65), action_dim (6),
              team_reward_alpha (0.2), seed (optional).
    """

    def __init__(self, cfg: dict):
        if not _SMACV2_AVAILABLE:
            raise ImportError(
                "SMACv2 is not installed. Run:\n"
                "  pip install git+https://github.com/oxwhirl/smacv2.git pysc2\n"
                "and ensure StarCraftII is installed at ~/StarCraftII/."
            )

        self.n_agents = cfg.get('n_agents', 3)
        self.n_enemies = cfg.get('n_enemies', 2)
        self.n_zones = cfg.get('n_zones', 3)
        self.max_steps = cfg.get('max_steps', 200)
        self.obs_dim = cfg.get('obs_dim', OBS_DIM)
        self.action_dim = cfg.get('action_dim', 6)
        self.alpha = cfg.get('team_reward_alpha', TEAM_ALPHA)
        self.map_name = cfg.get('map_name', 'protoss_5_vs_5')
        # Prefer the standard fixed-map env unless the config explicitly opts
        # into generated 10gen capability scenarios.
        self._use_capability_wrapper = bool(
            cfg.get('use_capability_wrapper', self.map_name.startswith('10gen_'))
        )
        self._capability_config = cfg.get('capability_config')
        self._manual_fixed_team = (
            not self._use_capability_wrapper and self.map_name.startswith('10gen_')
        )

        if self._use_capability_wrapper:
            if self._capability_config is None:
                self._capability_config = self._build_default_capability_config()
            self._env = StarCraftCapabilityEnvWrapper(
                capability_config=self._capability_config,
                map_name=self.map_name,
                seed=cfg.get('seed', None),
                debug=False,
                conic_fov=False,
                use_unit_ranges=True,
                min_attack_range=2,
                obs_own_pos=True,
                fully_observable=False,
            )
        elif self._manual_fixed_team:
            if self._capability_config is None:
                self._capability_config = self._build_default_capability_config()
            self._env = StarCraft2Env(
                map_name=self.map_name,
                capability_config=self._capability_config,
                seed=cfg.get('seed', None),
                debug=False,
                conic_fov=False,
                use_unit_ranges=True,
                min_attack_range=2,
                obs_own_pos=True,
                fully_observable=False,
            )
            self._start_pos_distribution = get_distribution(
                self._capability_config['start_positions']['dist_type']
            )({
                **self._capability_config['start_positions'],
                'env_key': 'start_positions',
                'n_units': self.n_agents,
                'n_enemies': self.n_enemies,
            })
        else:
            self._env = StarCraft2Env(
                map_name=self.map_name,
                seed=cfg.get('seed', None),
            )
        self._env_info = self._env.get_env_info()

        # Map size for coordinate normalization
        self._map_x = float(self._env_info.get('map_x', 32))
        self._map_y = float(self._env_info.get('map_y', 32))

        # Role assignment: unit_id → role_index (0=Strike, 1=Vanguard, 2=Support)
        # Built on first reset and held fixed across episodes.
        self._role_map: dict = {}       # unit_id → role_index
        self._agent_ids: list = []      # role_index → unit_id (length = n_agents)
        self._enemy_ids: list = []      # enemy slot → unit_id

        self._step_count = 0
        self._prev_agent_pos: np.ndarray = np.zeros((self.n_agents, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if self._use_capability_wrapper:
            self._reset_until_valid_roles()
        elif self._manual_fixed_team:
            self._env.reset(self._build_fixed_episode_config())
            self._build_role_map()
        else:
            self._env.reset()
            self._build_role_map()

        self._step_count = 0

        obs = self._build_obs_all()
        self._prev_agent_pos = self._get_agent_positions()
        return obs, {}

    def step(self, actions):
        """
        Parameters
        ----------
        actions : list[int]  length = n_agents, our 6-action space

        Returns
        -------
        obs_list, rewards (np.ndarray), done (bool), truncated (bool), info (dict)
        """
        self._step_count += 1
        smac_actions = self._translate_actions(actions)

        reward_smac, terminated, info = self._env.step(smac_actions)
        won = bool(info.get('battle_won', False))
        done = terminated or self._step_count >= self.max_steps

        obs = self._build_obs_all()

        # Role-specific reward shaping
        cur_pos = self._get_agent_positions()
        individual = self._compute_shaped_rewards(cur_pos, actions)
        self._prev_agent_pos = cur_pos

        # Mix role-specific shaping with the shared SMACv2 team reward.
        rewards = (1 - self.alpha) * individual + self.alpha * float(reward_smac)

        info['won'] = won
        info['step'] = self._step_count
        return obs, rewards, done, False, info

    def close(self):
        self._env.close()

    def __del__(self):
        try:
            self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    def _build_role_map(self):
        """
        Assign unit_ids to roles by unit_type.
        Priority order: Stalker=Strike(0), Zealot=Vanguard(1), Colossus=Support(2).
        If a type has multiple units, take the first by unit_id.
        If a type is absent, fill sequentially.
        """
        agents = self._env.agents  # dict {unit_id: Unit}
        enemies = self._env.enemies

        buckets = {0: [], 1: [], 2: []}  # role_index → list of unit_ids
        unmatched = []
        for uid, unit in sorted(agents.items()):
            ut = getattr(unit, 'unit_type', -1)
            if ut == UNIT_TYPE_STALKER:
                buckets[0].append(uid)
            elif ut == UNIT_TYPE_ZEALOT:
                buckets[1].append(uid)
            elif ut == UNIT_TYPE_COLOSSUS:
                buckets[2].append(uid)
            else:
                unmatched.append(uid)

        # Fill missing roles from unmatched
        for role in range(3):
            if not buckets[role] and unmatched:
                buckets[role].append(unmatched.pop(0))

        self._role_map = {}
        self._agent_ids = []
        for role in range(self.n_agents):
            if buckets[role]:
                uid = buckets[role][0]
            else:
                # Fallback: any remaining unit
                uid = list(sorted(agents.keys()))[role % len(agents)]
            self._agent_ids.append(uid)
            self._role_map[uid] = role

        # Enemy slot assignment: sorted by unit_id, take first n_enemies
        self._enemy_ids = sorted(enemies.keys())[:self.n_enemies]

    def _reset_until_valid_roles(self):
        """Retry generated SMACv2 episodes until ally team contains all 3 role unit types."""
        last_seen = None
        for _ in range(MAX_RESET_RETRIES):
            self._env.reset()
            self._build_role_map()
            ally_types = {
                getattr(unit, 'unit_type', None)
                for unit in self._env.agents.values()
            }
            last_seen = sorted(ally_types)
            if all(unit_type in ally_types for unit_type in ROLE_UNIT_TYPES):
                return

        raise RuntimeError(
            "Failed to sample a valid ally composition containing Stalker, Zealot, "
            f"and Colossus after {MAX_RESET_RETRIES} resets. Last ally unit types: {last_seen}"
        )

    def _build_default_capability_config(self) -> dict:
        """Default SMACv2 capability config for generated 10gen Protoss scenarios."""
        map_x = int(self._env_info.get('map_x', DEFAULT_MAP_SIZE)) if hasattr(self, '_env_info') else DEFAULT_MAP_SIZE
        map_y = int(self._env_info.get('map_y', DEFAULT_MAP_SIZE)) if hasattr(self, '_env_info') else DEFAULT_MAP_SIZE
        return {
            "n_units": self.n_agents,
            "n_enemies": self.n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams",
                "n_units": self.n_agents,
                "n_enemies": self.n_enemies,
                "unit_types": ["stalker", "zealot", "colossus"],
                "weights": [1 / 3, 1 / 3, 1 / 3],
                "observe": True,
                "exception_unit_types": ["colossus"],
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": map_x,
                "map_y": map_y,
            },
        }

    def _build_fixed_episode_config(self) -> dict:
        """Build a deterministic team composition for generated 10gen Protoss maps."""
        reset_config = self._start_pos_distribution.generate()
        reset_config['team_gen'] = {
            'ally_team': FIXED_ALLY_TEAM.copy(),
            'enemy_team': FIXED_ENEMY_TEAM.copy(),
            'id': 0,
        }
        return reset_config

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_obs_all(self) -> list:
        return [self._build_obs(role_idx) for role_idx in range(self.n_agents)]

    def _build_obs(self, role_idx: int) -> np.ndarray:
        o = np.zeros(OBS_DIM, dtype=np.float32)
        agents = self._env.agents
        enemies = self._env.enemies

        uid = self._agent_ids[role_idx]
        unit = agents.get(uid)
        if unit is None or not unit.health:
            # Agent is dead — return zero obs
            return o

        pos = self._norm_pos(unit.pos.x, unit.pos.y)
        health = _safe_ratio(unit.health, unit.health_max)
        shield = _safe_ratio(unit.shield, unit.shield_max)
        role_oh = np.zeros(3, dtype=np.float32)
        role_oh[role_idx] = 1.0

        # self_tok [0:8]
        o[0] = pos[0]
        o[1] = pos[1]
        o[2] = health
        o[3] = shield
        o[4:7] = role_oh
        o[7] = 0.0  # padding

        # ally_tok [8:24]  — other 2 agents
        ally_roles = [r for r in range(self.n_agents) if r != role_idx]
        for slot, ar in enumerate(ally_roles):
            base = 8 + slot * 8
            auid = self._agent_ids[ar]
            aunit = agents.get(auid)
            if aunit is not None and aunit.health > 0:
                apos = self._norm_pos(aunit.pos.x, aunit.pos.y)
                ah = _safe_ratio(aunit.health, aunit.health_max)
                as_ = _safe_ratio(aunit.shield, aunit.shield_max)
                at_oh = np.zeros(3, dtype=np.float32)
                at_oh[ar] = 1.0
                o[base + 0] = apos[0] - pos[0]
                o[base + 1] = apos[1] - pos[1]
                o[base + 2] = ah
                o[base + 3] = as_
                o[base + 4:base + 7] = at_oh
                o[base + 7] = 1.0  # alive
            else:
                o[base + 7] = 0.0  # dead → all zeros, alive=0

        # enemy_tok [24:40]  — up to 2 enemies, by priority slot
        for slot in range(N_ENEMY_SLOTS):
            base = 24 + slot * 8
            if slot >= len(self._enemy_ids):
                o[58 + 2 + slot] = 1.0  # enemy_mask = padding
                continue
            euid = self._enemy_ids[slot]
            eunit = enemies.get(euid)
            if eunit is not None and eunit.health > 0:
                epos = self._norm_pos(eunit.pos.x, eunit.pos.y)
                eh = _safe_ratio(eunit.health, eunit.health_max)
                es = _safe_ratio(eunit.shield, eunit.shield_max)
                o[base + 0] = epos[0] - pos[0]
                o[base + 1] = epos[1] - pos[1]
                o[base + 2] = eh
                o[base + 3] = es
                o[base + 4:base + 7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
                o[base + 7] = 1.0  # alive
            else:
                o[base + 7] = 0.0

        # zone_tok [40:58]  — 3 fixed landmarks
        for z in range(N_ZONE_SLOTS):
            base = 40 + z * 6
            zpos = ZONE_POSITIONS_NORM[z]
            o[base + 0] = zpos[0] - pos[0]
            o[base + 1] = zpos[1] - pos[1]
            o[base + 2] = 0.0   # owned (no capture mechanic in SMACv2 base)
            o[base + 3] = 0.0   # contest
            o[base + 4] = 1.0   # value (uniform)
            o[base + 5] = 0.0   # padding

        # masks [58:65]  — 0=valid, 1=padding
        # ally/enemy/zone masks: set padding only for dead/absent slots
        # ally_mask [58:60]: already 0 (set above for dead allies via alive flag)
        # enemy_mask [60:62]
        for slot in range(N_ENEMY_SLOTS):
            if slot < len(self._enemy_ids):
                euid = self._enemy_ids[slot]
                eunit = enemies.get(euid)
                if eunit is None or eunit.health <= 0:
                    o[60 + slot] = 1.0  # padding (dead)
            else:
                o[60 + slot] = 1.0  # no unit in this slot
        # zone_mask [62:65]: always valid (fixed landmarks)
        o[62:65] = 0.0

        return o

    # ------------------------------------------------------------------
    # Action translation
    # ------------------------------------------------------------------

    def _translate_actions(self, actions: list) -> list:
        """Map our 6-action space to SMACv2 discrete actions."""
        avail = self._env.get_avail_actions()  # list of arrays per agent
        smac_actions = []

        for role_idx, act in enumerate(actions):
            agent_avail = avail[role_idx] if role_idx < len(avail) else None

            # Dead/inactive agents are only allowed to no-op.
            if agent_avail is not None and len(agent_avail) > ACTION_NOOP and agent_avail[ACTION_NOOP]:
                smac_actions.append(ACTION_NOOP)
                continue

            if act < 5:
                # idle or move
                smac_act = OUR_ACTION_TO_SMAC[act]
                # Fallback to stop if this action is unavailable for a live agent
                if agent_avail is not None and smac_act < len(agent_avail) and not agent_avail[smac_act]:
                    smac_act = ACTION_STOP if len(agent_avail) > ACTION_STOP and agent_avail[ACTION_STOP] else ACTION_NOOP
            else:
                # attack: find first available attack action
                smac_act = ACTION_STOP if agent_avail is not None and len(agent_avail) > ACTION_STOP and agent_avail[ACTION_STOP] else ACTION_NOOP
                if agent_avail is not None:
                    for a_idx in range(ATTACK_BASE, len(agent_avail)):
                        if agent_avail[a_idx]:
                            smac_act = a_idx
                            break

            smac_actions.append(smac_act)

        return smac_actions

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_shaped_rewards(self, cur_pos: np.ndarray, actions: list) -> np.ndarray:
        """
        Compute per-agent role-specific shaped rewards.
        Shared team reward is mixed in once at the end of step().
        """
        agents = self._env.agents
        enemies = self._env.enemies
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        alive_enemy_pos = []
        for euid in self._enemy_ids:
            eu = enemies.get(euid)
            if eu is not None and eu.health > 0:
                alive_enemy_pos.append(self._norm_pos(eu.pos.x, eu.pos.y))

        for role_idx in range(self.n_agents):
            uid = self._agent_ids[role_idx]
            unit = agents.get(uid)
            if unit is None or unit.health <= 0:
                rewards[role_idx] = -0.1
                continue

            pos = cur_pos[role_idx]
            prev = self._prev_agent_pos[role_idx]

            if role_idx == 0:  # Strike (Stalker) — ranged DPS
                if alive_enemy_pos:
                    nearest = min(alive_enemy_pos, key=lambda p: _dist(pos, p))
                    d = _dist(pos, nearest)
                    rewards[role_idx] += APPROACH_BONUS * max(0.0, 0.5 - d)
                    if d < 0.1:  # too close penalty
                        rewards[role_idx] -= TOO_CLOSE_PENALTY
                if int(actions[role_idx]) == 5:
                    rewards[role_idx] += ATTACK_BONUS

            elif role_idx == 1:  # Vanguard (Zealot) — melee tank
                if alive_enemy_pos:
                    nearest = min(alive_enemy_pos, key=lambda p: _dist(pos, p))
                    d = _dist(pos, nearest)
                    rewards[role_idx] += APPROACH_BONUS * max(0.0, 0.5 - d)
                    if d < 0.08:  # melee range
                        rewards[role_idx] += MELEE_BONUS
                if int(actions[role_idx]) == 5:
                    rewards[role_idx] += ATTACK_BONUS

            elif role_idx == 2:  # Support (Colossus) — area/positioning
                # Stay near lowest-HP ally
                alive_allies = [
                    r for r in range(self.n_agents)
                    if r != role_idx and agents.get(self._agent_ids[r]) is not None
                    and agents[self._agent_ids[r]].health > 0
                ]
                if alive_allies:
                    target = min(
                        alive_allies,
                        key=lambda r: _safe_ratio(
                            agents[self._agent_ids[r]].health,
                            agents[self._agent_ids[r]].health_max
                        )
                    )
                    tpos = cur_pos[target]
                    prev_d = _dist(prev, tpos)
                    curr_d = _dist(pos, tpos)
                    rewards[role_idx] += SUPPORT_PROXIMITY_BONUS * (prev_d - curr_d)
                # Splash coverage bonus: enemies within short range
                splash_count = sum(1 for ep in alive_enemy_pos if _dist(pos, ep) < 0.2)
                rewards[role_idx] += SPLASH_BONUS * splash_count

        return rewards

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _norm_pos(self, x: float, y: float) -> np.ndarray:
        return np.array([x / self._map_x, y / self._map_y], dtype=np.float32)

    def _get_agent_positions(self) -> np.ndarray:
        agents = self._env.agents
        pos = np.zeros((self.n_agents, 2), dtype=np.float32)
        for role_idx in range(self.n_agents):
            uid = self._agent_ids[role_idx]
            unit = agents.get(uid)
            if unit is not None:
                pos[role_idx] = self._norm_pos(unit.pos.x, unit.pos.y)
        return pos


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _safe_ratio(val, max_val) -> float:
    if max_val is None or max_val == 0:
        return 0.0
    return float(val) / float(max_val)
