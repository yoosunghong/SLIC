"""
SMACv2 environment wrapper — Terran composition.

Adapts StarCraft2Env to the same interface as CooperativeGridEnv:
  reset() → (obs_list, info)
  step(actions) → (obs_list, rewards, done, truncated, info)

Observation layout (65-dim, identical slot positions to MiniGrid PoC):
  self_tok   [0:8]   pos(2) + health(1) + pad(1) + unit_type_onehot(3) + pad(1)
  ally_tok   [8:24]  2 × 8: rel_pos(2)+health(1)+pad(1)+unit_type_onehot(3)+alive(1)
  enemy_tok  [24:40] 2 × 8: rel_pos(2)+health(1)+pad(1)+type_conf(3)+alive(1)
  zone_tok   [40:58] 3 × 6: rel_pos(2)+owned(1)+contest(1)+value(1)+pad(1)
  ally_mask  [58:60] 1=padding
  enemy_mask [60:62] 1=padding
  zone_mask  [62:65] 1=padding
  TOTAL: 65

Note: Terran units have no shield. The shield slot (index 3 in each 8-dim token)
is always 0.0. This preserves full compatibility with the validation_poc model
architecture (N_ENEMY_SLOTS=2, all slice constants unchanged).

Role mapping (Terran unit types):
  Strike   → Marine   (SC2 unit_type 48, ranged light infantry)
  Vanguard → Marauder (SC2 unit_type 51, frontline heavy infantry, slows enemies)
  Support  → Medivac  (SC2 unit_type 54, biological healer, no ground attack)

Ally team  : Marine × 1, Marauder × 1, Medivac × 1
Enemy team : Marine × 2, Marauder × 1

Enemy observation: 3 actual enemies, but only the 2 closest are shown in the
observation (slots 0–1). Masks are set accordingly. This keeps obs_dim=65 and
model weights fully compatible.

Action mapping (our 6 → SMACv2 discrete):
  0 → stop (1)
  1 → move N (2)
  2 → move S (3)
  3 → move E (4)
  4 → move W (5)
  5 → attack nearest enemy (first available attack action ≥ 6)
       For Medivac (no ground attack), falls back to stop.
"""
import os
import numpy as np

# SC2 is installed at a non-default path on this machine.
os.environ.setdefault("SC2PATH", "D:/StarCraft II")

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

# ---------------------------------------------------------------------------
# Terran unit type IDs (SC2 internal IDs)
# ---------------------------------------------------------------------------
UNIT_TYPE_MARINE   = 48   # Strike
UNIT_TYPE_MARAUDER = 51   # Vanguard
UNIT_TYPE_MEDIVAC  = 54   # Support

ROLE_UNIT_TYPES = [UNIT_TYPE_MARINE, UNIT_TYPE_MARAUDER, UNIT_TYPE_MEDIVAC]
ROLE_NAMES      = ['Strike', 'Vanguard', 'Support']
FIXED_ALLY_TEAM  = ['marine', 'marauder', 'medivac']
FIXED_ENEMY_TEAM = ['marine', 'marine', 'marauder']

# ---------------------------------------------------------------------------
# SMACv2 base action indices
# ---------------------------------------------------------------------------
ACTION_NOOP   = 0
ACTION_STOP   = 1
ACTION_MOVE_N = 2
ACTION_MOVE_S = 3
ACTION_MOVE_E = 4
ACTION_MOVE_W = 5
ATTACK_BASE   = 6   # attack_enemy_i = ATTACK_BASE + enemy_index

OUR_ACTION_TO_SMAC = [ACTION_STOP, ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_E, ACTION_MOVE_W]

# ---------------------------------------------------------------------------
# Observation constants (must stay in sync with validation_poc/models/*.py)
# ---------------------------------------------------------------------------
OBS_DIM        = 65
N_ALLY_SLOTS   = 2
N_ENEMY_SLOTS  = 2   # shown in obs; 2 closest enemies selected from 3 actual
N_ZONE_SLOTS   = 3

# Fixed landmark positions (normalized 0–1) used as zone_tok proxies.
ZONE_POSITIONS_NORM = np.array([
    [0.15, 0.15],   # top-left
    [0.85, 0.15],   # top-right
    [0.50, 0.50],   # center
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Reward shaping weights
# ---------------------------------------------------------------------------
APPROACH_BONUS          = 0.005   # all combat roles: reward for closing on enemies
ATTACK_BONUS            = 0.05    # all combat roles: reward for issuing attack action
FRONTLINE_BONUS         = 0.05    # Marauder: bonus for being within frontline range
TOO_CLOSE_PENALTY       = 0.02    # Marine: penalty for being inside melee range
SUPPORT_PROXIMITY_BONUS = 0.01    # Medivac: reward per unit of distance closed to target
HEAL_RANGE_BONUS        = 0.03    # Medivac: bonus for being within heal range of target
REAR_PENALTY            = 0.02    # Medivac: penalty for wandering into enemy contact
TEAM_ALPHA              = 0.2

# Normalized distance thresholds (32-tile map → divide by 32)
MARINE_ATTACK_RANGE  = 0.156   # 5 tiles / 32
MARINE_TOO_CLOSE     = 0.063   # 2 tiles / 32
MARAUDER_FRONT_RANGE = 0.188   # 6 tiles / 32 (Marauder attack range)
MEDIVAC_HEAL_RANGE   = 0.125   # 4 tiles / 32
ENEMY_DANGER_DIST    = 0.10    # Medivac rear penalty threshold

DEFAULT_MAP_SIZE  = 32
MAX_RESET_RETRIES = 20


def _safe_unit_type_onehot(unit_type: int) -> np.ndarray:
    """Return [1,0,0]=Marine, [0,1,0]=Marauder, [0,0,1]=Medivac, [0,0,0]=unknown."""
    oh = np.zeros(3, dtype=np.float32)
    if unit_type == UNIT_TYPE_MARINE:
        oh[0] = 1.0
    elif unit_type == UNIT_TYPE_MARAUDER:
        oh[1] = 1.0
    elif unit_type == UNIT_TYPE_MEDIVAC:
        oh[2] = 1.0
    return oh


class SMACWrapper:
    """
    Wraps StarCraft2Env into the CooperativeGridEnv interface.

    Ally team  : Marine (Strike) × 1, Marauder (Vanguard) × 1, Medivac (Support) × 1
    Enemy team : Marine × 2, Marauder × 1  (3 enemies total)

    The observation exposes only the 2 closest enemies to each agent, keeping
    obs_dim=65 and all model slice constants identical to the MiniGrid PoC.

    Parameters
    ----------
    cfg : dict
        Keys: map_name, n_agents (3), n_enemies (3), n_zones (3),
              max_steps (200), obs_dim (65), action_dim (6),
              team_reward_alpha (0.2), seed (optional).
    """

    def __init__(self, cfg: dict):
        if not _SMACV2_AVAILABLE:
            raise ImportError(
                "SMACv2 is not installed. Run:\n"
                "  pip install git+https://github.com/oxwhirl/smacv2.git pysc2\n"
                "and ensure StarCraftII is installed at ~/StarCraftII/."
            )

        self.n_agents  = cfg.get('n_agents', 3)
        self.n_enemies = cfg.get('n_enemies', 3)
        self.n_zones   = cfg.get('n_zones', 3)
        self.max_steps = cfg.get('max_steps', 200)
        self.obs_dim   = cfg.get('obs_dim', OBS_DIM)
        self.action_dim = cfg.get('action_dim', 6)
        self.alpha     = cfg.get('team_reward_alpha', TEAM_ALPHA)
        self.map_name  = cfg.get('map_name', '10gen_terran')

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
        self._map_x   = float(self._env_info.get('map_x', DEFAULT_MAP_SIZE))
        self._map_y   = float(self._env_info.get('map_y', DEFAULT_MAP_SIZE))

        self._role_map:   dict = {}   # unit_id → role_index
        self._agent_ids:  list = []   # role_index → unit_id
        self._enemy_ids:  list = []   # all enemy unit_ids (up to n_enemies)

        self._step_count = 0
        self._prev_agent_pos = np.zeros((self.n_agents, 2), dtype=np.float32)

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
        won  = bool(info.get('battle_won', False))
        done = terminated or self._step_count >= self.max_steps

        obs = self._build_obs_all()

        cur_pos    = self._get_agent_positions()
        individual = self._compute_shaped_rewards(cur_pos, actions)
        self._prev_agent_pos = cur_pos

        rewards = (1 - self.alpha) * individual + self.alpha * float(reward_smac)

        info['won']  = won
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
        Marine → role 0 (Strike), Marauder → role 1 (Vanguard), Medivac → role 2 (Support).
        Falls back to sequential assignment if a type is absent.
        """
        agents  = self._env.agents
        enemies = self._env.enemies

        buckets: dict = {0: [], 1: [], 2: []}
        unmatched: list = []
        for uid, unit in sorted(agents.items()):
            ut = getattr(unit, 'unit_type', -1)
            if ut == UNIT_TYPE_MARINE:
                buckets[0].append(uid)
            elif ut == UNIT_TYPE_MARAUDER:
                buckets[1].append(uid)
            elif ut == UNIT_TYPE_MEDIVAC:
                buckets[2].append(uid)
            else:
                unmatched.append(uid)

        for role in range(3):
            if not buckets[role] and unmatched:
                buckets[role].append(unmatched.pop(0))

        self._role_map  = {}
        self._agent_ids = []
        for role in range(self.n_agents):
            if buckets[role]:
                uid = buckets[role][0]
            else:
                uid = list(sorted(agents.keys()))[role % len(agents)]
            self._agent_ids.append(uid)
            self._role_map[uid] = role

        self._enemy_ids = sorted(enemies.keys())[:self.n_enemies]

    def _reset_until_valid_roles(self):
        """Retry generated episodes until ally team contains Marine, Marauder, and Medivac."""
        last_seen = None
        for _ in range(MAX_RESET_RETRIES):
            self._env.reset()
            self._build_role_map()
            ally_types = {
                getattr(unit, 'unit_type', None)
                for unit in self._env.agents.values()
            }
            last_seen = sorted(ally_types)
            if all(ut in ally_types for ut in ROLE_UNIT_TYPES):
                return
        raise RuntimeError(
            "Failed to sample a valid ally composition containing Marine, Marauder, "
            f"and Medivac after {MAX_RESET_RETRIES} resets. Last ally unit types: {last_seen}"
        )

    def _build_default_capability_config(self) -> dict:
        """Default SMACv2 capability config for generated 10gen Terran scenarios."""
        map_x = int(self._env_info.get('map_x', DEFAULT_MAP_SIZE)) if hasattr(self, '_env_info') else DEFAULT_MAP_SIZE
        map_y = int(self._env_info.get('map_y', DEFAULT_MAP_SIZE)) if hasattr(self, '_env_info') else DEFAULT_MAP_SIZE
        return {
            "n_units":   self.n_agents,
            "n_enemies": self.n_enemies,
            "team_gen": {
                "dist_type":            "weighted_teams",
                "n_units":              self.n_agents,
                "n_enemies":            self.n_enemies,
                "unit_types":           ["marine", "marauder", "medivac"],
                "weights":              [1 / 3, 1 / 3, 1 / 3],
                "observe":              True,
                "exception_unit_types": ["medivac"],   # Medivac is non-combat
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p":     0.5,
                "map_x": map_x,
                "map_y": map_y,
            },
        }

    def _build_fixed_episode_config(self) -> dict:
        """Build a deterministic Terran team composition for 10gen maps."""
        reset_config = self._start_pos_distribution.generate()
        reset_config['team_gen'] = {
            'ally_team':  FIXED_ALLY_TEAM.copy(),
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
        agents  = self._env.agents
        enemies = self._env.enemies

        uid  = self._agent_ids[role_idx]
        unit = agents.get(uid)
        if unit is None or not unit.health:
            return o   # dead agent → zero obs

        pos     = self._norm_pos(unit.pos.x, unit.pos.y)
        health  = _safe_ratio(unit.health, unit.health_max)
        role_oh = np.zeros(3, dtype=np.float32)
        role_oh[role_idx] = 1.0

        # self_tok [0:8]  — pos(2) + health(1) + pad(1) + role_onehot(3) + pad(1)
        o[0] = pos[0]
        o[1] = pos[1]
        o[2] = health
        o[3] = 0.0           # shield slot — always 0 for Terran
        o[4:7] = role_oh
        o[7] = 0.0           # padding

        # ally_tok [8:24]  — other 2 agents
        ally_roles = [r for r in range(self.n_agents) if r != role_idx]
        for slot, ar in enumerate(ally_roles):
            base  = 8 + slot * 8
            auid  = self._agent_ids[ar]
            aunit = agents.get(auid)
            if aunit is not None and aunit.health > 0:
                apos   = self._norm_pos(aunit.pos.x, aunit.pos.y)
                ah     = _safe_ratio(aunit.health, aunit.health_max)
                at_oh  = np.zeros(3, dtype=np.float32)
                at_oh[ar] = 1.0
                o[base + 0] = apos[0] - pos[0]
                o[base + 1] = apos[1] - pos[1]
                o[base + 2] = ah
                o[base + 3] = 0.0     # shield — always 0 for Terran
                o[base + 4:base + 7] = at_oh
                o[base + 7] = 1.0     # alive
            # dead ally: all zeros, alive=0

        # enemy_tok [24:40]  — 2 closest enemies from the 3 actual enemies
        obs_enemy_ids = self._closest_enemy_ids(pos)
        for slot, euid in enumerate(obs_enemy_ids):
            base  = 24 + slot * 8
            eunit = enemies.get(euid)
            if eunit is not None and eunit.health > 0:
                epos = self._norm_pos(eunit.pos.x, eunit.pos.y)
                eh   = _safe_ratio(eunit.health, eunit.health_max)
                o[base + 0] = epos[0] - pos[0]
                o[base + 1] = epos[1] - pos[1]
                o[base + 2] = eh
                o[base + 3] = 0.0   # shield slot (Terran enemies also have no shield)
                # type_conf: uniform since we don't expose enemy role to policy
                o[base + 4:base + 7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
                o[base + 7] = 1.0   # alive
            else:
                o[60 + slot] = 1.0  # enemy_mask: dead/absent slot

        # Fill mask for any unpopulated enemy slots
        for slot in range(len(obs_enemy_ids), N_ENEMY_SLOTS):
            o[60 + slot] = 1.0

        # zone_tok [40:58]  — 3 fixed landmarks
        for z in range(N_ZONE_SLOTS):
            base = 40 + z * 6
            zpos = ZONE_POSITIONS_NORM[z]
            o[base + 0] = zpos[0] - pos[0]
            o[base + 1] = zpos[1] - pos[1]
            o[base + 2] = 0.0   # owned
            o[base + 3] = 0.0   # contest
            o[base + 4] = 1.0   # value (uniform)
            o[base + 5] = 0.0   # padding

        # zone_mask [62:65]: always valid
        o[62:65] = 0.0

        return o

    def _closest_enemy_ids(self, pos: np.ndarray) -> list:
        """Return up to N_ENEMY_SLOTS enemy unit IDs sorted by distance to pos."""
        enemies = self._env.enemies
        alive = []
        for euid in self._enemy_ids:
            eu = enemies.get(euid)
            if eu is not None and eu.health > 0:
                epos = self._norm_pos(eu.pos.x, eu.pos.y)
                alive.append((euid, _dist(pos, epos)))
        alive.sort(key=lambda x: x[1])
        return [euid for euid, _ in alive[:N_ENEMY_SLOTS]]

    # ------------------------------------------------------------------
    # Action translation
    # ------------------------------------------------------------------

    def _translate_actions(self, actions: list) -> list:
        """Map our 6-action space to SMACv2 discrete actions."""
        avail = self._env.get_avail_actions()
        smac_actions = []

        for role_idx, act in enumerate(actions):
            agent_avail = avail[role_idx] if role_idx < len(avail) else None

            # Dead agents must no-op.
            if agent_avail is not None and len(agent_avail) > ACTION_NOOP and agent_avail[ACTION_NOOP]:
                smac_actions.append(ACTION_NOOP)
                continue

            if act < 5:
                smac_act = OUR_ACTION_TO_SMAC[act]
                if agent_avail is not None and smac_act < len(agent_avail) and not agent_avail[smac_act]:
                    smac_act = ACTION_STOP if (len(agent_avail) > ACTION_STOP and agent_avail[ACTION_STOP]) else ACTION_NOOP
            else:
                # Attack: find first available attack action.
                # Medivac has no ground attack — falls back to stop naturally.
                smac_act = ACTION_STOP if (agent_avail is not None and len(agent_avail) > ACTION_STOP and agent_avail[ACTION_STOP]) else ACTION_NOOP
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
        Per-agent role-specific shaped rewards.
        Team reward is mixed in at the call site in step().

        Role 0 — Strike (Marine):
            Ranged DPS. Reward approach to attack range; penalize entering melee.
        Role 1 — Vanguard (Marauder):
            Frontline. Reward closing to frontline range and attacking.
        Role 2 — Support (Medivac):
            Healer. Reward closing on the lowest-HP ally and being in heal range.
            SC2 heal is automatic — no explicit heal action needed.
        """
        agents  = self._env.agents
        enemies = self._env.enemies
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        alive_enemy_pos = []
        for euid in self._enemy_ids:
            eu = enemies.get(euid)
            if eu is not None and eu.health > 0:
                alive_enemy_pos.append(self._norm_pos(eu.pos.x, eu.pos.y))

        for role_idx in range(self.n_agents):
            uid  = self._agent_ids[role_idx]
            unit = agents.get(uid)
            if unit is None or unit.health <= 0:
                rewards[role_idx] = -0.1
                continue

            pos  = cur_pos[role_idx]
            prev = self._prev_agent_pos[role_idx]

            if role_idx == 0:
                # ---- Strike (Marine) — ranged DPS, maintain safe range ----
                if alive_enemy_pos:
                    nearest = min(alive_enemy_pos, key=lambda p: _dist(pos, p))
                    d = _dist(pos, nearest)
                    # Reward closing to within attack range
                    rewards[role_idx] += APPROACH_BONUS * max(0.0, MARINE_ATTACK_RANGE - d)
                    # Penalty for entering melee range (Marine should kite)
                    if d < MARINE_TOO_CLOSE:
                        rewards[role_idx] -= TOO_CLOSE_PENALTY
                if int(actions[role_idx]) == 5:
                    rewards[role_idx] += ATTACK_BONUS

            elif role_idx == 1:
                # ---- Vanguard (Marauder) — frontline, close in and attack ----
                if alive_enemy_pos:
                    nearest = min(alive_enemy_pos, key=lambda p: _dist(pos, p))
                    d = _dist(pos, nearest)
                    rewards[role_idx] += APPROACH_BONUS * max(0.0, MARAUDER_FRONT_RANGE - d)
                    # Extra bonus for being in frontline contact range
                    if d < MARAUDER_FRONT_RANGE:
                        rewards[role_idx] += FRONTLINE_BONUS
                if int(actions[role_idx]) == 5:
                    rewards[role_idx] += ATTACK_BONUS

            elif role_idx == 2:
                # ---- Support (Medivac) — healer, track lowest-HP ally ----
                alive_allies = [
                    r for r in range(self.n_agents)
                    if r != role_idx
                    and agents.get(self._agent_ids[r]) is not None
                    and agents[self._agent_ids[r]].health > 0
                ]
                if alive_allies:
                    target = min(
                        alive_allies,
                        key=lambda r: _safe_ratio(
                            agents[self._agent_ids[r]].health,
                            agents[self._agent_ids[r]].health_max,
                        )
                    )
                    tpos   = cur_pos[target]
                    prev_d = _dist(prev, tpos)
                    curr_d = _dist(pos, tpos)
                    # Reward for moving toward the target
                    rewards[role_idx] += SUPPORT_PROXIMITY_BONUS * max(0.0, prev_d - curr_d)
                    # Bonus for being within SC2 heal range
                    if curr_d < MEDIVAC_HEAL_RANGE:
                        rewards[role_idx] += HEAL_RANGE_BONUS

                # Penalty for getting too close to enemies (Medivac must stay rear)
                for ep in alive_enemy_pos:
                    if _dist(pos, ep) < ENEMY_DANGER_DIST:
                        rewards[role_idx] -= REAR_PENALTY
                        break   # one penalty per step regardless of enemy count

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
            uid  = self._agent_ids[role_idx]
            unit = agents.get(uid)
            if unit is not None:
                pos[role_idx] = self._norm_pos(unit.pos.x, unit.pos.y)
        return pos


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _safe_ratio(val, max_val) -> float:
    if max_val is None or max_val == 0:
        return 0.0
    return float(val) / float(max_val)
