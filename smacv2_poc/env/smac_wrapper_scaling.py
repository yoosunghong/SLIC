"""
SMACv2 environment wrapper — SCALING study version.

Extends smac_wrapper.py to support:
  - Arbitrary team size (n_agents up to 9)
  - Arbitrary role composition (e.g. 3 Strike / 2 Vanguard / 1 Support)
  - Pad-to-max observation (single obs_dim=182 across all configs)

Observation layout (182-dim, padded to the 9v9 ceiling):
  self_tok   [0:8]      pos(2)+health(1)+pad(1)+role_onehot(3)+pad(1)
  ally_tok   [8:72]     8 slots × 8: rel_pos(2)+health(1)+pad(1)+role_onehot(3)+alive(1)
  enemy_tok  [72:144]   9 slots × 8: rel_pos(2)+health(1)+pad(1)+type_conf(3)+alive(1)
  zone_tok   [144:162]  3 slots × 6: rel_pos(2)+owned(1)+contest(1)+value(1)+pad(1)
  ally_mask  [162:170]  8  (1 = padded/dead)
  enemy_mask [170:179]  9
  zone_mask  [179:182]  3
  TOTAL: 182

For team sizes below the ceiling, unused slots are zero-filled and masked.
All attention masks correctly exclude padded slots; _safe_mask() in the
model still forces slot-0 unmask if a whole set is empty.

Role mapping (Terran unit types) — identical to smac_wrapper.py:
  Strike   → Marine   (unit_type 48)
  Vanguard → Marauder (unit_type 51)
  Support  → Medivac  (unit_type 54)

Composition is provided via cfg['composition'] = {'Strike': k0, 'Vanguard': k1, 'Support': k2}.
Enemy team mirrors ally composition by default for symmetric matchups.
"""
import os
import numpy as np

os.environ.setdefault("SC2PATH", "C:/Program Files (x86)/StarCraft II")

try:
    from smacv2.env import StarCraft2Env
    from smacv2.env.starcraft2.distributions import get_distribution
    _SMACV2_AVAILABLE = True
except ImportError:
    _SMACV2_AVAILABLE = False
    StarCraft2Env = None
    get_distribution = None

# ---------------------------------------------------------------------------
# Unit type IDs (SC2)
# ---------------------------------------------------------------------------
UNIT_TYPE_MARINE   = 48
UNIT_TYPE_MARAUDER = 51
UNIT_TYPE_MEDIVAC  = 54

ROLE_UNIT_TYPES = [UNIT_TYPE_MARINE, UNIT_TYPE_MARAUDER, UNIT_TYPE_MEDIVAC]
ROLE_NAMES      = ['Strike', 'Vanguard', 'Support']
ROLE_UNIT_STR   = ['marine', 'marauder', 'medivac']

# ---------------------------------------------------------------------------
# Action IDs
# ---------------------------------------------------------------------------
ACTION_NOOP   = 0
ACTION_STOP   = 1
ACTION_MOVE_N = 2
ACTION_MOVE_S = 3
ACTION_MOVE_E = 4
ACTION_MOVE_W = 5
ATTACK_BASE   = 6
OUR_ACTION_TO_SMAC = [ACTION_STOP, ACTION_MOVE_N, ACTION_MOVE_S, ACTION_MOVE_E, ACTION_MOVE_W]

# ---------------------------------------------------------------------------
# Padded observation ceiling (must match scaling models)
# ---------------------------------------------------------------------------
MAX_ALLY_SLOTS  = 8     # 9v9 → 8 other allies per agent
MAX_ENEMY_SLOTS = 9
N_ZONE_SLOTS    = 3

SELF_DIM  = 8
ALLY_DIM  = 8
ENEMY_DIM = 8
ZONE_DIM  = 6

_SELF_END   = SELF_DIM                                                   # 8
_ALLY_END   = _SELF_END  + MAX_ALLY_SLOTS  * ALLY_DIM                    # 72
_ENEMY_END  = _ALLY_END  + MAX_ENEMY_SLOTS * ENEMY_DIM                   # 144
_ZONE_END   = _ENEMY_END + N_ZONE_SLOTS    * ZONE_DIM                    # 162
_AMASK_END  = _ZONE_END  + MAX_ALLY_SLOTS                                # 170
_EMASK_END  = _AMASK_END + MAX_ENEMY_SLOTS                               # 179
OBS_DIM     = _EMASK_END + N_ZONE_SLOTS                                  # 182

ZONE_POSITIONS_NORM = np.array([
    [0.15, 0.15],
    [0.85, 0.15],
    [0.50, 0.50],
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Reward shaping (identical to smac_wrapper.py)
# ---------------------------------------------------------------------------
APPROACH_BONUS          = 0.02
ATTACK_BONUS            = 0.05
FRONTLINE_BONUS         = 0.05
TOO_CLOSE_PENALTY       = 0.02
SUPPORT_PROXIMITY_BONUS = 0.01
HEAL_RANGE_BONUS        = 0.05
REAR_PENALTY            = 0.02
TEAM_ALPHA              = 0.2

MARINE_ATTACK_RANGE  = 0.156
MARINE_TOO_CLOSE     = 0.063
MARAUDER_FRONT_RANGE = 0.188
MEDIVAC_HEAL_RANGE   = 0.125
ENEMY_DANGER_DIST    = 0.10

DEFAULT_MAP_SIZE  = 32
MAX_RESET_RETRIES = 30
MAX_ALLY_SPREAD   = 0.30   # slightly looser for larger teams


class ScalingSMACWrapper:
    """
    Parametrized SMACv2 wrapper for the scaling study.

    cfg keys:
      map_name            (default '10gen_terran')
      composition         dict {'Strike': k0, 'Vanguard': k1, 'Support': k2}
      n_enemies           (default: sum of composition counts = mirror)
      enemy_composition   optional dict; if absent mirrors ally composition
      max_steps           (default 200)
      team_reward_alpha   (default 0.2)
      action_dim          (default 6)
      seed                optional
    """

    def __init__(self, cfg: dict):
        if not _SMACV2_AVAILABLE:
            raise ImportError("SMACv2 not installed.")

        composition = cfg['composition']
        self.composition = {r: int(composition.get(r, 0)) for r in ROLE_NAMES}
        self.n_agents = sum(self.composition.values())
        if self.n_agents == 0:
            raise ValueError("composition sums to 0 agents")
        if self.n_agents > MAX_ALLY_SLOTS + 1:
            raise ValueError(f"n_agents={self.n_agents} exceeds ceiling {MAX_ALLY_SLOTS+1}")

        enemy_comp = cfg.get('enemy_composition', self.composition)
        self.enemy_composition = {r: int(enemy_comp.get(r, 0)) for r in ROLE_NAMES}
        self.n_enemies = int(cfg.get('n_enemies', sum(self.enemy_composition.values())))
        if self.n_enemies > MAX_ENEMY_SLOTS:
            raise ValueError(f"n_enemies={self.n_enemies} exceeds ceiling {MAX_ENEMY_SLOTS}")

        self.max_steps  = cfg.get('max_steps', 200)
        self.obs_dim    = OBS_DIM
        self.action_dim = cfg.get('action_dim', 6)
        self.alpha      = cfg.get('team_reward_alpha', TEAM_ALPHA)
        self.map_name   = cfg.get('map_name', '10gen_terran')

        # Build the fixed ally/enemy unit lists in role order [Strike..., Vanguard..., Support...]
        self._ally_team_str  = self._comp_to_team_list(self.composition)
        self._enemy_team_str = self._comp_to_team_list(self.enemy_composition)
        # agent_roles[i] = role_idx for unit i in team list order
        self._agent_roles = self._comp_to_role_list(self.composition)

        self._capability_config = self._build_capability_config()
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
            'n_units':   self.n_agents,
            'n_enemies': self.n_enemies,
        })

        self._env_info = self._env.get_env_info()
        self._map_x = float(self._env_info.get('map_x', DEFAULT_MAP_SIZE))
        self._map_y = float(self._env_info.get('map_y', DEFAULT_MAP_SIZE))

        self._agent_ids: list = []         # index i → unit_id for agent i (role = self._agent_roles[i])
        self._enemy_ids: list = []
        self._step_count = 0
        self._prev_agent_pos = np.zeros((self.n_agents, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _comp_to_team_list(comp: dict) -> list:
        team = []
        for role_idx, name in enumerate(ROLE_NAMES):
            team.extend([ROLE_UNIT_STR[role_idx]] * comp.get(name, 0))
        return team

    @staticmethod
    def _comp_to_role_list(comp: dict) -> list:
        roles = []
        for role_idx, name in enumerate(ROLE_NAMES):
            roles.extend([role_idx] * comp.get(name, 0))
        return roles

    def _build_capability_config(self) -> dict:
        return {
            "n_units":   self.n_agents,
            "n_enemies": self.n_enemies,
            "team_gen": {
                "dist_type":            "weighted_teams",
                "n_units":              self.n_agents,
                "n_enemies":            self.n_enemies,
                "unit_types":           ["marine", "marauder", "medivac"],
                "weights":              [1/3, 1/3, 1/3],
                "observe":              True,
                "exception_unit_types": ["medivac"],
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p":     0.5,
                "map_x": DEFAULT_MAP_SIZE,
                "map_y": DEFAULT_MAP_SIZE,
            },
        }

    def _build_fixed_episode_config(self) -> dict:
        rc = self._start_pos_distribution.generate()
        rc['team_gen'] = {
            'ally_team':  self._ally_team_str.copy(),
            'enemy_team': self._enemy_team_str.copy(),
            'id': 0,
        }
        return rc

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        for _ in range(MAX_RESET_RETRIES):
            self._env.reset(self._build_fixed_episode_config())
            self._build_id_maps()
            if self._ally_spread() <= MAX_ALLY_SPREAD:
                break
        self._step_count = 0
        self._prev_agent_pos = self._get_agent_positions()
        return self._build_obs_all(), {}

    def step(self, actions):
        self._step_count += 1
        smac_actions = self._translate_actions(actions)
        reward_smac, terminated, info = self._env.step(smac_actions)
        won  = bool(info.get('battle_won', False))
        done = terminated or self._step_count >= self.max_steps

        obs = self._build_obs_all()
        cur_pos = self._get_agent_positions()
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
    def _build_id_maps(self):
        """
        Assign SC2 unit_ids to slots so that _agent_ids[i] has role = _agent_roles[i].

        SMACv2's ally_team is given in order, but internal unit_ids are sorted.
        We bucket unit_ids by unit_type and assign in role order.
        """
        agents = self._env.agents
        buckets = {0: [], 1: [], 2: []}
        for uid, unit in sorted(agents.items()):
            ut = getattr(unit, 'unit_type', -1)
            if ut == UNIT_TYPE_MARINE:
                buckets[0].append(uid)
            elif ut == UNIT_TYPE_MARAUDER:
                buckets[1].append(uid)
            elif ut == UNIT_TYPE_MEDIVAC:
                buckets[2].append(uid)

        self._agent_ids = []
        cursor = {0: 0, 1: 0, 2: 0}
        for role_idx in self._agent_roles:
            b = buckets[role_idx]
            if cursor[role_idx] < len(b):
                self._agent_ids.append(b[cursor[role_idx]])
                cursor[role_idx] += 1
            else:
                # Fallback: composition mismatch — reuse any available unit
                leftovers = [u for u in agents.keys() if u not in self._agent_ids]
                self._agent_ids.append(leftovers[0] if leftovers else list(agents.keys())[0])

        self._enemy_ids = sorted(self._env.enemies.keys())[:self.n_enemies]

    def _ally_spread(self) -> float:
        positions = []
        for uid in self._agent_ids:
            u = self._env.agents.get(uid)
            if u is not None and u.health > 0:
                positions.append(self._norm_pos(u.pos.x, u.pos.y))
        if len(positions) < 2:
            return 0.0
        max_d = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                max_d = max(max_d, _dist(positions[i], positions[j]))
        return max_d

    # ------------------------------------------------------------------
    def _build_obs_all(self) -> list:
        return [self._build_obs(i) for i in range(self.n_agents)]

    def _build_obs(self, agent_idx: int) -> np.ndarray:
        o = np.zeros(OBS_DIM, dtype=np.float32)
        agents, enemies = self._env.agents, self._env.enemies
        role_idx = self._agent_roles[agent_idx]

        # Init all mask slots to 1 (padded); flip to 0 for valid slots below.
        o[_ZONE_END:_AMASK_END] = 1.0   # ally_mask
        o[_AMASK_END:_EMASK_END] = 1.0  # enemy_mask
        # zone_mask stays 0 (always valid)

        uid = self._agent_ids[agent_idx]
        unit = agents.get(uid)
        if unit is None or not unit.health:
            return o

        pos = self._norm_pos(unit.pos.x, unit.pos.y)
        health = _safe_ratio(unit.health, unit.health_max)
        role_oh = np.zeros(3, dtype=np.float32); role_oh[role_idx] = 1.0

        # self_tok
        o[0], o[1], o[2], o[3] = pos[0], pos[1], health, 0.0
        o[4:7] = role_oh
        o[7] = 0.0

        # ally_tok — all other agents, in agent_idx order
        slot = 0
        for other in range(self.n_agents):
            if other == agent_idx:
                continue
            if slot >= MAX_ALLY_SLOTS:
                break
            base = _SELF_END + slot * ALLY_DIM
            auid = self._agent_ids[other]
            aunit = agents.get(auid)
            if aunit is not None and aunit.health > 0:
                apos = self._norm_pos(aunit.pos.x, aunit.pos.y)
                ah = _safe_ratio(aunit.health, aunit.health_max)
                ar = self._agent_roles[other]
                oh = np.zeros(3, dtype=np.float32); oh[ar] = 1.0
                o[base+0] = apos[0] - pos[0]
                o[base+1] = apos[1] - pos[1]
                o[base+2] = ah
                o[base+3] = 0.0
                o[base+4:base+7] = oh
                o[base+7] = 1.0
                o[_ZONE_END + slot] = 0.0  # unmask
            # dead/absent ally: obs zero, mask stays 1
            slot += 1

        # enemy_tok — closest MAX_ENEMY_SLOTS enemies
        obs_enemy_ids = self._closest_enemy_ids(pos, MAX_ENEMY_SLOTS)
        for s, euid in enumerate(obs_enemy_ids):
            base = _ALLY_END + s * ENEMY_DIM
            eu = enemies.get(euid)
            if eu is not None and eu.health > 0:
                epos = self._norm_pos(eu.pos.x, eu.pos.y)
                eh = _safe_ratio(eu.health, eu.health_max)
                o[base+0] = epos[0] - pos[0]
                o[base+1] = epos[1] - pos[1]
                o[base+2] = eh
                o[base+3] = 0.0
                o[base+4:base+7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
                o[base+7] = 1.0
                o[_AMASK_END + s] = 0.0  # unmask

        # zone_tok
        for z in range(N_ZONE_SLOTS):
            base = _ENEMY_END + z * ZONE_DIM
            zpos = ZONE_POSITIONS_NORM[z]
            o[base+0] = zpos[0] - pos[0]
            o[base+1] = zpos[1] - pos[1]
            o[base+2] = 0.0
            o[base+3] = 0.0
            o[base+4] = 1.0
            o[base+5] = 0.0

        return o

    def _closest_enemy_ids(self, pos: np.ndarray, k: int) -> list:
        enemies = self._env.enemies
        alive = []
        for euid in self._enemy_ids:
            eu = enemies.get(euid)
            if eu is not None and eu.health > 0:
                ep = self._norm_pos(eu.pos.x, eu.pos.y)
                alive.append((euid, _dist(pos, ep)))
        alive.sort(key=lambda x: x[1])
        return [euid for euid, _ in alive[:k]]

    # ------------------------------------------------------------------
    def _translate_actions(self, actions: list) -> list:
        avail = self._env.get_avail_actions()
        out = []
        for i, act in enumerate(actions):
            aa = avail[i] if i < len(avail) else None
            if aa is not None and len(aa) > ACTION_NOOP and aa[ACTION_NOOP]:
                out.append(ACTION_NOOP); continue
            if act < 5:
                smac = OUR_ACTION_TO_SMAC[act]
                if aa is not None and smac < len(aa) and not aa[smac]:
                    smac = ACTION_STOP if (len(aa) > ACTION_STOP and aa[ACTION_STOP]) else ACTION_NOOP
            else:
                smac = ACTION_STOP if (aa is not None and len(aa) > ACTION_STOP and aa[ACTION_STOP]) else ACTION_NOOP
                if aa is not None:
                    for a_idx in range(ATTACK_BASE, len(aa)):
                        if aa[a_idx]:
                            smac = a_idx; break
            out.append(smac)
        return out

    # ------------------------------------------------------------------
    def _compute_shaped_rewards(self, cur_pos: np.ndarray, actions: list) -> np.ndarray:
        agents, enemies = self._env.agents, self._env.enemies
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        alive_enemy_pos = []
        for euid in self._enemy_ids:
            eu = enemies.get(euid)
            if eu is not None and eu.health > 0:
                alive_enemy_pos.append(self._norm_pos(eu.pos.x, eu.pos.y))

        for i in range(self.n_agents):
            uid = self._agent_ids[i]
            unit = agents.get(uid)
            if unit is None or unit.health <= 0:
                rewards[i] = -0.1
                continue
            pos  = cur_pos[i]
            prev = self._prev_agent_pos[i]
            role = self._agent_roles[i]

            if role == 0:
                # Strike (Marine) — ranged DPS
                if alive_enemy_pos:
                    near = min(alive_enemy_pos, key=lambda p: _dist(pos, p))
                    d = _dist(pos, near)
                    rewards[i] += APPROACH_BONUS * max(0.0, MARINE_ATTACK_RANGE - d)
                    if d < MARINE_TOO_CLOSE:
                        rewards[i] -= TOO_CLOSE_PENALTY
                if int(actions[i]) == 5 and alive_enemy_pos:
                    nd = min(_dist(pos, ep) for ep in alive_enemy_pos)
                    if nd <= MARINE_ATTACK_RANGE:
                        rewards[i] += ATTACK_BONUS
            elif role == 1:
                # Vanguard (Marauder) — frontline
                if alive_enemy_pos:
                    near = min(alive_enemy_pos, key=lambda p: _dist(pos, p))
                    d = _dist(pos, near)
                    rewards[i] += APPROACH_BONUS * max(0.0, MARAUDER_FRONT_RANGE - d)
                    if d < MARAUDER_FRONT_RANGE:
                        rewards[i] += FRONTLINE_BONUS
                if int(actions[i]) == 5 and alive_enemy_pos:
                    nd = min(_dist(pos, ep) for ep in alive_enemy_pos)
                    if nd <= MARAUDER_FRONT_RANGE:
                        rewards[i] += ATTACK_BONUS
            else:
                # Support (Medivac) — heal the lowest-HP non-Support ally
                candidates = [
                    j for j in range(self.n_agents)
                    if j != i
                    and self._agent_roles[j] != 2
                    and agents.get(self._agent_ids[j]) is not None
                    and agents[self._agent_ids[j]].health > 0
                ]
                # Fallback: if no non-Support ally alive, any ally
                if not candidates:
                    candidates = [
                        j for j in range(self.n_agents)
                        if j != i
                        and agents.get(self._agent_ids[j]) is not None
                        and agents[self._agent_ids[j]].health > 0
                    ]
                if candidates:
                    target = min(
                        candidates,
                        key=lambda j: _safe_ratio(
                            agents[self._agent_ids[j]].health,
                            agents[self._agent_ids[j]].health_max,
                        )
                    )
                    tpos = cur_pos[target]
                    prev_d = _dist(prev, tpos)
                    curr_d = _dist(pos,  tpos)
                    rewards[i] += SUPPORT_PROXIMITY_BONUS * max(0.0, prev_d - curr_d)
                    if curr_d < MEDIVAC_HEAL_RANGE:
                        rewards[i] += HEAL_RANGE_BONUS
                for ep in alive_enemy_pos:
                    if _dist(pos, ep) < ENEMY_DANGER_DIST:
                        rewards[i] -= REAR_PENALTY
                        break
        return rewards

    # ------------------------------------------------------------------
    def _norm_pos(self, x, y) -> np.ndarray:
        return np.array([x / self._map_x, y / self._map_y], dtype=np.float32)

    def _get_agent_positions(self) -> np.ndarray:
        pos = np.zeros((self.n_agents, 2), dtype=np.float32)
        for i in range(self.n_agents):
            u = self._env.agents.get(self._agent_ids[i])
            if u is not None:
                pos[i] = self._norm_pos(u.pos.x, u.pos.y)
        return pos


def _dist(a, b) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _safe_ratio(v, m) -> float:
    if m is None or m == 0:
        return 0.0
    return float(v) / float(m)
