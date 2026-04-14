"""
Cooperative 3-role grid environment.
15×15 grid, 3 agents (Strike/Vanguard/Support), 2 enemies, 3 capture zones.

Observation per agent: 65-dim flat vector
  self_tok    [0:8]   pos(2) + vel(2) + health(1) + role_onehot(3)
  ally_tok    [8:24]  2 allies × 8: rel_pos(2)+health(1)+alive(1)+role_onehot(3)+pad(1)
  enemy_tok  [24:40]  2 enemies × 8: rel_pos(2)+health(1)+alive(1)+conf(3)+pad(1)
  zone_tok   [40:58]  3 zones × 6: rel_pos(2)+occupancy(1)+progress(1)+assigned(1)+value(1)
  ally_mask  [58:60]  1=padding (2 ally slots)
  enemy_mask [60:62]  1=padding (2 enemy slots)
  zone_mask  [62:65]  1=padding (3 zone slots)

Actions (6): 0=up 1=down 2=left 3=right 4=stay 5=special
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROLE_STRIKE = 0
ROLE_VANGUARD = 1
ROLE_SUPPORT = 2
ROLE_NAMES = ['Strike', 'Vanguard', 'Support']

# Zone positions (row, col) on 15×15 grid
ZONE_CENTERS = [(2, 2), (2, 12), (7, 7)]  # top-left, top-right, center
ZONE_RADIUS = 1  # Manhattan radius for capture

# Action deltas: up, down, left, right, stay, special
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]

MAX_HEALTH = 10.0
ATTACK_DAMAGE = 2.0
HEAL_AMOUNT = 2.0
ENEMY_DAMAGE = 1.0


class CooperativeGridEnv(gym.Env):
    """Custom 3-role cooperative grid environment."""

    metadata = {'render_modes': []}

    def __init__(self, cfg: dict):
        super().__init__()
        self.grid_size = cfg.get('grid_size', 15)
        self.n_agents = cfg.get('n_agents', 3)
        self.n_enemies = cfg.get('n_enemies', 2)
        self.n_zones = cfg.get('n_zones', 3)
        self.max_steps = cfg.get('max_steps', 200)
        self.win_hold_steps = cfg.get('win_hold_steps', 20)
        self.alpha = cfg.get('team_reward_alpha', 0.2)
        self.obs_dim = cfg.get('obs_dim', 65)
        self.action_dim = cfg.get('action_dim', 6)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)

        self.zone_centers = [np.array(z, dtype=np.float32) for z in ZONE_CENTERS]
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        self.hold_count = 0  # consecutive steps team holds 2+ zones

        # Agent positions (row, col), one per role
        self.agent_pos = self._rng.integers(4, self.grid_size - 4, size=(self.n_agents, 2)).astype(np.float32)
        self.agent_vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_health = np.full(self.n_agents, MAX_HEALTH, dtype=np.float32)
        self.agent_alive = np.ones(self.n_agents, dtype=bool)

        # Enemy positions — start in bottom-right quadrant
        self.enemy_pos = self._rng.integers(self.grid_size - 6, self.grid_size - 1, size=(self.n_enemies, 2)).astype(np.float32)
        self.enemy_health = np.full(self.n_enemies, MAX_HEALTH, dtype=np.float32)
        self.enemy_alive = np.ones(self.n_enemies, dtype=bool)

        # Zone state
        self.zone_progress = np.zeros(self.n_zones, dtype=np.float32)
        self.zone_captured = np.zeros(self.n_zones, dtype=bool)

        # Support tracks lowest-HP ally
        self.support_target = 0  # ally index (excluding self)

        obs = [self._get_obs(i) for i in range(self.n_agents)]
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, actions):
        assert len(actions) == self.n_agents
        self.step_count += 1

        prev_pos = self.agent_pos.copy()
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Move agents
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                continue
            delta = ACTION_DELTAS[int(actions[i])]
            new_pos = self.agent_pos[i] + np.array(delta, dtype=np.float32)
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            vel = new_pos - self.agent_pos[i]
            self.agent_vel[i] = vel
            self.agent_pos[i] = new_pos

        # Move enemies (simple chase nearest alive agent)
        self._step_enemies()

        # Special action effects
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                continue
            if int(actions[i]) == 5:
                if i == ROLE_SUPPORT:
                    self._do_heal(i)
                else:
                    self._do_attack(i)

        # Enemy attacks adjacent agents
        for e in range(self.n_enemies):
            if not self.enemy_alive[e]:
                continue
            for i in range(self.n_agents):
                if not self.agent_alive[i]:
                    continue
                dist = np.sum(np.abs(self.agent_pos[i] - self.enemy_pos[e]))
                if dist <= 1:
                    self.agent_health[i] -= ENEMY_DAMAGE
                    if self.agent_health[i] <= 0:
                        self.agent_health[i] = 0
                        self.agent_alive[i] = False

        # Update zone progress
        self._update_zones()

        # Compute individual rewards
        rewards = self._compute_rewards(prev_pos, actions)

        # Team reward mixing
        team_avg = rewards.mean()
        rewards = (1 - self.alpha) * rewards + self.alpha * team_avg

        # Check win condition: hold 2+ zones for win_hold_steps
        zones_held = int(np.sum(self.zone_captured))
        if zones_held >= 2:
            self.hold_count += 1
        else:
            self.hold_count = 0

        won = self.hold_count >= self.win_hold_steps
        done = won or self.step_count >= self.max_steps or not np.any(self.agent_alive)

        obs = [self._get_obs(i) for i in range(self.n_agents)]
        info = {'won': won, 'zones_held': zones_held, 'step': self.step_count}
        return obs, rewards, done, False, info

    # ------------------------------------------------------------------
    def _step_enemies(self):
        for e in range(self.n_enemies):
            if not self.enemy_alive[e]:
                continue
            alive_agents = [i for i in range(self.n_agents) if self.agent_alive[i]]
            if not alive_agents:
                continue
            # Chase nearest
            dists = [np.sum(np.abs(self.agent_pos[i] - self.enemy_pos[e])) for i in alive_agents]
            target = alive_agents[int(np.argmin(dists))]
            diff = self.agent_pos[target] - self.enemy_pos[e]
            if self._rng.random() < 0.7:  # 70% chase, 30% random
                if abs(diff[0]) >= abs(diff[1]):
                    move = np.array([np.sign(diff[0]), 0], dtype=np.float32)
                else:
                    move = np.array([0, np.sign(diff[1])], dtype=np.float32)
            else:
                move = self._rng.choice([[-1,0],[1,0],[0,-1],[0,1],[0,0]]).astype(np.float32)
            new_pos = np.clip(self.enemy_pos[e] + move, 0, self.grid_size - 1)
            self.enemy_pos[e] = new_pos

    def _do_attack(self, agent_idx: int):
        pos = self.agent_pos[agent_idx]
        attack_range = 2 if agent_idx == ROLE_STRIKE else 1
        for e in range(self.n_enemies):
            if not self.enemy_alive[e]:
                continue
            dist = np.sum(np.abs(pos - self.enemy_pos[e]))
            if dist <= attack_range:
                self.enemy_health[e] -= ATTACK_DAMAGE
                if self.enemy_health[e] <= 0:
                    self.enemy_health[e] = 0
                    self.enemy_alive[e] = False

    def _do_heal(self, agent_idx: int):
        alive = [i for i in range(self.n_agents) if self.agent_alive[i]]
        if not alive:
            return
        target = min(alive, key=lambda i: self.agent_health[i])
        dist = np.sum(np.abs(self.agent_pos[agent_idx] - self.agent_pos[target]))
        if dist <= 2:
            self.agent_health[target] = min(MAX_HEALTH, self.agent_health[target] + HEAL_AMOUNT)

    def _update_zones(self):
        for z in range(self.n_zones):
            zc = self.zone_centers[z]
            # Count alive agents in zone
            agents_in = sum(
                1 for i in range(self.n_agents)
                if self.agent_alive[i] and np.sum(np.abs(self.agent_pos[i] - zc)) <= ZONE_RADIUS
            )
            if agents_in > 0:
                self.zone_progress[z] = min(1.0, self.zone_progress[z] + 0.05 * agents_in)
            else:
                self.zone_progress[z] = max(0.0, self.zone_progress[z] - 0.02)
            self.zone_captured[z] = self.zone_progress[z] >= 1.0

    def _compute_rewards(self, prev_pos, actions) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            if not self.agent_alive[i]:
                rewards[i] = -0.1
                continue
            pos = self.agent_pos[i]
            role = i  # agent index == role index

            # Zone approach bonus: reward moving closer to nearest uncaptured zone
            uncaptured = [z for z in range(self.n_zones) if not self.zone_captured[z]]
            if uncaptured:
                nearest_z = min(uncaptured, key=lambda z: np.sum(np.abs(pos - self.zone_centers[z])))
                prev_dist = np.sum(np.abs(prev_pos[i] - self.zone_centers[nearest_z]))
                curr_dist = np.sum(np.abs(pos - self.zone_centers[nearest_z]))
                rewards[i] += 0.01 * (prev_dist - curr_dist)

            # Capture bonus
            for z in range(self.n_zones):
                in_zone = np.sum(np.abs(pos - self.zone_centers[z])) <= ZONE_RADIUS
                if in_zone:
                    rewards[i] += 0.05 * self.zone_progress[z]

            if role == ROLE_STRIKE:
                # Approach bonus to nearest enemy
                alive_e = [e for e in range(self.n_enemies) if self.enemy_alive[e]]
                if alive_e:
                    nearest_e = min(alive_e, key=lambda e: np.sum(np.abs(pos - self.enemy_pos[e])))
                    rewards[i] += 0.005 * max(0, 5 - np.sum(np.abs(pos - self.enemy_pos[nearest_e])))
                # Too-close penalty (dist < 2, wants range)
                if alive_e:
                    min_dist = min(np.sum(np.abs(pos - self.enemy_pos[e])) for e in alive_e)
                    if min_dist < 2:
                        rewards[i] -= 0.02

            elif role == ROLE_VANGUARD:
                # Melee bonus: dist < 2 while in zone
                alive_e = [e for e in range(self.n_enemies) if self.enemy_alive[e]]
                in_any_zone = any(np.sum(np.abs(pos - self.zone_centers[z])) <= ZONE_RADIUS for z in range(self.n_zones))
                if in_any_zone and alive_e:
                    min_dist = min(np.sum(np.abs(pos - self.enemy_pos[e])) for e in alive_e)
                    if min_dist < 2:
                        rewards[i] += 0.05

            elif role == ROLE_SUPPORT:
                # Track lowest-HP ally
                alive_allies = [j for j in range(self.n_agents) if j != i and self.agent_alive[j]]
                if alive_allies:
                    target = min(alive_allies, key=lambda j: self.agent_health[j])
                    # Approach-to-target bonus
                    prev_d = np.sum(np.abs(prev_pos[i] - self.agent_pos[target]))
                    curr_d = np.sum(np.abs(pos - self.agent_pos[target]))
                    rewards[i] += 0.01 * (prev_d - curr_d)
                    # Heal bonus (action 5 while close)
                    if int(actions[i]) == 5 and curr_d <= 2:
                        rewards[i] += 0.05
                    # Rear-arc: support should be behind (higher row) the target
                    if pos[0] > self.agent_pos[target][0]:
                        rewards[i] += 0.005

        # Zone capture team bonus
        n_captured = int(np.sum(self.zone_captured))
        if n_captured >= 2:
            rewards += 0.1

        return rewards

    # ------------------------------------------------------------------
    def _get_obs(self, agent_idx: int) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        pos = self.agent_pos[agent_idx]
        g = float(self.grid_size)

        # self_tok [0:8]
        role_oh = np.zeros(3, dtype=np.float32)
        role_oh[agent_idx] = 1.0
        obs[0] = pos[0] / g
        obs[1] = pos[1] / g
        obs[2] = self.agent_vel[agent_idx][0]
        obs[3] = self.agent_vel[agent_idx][1]
        obs[4] = self.agent_health[agent_idx] / MAX_HEALTH
        obs[5:8] = role_oh

        # ally_tok [8:24]  2 allies × 8 dims
        ally_indices = [j for j in range(self.n_agents) if j != agent_idx]
        for slot, j in enumerate(ally_indices):
            base = 8 + slot * 8
            if self.agent_alive[j]:
                rel = (self.agent_pos[j] - pos) / g
                ally_role_oh = np.zeros(3, dtype=np.float32)
                ally_role_oh[j] = 1.0
                obs[base + 0] = rel[0]
                obs[base + 1] = rel[1]
                obs[base + 2] = self.agent_health[j] / MAX_HEALTH
                obs[base + 3] = 1.0  # alive
                obs[base + 4:base + 7] = ally_role_oh
                obs[base + 7] = 0.0  # padding
            else:
                obs[base + 3] = 0.0  # not alive → zero rest

        # enemy_tok [24:40]  2 enemies × 8 dims
        for slot in range(self.n_enemies):
            base = 24 + slot * 8
            if self.enemy_alive[slot]:
                rel = (self.enemy_pos[slot] - pos) / g
                obs[base + 0] = rel[0]
                obs[base + 1] = rel[1]
                obs[base + 2] = self.enemy_health[slot] / MAX_HEALTH
                obs[base + 3] = 1.0  # alive
                # class_confidence: uniform (enemy role unknown)
                obs[base + 4:base + 7] = np.array([0.33, 0.33, 0.34], dtype=np.float32)
                obs[base + 7] = 0.0

        # zone_tok [40:58]  3 zones × 6 dims
        for z in range(self.n_zones):
            base = 40 + z * 6
            zc = self.zone_centers[z]
            rel = (zc - pos) / g
            # occupancy: fraction of agents in zone
            occupancy = sum(
                1 for i in range(self.n_agents)
                if self.agent_alive[i] and np.sum(np.abs(self.agent_pos[i] - zc)) <= ZONE_RADIUS
            ) / float(self.n_agents)
            obs[base + 0] = rel[0]
            obs[base + 1] = rel[1]
            obs[base + 2] = occupancy
            obs[base + 3] = self.zone_progress[z]
            obs[base + 4] = 1.0 if self.zone_captured[z] else 0.0
            obs[base + 5] = 1.0  # zone value (uniform)

        # masks [58:65]  0=valid, 1=padding
        # All ally/enemy/zone slots are always valid (no padding in fixed 3-agent env)
        obs[58:60] = 0.0  # ally mask — both slots valid
        obs[60:62] = 0.0  # enemy mask
        obs[62:65] = 0.0  # zone mask

        return obs
