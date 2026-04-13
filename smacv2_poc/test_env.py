"""Quick smoke test — verifies SMACv2 + SC2 are wired up correctly."""
import os
os.environ.setdefault("SC2PATH", "D:/StarCraft II")

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

CAPABILITY_CONFIG = {
    "n_units": 10,
    "n_enemies": 10,
    "team_gen": {
        "dist_type": "weighted_teams",
        "unit_types": ["stalker", "zealot", "colossus"],
        "weights": [0.45, 0.45, 0.1],
        "observe": True,
        "exception_unit_types": ["colossus"],
    },
    "start_positions": {
        "dist_type": "surrounded_and_reflect",
        "p": 0.5,
        "map_x": 32,
        "map_y": 32,
    },
}


def main():
    print("Creating env...")
    env = StarCraftCapabilityEnvWrapper(
        capability_config=CAPABILITY_CONFIG,
        map_name="10gen_protoss",
        debug=False,
        conic_fov=False,
        use_unit_ranges=True,
        min_attack_range=2,
        obs_own_pos=True,
    )

    print("Resetting...")
    env.reset()

    info = env.get_env_info()
    print(f"  n_agents : {info['n_agents']}")
    print(f"  n_actions: {info['n_actions']}")
    print(f"  obs_shape: {info['obs_shape']}")
    print(f"  state_shape: {info['state_shape']}")
    print("OK")
    env.close()


if __name__ == "__main__":
    main()
