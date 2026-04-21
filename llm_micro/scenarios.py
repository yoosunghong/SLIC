"""Scenario generator for the LLM routing-legibility probe.

The goal is not to leak the answer in the prompt. Instead, the planner sees a
team roster plus a tactical state and must decide which specialist to dispatch.

- explicit: canonical role labels are visible (Strike / Vanguard / Support)
- implicit: the same specialists are described only through randomized aliases
  (Agent_X / Agent_Y / Agent_Z), so the underlying role remains recoverable only
  through the stored alias map
"""
from dataclasses import dataclass
from random import Random
from typing import Literal

ContextRole = Literal["Strike", "Vanguard", "Support"]
Formation = Literal["Clustered", "Spread"]
Condition = Literal["explicit", "implicit"]

EXPLICIT_LABELS = ["Strike", "Vanguard", "Support"]
IMPLICIT_LABELS = ["Agent_X", "Agent_Y", "Agent_Z"]

ROLE_DESCRIPTIONS = {
    "Strike": (
        "Carries a long-range rifle, creates pressure from standoff distance, "
        "and loses efficiency if forced into close contact."
    ),
    "Vanguard": (
        "Wears heavy armor, thrives in close-quarters contact, and can pin or "
        "body-block a threatening enemy."
    ),
    "Support": (
        "Carries field medical gear, restores allied health, contributes little "
        "direct damage, and works best behind teammates."
    ),
}


@dataclass
class Scenario:
    context_role: ContextRole
    formation: Formation
    condition: Condition
    seed: int

    def _rng(self) -> Random:
        return Random(f"{self.context_role}|{self.formation}|{self.condition}|{self.seed}")

    def surface_to_role(self) -> dict[str, str]:
        if self.condition == "explicit":
            return {role: role for role in EXPLICIT_LABELS}

        rng = self._rng()
        aliases = IMPLICIT_LABELS[:]
        roles = EXPLICIT_LABELS[:]
        rng.shuffle(aliases)
        rng.shuffle(roles)
        return dict(zip(aliases, roles))

    def role_to_surface(self) -> dict[str, str]:
        return {role: surface for surface, role in self.surface_to_role().items()}

    def valid_dispatch_names(self) -> list[str]:
        return list(self.surface_to_role().keys())

    def expected_dispatch(self) -> str:
        return self.role_to_surface()[self.context_role]

    def render_state(self) -> str:
        if self.context_role == "Strike":
            if self.formation == "Clustered":
                return (
                    "Formation: Clustered. Two allies are grouped behind cover near zone Z1 while enemy "
                    "marksmen hold a medium-range firing angle across the lane. A close push would expose "
                    "the team, so the reserve specialist must project force from distance and break the "
                    "enemy firing line."
                )
            return (
                "Formation: Spread. Allies are covering three lanes and two enemy shooters are pinning "
                "the open side of the map from long range. The reserve specialist must pressure exposed "
                "shooters without overcommitting into melee range."
            )

        if self.context_role == "Vanguard":
            if self.formation == "Clustered":
                return (
                    "Formation: Clustered. An enemy bruiser has breached the choke and is now contesting "
                    "space right on top of the allied group near the objective. The reserve specialist must "
                    "step into contact, absorb pressure, and keep the frontline from collapsing."
                )
            return (
                "Formation: Spread. One ally is isolated on the flank while a fast melee threat is "
                "closing the gap. The reserve specialist must intercept, body-block, and hold the line "
                "long enough for the team to stabilize."
            )

        if self.formation == "Clustered":
            return (
                "Formation: Clustered. Two allies near zone Z1 have both been chipped down by sustained "
                "fire and will lose the attrition battle if not stabilized soon. The reserve specialist "
                "must restore team durability rather than add raw damage."
            )
        return (
            "Formation: Spread. One ally on the far lane is in critical condition and retreating, while "
            "the other two allies are still holding their positions. The reserve specialist must keep the "
            "wounded ally alive without needing to win a duel directly."
        )


def build_agent_roster(scenario: Scenario) -> str:
    rng = scenario._rng()
    labels = scenario.valid_dispatch_names()
    rng.shuffle(labels)
    surface_to_role = scenario.surface_to_role()

    header = "Available sub-agents:"
    lines = [header]
    for label in labels:
        role = surface_to_role[label]
        lines.append(f"- {label}: {ROLE_DESCRIPTIONS[role]}")
    return "\n".join(lines)


def build_prompt(scenario: Scenario) -> str:
    roster = build_agent_roster(scenario)
    state = scenario.render_state()
    choices = " | ".join(scenario.valid_dispatch_names())

    return f"""You are the planner for a 3-agent cooperative team.

{roster}

Current tactical state:
{state}

Choose exactly one reserve specialist to dispatch right now.
Base the choice on the roster descriptions and the tactical state, not on list order.
Respond in this exact format:
DISPATCH: <name>
REASON: <one short sentence>

Valid names: {choices}
Never leave <name> blank.
"""
