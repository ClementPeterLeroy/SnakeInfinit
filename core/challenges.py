"""Challenge definitions share the same Game engine and rules."""
from dataclasses import dataclass
from core.config import GameRules


@dataclass(frozen=True)
class Challenge:
    id: str
    name: str
    objective: str
    difficulty: int
    rules: GameRules


CHALLENGES = (
    Challenge("centurion", "THE CENTURION", "Reach 70 points without dying.", 3,
              GameRules(mode="challenge", challenge_id="centurion", score_target=70)),
    Challenge("star_hunter", "STAR HUNTER", "Collect 10 stars in one game.", 2,
              GameRules(mode="challenge", challenge_id="star_hunter", stars_enabled=True,
                        star_target=10, star_interval=(10, 15))),
    Challenge("speed_demon", "SPEED DEMON", "50 SEGMENTS · 20 BLOCKS/SEC · SURVIVE 45 SECONDS", 5,
              GameRules(mode="challenge", challenge_id="speed_demon", initial_length=50,
                        initial_speed=20, survival_target=45)),
    Challenge("labyrinth", "THE LABYRINTH", "Reach 30 points. Walls accumulate permanently.", 3,
              GameRules(mode="challenge", challenge_id="labyrinth", score_target=30,
                        hazards_enabled=True, wall_enabled=True)),
    Challenge("minefield", "MINEFIELD", "Reach 25 points. Avoid mines and 3×3 explosions.", 4,
              GameRules(mode="challenge", challenge_id="minefield", score_target=25,
                        hazards_enabled=True, mines_enabled=True)),
)
BY_ID = {challenge.id: challenge for challenge in CHALLENGES}
COMING_SOON = ("AGAINST THE CLOCK", "CLAUSTROPHOBIA", "NO TURNING BACK",
               "THE GAUNTLET", "ENDURANCE", "THE HUNTER")
