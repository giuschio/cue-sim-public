import numpy as np

from typing import NamedTuple

from cuesim.utils import random_2d


class Shot(NamedTuple):
    cue_velocity: np.ndarray
    cue_impact_position: np.ndarray
    target_velocity: np.ndarray
    alignment: float
    cue_reflections: int


def normalize(vec):
    return vec / np.linalg.norm(vec)


def collide(p1, p2, r):
    return np.linalg.norm(p1 - p2) < (2 * r)


def get_annotated_state(state):
    # assume the first ball is the cue ball, all other positions are target balls
    state = state.reshape(-1, 3)
    info = {"state": {"cue_ball": {"position": state[0, 0:2]}}}
    tg_balls = {
        f"target_ball_{i}": {"position": state[i, 0:2]} for i in range(1, len(state))
    }
    info["state"].update(tg_balls)

    for idx, key in enumerate(info["state"].keys()):
        info["state"][key]["pocketed"] = state[idx, 2] > 0

    return info


def closest_point_on_segment(p, segment):
    # Convert points to numpy arrays for easier vector operations
    p = np.array(p)
    p1 = np.array(segment[0])
    p2 = np.array(segment[1])

    # Calculate the direction vector from p1 to p2
    d = p2 - p1

    # Compute the dot products needed for the projection scalar t
    pa_p1 = p - p1
    t = np.dot(pa_p1, d) / np.dot(d, d)

    # Clamp t to the [0, 1] interval
    t = max(0, min(1, t))

    # Calculate the closest point on the segment
    closest_point = p1 + t * d

    return closest_point


def compute_direct_shot(
    cue_position, target_position, pocket_position, obstacles, ball_rad
) -> Shot:
    eps = 1e-5
    target_velocity = normalize(pocket_position - target_position)
    cue_impact_position = target_position - 2 * ball_rad * target_velocity
    cue_velocity = normalize(cue_impact_position - cue_position)

    cue_preimpact_position = cue_impact_position - eps * cue_velocity

    shot = Shot(
        cue_velocity=cue_velocity,
        cue_impact_position=cue_impact_position,
        target_velocity=target_velocity,
        alignment=abs(np.dot(cue_velocity, target_velocity)),
        cue_reflections=0,
    )
    path = (cue_position, cue_impact_position)

    collisions = list()
    collisions.append(collide(target_position, cue_preimpact_position, ball_rad))

    for obstacle in obstacles:
        closest_point = closest_point_on_segment(obstacle, path)
        collisions.append(collide(obstacle, closest_point, ball_rad))

    path = (target_position, pocket_position)
    for obstacle in obstacles:
        closest_point = closest_point_on_segment(obstacle, path)
        collisions.append(collide(obstacle, closest_point, ball_rad))

    if sum(collisions) > 0:
        return None
    else:
        return shot


def compute_direct_shots(state_info, physics):
    state = state_info["state"]
    targets = {
        k: state[k]["position"]
        for k in state.keys()
        if "target" in k and not state[k]["pocketed"]
    }
    pockets = physics["pocket_positions"]
    cue = state["cue_ball"]["position"]
    rad = physics["ball_radius"]

    balls = state.keys()

    shots = list()
    for target_name, target_pos in targets.items():
        for pocket in pockets:

            ignored = ["cue_ball", target_name]
            obstacles = [state[k]["position"] for k in balls if k not in ignored]

            shot = compute_direct_shot(cue, target_pos, pocket, obstacles, rad)
            if shot is not None:
                shots.append(shot)

    return shots


class RandomAgent:
    def __init__(self, physics) -> None:
        self.rng = np.random.default_rng(0)

    def select_action(self, state):
        return np.asarray(random_2d(self.rng))


class HitClosestBallAgent:
    def __init__(self, physics) -> None:
        self.physics = physics

    def set_eval(self):
        pass

    def select_action(self, state, action):
        action = None
        info = get_annotated_state(state)
        state = info["state"]

        cue_pos = state["cue_ball"]["position"]
        target_keys = [k for k in state.keys() if "target" in k]

        target_poss = [state[k]["position"] for k in target_keys]
        target_dist = [np.linalg.norm((cue_pos - p)) for p in target_poss]

        closest_target = np.argmin(target_dist)

        action = target_poss[closest_target] - cue_pos
        action /= np.linalg.norm(action)

        return action


class DirectShotAgent:
    def __init__(self, physics) -> None:
        self.hit_closest_ball = HitClosestBallAgent(physics)
        self.physics = physics

    def set_eval(self):
        pass

    def get_actions(self, state) -> list:
        info = get_annotated_state(state)
        # if info["time"] < 0.001:
        #     return [self.hit_closest_ball(info)]
        actions = compute_direct_shots(info, self.physics)
        if len(actions) == 0:
            actions = [self.hit_closest_ball.select_action(state, None)]
        return actions

    def select_action(self, state, action):
        action = None
        # compute available actions
        actions = self.get_actions(state)
        if len(actions) > 1:
            alignments = np.array([shot.alignment for shot in actions])
            action = actions[np.argmax(alignments)]
        else:
            action = actions[0]

        if type(action) == Shot:
            action = action.cue_velocity

        return action


def get_banked_trajectory(initial, target, wall):
    x0, y0 = initial
    x1, y1 = target

    wall, wall_type = wall

    if wall_type == "vertical":
        r_y = (
            y1 - (abs(y0 - y1) / (1 + abs(x0 - wall) / abs(x1 - wall)))
            if y1 > y0
            else y1 + (abs(y0 - y1) / (1 + abs(x0 - wall) / abs(x1 - wall)))
        )
        r_x = wall
    else:  # "horizontal"
        r_x = (
            x1 - (abs(x0 - x1) / (1 + abs(y0 - wall) / abs(y1 - wall)))
            if x1 > x0
            else x1 + (abs(x0 - x1) / (1 + abs(y0 - wall) / abs(y1 - wall)))
        )
        r_y = wall

    reflection_point = [r_x, r_y]
    return reflection_point


def compute_banked_shot(
    cue_position, target_position, pocket_position, obstacles, ball_rad, wall
) -> dict:
    eps = 1e-5

    target_velocity = normalize(pocket_position - target_position)
    cue_impact_position = target_position - 2 * ball_rad * target_velocity
    reflection_point = get_banked_trajectory(cue_position, cue_impact_position, wall)

    cue_velocity = normalize(reflection_point - cue_position)
    impact_velocity = normalize(cue_impact_position - reflection_point)

    cue_preimpact_position = cue_impact_position - eps * impact_velocity

    shot = Shot(
        cue_velocity=cue_velocity,
        cue_impact_position=cue_impact_position,
        target_velocity=target_velocity,
        alignment=abs(np.dot(cue_velocity, target_velocity)),
        cue_reflections=1,
    )
    paths = [
        (cue_position, reflection_point),
        (reflection_point, cue_preimpact_position),
    ]
    collisions = list()
    collisions.append(collide(target_position, cue_preimpact_position, ball_rad))

    for obstacle in obstacles:
        for path in paths:
            closest_point = closest_point_on_segment(obstacle, path)
            collisions.append(collide(obstacle, closest_point, ball_rad))

    if sum(collisions) > 0:
        return None
    else:
        return shot


def compute_banked_shots(state_info, physics):
    state = state_info["state"]
    targets = {
        k: state[k]["position"]
        for k in state.keys()
        if "target" in k and not state[k]["pocketed"]
    }
    pockets = physics["pocket_positions"]
    cue = state["cue_ball"]["position"]

    balls = state.keys()
    w = physics["table_width"] / 2 - physics["ball_radius"] - 0.002
    h = physics["table_height"] / 2 - physics["ball_radius"] - 0.002

    walls = [
        (w, "vertical"),
        (-w, "vertical"),
        (h, "horizontal"),
        (-h, "horizontal"),
    ]

    rad = physics["ball_radius"]

    shots = list()
    for target_name, target_pos in targets.items():
        for pocket in pockets:

            ignored = ["cue_ball", target_name]
            obstacles = [state[k]["position"] for k in balls if k not in ignored]
            for w in walls:
                shot = compute_banked_shot(cue, target_pos, pocket, obstacles, rad, w)
                if shot is not None:
                    shots.append(shot)

    return shots


class AimedShotAgent:
    def __init__(self, physics) -> None:
        self.hit_closest_ball = HitClosestBallAgent(physics)
        self.physics = physics

    def get_actions(self, state):
        info = get_annotated_state(state)
        # if info["time"] < 0.001:
        #     return [self.hit_closest_ball(info)]
        actions = compute_direct_shots(info, self.physics)
        actions += compute_banked_shots(info, self.physics)

        if len(actions) == 0:
            actions = [self.hit_closest_ball.select_action(state, None)]
        return actions

    def select_action(self, state):
        # compute available actions
        actions = self.get_actions(state)
        if len(actions) > 1:
            alignments = np.array([shot.alignment for shot in actions])
            action = actions[np.argmax(alignments)]
        else:
            action = actions[0]

        if type(action) == Shot:
            action = action.cue_velocity

        return action


__available_heuristics = dict(
    random=RandomAgent,
    hit_closest_ball=HitClosestBallAgent,
    aimed_direct_shot=DirectShotAgent,
)


def get_heuristic(key, physics):
    H = __available_heuristics[key]
    return H(physics)
