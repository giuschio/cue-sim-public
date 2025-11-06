import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

from time import sleep


def is_point_inside_rectangle(point, rect_top_left, rect_bottom_right):
    return np.all(rect_top_left <= point) and np.all(point <= rect_bottom_right)


class BilliardsSimulator:
    TABLE_COLOR = (153, 255, 153)
    BALL_COLORS = {
        "cue": (255, 255, 255, 1),
        "target": (255, 0, 0, 1),
        "penalty": (0, 0, 0, 1),
    }
    WALL_THICKNESS = 0.004
    FONT_COLOR = (100, 150, 100)
    FONT_SIZE = 24

    def __init__(self, seed, options={}):
        self.__time_int = 0
        self.__rng = np.random.default_rng(seed)

        self.options = options
        self.options.update(options["physics"])

        self.n_target_balls = self.options["n_target_balls"]
        self.n_penalty_balls = self.options["n_penalty_balls"]
        self.space = pymunk.Space(threaded=True)
        self.space.threads = 2
        self.space.gravity = (0. + self.options["gravity_bias_x"], 0. + self.options["gravity_bias_y"])
        self.space.damping = self.options["damping"]
        self.velocity_eps = 0.05
        self.sim_timestep = self.options["dt"]
        self.__ballcheck_period = int(20 * 0.02 / self.sim_timestep)

        self.options = options
        self.w, self.h = self.options["table_width"], options["table_height"]

        self.balls = {}  # Dictionary to store ball references
        self.ball_ids = {}
        # self.__initial_state = None
        self._add_balls()
        self._add_walls()

        self._window_width_px = 800
        self._side_text = ""

        self.screen = None

        pocket_positions = [
            [-self.w / 2, -self.h / 2],
            [-self.w / 2, self.h / 2],
            [self.w / 2, -self.h / 2],
            [self.w / 2, self.h / 2],
        ]
        if self.options["physics"]["side_pockets"]:
            pocket_positions += [[0.0, -self.h / 2], [0.0, self.h / 2]]

        self.options["physics"]["pocket_positions"] = np.asarray(pocket_positions)

    def __init_screen(self):
        window_scaling = int(self._window_width_px / self.options["table_width"])
        self._window_height_px = int(window_scaling * self.options["table_height"])
        desired_size = (self._window_width_px, self._window_height_px)
        existing_surface = pygame.display.get_surface()
        if existing_surface is None or existing_surface.get_size() != desired_size:
            # Only create a new window if the size changed; otherwise reuse the existing one.
            self.screen = pygame.display.set_mode(desired_size)
        else:
            self.screen = existing_surface
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        # make sure we can see outside of the table
        visualization_scaling = 0.75
        self.draw_options.transform = pymunk.Transform.scaling(
            window_scaling * visualization_scaling
        ) @ pymunk.Transform.translation(
            self.w / 2 * (1 / visualization_scaling),
            self.h / 2 * (1 / visualization_scaling),
        )
        self.__font = pygame.font.Font(None, self.FONT_SIZE)

    def is_pocketed(self, position):
        toll = 0.001
        return not is_point_inside_rectangle(
            position,
            np.array([-self.w / 2 - toll, -self.h / 2 - toll]),
            np.array([self.w / 2 + toll, self.h / 2 + toll]),
        )

    def is_stationary(self, lin_vel):
        return bool(np.linalg.norm(lin_vel) < self.velocity_eps)

    def set_text(self, text):
        self._side_text = text

    @property
    def time(self):
        return self.__time_int * self.sim_timestep

    def reset_clock(self, time=0):
        self.__time_int = int(time / self.sim_timestep)

    @property
    def num_balls(self):
        return len(self.balls)

    def get_info(self):
        info = {
            "time": self.time,
            "state": {},
            "num_stationary": 0,
            "num_pocketed": 0,
            "num_target_pocketed": 0,
        }
        for name, shape in self.balls.items():
            lin_vel = np.asarray(shape.body.velocity)
            position = np.asarray(shape.body.position)
            stationary = self.is_stationary(lin_vel)
            pocketed = self.is_pocketed(position)
            info["state"][name] = {
                "position": position,
                "orientation": np.asarray(shape.body.angle),
                "velocity": lin_vel,
                "angular_velocity": np.asarray(shape.body.angular_velocity),
                "stationary": stationary,
                "pocketed": pocketed,
            }

            info["num_stationary"] += float(stationary)
            info["num_pocketed"] += float(pocketed)
            info["num_target_pocketed"] += float(pocketed) if "target" in name else 0.0

            info["pocket_positions"] = self.options["physics"]["pocket_positions"]
        return info

    def set_velocity(self, ball_name, vel, jitter=0.0):
        j = self.__rng.normal(0, jitter, 2)
        self.balls[ball_name].body.velocity = (vel[0] + j[0], vel[1] + j[1])

    def set_angular_velocity(self, ball_name, vel, jitter=0.0):
        j = self.__rng.normal(0, jitter)
        self.balls[ball_name].body.angular_velocity = vel + j

    def set_position(self, ball_name, pos, jitter=0.0):
        j = self.__rng.normal(0, jitter, 2)
        self.balls[ball_name].body.position = (pos[0] + j[0], pos[1] + j[1])
        self.balls[ball_name].body.velocity = (0.0, 0.0)

    def step(self):
        # returns done (bool)
        self.space.step(self.sim_timestep)
        self.__time_int += 1

        if self.__time_int % self.__ballcheck_period == 0:
            num_stationary = 0
            for name, shape in self.balls.items():
                position = np.asarray(shape.body.position)
                if self.is_pocketed(position):
                    shape.body.velocity = (0.0, 0.0)
                    shape.body.angular_velocity = 0.0
                    shape.body.position = (
                        self.options["ball_radius"] * 3.0 * self.ball_ids[name],
                        -self.options["table_height"] / 2.0
                        - self.options["ball_radius"] * 2.0
                        - self.WALL_THICKNESS,
                    )
                num_stationary += self.is_stationary(shape.body.velocity)
            if num_stationary == self.num_balls:
                return True
        return False

    def draw(self):
        if self.screen is None:
            self.__init_screen()
        self.screen.fill(self.TABLE_COLOR)  # Light green background
        self.space.debug_draw(self.draw_options)
        text_surface = self.__font.render(self._side_text, True, self.FONT_COLOR)
        self.screen.blit(text_surface, (5, 5))
        pygame.display.flip()
        sleep(self.options["physics"]["render_dt"])

    def _set_ball(self, name, position, velocity=(0, 0), angular_velocity=0):
        self.balls[name].body.position = position
        self.balls[name].body.velocity = velocity
        self.balls[name].body.angular_velocity = angular_velocity

    def _create_ball(self, position, color, velocity, name):
        body = pymunk.Body(
            1, pymunk.moment_for_circle(1, 0, self.options["ball_radius"])
        )
        body.position = position
        body.velocity = velocity
        shape = pymunk.Circle(body, self.options["ball_radius"])
        shape.elasticity = 1.0
        shape.friction = self.options["physics"]["friction"]
        # shape.color = pygame.color.THECOLORS[color]
        shape.color = color
        self.space.add(body, shape)
        self.balls[name] = shape  # Store the reference to the ball

    def _add_balls(self):
        # initialize all the ball ids
        target_names = [f"target_ball_{i+1}" for i in range(self.n_target_balls)]
        penalty_names = [f"penalty_ball_{i+1}" for i in range(self.n_penalty_balls)]
        ball_ids = ["cue_ball"] + target_names + penalty_names

        for idx, name in enumerate(ball_ids):
            color = self.BALL_COLORS[
                next((key for key in self.BALL_COLORS if key in name), None)
            ]
            self._create_ball(position=(0, 0), color=color, velocity=(0, 0), name=name)
            self.ball_ids[name] = idx

    def _add_walls(self):
        # Coordinates for the wall segments, leaving space for pockets
        hw = self.options["holes_width"]
        hw_2 = 1.4142 * hw
        if self.options["physics"]["side_pockets"]:
            wall_segments = [
                # Top wall
                ((-self.w / 2 + hw_2, -self.h / 2), (-hw, -self.h / 2)),
                ((hw, -self.h / 2), (self.w / 2 - hw_2, -self.h / 2)),
                # Bottom wall
                ((-self.w / 2 + hw_2, self.h / 2), (-hw, self.h / 2)),
                ((hw, self.h / 2), (self.w / 2 - hw_2, self.h / 2)),
                # Left wall
                ((-self.w / 2, -self.h / 2 + hw_2), (-self.w / 2, self.h / 2 - hw_2)),
                # Right wall
                ((self.w / 2, -self.h / 2 + hw_2), (self.w / 2, self.h / 2 - hw_2)),
            ]

        else:
            wall_segments = [
                # Top wall
                ((-self.w / 2 + hw_2, -self.h / 2), (self.w / 2 - hw_2, -self.h / 2)),
                # Bottom wall
                ((-self.w / 2 + hw_2, self.h / 2), (self.w / 2 - hw_2, self.h / 2)),
                # Left wall
                ((-self.w / 2, -self.h / 2 + hw_2), (-self.w / 2, self.h / 2 - hw_2)),
                # Right wall
                ((self.w / 2, -self.h / 2 + hw_2), (self.w / 2, self.h / 2 - hw_2)),
            ]

        for start, end in wall_segments:
            wall = pymunk.Segment(
                self.space.static_body, start, end, self.WALL_THICKNESS
            )
            wall.elasticity = 1.0
            wall.friction = self.options["physics"]["friction"]
            wall.filter = pymunk.ShapeFilter(group=1)
            self.space.add(wall)
