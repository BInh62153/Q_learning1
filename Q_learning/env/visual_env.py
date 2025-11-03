# env/visual_env.py
import pygame
import sys
import numpy as np
from env.driving_env import DrivingEnvironment

class VisualDrivingEnv(DrivingEnvironment):
    def __init__(self, difficulty=1, width=600, height=400, render_speed=30,
                 grid_w=20, grid_h=20):
        super().__init__(difficulty)
        self.state = None

        self.width = width
        self.height = height
        self.render_speed = render_speed

        # Grid params
        self.grid_w = grid_w
        self.grid_h = grid_h
        # cell size (float to map positions precisely)
        self.cell_w = self.width / float(self.grid_w)
        self.cell_h = self.height / float(self.grid_h)
        self.current_path = []  # list of grid nodes to draw (optional)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Driving Simulation")
        self.clock = pygame.time.Clock()

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_CAR = (50, 150, 255)
        self.COLOR_OBS = (220, 60, 60)
        self.COLOR_GOAL = (60, 220, 80)
        self.COLOR_PATH = (180, 180, 220)   # path color
        self.COLOR_GRID = (50, 50, 60)

        self.car_w, self.car_h = 40, 30
        self.obj_w, self.obj_h = 45, 45
        # actions order: right, up, down, left, stay
        self.actions = ["right", "up", "down", "left", "stay"]

        # state: [cx/width, cy/height, dx/width, dy/height]
        self.state_size = 4
        self.action_size = len(self.actions)

        self.previous_distance = None
        self.steps_count = 0
        self.max_steps = 1000

        # initial car position (x,y)
        self.car_pos = np.array([60.0, float(self.height) / 2.0], dtype=float)
        self.obstacles = []
        self.goal_positions = []

        self.reset()

    def add_obstacle(self, x, y, is_goal=False):
        self.obstacles.append({'pos': (float(x), float(y)), 'is_goal': bool(is_goal)})
        if is_goal:
            # keep goal_positions as list of tuples (x,y)
            self.goal_positions.append((float(x), float(y)))

    def reset(self):
        self.car_pos = np.array([60.0, float(self.height) / 2.0], dtype=float)
        self.previous_distance = None
        self.steps_count = 0
        self.obstacles = []
        self.goal_positions = []

        fixed_obstacles = [
            # you can add fixed obstacles here as (x,y)
            # (250, 100),
            # (300, 200),
        ]
        for x, y in fixed_obstacles:
            self.add_obstacle(x, y, is_goal=False)

        fixed_goals = [
            (self.width - 80, self.height // 2),
        ]
        for x, y in fixed_goals:
            self.add_obstacle(x, y, is_goal=True)

        # clear drawn path
        self.current_path = []

        state = self._get_state()
        if len(self.goal_positions) > 0:
            cx, cy = self.car_pos
            dists = [np.hypot(gx - cx, gy - cy) for gx, gy in self.goal_positions]
            self.previous_distance = float(min(dists))
        return state

    def _get_state(self):
        cx, cy = self.car_pos
        W, H = self.width, self.height

        # ---- nearest goal ----
        if len(self.goal_positions) > 0:
            dists = [np.hypot(gx - cx, gy - cy) for gx, gy in self.goal_positions]
            idx = int(np.argmin(dists))
            gx, gy = self.goal_positions[idx]
            dx, dy = gx - cx, gy - cy
            goal_dist = dists[idx]
        else:
            dx = dy = 0.0
            goal_dist = 9999

        # ---- nearest obstacle ----
        obs_dist = 9999
        if len(self.obstacles) > 0:
            d = [np.hypot(o['pos'][0] - cx, o['pos'][1] - cy) for o in self.obstacles]
            obs_dist = min(d)

        max_dist = np.hypot(W, H)

        return np.array([
            cx / W,
            cy / H,
            dx / W,
            dy / H,
            goal_dist / max_dist,
            obs_dist / max_dist
        ], dtype=float)

    def get_available_actions(self):
        return self.actions

    def step(self, action):
        # resolve action index
        if isinstance(action, int):
            action_idx = int(action)
        else:
            try:
                action_idx = self.actions.index(action)
            except ValueError:
                action_idx = self.actions.index("stay")

        old_pos = self.car_pos.copy()
        speed = 10.0

        # compute intended movement (dx, dy)
        intended_dx = 0.0
        intended_dy = 0.0
        if action_idx == 0:      # right
            intended_dx = speed
        elif action_idx == 1:    # up
            intended_dy = -speed
        elif action_idx == 2:    # down
            intended_dy = speed
        elif action_idx == 3:    # left
            intended_dx = -speed
        elif action_idx == 4:    # stay
            intended_dx = 0.0
            intended_dy = 0.0

        # apply movement and clip to arena
        self.car_pos[0] += intended_dx
        self.car_pos[1] += intended_dy

        # detect wall hit by checking if clipping changed position
        clipped_x = np.clip(self.car_pos[0], 10, self.width - self.car_w - 10)
        clipped_y = np.clip(self.car_pos[1], 10, self.height - self.car_h - 10)
        wall_hit = False
        if clipped_x != self.car_pos[0] or clipped_y != self.car_pos[1]:
            wall_hit = True
        # enforce clipping
        self.car_pos[0] = clipped_x
        self.car_pos[1] = clipped_y

        reward = -0.01
        done = False
        info = {'collision': False, 'goal': False, 'wall_hit': False}

        self.steps_count += 1

        if wall_hit:
            reward -= 0.1
            info['wall_hit'] = True

        # distance-based shaping towards nearest goal (if any)
        cx, cy = float(self.car_pos[0]), float(self.car_pos[1])
        if len(self.goal_positions) > 0:
            dists = [np.hypot(gx - cx, gy - cy) for gx, gy in self.goal_positions]
            current_distance = float(min(dists))
            if self.previous_distance is not None:
                distance_delta = float(self.previous_distance - current_distance)
                # small reward for moving closer, small penalty for moving away
                reward += float(distance_delta) * 0.02
            self.previous_distance = current_distance

            # proximity bonuses
            if current_distance < 80:
                reward += 0.05
            if current_distance < 50:
                reward += 0.1

        # collision / goal detection
        # check obstacles and goals
        for obj in list(self.obstacles):
            ox, oy = float(obj['pos'][0]), float(obj['pos'][1])
            collision_margin_x = (self.obj_w / 2.0) + (self.car_w / 2.0) - 5.0
            collision_margin_y = (self.obj_h / 2.0) + (self.car_h / 2.0) - 5.0
            if abs(cx - ox) < collision_margin_x and abs(cy - oy) < collision_margin_y:
                if obj.get('is_goal', False):
                    base_goal_reward = 100.0
                    speed_bonus = max(0.0, (self.max_steps - self.steps_count) / 10.0)
                    reward = base_goal_reward + speed_bonus
                    done = True
                    info['goal'] = True
                    info['steps'] = int(self.steps_count)
                    # remove reached goal from both lists
                    try:
                        self.goal_positions.remove(obj['pos'])
                    except ValueError:
                        pass
                    try:
                        self.obstacles.remove(obj)
                    except ValueError:
                        pass
                else:
                    reward = -5.0
                    done = True
                    info['collision'] = True
                break

        # timeout
        if self.steps_count >= self.max_steps:
            reward -= 20.0
            done = True
            info['timeout'] = True

        # small penalty if goal far away initially (discourage doing nothing)
        if len(self.goal_positions) > 0 and self.previous_distance is not None and self.previous_distance > 300:
            reward -= 0.01

        state = self._get_state()
        return state, reward, done, info

    # ---------- GRID helpers ----------
    def world_to_grid(self, px, py):
        """Convert world pixel coord -> grid cell (int,int)"""
        gx = int(px / self.cell_w)
        gy = int(py / self.cell_h)
        gx = np.clip(gx, 0, self.grid_w - 1)
        gy = np.clip(gy, 0, self.grid_h - 1)
        return (gx, gy)

    def grid_to_world_center(self, gx, gy):
        """Return pixel center of cell"""
        cx = int((gx + 0.5) * self.cell_w)
        cy = int((gy + 0.5) * self.cell_h)
        return (cx, cy)

    def get_blocked_cells(self):
        """Return set of grid cells that are blocked by obstacles (and optionally near obstacles)."""
        blocked = set()
        margin_cells_x = max(1, int((self.obj_w/2) / self.cell_w))
        margin_cells_y = max(1, int((self.obj_h/2) / self.cell_h))
        for o in self.obstacles:
            ox, oy = o['pos']
            gx, gy = self.world_to_grid(ox, oy)
            # mark neighbors within margin to be safe
            for dx in range(-margin_cells_x, margin_cells_x + 1):
                for dy in range(-margin_cells_y, margin_cells_y + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        blocked.add((nx, ny))
        return blocked

    def draw_grid(self):
        # draw grid lines
        for i in range(1, self.grid_w):
            x = int(i * self.cell_w)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height), 1)
        for j in range(1, self.grid_h):
            y = int(j * self.cell_h)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y), 1)

    def draw_path_on_screen(self, path):
        """path: list of (gx,gy) grid coords. Draw path as circles + connecting lines."""
        if not path:
            return
        pts = [self.grid_to_world_center(gx, gy) for gx, gy in path]
        # draw lines
        for i in range(len(pts) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, pts[i], pts[i+1], 3)
        # draw nodes
        for p in pts:
            pygame.draw.circle(self.screen, self.COLOR_PATH, p, 6)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.COLOR_BG)

        # draw grid (optional)
        self.draw_grid()

        # draw path if available (grid nodes)
        if self.current_path:
            self.draw_path_on_screen(self.current_path)

        cx, cy = float(self.car_pos[0]), float(self.car_pos[1])
        if len(self.goal_positions) > 0:
            dists = [np.hypot(gx - cx, gy - cy) for gx, gy in self.goal_positions]
            idx = int(np.argmin(dists))
            nearest_goal = self.goal_positions[idx]
            pygame.draw.line(self.screen, (120,120,140), (int(cx), int(cy)), nearest_goal, 2)
            pygame.draw.circle(self.screen, (120,120,140), nearest_goal, self.obj_w//2 + 15, 2)

        for obj in self.obstacles:
            x, y = obj['pos']
            rect = pygame.Rect(int(x - self.obj_w//2), int(y - self.obj_h//2), self.obj_w, self.obj_h)
            if obj.get('is_goal', False):
                pygame.draw.rect(self.screen, self.COLOR_GOAL, rect)
                pygame.draw.rect(self.screen, (200, 255, 200), rect, 3)
            else:
                pygame.draw.rect(self.screen, self.COLOR_OBS, rect)
                pygame.draw.rect(self.screen, (255, 120, 120), rect, 3)

        car_rect = pygame.Rect(int(cx - self.car_w//2), int(cy - self.car_h//2), self.car_w, self.car_h)
        pygame.draw.rect(self.screen, self.COLOR_CAR, car_rect)
        pygame.draw.rect(self.screen, (150, 200, 255), car_rect, 3)

        arrow_points = [
            (int(cx + self.car_w//2), int(cy)),
            (int(cx + self.car_w//2 + 10), int(cy - 8)),
            (int(cx + self.car_w//2 + 10), int(cy + 8))
        ]
        pygame.draw.polygon(self.screen, (200, 230, 255), arrow_points)

        font = pygame.font.SysFont("arial", 20, bold=True)
        text = font.render(f"Goals: {len(self.goal_positions)}", True, (255, 255, 100))
        self.screen.blit(text, (10, 10))
        text = font.render(f"Steps: {self.steps_count}/{self.max_steps}", True, (200, 200, 255))
        self.screen.blit(text, (10, 40))

        if len(self.goal_positions) > 0:
            dists = [np.hypot(gx - cx, gy - cy) for gx, gy in self.goal_positions]
            nearest_dist = min(dists)
            if nearest_dist < 50:
                dist_color = (100, 255, 100)
            elif nearest_dist < 100:
                dist_color = (255, 255, 100)
            else:
                dist_color = (255, 150, 150)
            text = font.render(f"Distance: {int(nearest_dist)}", True, dist_color)
            self.screen.blit(text, (10, 70))

        small_font = pygame.font.SysFont("arial", 14)
        instr = small_font.render("Arrow keys to move | Find green goals! | Press H for hybrid run", True, (180, 180, 180))
        self.screen.blit(instr, (10, self.height - 25))

        pygame.display.flip()
        self.clock.tick(self.render_speed)

    def close(self):
        pygame.quit()
