# env/first_person_env.py
import pygame
import sys
import numpy as np
from env.driving_env import DrivingEnvironment

class FirstPersonDrivingEnv(DrivingEnvironment):
    """
    Góc nhìn thứ nhất - xe đứng yên ở giữa màn hình
    Các vật thể và mục tiêu di chuyển về phía xe
    """
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
        self.cell_w = self.width / float(self.grid_w)
        self.cell_h = self.height / float(self.grid_h)
        self.current_path = []

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("First Person AI Driving")
        self.clock = pygame.time.Clock()

        # Colors - darker theme for first person
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_CAR = (50, 150, 255)
        self.COLOR_OBS = (220, 60, 60)
        self.COLOR_GOAL = (60, 220, 80)
        self.COLOR_PATH = (180, 180, 220)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_ROAD = (50, 50, 55)
        self.COLOR_LANE = (80, 80, 90)

        # Xe cố định ở dưới giữa màn hình
        self.car_x = self.width // 2
        self.car_y = self.height - 80
        self.car_w, self.car_h = 50, 70
        
        # Lane position (3 lanes)
        self.car_lane = 1  # 0=left, 1=middle, 2=right
        self.lane_width = self.width // 3
        
        self.obj_w, self.obj_h = 45, 45
        self.actions = ["left", "stay", "right"]  # Đơn giản hóa actions

        self.state_size = 6  # [lane, nearest_obs_lane, nearest_obs_dist, goal_lane, goal_dist, speed]
        self.action_size = len(self.actions)

        self.previous_distance = None
        self.steps_count = 0
        self.max_steps = 1000
        
        # Obstacles và goals di chuyển
        self.obstacles = []
        self.goals = []
        self.world_speed = 3.0  # Tốc độ vật thể di chuyển về phía xe
        self.spawn_timer = 0
        self.spawn_interval = 60  # frames
        
        self.score = 0
        self.collision_count = 0

        self.reset()



    @property
    def car_pos(self):
        """
        Trả về vị trí pixel [x, y] của xe 
        để tương thích với AdvancedAgent.
        """
        # Tính toán x dựa trên lane hiện tại
        x = self._get_lane_x(self.car_lane)
        
        # y là cố định
        y = self.car_y
        
        return [x, y]
    
    @property
    def goal_positions(self):
        """
        Trả về danh sách các vị trí [x, y] của các mục tiêu (goals) 
        chưa được thu thập, để tương thích với AdvancedAgent.
        """
        # Lấy vị trí 'pos' từ mỗi mục tiêu 'goal' trong self.goals
        # CHỈ khi mục tiêu đó chưa được thu thập ('collected' == False)
        return [goal['pos'] for goal in self.goals if not goal['collected']]

    def _get_lane_x(self, lane):
        """Lấy vị trí x của lane (0, 1, 2)"""
        return lane * self.lane_width + self.lane_width // 2

    def add_obstacle(self, lane, distance, is_goal=False):
        """
        Thêm vật thể vào lane cụ thể
        lane: 0, 1, 2
        distance: khoảng cách từ đầu màn hình (y = 0)
        """
        x = self._get_lane_x(lane)
        y = -distance  # Spawn phía trên màn hình
        
        obj = {
            'pos': np.array([x, y], dtype=float),
            'lane': lane,
            'is_goal': bool(is_goal),
            'collected': False
        }
        
        if is_goal:
            self.goals.append(obj)
        else:
            self.obstacles.append(obj)

    def reset(self):
        self.car_lane = 1  # Bắt đầu ở lane giữa
        self.previous_distance = None
        self.steps_count = 0
        self.obstacles = []
        self.goals = []
        self.spawn_timer = 0
        self.score = 0
        self.collision_count = 0
        self.current_path = []

        # Spawn initial obstacles
        for i in range(3):
            lane = np.random.randint(0, 3)
            distance = 200 + i * 150
            is_goal = (i % 3 == 0)  # Mỗi 3 obstacle có 1 goal
            self.add_obstacle(lane, distance, is_goal)

        state = self._get_state()
        return state

    def _get_state(self):
        """
        State vector: [car_lane, nearest_obs_lane, nearest_obs_dist, 
                       nearest_goal_lane, nearest_goal_dist, world_speed]
        """
        car_lane_norm = self.car_lane / 2.0  # normalize to [0, 1]
        
        # Tìm obstacle gần nhất
        nearest_obs_lane = 1.0  # default middle
        nearest_obs_dist = 1.0  # default far
        
        if len(self.obstacles) > 0:
            dists = [abs(obj['pos'][1] - self.car_y) for obj in self.obstacles]
            idx = int(np.argmin(dists))
            nearest_obs = self.obstacles[idx]
            nearest_obs_lane = nearest_obs['lane'] / 2.0
            nearest_obs_dist = min(1.0, nearest_obs['pos'][1] / self.height)
        
        # Tìm goal gần nhất
        nearest_goal_lane = 1.0
        nearest_goal_dist = 1.0
        
        active_goals = [g for g in self.goals if not g['collected']]
        if len(active_goals) > 0:
            dists = [abs(g['pos'][1] - self.car_y) for g in active_goals]
            idx = int(np.argmin(dists))
            nearest_goal = active_goals[idx]
            nearest_goal_lane = nearest_goal['lane'] / 2.0
            nearest_goal_dist = min(1.0, nearest_goal['pos'][1] / self.height)
        
        speed_norm = self.world_speed / 10.0  # normalize speed
        
        return np.array([
            car_lane_norm,
            nearest_obs_lane, 
            nearest_obs_dist,
            nearest_goal_lane,
            nearest_goal_dist,
            speed_norm
        ], dtype=float)

    def get_available_actions(self):
        return self.actions

    def step(self, action):
        if isinstance(action, int):
            action_idx = action
        else:
            try:
                action_idx = self.actions.index(action)
            except ValueError:
                action_idx = 1  # stay

        # Di chuyển xe sang lane
        old_lane = self.car_lane
        
        if action_idx == 0:  # left
            self.car_lane = max(0, self.car_lane - 1)
        elif action_idx == 2:  # right
            self.car_lane = min(2, self.car_lane + 1)
        # action_idx == 1: stay

        reward = 0.01  # Small positive reward for surviving
        done = False
        info = {'collision': False, 'goal': False, 'timeout': False}

        self.steps_count += 1

        # Penalty for unnecessary lane changes
        if old_lane != self.car_lane:
            reward -= 0.02

        # Di chuyển tất cả vật thể về phía xe
        self.world_speed = min(5.0, 3.0 + self.steps_count / 500.0)  # Tăng tốc dần
        
        for obj in self.obstacles:
            obj['pos'][1] += self.world_speed
        
        for goal in self.goals:
            goal['pos'][1] += self.world_speed

        # Spawn vật thể mới
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self.spawn_interval = max(30, 60 - self.steps_count // 100)  # Spawn nhanh hơn theo thời gian
            
            # Random spawn obstacle or goal
            lane = np.random.randint(0, 3)
            is_goal = (np.random.random() < 0.3)  # 30% chance goal
            self.add_obstacle(lane, 0, is_goal)

        # Xóa vật thể đã qua xe
        self.obstacles = [o for o in self.obstacles if o['pos'][1] < self.height + 50]
        self.goals = [g for g in self.goals if g['pos'][1] < self.height + 50]

        # Kiểm tra va chạm với obstacles
        car_x = self._get_lane_x(self.car_lane)
        collision_margin = (self.obj_w + self.car_w) / 2 - 10
        
        for obj in self.obstacles[:]:
            ox, oy = obj['pos']
            # Check if obstacle is at car level
            if abs(oy - self.car_y) < self.car_h:
                if abs(ox - car_x) < collision_margin:
                    reward = -10.0
                    done = True
                    info['collision'] = True
                    self.collision_count += 1
                    break

        # Kiểm tra thu thập goals
        for goal in self.goals:
            if goal['collected']:
                continue
            gx, gy = goal['pos']
            if abs(gy - self.car_y) < self.car_h:
                if abs(gx - car_x) < collision_margin:
                    goal['collected'] = True
                    reward = 5.0
                    self.score += 1
                    info['goal'] = True

        # Reward for staying alive
        reward += 0.05

        # Bonus for high score
        if self.score > 0:
            reward += self.score * 0.01

        # Timeout
        if self.steps_count >= self.max_steps:
            done = True
            info['timeout'] = True
            reward += self.score * 2.0  # Big bonus for surviving

        state = self._get_state()
        return state, reward, done, info

    def world_to_grid(self, px, py):
        """Convert world pixel coord -> grid cell"""
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
        """Return set of grid cells blocked by obstacles"""
        blocked = set()
        for o in self.obstacles:
            ox, oy = o['pos']
            if oy < 0 or oy > self.height:
                continue
            gx, gy = self.world_to_grid(ox, oy)
            blocked.add((gx, gy))
        return blocked

    def draw_road(self):
        """Vẽ đường với 3 lane"""
        # Road background
        pygame.draw.rect(self.screen, self.COLOR_ROAD, 
                        (0, 0, self.width, self.height))
        
        # Lane markers (animated)
        marker_height = 40
        marker_gap = 60
        offset = (self.steps_count * 5) % (marker_height + marker_gap)
        
        for lane in range(1, 3):  # 2 lane dividers
            x = lane * self.lane_width
            y = -offset
            while y < self.height:
                pygame.draw.rect(self.screen, self.COLOR_LANE,
                               (x - 3, y, 6, marker_height))
                y += marker_height + marker_gap

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.COLOR_BG)

        # Vẽ đường
        self.draw_road()

        # Vẽ obstacles
        for obj in self.obstacles:
            x, y = obj['pos']
            if y < -50 or y > self.height + 50:
                continue
            
            rect = pygame.Rect(x - self.obj_w//2, y - self.obj_h//2, 
                             self.obj_w, self.obj_h)
            pygame.draw.rect(self.screen, self.COLOR_OBS, rect)
            pygame.draw.rect(self.screen, (255, 120, 120), rect, 3)
            
            # Warning indicator if close
            if y > self.car_y - 150 and y < self.car_y:
                dist_alpha = int(255 * (1 - (self.car_y - y) / 150))
                warning_color = (255, 200, 0, dist_alpha)
                pygame.draw.circle(self.screen, warning_color, 
                                 (int(x), int(y)), self.obj_w, 2)

        # Vẽ goals
        for goal in self.goals:
            if goal['collected']:
                continue
            x, y = goal['pos']
            if y < -50 or y > self.height + 50:
                continue
            
            rect = pygame.Rect(x - self.obj_w//2, y - self.obj_h//2,
                             self.obj_w, self.obj_h)
            pygame.draw.rect(self.screen, self.COLOR_GOAL, rect)
            pygame.draw.rect(self.screen, (200, 255, 200), rect, 3)
            
            # Goal indicator
            pygame.draw.circle(self.screen, (100, 255, 100),
                             (int(x), int(y)), self.obj_w//2 + 5, 2)

        # Vẽ xe (cố định ở dưới màn hình)
        car_x = self._get_lane_x(self.car_lane)
        car_rect = pygame.Rect(car_x - self.car_w//2, 
                               self.car_y - self.car_h//2,
                               self.car_w, self.car_h)
        
        # Shadow effect
        shadow_rect = pygame.Rect(car_rect.x + 3, car_rect.y + 3,
                                 car_rect.width, car_rect.height)
        pygame.draw.rect(self.screen, (10, 10, 20), shadow_rect, border_radius=5)
        
        # Car body
        pygame.draw.rect(self.screen, self.COLOR_CAR, car_rect, border_radius=5)
        pygame.draw.rect(self.screen, (150, 200, 255), car_rect, 3, border_radius=5)
        
        # Car details (windows, lights)
        window_rect = pygame.Rect(car_x - 20, self.car_y - 20, 40, 25)
        pygame.draw.rect(self.screen, (100, 150, 200), window_rect, border_radius=3)
        
        # Headlights
        pygame.draw.circle(self.screen, (255, 255, 200),
                         (car_x - 15, self.car_y - self.car_h//2 + 5), 4)
        pygame.draw.circle(self.screen, (255, 255, 200),
                         (car_x + 15, self.car_y - self.car_h//2 + 5), 4)

        # UI - Score và Stats
        font = pygame.font.SysFont("arial", 24, bold=True)
        
        # Score
        score_text = font.render(f"Score: {self.score}", True, (100, 255, 100))
        self.screen.blit(score_text, (10, 10))
        
        # Speed
        speed_text = font.render(f"Speed: {self.world_speed:.1f}", True, (255, 255, 100))
        self.screen.blit(speed_text, (10, 45))
        
        # Steps
        steps_text = font.render(f"Time: {self.steps_count}", True, (200, 200, 255))
        self.screen.blit(steps_text, (10, 80))
        
        # Collisions
        if self.collision_count > 0:
            coll_text = font.render(f"Collisions: {self.collision_count}", True, (255, 100, 100))
            self.screen.blit(coll_text, (10, 115))

        # Lane indicator
        lane_names = ["LEFT", "MIDDLE", "RIGHT"]
        lane_text = font.render(f"Lane: {lane_names[self.car_lane]}", True, (180, 180, 255))
        self.screen.blit(lane_text, (self.width - 200, 10))

        # Instructions
        small_font = pygame.font.SysFont("arial", 14)
        instr = small_font.render("← → to change lanes | Avoid red! Collect green!", 
                                 True, (180, 180, 180))
        self.screen.blit(instr, (10, self.height - 25))

        pygame.display.flip()
        self.clock.tick(self.render_speed)

    def close(self):
        pygame.quit()