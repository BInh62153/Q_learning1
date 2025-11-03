"""
agents/advanced_agent.py

Features:
- A* path planning (grid-based) -> path -> actions
- Hybrid: follow planned path for N steps then fallback to policy
- Tabular Q (discretized) for fast experiments / visualization
- Double DQN (PyTorch) with target network
- Prioritized Replay (proportional) - simple implementation
- Utilities: save/load, visualize Q / value map
"""

import numpy as np
import random
import os
import math
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# --- PyTorch for DQN parts ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -------------------------
# Utilities: A* (grid)
# -------------------------
from heapq import heappush, heappop
def astar_grid(start, goal, grid):
    """
    start, goal: (gx, gy)
    grid: 2D numpy array: 0 free, 1 blocked
    returns list of (gx,gy) from start->goal (inclusive), empty if no path
    """
    W = grid.shape[1]
    H = grid.shape[0]
    DIRS = [(1,0),(-1,0),(0,1),(0,-1)]
    openp = []
    heappush(openp, (0 + (abs(goal[0]-start[0]) + abs(goal[1]-start[1])), 0, start))
    came = {start: None}
    gscore = {start: 0}
    while openp:
        _, g, cur = heappop(openp)
        if cur == goal:
            break
        x,y = cur
        for dx,dy in DIRS:
            nx,ny = x+dx, y+dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] != 0:
                continue
            nxt = (nx, ny)
            tentative = g + 1
            if nxt not in gscore or tentative < gscore[nxt]:
                gscore[nxt] = tentative
                priority = tentative + abs(goal[0]-nx) + abs(goal[1]-ny)
                heappush(openp, (priority, tentative, nxt))
                came[nxt] = cur
    if goal not in came:
        return []
    path = []
    p = goal
    while p is not None:
        path.append(p)
        p = came[p]
    path.reverse()
    return path

def path_to_actions(path):
    """
    Convert grid path to action indices matching VisualDrivingEnv:
    We'll use: 0=left,1=right,2=up,3=down,4=stay (matches earlier env.actions order)
    """
    acts = []
    for i in range(len(path)-1):
        x1,y1 = path[i]
        x2,y2 = path[i+1]
        if x2 == x1 - 1 and y2 == y1:
            acts.append(0)  # left
        elif x2 == x1 + 1 and y2 == y1:
            acts.append(1)  # right
        elif x2 == x1 and y2 == y1 - 1:
            acts.append(2)  # up
        elif x2 == x1 and y2 == y1 + 1:
            acts.append(3)  # down
        else:
            acts.append(4)
    return acts

# -------------------------
# Prioritized Replay Buffer
# Proportional prioritized replay (simple)
# -------------------------
class PrioritizedReplay:
    def __init__(self, capacity=10000, alpha=0.6, beta0=0.4, beta_anneal=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta0
        self.beta_anneal = beta_anneal
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_prio = 1.0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        # importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        # anneal beta
        self.beta = min(1.0, self.beta + self.beta_anneal * batch_size)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p
            self.max_prio = max(self.max_prio, p)

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Simple DQN network (for grid states)
# -------------------------
if TORCH_AVAILABLE:
    class DQNet(nn.Module):
        def __init__(self, input_dim, hidden=256, output_dim=4): # SỬA: hidden=256, output_dim=4 là giá trị mặc định được giữ lại
            super().__init__()
            # SỬA LỚP ĐẦU TIÊN: Dùng input_dim thay vì hard-code
            self.fc1 = nn.Linear(input_dim, hidden) 
            self.fc2 = nn.Linear(hidden, hidden // 2) # Giảm kích thước lớp ẩn
            self.fc3 = nn.Linear(hidden // 2, output_dim) # Dùng output_dim
            # LƯU Ý: Nếu env.action_size = 3, output_dim sẽ là 3

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

# -------------------------
# Advanced Agent
# -------------------------
class AdvancedAgent:
    def __init__(self,
                 mode='dqn',            # 'tabular' or 'dqn'
                 state_size=4,          # <<< THÊM: Kích thước state từ môi trường (Phải là 6)
                 grid_shape=(20,20),
                 action_size=5,         # Kích thước hành động (Phải là 3)
                 tabular_bins=20,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 replay_capacity=20000,
                 batch_size=64,
                 device=None):
        self.mode = mode
        self.state_size = state_size    # <<< LƯU: Kích thước state
        self.grid_w, self.grid_h = grid_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Tabular Q
        self.tabular_bins = tabular_bins
        # SỬA: Sử dụng state_size để tạo Q-table đúng chiều
        self.tabular_Q = np.zeros([tabular_bins] * self.state_size + [action_size]) 

        # Prioritized replay
        self.replay = PrioritizedReplay(capacity=replay_capacity)

        # Double DQN (PyTorch)
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        if self.mode == 'dqn':
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch required for mode='dqn'")
            # SỬA: Dùng self.state_size (sẽ là 6) làm input_dim
            self.online_net = DQNet(input_dim=self.state_size, hidden=256, output_dim=action_size).to(self.device)
            self.target_net = DQNet(input_dim=self.state_size, hidden=256, output_dim=action_size).to(self.device)
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss(reduction='none')  # for PER we need per-sample loss

        # bookkeeping
        self.learn_steps = 0
        self.update_target_every = 1000  # steps
        self.min_replay_size = 500

        # for hybrid path-following
        self.follow_planned_steps = 4  # default how many steps to follow planned path
        self.current_planned = []  # cached path (list of grid nodes)
        self.planned_action_idx = 0

        # statistics
        self.stats = {'episodes': 0, 'rewards': [], 'steps': [], 'successes': []}

    # -------------------------
    # Tabular helpers
    # -------------------------
    def discretize_state(self, state):
        # state expected normalized approx [-1,1] or raw; we will scale to bins
        s = np.clip(np.array(state), -1, 1)
        idx = ((s + 1.0) * 0.5 * (self.tabular_bins - 1)).astype(int)
        idx = np.clip(idx, 0, self.tabular_bins - 1)
        return tuple(idx.tolist())

    def tabular_get(self, state, action=None):
        key = self.discretize_state(state)
        if action is None:
            return self.tabular_Q[key]
        else:
            return self.tabular_Q[key + (action,)]

    def tabular_update(self, state, action, reward, next_state, done, alpha=0.1):
        s_idx = self.discretize_state(state)
        ns_idx = self.discretize_state(next_state)
        q_old = self.tabular_Q[s_idx + (action,)]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.tabular_Q[ns_idx])
        self.tabular_Q[s_idx + (action,)] += alpha * (target - q_old)

    # -------------------------
    # A* integration
    # -------------------------
    def plan_path(self, env):
        """
        env must provide:
          - get_blocked_cells() -> set of (gx,gy)
          - grid_w, grid_h attributes OR use self.grid_w/self.grid_h
          - world_to_grid(px,py) and grid_to_world_center(gx,gy) optional
        We'll build grid array from get_blocked_cells()
        """
        W = getattr(env, 'grid_w', self.grid_w)
        H = getattr(env, 'grid_h', self.grid_h)
        grid = np.zeros((H, W), dtype=np.int32)
        blocked = env.get_blocked_cells() if hasattr(env, 'get_blocked_cells') else set()
        for (gx,gy) in blocked:
            if 0 <= gx < W and 0 <= gy < H:
                grid[gy, gx] = 1
        # start and goal grid
        sx, sy = env.world_to_grid(env.car_pos[0], env.car_pos[1])
        # LƯU Ý: Giả định env.goal_positions đã được sửa ở bước trước
        if len(env.goal_positions) == 0: 
            return []
        gx, gy = env.world_to_grid(env.goal_positions[0][0], env.goal_positions[0][1])
        path = astar_grid((sx, sy), (gx, gy), grid)
        self.current_planned = path
        self.planned_action_idx = 0
        return path

    def get_planned_action(self, env):
        """Return next action following current_planned; recompute if empty"""
        if not self.current_planned:
            self.plan_path(env)
        if len(self.current_planned) < 2:
            return None
        # find current agent cell index in planned path (allow mismatch)
        sx, sy = env.world_to_grid(env.car_pos[0], env.car_pos[1])
        # if cached idx valid
        if self.planned_action_idx < len(self.current_planned)-1:
            s_node = self.current_planned[self.planned_action_idx]
            # if agent moved ahead of planned_action_idx, advance
            if (sx, sy) == self.current_planned[self.planned_action_idx+1]:
                self.planned_action_idx += 1
        # recompute nearest node index
        # find nearest index of path to current pos
        dists = [abs(n[0]-sx) + abs(n[1]-sy) for n in self.current_planned]
        if len(dists) == 0:
            return None
        idx = int(np.argmin(dists))
        if idx >= len(self.current_planned)-1:
            return None
        # return action from path[idx] -> path[idx+1]
        action = path_to_actions(self.current_planned[idx: idx+2])[0]
        return action

    # -------------------------
    # Action selection (hybrid)
    # -------------------------
    def act(self, state, env=None, training=True, hybrid_prob=0.5):
        """
        If hybrid_prob>0 and env provided, with probability hybrid_prob follow planned path for some steps
        Else use selected policy (tabular or dqn)
        """
        # hybrid planned path
        if env is not None and random.random() < hybrid_prob:
            act = self.get_planned_action(env)
            if act is not None:
                return act

        # else policy
        if self.mode == 'tabular':
            s_idx = self.discretize_state(state)
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_size)
            return int(np.argmax(self.tabular_Q[s_idx]))
        elif self.mode == 'dqn':
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_size)
            # state -> tensor
            if not TORCH_AVAILABLE:
                return random.randrange(self.action_size)
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.online_net(st).cpu().numpy()[0]
                return int(np.argmax(q))
        else:
            return random.randrange(self.action_size)

    # -------------------------
    # Learning step for DQN (Double DQN)
    # -------------------------
    def learn_dqn_step(self):
        if len(self.replay) < self.min_replay_size:
            return None
        batch, indices, weights = self.replay.sample(self.batch_size)
        # convert to tensors
        states = torch.tensor(np.vstack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # online selects best action for next_state
        q_next_online = self.online_net(next_states)
        best_next_actions = q_next_online.argmax(dim=1, keepdim=True)  # indices

        # target evaluates
        q_next_target = self.target_net(next_states).gather(1, best_next_actions)

        q_targets = rewards + (1.0 - dones) * (self.gamma * q_next_target)

        q_values = self.online_net(states).gather(1, actions)

        # loss per sample
        loss_per = self.loss_fn(q_values, q_targets.detach())
        loss = (loss_per * weights).mean()

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities by TD-error (abs)
        td_errors = (q_targets.detach() - q_values).abs().detach().cpu().numpy().squeeze()
        new_priorities = np.abs(td_errors) + 1e-6
        self.replay.update_priorities(indices, new_priorities)

        # target update
        self.learn_steps += 1
        if self.learn_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss.item()

    # -------------------------
    # Store transition
    # -------------------------
    def store_transition(self, s, a, r, ns, done):
        # For tabular training we might store raw states; for DQN store numpy arrays
        sample = (np.array(s, dtype=np.float32), int(a), float(r), np.array(ns, dtype=np.float32), bool(done))
        self.replay.push(sample)

    # -------------------------
    # Utilities: save/load
    # -------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = {
            'mode': self.mode,
            'state_size': self.state_size, # <<< Thêm state_size vào metadata
            'grid_shape': (self.grid_w, self.grid_h),
            'action_size': self.action_size,
            'tabular_bins': self.tabular_bins,
            'epsilon': self.epsilon,
            'stats': self.stats
        }
        torch_part = {}
        if self.mode == 'dqn' and TORCH_AVAILABLE:
            torch_part['online'] = self.online_net.state_dict()
            torch_part['target'] = self.target_net.state_dict()
            torch_part['optimizer'] = self.optimizer.state_dict()
        data = {'meta': meta, 'tabular_Q': self.tabular_Q, 'torch': torch_part}
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(data, f)

    def load(self, path):
        if not os.path.exists(path): return False
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        meta = data.get('meta', {})
        self.mode = meta.get('mode', self.mode)
        self.state_size = meta.get('state_size', self.state_size) # <<< Load state_size
        self.tabular_Q = data.get('tabular_Q', self.tabular_Q)
        if self.mode == 'dqn' and TORCH_AVAILABLE:
            tp = data.get('torch', {})
            if 'online' in tp:
                self.online_net.load_state_dict(tp['online'])
            if 'target' in tp:
                self.target_net.load_state_dict(tp['target'])
        return True

    # -------------------------
    # Stats & logging
    # -------------------------
    def update_stats(self, episode_reward, steps, success):
        self.stats['episodes'] += 1
        self.stats['rewards'].append(episode_reward)
        self.stats['steps'].append(steps)
        self.stats['successes'].append(1 if success else 0)

    def get_stats(self, last_n=100):
        if self.stats['episodes'] == 0:
            return None
        rewards = self.stats['rewards'][-last_n:]
        steps = self.stats['steps'][-last_n:]
        successes = self.stats['successes'][-last_n:]
        return {
            'avg_reward': float(np.mean(rewards)),
            'avg_steps': float(np.mean(steps)),
            'success_rate': float(np.mean(successes)),
            'episodes': int(self.stats['episodes']),
            'epsilon': float(self.epsilon)
        }

    # -------------------------
    # Visualization: Q-table / value map
    # -------------------------
    def visualize_tabular_q(self, axis=0, action=None, cmap='viridis', savepath=None):
        """
        Visualize a 2D slice of tabular Q.
        axis: which two axes remain. Because tabular_Q is N-D, we collapse other dims
        """
        Q = self.tabular_Q
        if Q.ndim != self.state_size + 1:
            print(f"Cannot visualize: Tabular Q dim is {Q.ndim}, expected {self.state_size + 1}")
            return
        
        # Collapse all dims except the first two state dims
        axes_to_collapse = tuple(range(2, self.state_size))
        Q_collapsed = np.max(Q, axis=axes_to_collapse) if axes_to_collapse else Q

        if action is not None:
            arr = Q_collapsed[..., action]
        else:
            arr = Q_collapsed.max(axis=-1)
            
        plt.figure(figsize=(6,5))
        plt.imshow(arr.T, origin='lower', cmap=cmap)
        plt.colorbar()
        plt.title('Tabular Q (collapsed)')
        if savepath:
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()

    def visualize_value_map_dqn(self, env, savepath=None, cmap='viridis'):
        """
        For each grid cell, query DQN network for state near center and plot max action value.
        env must provide grid dims and grid_to_world_center(gx,gy) and _get_state() or similar.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DQN visualization")
        W = getattr(env, 'grid_w', self.grid_w)
        H = getattr(env, 'grid_h', self.grid_h)
        vals = np.zeros((W, H), dtype=np.float32)
        for gx in range(W):
            for gy in range(H):
                # create pseudo-state: world center -> env._get_state() expects car_pos as env.car_pos
                cx, cy = env.grid_to_world_center(gx, gy)
                # temporarily set car pos and compute state
                old_pos = env.car_pos.copy()
                env.car_pos = np.array([cx, cy], dtype=float)
                s = env._get_state()
                # Kiểm tra kích thước state trước khi chuyển thành tensor
                if len(s) != self.state_size:
                    # Nếu state size không khớp, bỏ qua ô này
                    vals[gx, gy] = 0.0
                    env.car_pos = old_pos
                    continue
                    
                st = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q = self.online_net(st).cpu().numpy()[0]
                vals[gx, gy] = np.max(q)
                env.car_pos = old_pos
        # plot
        plt.figure(figsize=(6,5))
        plt.imshow(vals.T, origin='lower', cmap=cmap)
        plt.colorbar()
        plt.title('DQN value map (max action)')
        if savepath:
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()