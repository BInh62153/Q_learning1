import numpy as np
import pickle
import os
from collections import deque


class QLearningAgent:
    def __init__(
        self,
        state_size=6,
        action_size=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        state_bins=20
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = learning_rate
        self.gamma = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.state_bins = state_bins

        self.bins = [
            np.linspace(0, 1, state_bins),   # car lane normalized
            np.linspace(0, 1, state_bins),   # nearest obs lane
            np.linspace(0, 1, state_bins),   # nearest obs dist
            np.linspace(0, 1, state_bins),   # goal lane
            np.linspace(0, 1, state_bins),   # goal dist
            np.linspace(0, 1, state_bins),   # speed norm
        ]
        # Q-table: (bin^state_size) × action
        self.Q = np.zeros([state_bins] * state_size + [action_size])


        # Experience replay buffer
        self.memory = deque(maxlen=5000)

        # Stats
        self.stats = {
            "episodes": 0,
            "rewards": [],
            "steps": [],
            "successes": []
        }

    # ============================
    # State discretizing
    # ============================
    def discretize(self, state):
        """
        Convert continuous state → discrete tuple index
        """
        idx = []
        for i, s in enumerate(state):
            idx.append(
                np.digitize(s, self.bins[i]) - 1
            )
        return tuple(np.clip(idx, 0, self.state_bins - 1))

    # ============================
    # Action selection
    # ============================
    def get_action(self, state, training=True):
        s = self.discretize(state)

        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.argmax(self.Q[s])

    # ============================
    # Q-Learning update
    # ============================
    def update(self, state, action, reward, next_state, done):
        s = self.discretize(state)
        ns = self.discretize(next_state)

        q_old = self.Q[s][action]

        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.Q[ns])

        self.Q[s][action] += self.lr * (q_target - q_old)

        # Store into experience replay
        self.memory.append((state, action, reward, next_state, done))

    # ============================
    # Experience replay
    # ============================
    def replay_experience(self, n_replays=3):
        if len(self.memory) == 0:
            return

        n_replays = min(n_replays, len(self.memory))
        samples = np.random.choice(len(self.memory), n_replays, replace=False)

        for idx in samples:
            state, action, reward, next_state, done = self.memory[idx]
            self.update(state, action, reward, next_state, done)

    # ============================
    # Epsilon decay
    # ============================
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # ============================
    # Statistics
    # ============================
    def update_stats(self, episode_reward, steps, success):
        self.stats["episodes"] += 1
        self.stats["rewards"].append(episode_reward)
        self.stats["steps"].append(steps)
        self.stats["successes"].append(1 if success else 0)

    def get_stats(self, last_n=100):
        if self.stats["episodes"] == 0:
            return None

        rewards = self.stats["rewards"][-last_n:]
        steps = self.stats["steps"][-last_n:]
        successes = self.stats["successes"][-last_n:]

        return {
            "avg_reward": np.mean(rewards),
            "avg_steps": np.mean(steps),
            "success_rate": np.mean(successes),
            "episodes": self.stats["episodes"]
        }

    # ============================
    # Curriculum reset
    # ============================
    def reset_for_new_difficulty(self, keep_knowledge=True):
        """
        keep_knowledge = True → giữ Q table
        False → reset Q
        """
        if not keep_knowledge:
            self.Q = np.zeros_like(self.Q)

    # ============================
    # Save / Load
    # ============================
    def save(self, path):
        data = {
            "Q": self.Q,
            "epsilon": self.epsilon,
            "stats": self.stats,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.Q = data.get("Q", self.Q)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.stats = data.get("stats", self.stats)
        return True
