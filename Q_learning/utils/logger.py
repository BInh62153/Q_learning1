import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class Logger:
    """
    Logger để tracking và visualize training progress
    """
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Data storage
        self.episodes = []
        self.rewards = []
        self.steps = []
        self.epsilons = []
        self.successes = []
        self.difficulties = []
        
        # Moving averages
        self.window_size = 100
        
        # Session info
        self.session_start = datetime.now()
        self.session_file = f"{log_dir}/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
    
    def log_episode(self, episode, reward, steps, epsilon, success, difficulty=1):
        """
        Log một episode
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.epsilons.append(epsilon)
        self.successes.append(1 if success else 0)
        self.difficulties.append(difficulty)
    
    def get_moving_average(self, data, window=None):
        """
        Calculate moving average
        """
        if window is None:
            window = self.window_size
        
        if len(data) < window:
            return data
        
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')
    
    def save_plots(self, filepath=None):
        """
        Vẽ và lưu training plots
        """
        if len(self.episodes) < 10:
            return
        
        if filepath is None:
            filepath = f"{self.log_dir}/training_plots.png"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # 1. Rewards
        ax = axes[0, 0]
        ax.plot(self.episodes, self.rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(self.rewards) >= self.window_size:
            moving_avg = self.get_moving_average(self.rewards)
            ax.plot(self.episodes[self.window_size-1:], moving_avg, 
                   color='red', linewidth=2, label=f'{self.window_size}-Episode MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Steps
        ax = axes[0, 1]
        ax.plot(self.episodes, self.steps, alpha=0.3, color='green', label='Steps')
        if len(self.steps) >= self.window_size:
            moving_avg = self.get_moving_average(self.steps)
            ax.plot(self.episodes[self.window_size-1:], moving_avg,
                   color='darkgreen', linewidth=2, label=f'{self.window_size}-Episode MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Epsilon
        ax = axes[0, 2]
        ax.plot(self.episodes, self.epsilons, color='orange', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon)')
        ax.grid(True, alpha=0.3)
        
        # 4. Success Rate
        ax = axes[1, 0]
        if len(self.successes) >= self.window_size:
            success_rate = self.get_moving_average(self.successes, window=100)
            ax.plot(self.episodes[99:], success_rate, color='purple', linewidth=2)
            ax.fill_between(self.episodes[99:], 0, success_rate, alpha=0.3, color='purple')
        else:
            ax.plot(self.episodes, self.successes, color='purple', marker='o')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate (100-episode window)')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # 5. Difficulty Progress
        ax = axes[1, 1]
        ax.plot(self.episodes, self.difficulties, color='brown', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Difficulty Level')
        ax.set_title('Curriculum Difficulty')
        ax.grid(True, alpha=0.3)
        
        # 6. Cumulative Reward
        ax = axes[1, 2]
        cumulative_rewards = np.cumsum(self.rewards)
        ax.plot(self.episodes, cumulative_rewards, color='teal', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Rewards')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {filepath}")
    
    def save_session_data(self):
        """
        Lưu session data dưới dạng JSON
        """
        session_data = {
            'start_time': self.session_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_episodes': len(self.episodes),
            'data': {
                'episodes': self.episodes,
                'rewards': self.rewards,
                'steps': self.steps,
                'epsilons': self.epsilons,
                'successes': self.successes,
                'difficulties': self.difficulties
            },
            'statistics': self.get_statistics()
        }
        
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session data saved to {self.session_file}")
    
    def get_statistics(self):
        """
        Tính toán statistics tổng hợp
        """
        if len(self.rewards) == 0:
            return {}
        
        recent_100 = min(100, len(self.rewards))
        
        stats = {
            'total_episodes': len(self.episodes),
            'average_reward': float(np.mean(self.rewards)),
            'average_steps': float(np.mean(self.steps)),
            'recent_100_avg_reward': float(np.mean(self.rewards[-recent_100:])),
            'recent_100_avg_steps': float(np.mean(self.steps[-recent_100:])),
            'best_reward': float(np.max(self.rewards)),
            'worst_reward': float(np.min(self.rewards)),
            'total_successes': int(np.sum(self.successes)),
            'success_rate': float(np.mean(self.successes)),
            'final_epsilon': float(self.epsilons[-1]) if self.epsilons else 1.0,
            'max_difficulty_reached': int(np.max(self.difficulties)) if self.difficulties else 1
        }
        
        return stats
    
    def print_summary(self):
        """
        In ra summary của training session
        """
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Duration: {datetime.now() - self.session_start}")
        print(f"Total Episodes: {stats.get('total_episodes', 0)}")
        print(f"Best Reward: {stats.get('best_reward', 0):.2f}")
        print(f"Average Reward: {stats.get('average_reward',0):.1f}")
        print(f"Total Successes: {stats.get('total_successes', 0)}")
        print(f"Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"Max Difficulty: {stats.get('max_difficulty_reached', 1)}")
        print(f"Final Epsilon: {stats.get('final_epsilon', 1.0):.4f}")
        print("="*60 + "\n")