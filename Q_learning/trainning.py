import numpy as np
import matplotlib.pyplot as plt
from env.visual_env import VisualDrivingEnv

#from agents.q_learning import QLearningAgent
from agents.advanced_agent import AdvancedAgent


from utils.logger import Logger
import os

class CurriculumTrainer:
    """
    Trainer với Curriculum Learning: bắt đầu dễ, tăng dần độ khó
    """
    def __init__(self, 
                 max_episodes=10000,
                 max_steps_per_episode=500,
                 save_interval=100,
                 render_interval=50,
                 model_dir='models',
                 log_dir='logs'):
        
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval = save_interval
        self.render_interval = render_interval
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Tạo directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger(log_dir)
        
        # Curriculum settings
        self.current_difficulty = 1
        self.max_difficulty = 5
        self.difficulty_threshold = 0.7  # success rate để tăng difficulty
        
    def train(self, 
              agent=None,
              env=None,
              start_episode=0,
              curriculum_learning=True,
              render=True):
        """
        Main training loop với curriculum learning
        
        Args:
            agent: QLearningAgent instance
            env: Environment instance
            start_episode: Episode bắt đầu (để continue training)
            curriculum_learning: Có dùng curriculum learning không
            render: Có render không
        """
        
        # Initialize agent nếu chưa có
        if agent is None:
            agent = AdvancedAgent(#QLearningAgent
                #state_size=4,
                #action_size=5,
                #learning_rate=0.1,
                #discount_factor=0.95,
                #epsilon=1.0,
                #epsilon_min=0.01,
                #epsilon_decay=0.995,
                #state_bins=20

                state_dim=4,
                action_dim=5,
                ddqn=True,
                prioritized=True,
                use_dqn=True
            )
        
        # Initialize environment
        if env is None:
            env = VisualDrivingEnv(
                difficulty=self.current_difficulty,
                width=600,
                height=400,
                render_speed=30
            )
        
        print("Starting Training...")
        print(f"Max Episodes: {self.max_episodes}")
        print(f"Starting Difficulty: {self.current_difficulty}")
        print(f"Curriculum Learning: {curriculum_learning}")
        print("-" * 60)
        
        best_reward = -float('inf')
        consecutive_successes = 0
        
        for episode in range(start_episode, self.max_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            # Episode loop
            while not done and steps < self.max_steps_per_episode:
                # Get action
                action = agent.get_action(state, training=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Update agent
                agent.update(state, action, reward, next_state, done)
                
                # Accumulate reward
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Render
                if render and (episode % self.render_interval == 0 or episode < 5):
                    env.render()
            
            # Experience replay sau mỗi episode
            if episode % 5 == 0:  # mỗi 5 episodes
                agent.replay_experience(n_replays=3)
            
            # Track success
            success = info.get('goal', False)
            if success:
                consecutive_successes += 1
            else:
                consecutive_successes = 0
            
            # Update agent stats
            agent.update_stats(episode_reward, steps, success)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Log progress
            stats = agent.get_stats()
            if stats:
                self.logger.log_episode(
                    episode=episode,
                    reward=episode_reward,
                    steps=steps,
                    epsilon=agent.epsilon,
                    success=success,
                    difficulty=self.current_difficulty
                )
                
                # Print progress
                if episode % 10 == 0:
                    print(f"Episode {episode:5d} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Steps: {steps:3d} | "
                          f"ε: {agent.epsilon:.3f} | "
                          f"Success Rate: {stats['success_rate']:.2%} | "
                          f"Difficulty: {self.current_difficulty}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(f"{self.model_dir}/best_model.pkl")
            
            # Save checkpoint
            if episode % self.save_interval == 0 and episode > 0:
                agent.save(f"{self.model_dir}/checkpoint_ep{episode}.pkl")
                self.logger.save_plots(f"{self.log_dir}/training_plots_ep{episode}.png")
            
            # CURRICULUM LEARNING: Tăng difficulty khi agent giỏi
            if curriculum_learning and episode > 100:  # sau 100 episodes
                if stats and stats['success_rate'] > self.difficulty_threshold:
                    if self.current_difficulty < self.max_difficulty:
                        self.current_difficulty += 1
                        print(f"\n🎓 LEVEL UP! Difficulty increased to {self.current_difficulty}")
                        print(f"   Success rate: {stats['success_rate']:.2%}")
                        
                        # Reset environment với difficulty mới
                        env.close()
                        env = VisualDrivingEnv(
                            difficulty=self.current_difficulty,
                            width=600,
                            height=400,
                            render_speed=30
                        )
                        
                        # Reset agent (giữ lại knowledge)
                        agent.reset_for_new_difficulty(keep_knowledge=True)
                        
                        # Reset consecutive successes
                        consecutive_successes = 0
                        print("-" * 60)
        
        # Final save
        agent.save(f"{self.model_dir}/final_model.pkl")
        self.logger.save_plots(f"{self.log_dir}/final_training_plots.png")
        
        print("\nTraining completed!")
        print(f"Total episodes: {self.max_episodes}")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final difficulty: {self.current_difficulty}")
        
        env.close()
        return agent

def continue_training(checkpoint_path, additional_episodes=1000):
    """
    Continue training từ checkpoint
    """
    # Load agent
    agent = AdvancedAgent()
    #agent = QLearningAgent()
    if not agent.load(checkpoint_path):
        print("Failed to load checkpoint")
        return None
    
    # Get current episode count
    start_episode = agent.stats['episodes']
    
    # Create trainer
    trainer = CurriculumTrainer(
        max_episodes=start_episode + additional_episodes
    )
    
    # Continue training
    trained_agent = trainer.train(
        agent=agent,
        start_episode=start_episode,
        curriculum_learning=True
    )
    
    return trained_agent

if __name__ == "__main__":
    # Cấu hình training
    trainer = CurriculumTrainer(
        max_episodes=5000,
        max_steps_per_episode=500,
        save_interval=100,
        render_interval=50
    )
    
    # Bắt đầu training
    agent = trainer.train(
        curriculum_learning=True,
        render=True
    )