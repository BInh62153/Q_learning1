# main.py
"""
Main script để chạy First Person Driving với nhiều options
Đặt file này ở root directory của project
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pygame
from first_person_trainer import FirstPersonTrainer
from agents.advanced_agent import AdvancedAgent
from env.first_person_env import FirstPersonDrivingEnv


# --- Utility để lấy State/Action Size (Đã sửa lỗi TypeError) ---
def get_env_sizes():
    """Tạo environment tạm thời để xác định kích thước state và action chính xác."""
    
    # SỬA LỖI: Khởi tạo an toàn bằng cách không truyền các tham số render/width/height.
    # Thử khởi tạo với difficulty=1 (thường là tham số cần thiết trong các môi trường RL)
    try:
        env = FirstPersonDrivingEnv(difficulty=1)
    except TypeError:
        # Nếu vẫn báo lỗi, thử khởi tạo không tham số
        try:
            env = FirstPersonDrivingEnv()
        except Exception as e:
            # Nếu không thể khởi tạo, báo lỗi rõ ràng và dừng chương trình
            raise RuntimeError(f"Không thể khởi tạo FirstPersonDrivingEnv để lấy kích thước: {e}")

    state_size = env.state_size
    action_size = env.action_size
    env.close()
    return state_size, action_size

STATE_SIZE, ACTION_SIZE = get_env_sizes()
# ----------------------------------------


def train_new(args):
    """Train agent mới từ đầu"""
    print("Starting NEW training...")
    
    trainer = FirstPersonTrainer(
        max_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_interval=args.save_interval,
        render_interval=args.render_interval,
        mode=args.mode,
        model_dir='models',
        log_dir='logs',
        # THÊM: Truyền kích thước state và action vào trainer để nó tạo Agent đúng
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )
    
    agent = trainer.train(
        render=args.render,
        use_hybrid=args.hybrid
    )
    
    return agent


def train_continue(args):
    """Continue training từ checkpoint"""
    print(f"Continuing training from {args.checkpoint}...")
    
    # Load agent
    # SỬA: Truyền STATE_SIZE và ACTION_SIZE
    agent = AdvancedAgent(mode=args.mode, state_size=STATE_SIZE, action_size=ACTION_SIZE)
    if not agent.load(args.checkpoint):
        print(" Failed to load checkpoint!")
        return None
    
    print(f" Loaded checkpoint from {args.checkpoint}")
    
    # Get current episode count
    start_episode = agent.stats.get('episodes', 0)
    
    # Create trainer
    trainer = FirstPersonTrainer(
        max_episodes=start_episode + args.episodes,
        mode=args.mode,
        model_dir='models',
        log_dir='logs',
        # THÊM: Truyền kích thước state và action
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )
    
    # Continue training
    trained_agent = trainer.train(
        agent=agent,
        start_episode=start_episode,
        render=args.render,
        use_hybrid=args.hybrid
    )
    
    return trained_agent


def evaluate_model(args):
    """Evaluate trained model"""
    print(f" Evaluating model from {args.checkpoint}...")
    
    # Load agent
    # SỬA: Truyền STATE_SIZE và ACTION_SIZE
    agent = AdvancedAgent(mode=args.mode, state_size=STATE_SIZE, action_size=ACTION_SIZE)
    if not agent.load(args.checkpoint):
        print("Failed to load model!")
        return
    
    print(" Model loaded successfully")
    
    # Create trainer for evaluation
    trainer = FirstPersonTrainer(
        mode=args.mode,
        # THÊM: Truyền kích thước state và action
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )
    
    # Run evaluation
    results = trainer.evaluate(
        agent=agent,
        n_episodes=args.eval_episodes,
        render=args.render
    )
    
    return results


def play_manual(args):
    """Chơi thủ công (manual control)"""
    print("  Manual Play Mode")
    print("=" * 50)
    print("Controls:")
    print("  ← → : Change lanes")
    print("  ESC : Quit")
    print("=" * 50)
    
    # Khởi tạo Env có tham số đầy đủ (vì đây là chế độ chơi, cần hiển thị)
    env = FirstPersonDrivingEnv(
        difficulty=1,
        width=600,
        height=400,
        render_speed=60
    )
    
    # SỬA: Thay thế action 1 (stay) bằng hành động mặc định phù hợp với env.action_size=3
    # Trong FirstPersonDrivingEnv: 0=LEFT, 1=STRAIGHT (hoặc TĂNG TỐC), 2=RIGHT
    # Giả định 1 là STRAIGHT
    
    running = True
    while running:
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n New game started!")
        
        while not done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
            
            # Handle input
            keys = pygame.key.get_pressed()
            action = 1  # STRAIGHT (Giả định)
            
            if keys[pygame.K_LEFT]:
                action = 0  # left
            elif keys[pygame.K_RIGHT]:
                action = 2  # right
            
            # Step
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render
            env.render()
            
            # Check done
            if done:
                print(f"\n{'='*50}")
                print(f" 🏁 Game Over!")
                print(f"{'='*50}")
                print(f"  Final Score: {env.score}")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Steps Survived: {env.steps_count}")
                print(f"  Collisions: {env.collision_count}")
                print(f"  Final Speed: {env.world_speed:.1f}")
                print(f"{'='*50}")
                
                # Ask to play again
                print("\nPress SPACE to play again, ESC to quit")
                waiting = True
                while waiting and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                waiting = False
                                running = False
    
    env.close()
    print(" Thanks for playing!")


def demo_agent(args):
    """Demo trained agent"""
    print(f"  Demo AI Agent from {args.checkpoint}...")
    
    # Load agent
    # SỬA: Truyền STATE_SIZE và ACTION_SIZE
    agent = AdvancedAgent(mode=args.mode, state_size=STATE_SIZE, action_size=ACTION_SIZE)
    if not agent.load(args.checkpoint):
        print(" Failed to load model!")
        return
    
    print(" Model loaded")
    print("=" * 50)
    print("Watch the AI play!")
    print("Press ESC to quit")
    print("=" * 50)
    
    env = FirstPersonDrivingEnv(
        difficulty=1,
        width=600,
        height=400,
        render_speed=30
    )
    
    episode = 0
    running = True
    
    while running:
        state = env.reset()
        done = False
        total_reward = 0
        episode += 1
        
        print(f"\n Episode {episode} started...")
        
        while not done and running:
            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
            
            # AI action (no exploration)
            # LƯU Ý: agent.act() trong chế độ 'dqn' không sử dụng 'env' nếu hybrid_prob=0.0
            action = agent.act(state, env=None, training=False, hybrid_prob=0.0)
            
            # Step
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render
            env.render()
        
        if running:
            print(f"Episode {episode} ended - Score: {env.score}, Reward: {total_reward:.2f}, Steps: {env.steps_count}")
            
            # Short pause before next episode
            import time
            time.sleep(1)
    
    env.close()
    print(" Demo ended!")


def quick_train():
    """Quick training với cấu hình mặc định"""
    print(" Quick Training Mode")
    print("=" * 50)
    print("Training with default configuration:")
    print("  • Mode: DQN")
    print("  • Episodes: 1000")
    print("  • Hybrid A*: Enabled")
    print("  • State Size:", STATE_SIZE)
    print("  • Action Size:", ACTION_SIZE)
    print("=" * 50)
    
    trainer = FirstPersonTrainer(
        max_episodes=1000,
        max_steps_per_episode=1000,
        save_interval=100,
        render_interval=50,
        mode='dqn',
        model_dir='models',
        log_dir='logs',
        # THÊM: Truyền kích thước state và action
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )
    
    agent = trainer.train(
        render=True,
        use_hybrid=True
    )
    
    print("\n Quick training completed!")
    return agent


def main():
    parser = argparse.ArgumentParser(
        description=" First Person Driving - RL Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (1000 episodes with defaults)
  python main.py quick
  
  # Train new agent
  python main.py train --mode dqn --episodes 2000
  
  # Continue training
  python main.py continue models/checkpoint_ep500.pkl --episodes 1000
  
  # Evaluate model
  python main.py eval models/best_score_model.pkl --eval-episodes 10
  
  # Demo AI playing
  python main.py demo models/best_score_model.pkl
  
  # Play manually
  python main.py play
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Quick train
    quick_parser = subparsers.add_parser('quick', help='Quick training with defaults')
    
    # Train new
    train_parser = subparsers.add_parser('train', help='Train new agent')
    train_parser.add_argument('--mode', type=str, default='dqn', 
                             choices=['dqn', 'tabular'],
                             help='Learning mode (default: dqn)')
    train_parser.add_argument('--episodes', type=int, default=2000,
                             help='Number of episodes (default: 2000)')
    train_parser.add_argument('--max-steps', type=int, default=1000,
                             help='Max steps per episode (default: 1000)')
    train_parser.add_argument('--save-interval', type=int, default=100,
                             help='Save checkpoint every N episodes (default: 100)')
    train_parser.add_argument('--render-interval', type=int, default=50,
                             help='Render every N episodes (default: 50)')
    train_parser.add_argument('--no-render', dest='render', action='store_false',
                             help='Disable rendering (faster training)')
    train_parser.add_argument('--no-hybrid', dest='hybrid', action='store_false',
                             help='Disable A* hybrid mode')
    train_parser.set_defaults(render=True, hybrid=True)
    
    # Continue training
    continue_parser = subparsers.add_parser('continue', help='Continue training from checkpoint')
    continue_parser.add_argument('checkpoint', type=str,
                                help='Path to checkpoint file')
    continue_parser.add_argument('--mode', type=str, default='dqn',
                                choices=['dqn', 'tabular'],
                                help='Learning mode (default: dqn)')
    continue_parser.add_argument('--episodes', type=int, default=1000,
                                help='Additional episodes to train (default: 1000)')
    continue_parser.add_argument('--no-render', dest='render', action='store_false')
    continue_parser.add_argument('--no-hybrid', dest='hybrid', action='store_false')
    continue_parser.set_defaults(render=True, hybrid=True)
    
    # Evaluate
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('checkpoint', type=str,
                            help='Path to model file')
    eval_parser.add_argument('--mode', type=str, default='dqn',
                            choices=['dqn', 'tabular'],
                            help='Learning mode (default: dqn)')
    eval_parser.add_argument('--eval-episodes', type=int, default=10,
                            help='Number of evaluation episodes (default: 10)')
    eval_parser.add_argument('--no-render', dest='render', action='store_false')
    eval_parser.set_defaults(render=True)
    
    # Demo
    demo_parser = subparsers.add_parser('demo', help='Demo trained agent playing')
    demo_parser.add_argument('checkpoint', type=str,
                            help='Path to model file')
    demo_parser.add_argument('--mode', type=str, default='dqn',
                            choices=['dqn', 'tabular'],
                            help='Learning mode (default: dqn)')
    
    # Manual play
    play_parser = subparsers.add_parser('play', help='Play manually (human control)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'quick':
        quick_train()
    elif args.command == 'train':
        train_new(args)
    elif args.command == 'continue':
        train_continue(args)
    elif args.command == 'eval':
        evaluate_model(args)
    elif args.command == 'demo':
        demo_agent(args)
    elif args.command == 'play':
        play_manual(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()