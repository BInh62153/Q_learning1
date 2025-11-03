# test_setup.py
"""
Script để test xem mọi thứ đã setup đúng chưa
Chạy file này trước khi training
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("TESTING SETUP")
print("=" * 70)

# Test 1: Import modules
print("\n  Testing imports...")
try:
    import pygame
    print("    pygame imported")
except ImportError as e:
    print(f"    pygame import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("    numpy imported")
except ImportError as e:
    print(f"    numpy import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"    torch imported (version {torch.__version__})")
    if torch.cuda.is_available():
        print(f"    CUDA available! GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("     CUDA not available (will use CPU)")
except ImportError as e:
    print(f"    torch import failed: {e}")
    print("   → DQN mode won't work, but tabular mode will")

try:
    import matplotlib
    print("    matplotlib imported")
except ImportError as e:
    print(f"    matplotlib import failed: {e}")
    print("   → Visualization won't work")

# Test 2: Import project modules
print("\n  Testing project modules...")
try:
    from env.first_person_env import FirstPersonDrivingEnv
    print("    FirstPersonDrivingEnv imported")
except ImportError as e:
    print(f"    FirstPersonDrivingEnv import failed: {e}")
    sys.exit(1)

try:
    from agents.advanced_agent import AdvancedAgent
    print("    AdvancedAgent imported")
except ImportError as e:
    print(f"    AdvancedAgent import failed: {e}")
    sys.exit(1)

try:
    from first_person_trainer import FirstPersonTrainer
    print("    FirstPersonTrainer imported")
except ImportError as e:
    print(f"    FirstPersonTrainer import failed: {e}")
    sys.exit(1)

try:
    from utils.logger import Logger
    print("    Logger imported")
except ImportError as e:
    print(f"    Logger import failed: {e}")
    print("   → Logging might not work properly")

# Test 3: Check directories
print("\n Checking directories...")
dirs = ['models', 'logs', 'env', 'agents', 'utils']
for d in dirs:
    if os.path.exists(d):
        print(f"    {d}/ exists")
    else:
        print(f"    {d}/ not found, will be created during training")

# Test 4: Create environment
print("\n Testing environment creation...")
try:
    env = FirstPersonDrivingEnv(width=600, height=400)
    state = env.reset()
    print(f"    Environment created")
    print(f"   • State shape: {state.shape}")
    print(f"   • Action space: {env.actions}")
    print(f"   • Grid size: {env.grid_w}x{env.grid_h}")
    
    # Test step
    next_state, reward, done, info = env.step(1)
    print(f"    Environment step works")
    print(f"   • Next state shape: {next_state.shape}")
    print(f"   • Reward: {reward:.2f}")
    print(f"   • Done: {done}")
    
    env.close()
except Exception as e:
    print(f"   Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create agent (tabular)
print("\n  Testing tabular agent...")
try:
    agent = AdvancedAgent(
        mode='tabular',
        grid_shape=(20, 20),
        action_size=3
    )
    print(f"    Tabular agent created")
    print(f"   • Q-table shape: {agent.tabular_Q.shape}")
    
    # Test action
    state = np.array([0.5, 0.5, 0.7, 0.3, 0.8, 0.5])
    action = agent.act(state, training=True)
    print(f"   Agent can select action: {action}")
except Exception as e:
    print(f"    Tabular agent test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Create agent (DQN)
print("\n  Testing DQN agent...")
try:
    agent = AdvancedAgent(
        mode='dqn',
        grid_shape=(20, 20),
        action_size=3
    )
    print(f"     DQN agent created")
    print(f"   • Device: {agent.device}")
    print(f"   • Network: {type(agent.online_net).__name__}")
    
    # Test action
    state = np.array([0.5, 0.5, 0.7, 0.3, 0.8, 0.5])
    action = agent.act(state, training=True)
    print(f"    Agent can select action: {action}")
    
    # Test learning
    agent.store_transition(state, action, 1.0, state, False)
    if len(agent.replay) >= agent.min_replay_size:
        loss = agent.learn_dqn_step()
        print(f"    Learning works (loss: {loss:.4f})")
    else:
        print(f"     Not enough samples for learning yet")
        
except Exception as e:
    print(f"    DQN agent test failed: {e}")
    print("   → You can still use tabular mode")
    import traceback
    traceback.print_exc()

# Test 7: Quick render test
print("\n  Testing rendering (5 seconds)...")
try:
    env = FirstPersonDrivingEnv(width=600, height=400, render_speed=60)
    state = env.reset()
    print("     Window should appear for 5 seconds...")
    
    import time
    start_time = time.time()
    frames = 0
    
    while time.time() - start_time < 5:
        # Random action
        action = np.random.randint(0, 3)
        state, reward, done, info = env.step(action)
        env.render()
        frames += 1
        
        if done:
            state = env.reset()
    
    env.close()
    fps = frames / 5.0
    print(f"   Rendering works! (Average FPS: {fps:.1f})")
    
except Exception as e:
    print(f"    Rendering test failed: {e}")
    print("   → Training with --no-render might still work")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("SETUP TEST COMPLETE")
print("=" * 70)
print("\n All critical tests passed!")
print("\nYou can now run:")
print("  • python main.py play          # Play manually")
print("  • python main.py quick         # Quick training")
print("  • python main.py train --mode dqn --episodes 2000")
print("\n" + "=" * 70)