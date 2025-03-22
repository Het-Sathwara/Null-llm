import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import logging
import traceback
import json

# Import the fixed modules
from trading_env import TradingEnv
from model import TDQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Project/logs/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create directories
os.makedirs("Project/logs", exist_ok=True)
os.makedirs("Project/models", exist_ok=True)
os.makedirs("Project/plots", exist_ok=True)

# Set TensorFlow options for better performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Hide info messages, show warnings
tf.keras.backend.set_floatx('float32')  # Use float32 for better performance

try:
    logging.info(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"Using GPU: {gpus}")
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logging.info("Using CPU for training")
except Exception as e:
    logging.error(f"Error initializing TensorFlow: {str(e)}")
    logging.error(traceback.format_exc())
    sys.exit(1)

# Training parameters
EPISODES = 100           # Total training episodes (reduced from 500)
INITIAL_BALANCE = 100000 # Starting balance
BATCH_SIZE = 32         # Batch size for training (reduced from 64)
BUFFER_SIZE = 50000     # Replay buffer size (reduced from 100000)
GAMMA = 0.95            # Discount factor (reduced from 0.99)
EPSILON = 1.0           # Initial exploration rate
EPSILON_MIN = 0.1       # Minimum exploration rate
EPSILON_DECAY = 0.99    # Epsilon decay rate (increased from 0.995)
LEARNING_RATE = 0.0001  # Learning rate (reduced from 0.0005)
TARGET_UPDATE_FREQ = 5  # Update target network every N episodes (reduced from 10)
EVAL_INTERVAL = 5       # Evaluate every N episodes (reduced from 25)
SAVE_INTERVAL = 10      # Save model every N episodes (reduced from 50)

# Save hyperparameters
hyperparams = {
    'episodes': EPISODES,
    'initial_balance': INITIAL_BALANCE,
    'batch_size': BATCH_SIZE,
    'buffer_size': BUFFER_SIZE,
    'gamma': GAMMA,
    'epsilon': EPSILON,
    'epsilon_min': EPSILON_MIN,
    'epsilon_decay': EPSILON_DECAY,
    'learning_rate': LEARNING_RATE,
    'target_update_freq': TARGET_UPDATE_FREQ
}

with open('Project/models/hyperparameters.json', 'w') as f:
    json.dump(hyperparams, f, indent=4)

logging.info("Loading data...")
# Read the processed data
try:
    train_data = pd.read_csv("Project/data/processed/train_scaled.csv", index_col="Date", parse_dates=True)
    test_data = pd.read_csv("Project/data/processed/test_scaled.csv", index_col="Date", parse_dates=True)
except FileNotFoundError:
    logging.error("Processed data files not found. Please run the data preparation script first.")
    sys.exit(1)

logging.info(f"Training data shape: {train_data.shape}")
logging.info(f"Testing data shape: {test_data.shape}")
logging.info(f"Training data date range: {train_data.index.min()} to {train_data.index.max()}")
logging.info(f"Testing data date range: {test_data.index.min()} to {test_data.index.max()}")

logging.info("\nInitializing environment and agent...")
# Create trading environment
env = TradingEnv(train_data, initial_balance=INITIAL_BALANCE)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
logging.info(f"State size: {state_size}")
logging.info(f"Action size: {action_size}")

# Create agent
agent = TDQNAgent(
    state_size=state_size,
    action_size=action_size,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    learning_rate=LEARNING_RATE
)

# Training metrics
episode_rewards = []
episode_losses = []
episode_epsilons = []
best_reward = -np.inf

logging.info("\nStarting training...")
try:
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.replay_buffer) >= BATCH_SIZE:
                loss = agent.train()
                episode_loss += loss
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            
            # Optional: Render environment
            if episode % EVAL_INTERVAL == 0:
                env.render()
        
        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network(1.0)  # Hard update
        
        # Save metrics
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / step if step > 0 else 0)
        episode_epsilons.append(agent.epsilon)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = os.path.join("Project", "models", "best_model.weights.h5")
            agent.save(model_path)
            
        # Regular model saves
        if episode % SAVE_INTERVAL == 0:
            model_path = os.path.join("Project", "models", f"model_episode_{episode}.weights.h5")
            agent.save(model_path)
        
        # Log progress
        logging.info(f"\nEpisode {episode + 1}/{EPISODES}")
        logging.info(f"Total Reward: {episode_reward:.2f}")
        logging.info(f"Average Loss: {episode_losses[-1]:.6f}")
        logging.info(f"Epsilon: {agent.epsilon:.4f}")
        logging.info(f"Final Portfolio Value: ${info['portfolio_value']:.2f}")
        logging.info(f"Return: {info['return']:.2f}%")
        
        # Plot training progress
        if episode % EVAL_INTERVAL == 0:
            plt.figure(figsize=(15, 5))
            
            # Plot rewards
            plt.subplot(131)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            
            # Plot losses
            plt.subplot(132)
            plt.plot(episode_losses)
            plt.title('Average Loss per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            
            # Plot epsilon
            plt.subplot(133)
            plt.plot(episode_epsilons)
            plt.title('Epsilon Decay')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            
            plt.tight_layout()
            plt.savefig(f"Project/plots/training_progress_episode_{episode}.png")
            plt.close()

except KeyboardInterrupt:
    logging.info("\nTraining interrupted by user")
except Exception as e:
    logging.error(f"\nError during training: {str(e)}")
    logging.error(traceback.format_exc())
finally:
    # Save final model
    model_path = os.path.join("Project", "models", "final_model.weights.h5")
    agent.save(model_path)
    logging.info("\nTraining completed!")
    
    # Save training history
    history = {
        'rewards': episode_rewards,
        'losses': episode_losses,
        'epsilons': episode_epsilons
    }
    
    with open('Project/models/training_history.json', 'w') as f:
        json.dump(history, f)
        
    # Final plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(132)
    plt.plot(episode_losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(133)
    plt.plot(episode_epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig("Project/plots/final_training_progress.png")
    plt.close()