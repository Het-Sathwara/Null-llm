import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Input, Dropout
from tensorflow.keras.models import Model
from collections import deque
import random
import time
import os

class ReplayBuffer:
    """Experience replay buffer (Sec 4.1 of paper)"""
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            # Return random sample with replacement if buffer isn't large enough
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class TDQNAgent:
    """Trading Deep Q-Network Agent (Paper Sections 4.1-4.3)"""
    def __init__(self, state_size, action_size,
                 gamma=0.99,           # Discount factor
                 epsilon=1.0,          # Initial exploration rate
                 epsilon_min=0.1,      # Min exploration rate
                 epsilon_decay=0.995,  # Decay rate
                 batch_size=64,        # Training batch size
                 buffer_size=10000,    # Replay buffer size
                 learning_rate=0.001,  # Learning rate
                 tau=0.001):          # Soft update rate
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau  # For soft updates
        
        # Main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network(1.0)  # Full copy initially
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.total_steps = 0
        self.train_count = 0

    def _build_model(self):
        """Build DNN architecture (Paper Section 4.1)"""
        inputs = Input(shape=(self.state_size,))
        
        # First layer
        x = Dense(128, kernel_initializer='he_uniform')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(0.2)(x)  # Add dropout for regularization
        
        # Second layer
        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(0.2)(x)  # Add dropout for regularization
        
        # Output layer - Linear activation for Q-values
        outputs = Dense(self.action_size, activation='linear',
                      kernel_initializer='he_uniform')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=tf.keras.losses.Huber(),  # Huber loss is more robust to outliers
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target_network(self, tau=None):
        """
        Update target network weights
        If tau=1.0, performs a hard update (full copy)
        Otherwise, performs a soft update with the provided tau
        """
        if tau is None:
            tau = self.tau
            
        if tau >= 1.0:
            # Hard update - full copy
            self.target_model.set_weights(self.model.get_weights())
        else:
            # Soft update - weighted average
            weights = []
            target_weights = self.target_model.get_weights()
            model_weights = self.model.get_weights()
            
            for i in range(len(target_weights)):
                weights.append(tau * model_weights[i] + (1 - tau) * target_weights[i])
                
            self.target_model.set_weights(weights)

    def act(self, state):
        """Epsilon-greedy action selection"""
        self.total_steps += 1
        
        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        # Exploitation
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)

    def train(self):
        """Train using experience replay (Sec 4.1)"""
        if len(self.replay_buffer) < self.batch_size:
            return 0  # Return zero loss if not enough samples
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays and reshape
        states = np.array([np.array(state) for state in states])
        next_states = np.array([np.array(next_state) for next_state in next_states])
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=np.float32)
        
        # Handle NaN values in states or next_states
        if np.isnan(states).any() or np.isnan(next_states).any():
            print("Warning: NaN values detected in states or next_states", flush=True)
            # Replace NaN values with zeros
            states = np.nan_to_num(states, nan=0.0)
            next_states = np.nan_to_num(next_states, nan=0.0)
        
        # Get current Q values for the actions taken
        current_q_values = self.model.predict(states, verbose=0)
        
        # DDQN: Use main network to select actions, target network to estimate values
        next_q_values_main = self.model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)
        
        # Prepare targets for all transitions in the batch
        targets = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # Terminal state - only reward, no future value
                targets[i][actions[i]] = rewards[i]
            else:
                # Non-terminal state - reward + discounted future value
                # Select best action using MAIN network (DDQN)
                best_action = np.argmax(next_q_values_main[i])
                # Get Q value for that action from TARGET network
                next_q = next_q_values_target[i][best_action]
                # Add numerical stability by clipping values
                targets[i][actions[i]] = np.clip(
                    rewards[i] + self.gamma * next_q, 
                    -10, 
                    10
                )
        
        # Train the model and get the loss
        history = self.model.fit(
            states, 
            targets, 
            epochs=1, 
            batch_size=self.batch_size,
            verbose=0
        )
        
        loss = history.history['loss'][0]
        self.train_count += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Perform soft update of target network
        if self.train_count % 4 == 0:  # Every 4 training steps
            self.update_target_network()
            
        return loss
    
    def save(self, path):
        """Save model weights"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Ensure path ends with .weights.h5
            if not path.endswith('.weights.h5'):
                path = os.path.splitext(path)[0] + '.weights.h5'
            
            self.model.save_weights(path)
            print(f"Model weights saved to {path}", flush=True)
        except Exception as e:
            print(f"Error saving model: {e}", flush=True)

    def load(self, path):
        """Load model weights"""
        try:
            # Ensure path ends with .weights.h5
            if not path.endswith('.weights.h5'):
                path = os.path.splitext(path)[0] + '.weights.h5'
            
            self.model.load_weights(path)
            self.update_target_network(1.0)  # Hard update
            print(f"Model weights loaded from {path}", flush=True)
            return True
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            return False