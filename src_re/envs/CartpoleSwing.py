import gymnasium as gym
import numpy as np

class CartPoleSwing(gym.Wrapper):
    '''
    Modify CartPole so the cart keeps moving back and forth.
    '''
    def __init__(self, env, w_a=1.0, w_u=0.5, w_s=2.5):
        super().__init__(env)
        self.direction = 1  # 1 for right, -1 for left
        self.w_a = w_a  # Weight for alive bonus
        self.w_u = w_u  # Weight for upright bonus (reduced)
        self.w_s = w_s  # Weight for swing bonus (increased)
        self.target_margin = 0.3  # Distance from the edge to trigger direction change (reduced)
        self.target_left = -2.4 + self.target_margin
        self.target_right = 2.4 - self.target_margin
        self.last_x = 0  # Track previous x position to calculate velocity
        self.reached_target = False  # Flag to track if cart reached target
        self.time_in_center = 0  # Count timesteps spent near center

    def compute_reward(self, obs, action, terminated, truncated, info):
        # obs = [x, x_dot, theta, theta_dot]
        x, x_dot, theta, theta_dot = obs
        
        # Basic rewards
        alive_bonus = 0.1
        
        # Upright pole bonus (smaller theta is better) - reduced weight
        upright_bonus = 0.1 * (1 - abs(theta) / np.pi)
        
        # Direction-based reward
        if self.direction == 1:  # Moving right
            target = self.target_right
        else:  # Moving left
            target = self.target_left
            
        # Distance to target - make the penalty stronger for staying in the center
        distance_to_target = abs(x - target)
        distance_normalized = distance_to_target / 4.8  # Normalize by total track length
        
        # Strong exponential penalty for distance from target
        target_bonus = -0.5 * (distance_normalized ** 2)
        
        # Center penalty - extra penalty for staying near the center
        center_distance = abs(x)
        if center_distance < 0.5:  # If within 0.5 units of center
            target_bonus -= 0.5 * (1.0 - center_distance)  # Additional penalty
            
        # Direction bonus based on making progress toward target
        direction_bonus = 0.0
        making_progress = (self.direction == 1 and x_dot > 0) or (self.direction == -1 and x_dot < 0)
        
        if making_progress:
            # Larger reward for moving in the correct direction
            direction_bonus = 1.0 * abs(x_dot)
        else:
            # Strong penalty for moving in the wrong direction
            direction_bonus = -1.5 * abs(x_dot)
            
        # Check if target reached and switch direction - larger reward
        if (self.direction == 1 and x >= self.target_right) or (self.direction == -1 and x <= self.target_left):
            if not self.reached_target:
                self.reached_target = True
                target_bonus += 2.0  # Larger bonus for reaching target
            self.direction *= -1  # Switch direction
        else:
            self.reached_target = False
            
        total_reward = (self.w_a * alive_bonus + 
                        self.w_u * upright_bonus + 
                        self.w_s * (target_bonus + direction_bonus))
        
        return total_reward

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Custom termination only for falling off the track
        x = obs[0]
        terminated = bool(x < -2.4 or x > 2.4)
        
        # Track time spent near center
        if abs(x) < 0.5:
            self.time_in_center += 1
        else:
            self.time_in_center = 0
            
        # Custom reward
        reward = self.compute_reward(obs, action, terminated, truncated, info)
        
        # Add strong penalty for staying in center too long
        if self.time_in_center > 10:  # If stuck in center for 10+ steps
            reward -= 0.1 * self.time_in_center  # Increasing penalty the longer it stays
        
        # Store current x for next step
        self.last_x = x
        
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.direction = 1 if np.random.random() > 0.5 else -1  # Random initial direction
        self.last_x = obs[0]
        self.reached_target = False
        self.time_in_center = 0  # Reset center time counter
        return obs, info
    
    def update_weights(self, w_a=None, w_c=None, w_s=None):
        """Update reward weights during training"""
        if w_a is not None:
            self.w_a = w_a
        if w_c is not None:
            self.w_c = w_c
        if w_s is not None:
            self.w_s = w_s