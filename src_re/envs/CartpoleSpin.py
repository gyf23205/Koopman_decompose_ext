import gymnasium as gym
import numpy as np

class CartPoleSpin(gym.Wrapper):
    """
    Modify CartPole so the pole can spin freely.
    Termination from pole angle is removed, reward reshaped.
    """
    def __init__(self, env, w_a=1.0, w_c=3.0, w_s=1.0):
        super().__init__(env)
        self.w_a = w_a
        self.w_c = w_c
        self.w_s = w_s
        self.initial_w_s = w_s  # Store initial value for reference
        
        # Track angular velocity history for consistency reward
        self.theta_dot_history = []
        self.history_length = 20  # Number of steps to consider for consistency

    def compute_reward(self, obs, action, terminated, truncated, info):
        # obs = [x, x_dot, theta, theta_dot]
        x, x_dot, theta, theta_dot = obs

        # Keep track of angular velocity
        self.theta_dot_history.append(theta_dot)
        if len(self.theta_dot_history) > self.history_length:
            self.theta_dot_history.pop(0)

        # Basic rewards
        center_penalty = - (x / 2.4) ** 2
        alive_bonus = 0.01
        
        # Directional consistency reward
        if len(self.theta_dot_history) >= 3:  # Need at least 3 samples
            # Check if most recent velocities have the same sign (same direction)
            signs = [1 if vel > 0 else -1 if vel < 0 else 0 for vel in self.theta_dot_history[-3:]]
            if all(s == signs[0] and s != 0 for s in signs):
                # Reward consistent direction + speed
                consistency_reward = 0.1 * abs(theta_dot)
                # Additional bonus for maintaining direction over longer periods
                if len(self.theta_dot_history) >= self.history_length:
                    all_signs = [1 if vel > 0 else -1 if vel < 0 else 0 for vel in self.theta_dot_history]
                    if all(s == all_signs[0] and s != 0 for s in all_signs):
                        consistency_reward += 0.05  # Bonus for long-term consistency
            else:
                consistency_reward = -0.02 * abs(theta_dot)  # Penalty for direction changes
        else:
            consistency_reward = 0.05 * abs(theta_dot)  # Initial spin reward

        return self.w_a * alive_bonus + self.w_c * center_penalty + self.w_s * consistency_reward

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = self.env.step(action)

        # Remove pole-angle termination condition
        x, _, theta, _ = obs
        terminated = bool(
            x < -2.4 or x > 2.4   # keep cart bounds only
        )

        reward = self.compute_reward(obs, action, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        # Clear history on reset
        self.theta_dot_history = []
        return self.env.reset(*args, **kwargs)

    def update_w_s(self, new_w_s):
        """Update the w_s parameter during training"""
        self.w_s = new_w_s


# # Usage
# base = gym.make("CartPole-v1")
# env = CartPoleSpin(base)

# obs, info = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     print(reward, obs)
