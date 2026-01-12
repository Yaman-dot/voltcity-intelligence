# ============================================================
# Reinforcement Learning Agent for Student Performance Support
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import joblib
from tqdm import tqdm


class ActionRecommender:
    def __init__(self, data_path):
        self.df = joblib.load(f"{data_path}")

        required_cols = [
            "Battery Capacity (kWh)",
            "Charging Station Location_Chicago",
            "Charging Station Location_Houston",
            "Charging Station Location_Los Angeles",
            "Charging Station Location_New York",
            "Charging Station Location_San Francisco",
            "Charger Type",
            "Time of Day"
        ]
        
        self.ACTION_NAMES = {}
        self.n_actions = None
    def preprocess_Train(self):
        # Discretizing Continuous features
        self.df["Battery_Level"] = pd.qcut(self.df["Battery Capacity (kWh)"], 2, labels=[0, 1]).astype(int)
        location_cols = [
            "Charging Station Location_Chicago",
            "Charging Station Location_Houston",
            "Charging Station Location_Los Angeles",
            "Charging Station Location_New York",
            "Charging Station Location_San Francisco"
        ]
        self.df["Location_State"] = self.df[location_cols].idxmax(axis=1).str.replace("Charging Station Location_", "")
        location_mapping = {
            "Chicago": 0,
            "Houston": 1,
            "Los Angeles": 2,
            "New York": 3,
            "San Francisco": 4
        }
        self.df["Location_State"] = self.df["Location_State"].map(location_mapping)
        
        self.df["State"] = list(
            zip(
                self.df["Battery_Level"],
                self.df["Location_State"],
                self.df["Charger Type"],
                self.df["Time of Day"]
            )
        )
        self.ACTION_NAMES = {
            0: "Do Nothing",
            1: "Recommend Changing Location",
            2: "Delay Charging"
        }
        self.n_actions = 3

        self.df["Reward"] = 2 - (self.df["Charging Cost (USD)"] * 3)
        self.states = list(itertools.product([0,1],
                                [0,1,2,3,4],
                                [0,1,2],
                                [0,1,2,3], repeat=1))
        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)
        self.Q = np.zeros((self.n_states, self.n_actions))

        # ============================================================
        # 6. Q-learning Hyperparameters
        # ============================================================
        episodes = 10000
        alpha = 0.2
        gamma = 0.0        # single-step learning
        epsilon = 1.0
        epsilon_min = 0.05
        epsilon_decay = 0.97

        reward_history = []
        q_history = []

        # Group data for fast reward sampling
        grouped = {}
        for s in self.states:
            for a in range(self.n_actions):
                grouped[(s, a)] = self.df[self.df["State"] == s]
        # Initialize progress bar
        pbar = tqdm(range(episodes), desc="Training RL Agent")

        for ep in pbar:

            # Random state (offline RL from dataset)
            s = random.choice(self.states)
            s_idx = self.state_to_index[s]

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                a = np.random.randint(self.n_actions)
            else:
                a = int(np.argmax(self.Q[s_idx]))

            # Reward sampling
            candidates = grouped[(s, a)]
            if len(candidates) == 0:
                r = 0.0
            else:
                r = float(candidates.sample(1)["Reward"].iloc[0])

            # Q update
            self.Q[s_idx, a] += alpha * (r - self.Q[s_idx, a])

            reward_history.append(r)
            q_history.append(self.Q.copy())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    def recommend_action(self, battery_capacity, location_idx, charger_type, time_of_day):
        """
        Input raw student values
        Output recommended action name
        """

        # Convert input to low(0) / high(1) using dataset medians
        battery_level  = int(battery_capacity >= self.df["Battery Capacity (kWh)"].median())
        # Create location one-hot
        location_one_hot = [0, 0, 0, 0, 0]
        location_one_hot[location_idx] = 1

        state = (
            battery_level,
            location_idx,
            int(charger_type),
            int(time_of_day)
        )


        if state not in self.state_to_index:
            print(f"Warning: State {state} not found in training data.")
            print(f"Available states: {len(self.state_to_index)}")
            return "Do Nothing"  # Default action
        state_idx = self.state_to_index[state]
        best_action = int(np.argmax(self.Q[state_idx]))

        return self.ACTION_NAMES[best_action]