import torch
import numpy as np
from train_hanamikoji import HanamikojiEnv, PPO, ACTION_MAP, GEISHAS, CONFIG
import random
from copy import deepcopy

# --- Agents ---

class RandomAgent:
    def select_action(self, env, player_idx):
        mask = env.get_valid_actions(player_idx)
        valid_indices = np.where(mask == 1)[0]
        return np.random.choice(valid_indices)

class HeuristicAgent:
    def select_action(self, env, player_idx):
        mask = env.get_valid_actions(player_idx)
        valid_indices = np.where(mask == 1)[0]
        
        best_action = None
        best_score = -float('inf')
        
        # Simple Heuristic: Evaluate immediate material gain/loss
        # This is a simplified version of the JS "Smart AI"
        
        for action_idx in valid_indices:
            score = self.evaluate_move(env, player_idx, action_idx)
            # Add small random noise to break ties
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action

    def evaluate_move(self, env, player_idx, action_idx):
        action_type, action_data = ACTION_MAP[action_idx]
        
        # Get current hand cards (values)
        my_hand = env.p0_hand if player_idx == 0 else env.p1_hand
        # Note: In env, hand is list of geisha_ids (0-6). Value is GEISHAS[id].
        
        score = 0
        
        if action_type == 'SECRET':
            # Keep high value cards
            card_idx = action_data[0]
            card_id = my_hand[card_idx]
            score += GEISHAS[card_id] * 0.5
            
        elif action_type == 'DISCARD':
            # Discard low value or useless cards
            # Negative score for value lost
            c1 = my_hand[action_data[0]]
            c2 = my_hand[action_data[1]]
            score -= (GEISHAS[c1] + GEISHAS[c2])
            
        elif action_type == 'GIFT':
            # Offer 3 cards. Opponent takes 1. We get 2.
            # Assume opponent takes best. We get rest.
            # Maximize value of remaining 2.
            cards = [my_hand[i] for i in action_data]
            values = [GEISHAS[c] for c in cards]
            # Op takes max
            max_v = max(values)
            sum_v = sum(values)
            score += (sum_v - max_v)
            
        elif action_type == 'COMPETITION':
            # Offer 2 piles. Op takes better. We get worse.
            # Maximize min pile.
            p1_indices = action_data[0]
            p2_indices = action_data[1]
            
            v1 = sum([GEISHAS[my_hand[i]] for i in p1_indices])
            v2 = sum([GEISHAS[my_hand[i]] for i in p2_indices])
            
            score += min(v1, v2)
            
        elif action_type == 'RESPONSE_GIFT':
            # Pick best card
            # env.pending_action['cards'] has the cards
            choice = action_data
            cards = env.pending_action['cards']
            score += GEISHAS[cards[choice]]
            
        elif action_type == 'RESPONSE_COMP':
            # Pick best pile
            choice = action_data
            piles = env.pending_action['piles']
            pile = piles[choice]
            score += sum([GEISHAS[c] for c in pile])
            
        return score

class PPOAgent:
    def __init__(self, model_path):
        state_dim = 68
        action_dim = len(ACTION_MAP)
        self.ppo = PPO(state_dim, action_dim)
        try:
            self.ppo.load(model_path)
            print(f"Loaded PPO model from {model_path}")
        except:
            print(f"Could not load model from {model_path}, using random weights!")
            
    def select_action(self, env, player_idx):
        state = env.get_state(player_idx)
        mask = env.get_valid_actions(player_idx)
        action, _ = self.ppo.select_action(state, mask)
        return action

# --- Evaluation Loop ---

def evaluate(agent1, agent2, num_games=100):
    # Agent1 is P0, Agent2 is P1
    wins = {0: 0, 1: 0, 'draw': 0}
    
    for _ in range(num_games):
        env = HanamikojiEnv()
        state = env.reset()
        done = False
        
        while not done:
            current_player = env.turn if env.phase == 'ACTION' else (1 - env.pending_action['source'])
            
            if current_player == 0:
                action = agent1.select_action(env, 0)
            else:
                action = agent2.select_action(env, 1)
                
            _, _, done, _ = env.step(action)
            
        # Check winner
        p0_score = env.calculate_reward(0)
        if p0_score > 0: wins[0] += 1
        elif p0_score < 0: wins[1] += 1
        else: wins['draw'] += 1
        
    return wins

import pickle
import hashlib
import os

class MCTSAgent:
    def __init__(self, simulations=100):
        self.simulations = simulations
        
    def select_action(self, env, player_idx):
        mask = env.get_valid_actions(player_idx)
        valid_indices = np.where(mask == 1)[0]
        
        if len(valid_indices) == 0: return 0 
        if len(valid_indices) == 1: return valid_indices[0]
        
        scores = {}
        # Rollout MCTS
        for action_idx in valid_indices:
            total_score = 0
            for _ in range(self.simulations):
                sim_env = deepcopy(env)
                _, _, done, _ = sim_env.step(action_idx)
                while not done:
                    curr = sim_env.turn if sim_env.phase == 'ACTION' else (1 - sim_env.pending_action['source'])
                    mask_sim = sim_env.get_valid_actions(curr)
                    valid_sim = np.where(mask_sim == 1)[0]
                    if len(valid_sim) == 0: break
                    a_sim = np.random.choice(valid_sim)
                    _, _, done, _ = sim_env.step(a_sim)
                
                res = sim_env.calculate_reward(0)
                if player_idx == 1: res = -res
                total_score += res
            scores[action_idx] = total_score
            
        best_action = max(scores, key=scores.get)
        return best_action

if __name__ == "__main__":
    print("Initializing Agents...")
    ppo_agent = PPOAgent('hanamikoji_ai.pth')
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()
    # Use Standard MCTS
    mcts_agent = MCTSAgent(simulations=50) 
    
    try:
        # 1. PPO vs Random
        print("\n--- PPO vs Random (100 games) ---")
        res = evaluate(ppo_agent, random_agent, 100)
        print(f"PPO Wins: {res[0]}, Random Wins: {res[1]}, Draws: {res['draw']}")
        
        # 2. PPO vs Heuristic
        print("\n--- PPO vs Heuristic (100 games) ---")
        res = evaluate(ppo_agent, heuristic_agent, 100)
        print(f"PPO Wins: {res[0]}, Heuristic Wins: {res[1]}, Draws: {res['draw']}")
        
        # 3. PPO vs MCTS
        print("\n--- PPO vs MCTS (50 games) ---")
        res = evaluate(ppo_agent, mcts_agent, 50)
        print(f"PPO Wins: {res[0]}, MCTS Wins: {res[1]}, Draws: {res['draw']}")
        
        # 4. MCTS vs Heuristic
        print("\n--- MCTS vs Heuristic (50 games) ---")
        res = evaluate(mcts_agent, heuristic_agent, 50)
        print(f"MCTS Wins: {res[0]}, Heuristic Wins: {res[1]}, Draws: {res['draw']}")
        
        # 5. Heuristic vs Random
        print("\n--- Heuristic vs Random (100 games) ---")
        res = evaluate(heuristic_agent, random_agent, 100)
        print(f"Heuristic Wins: {res[0]}, Random Wins: {res[1]}, Draws: {res['draw']}")

    except KeyboardInterrupt:
        print("Interrupted.")
        
    print("Evaluation Complete.")
