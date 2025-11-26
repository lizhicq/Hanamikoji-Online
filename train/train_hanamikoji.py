import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import copy
from collections import deque
import time
import os
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    'lr': 3e-4,
    'gamma': 0.99,
    'eps_clip': 0.2,
    'k_epochs': 4,
    'batch_size': 64,
    'hidden_dim': 256,
    'update_timestep': 2000,
    'max_episodes': 1000000,
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    'save_interval': 1000,
    'model_path': 'hanamikoji_ai.pth'
}

print(f"Training on device: {CONFIG['device']}")

# --- Game Constants ---
GEISHAS = [2, 2, 2, 3, 3, 4, 5] # Values
ACTIONS = ['SECRET', 'DISCARD', 'GIFT', 'COMPETITION']
# Action Space:
# We need to flatten all possible moves into a discrete action space.
# Secret: 6 cards = 6 moves
# Discard: C(6,2) = 15 moves
# Gift: C(6,3) = 20 moves
# Competition: C(6,4)*3 = 45 moves
# Response (Gift): 3 choices (indices 0,1,2 of offered cards) -> actually max 3
# Response (Comp): 2 choices (indices 0,1 of piles)
# Total Action Space Size = 6 + 15 + 20 + 45 + 3 + 2 = 91 (Approx)
# To simplify, we can use a masked action space.

# Mapping actions to indices
# 0-5: Secret (Card Index)
# 6-20: Discard (Pair Indices)
# 21-40: Gift (Triple Indices)
# 41-85: Competition (4 Cards + Split)
# 86-88: Response Gift (Choice 0, 1, 2)
# 89-90: Response Comp (Choice 0, 1)

# Helper to generate combinations
from itertools import combinations

def get_combinations(lst, k):
    return list(combinations(lst, k))

# Precompute action mappings
ACTION_MAP = []
# Secret
for i in range(6): ACTION_MAP.append(('SECRET', [i]))
# Discard
for c in get_combinations(range(6), 2): ACTION_MAP.append(('DISCARD', list(c)))
# Gift
for c in get_combinations(range(6), 3): ACTION_MAP.append(('GIFT', list(c)))
# Competition
# For 4 cards, there are 3 ways to split into 2+2.
# {a,b,c,d} -> ({a,b},{c,d}), ({a,c},{b,d}), ({a,d},{b,c})
for c in get_combinations(range(6), 4):
    # c is tuple of 4 indices
    # Split 1: 0,1 vs 2,3
    ACTION_MAP.append(('COMPETITION', [[c[0], c[1]], [c[2], c[3]]]))
    # Split 2: 0,2 vs 1,3
    ACTION_MAP.append(('COMPETITION', [[c[0], c[2]], [c[1], c[3]]]))
    # Split 3: 0,3 vs 1,2
    ACTION_MAP.append(('COMPETITION', [[c[0], c[3]], [c[1], c[2]]]))

# Response Gift
for i in range(3): ACTION_MAP.append(('RESPONSE_GIFT', i))
# Response Comp
for i in range(2): ACTION_MAP.append(('RESPONSE_COMP', i))

ACTION_DIM = len(ACTION_MAP) # Should be 6+15+20+45+3+2 = 91

# --- Game Environment ---
class HanamikojiEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # 7 Geishas: 0-6. Values: 2,2,2,3,3,4,5
        self.favors = [-1] * 7 # -1: Neutral, 0: P0, 1: P1
        self.round = 1
        self.start_round()
        return self.get_state(0)

    def start_round(self):
        # Deck: 21 cards.
        self.deck = []
        for gid, val in enumerate(GEISHAS):
            for _ in range(val):
                self.deck.append(gid)
        random.shuffle(self.deck)
        
        self.removed_card = self.deck.pop()
        
        self.p0_hand = [self.deck.pop() for _ in range(6)]
        self.p1_hand = [self.deck.pop() for _ in range(6)]
        
        self.p0_actions = {k: False for k in ACTIONS}
        self.p1_actions = {k: False for k in ACTIONS}
        
        self.p0_side = {g: [] for g in range(7)}
        self.p1_side = {g: [] for g in range(7)}
        
        self.p0_secret = None
        self.p1_secret = None
        
        self.p0_discard = []
        self.p1_discard = []
        
        self.turn = (self.round - 1) % 2 # Alternate starting player
        self.phase = 'ACTION' # ACTION, RESPONSE
        self.pending_action = None # {type, source, cards/piles}
        
        self.history = []
        
        # Initial draw for the starting player
        self.draw_card(self.turn)
        
        return self.get_state(0)

    def get_state(self, player_idx):
        # Construct a vector representation of the state from player_idx perspective
        # 1. My Hand (One-hot x 6 cards? No, count of each Geisha type is better) -> 7 ints
        # 2. My Actions Used (4 bools)
        # 3. Op Actions Used (4 bools)
        # 4. My Side (Count per Geisha) -> 7 ints
        # 5. Op Side (Count per Geisha) -> 7 ints
        # 6. My Discard (Count per Geisha) -> 7 ints
        # 7. Op Discard (Count per Geisha) -> 7 ints
        # 8. My Secret (One-hot or 0 if none) -> 8 ints (7 types + 1 none)
        # 9. Pending Action info (if any)
        # New: Add Favor state (7 ints)
        # From player perspective: 1 if My Favor, -1 if Op Favor, 0 if Neutral
        
        my_hand = self.p0_hand if player_idx == 0 else self.p1_hand
        op_hand = self.p1_hand if player_idx == 0 else self.p0_hand # Hidden!
        
        my_actions = self.p0_actions if player_idx == 0 else self.p1_actions
        op_actions = self.p1_actions if player_idx == 0 else self.p0_actions
        
        my_side = self.p0_side if player_idx == 0 else self.p1_side
        op_side = self.p1_side if player_idx == 0 else self.p0_side
        
        my_discard = self.p0_discard if player_idx == 0 else self.p1_discard
        op_discard = self.p1_discard if player_idx == 0 else self.p0_discard
        
        my_secret = self.p0_secret if player_idx == 0 else self.p1_secret
        
        # Feature Vector Construction
        features = []
        
        # My Hand Counts (7)
        counts = [0]*7
        for c in my_hand: counts[c] += 1
        features.extend(counts)
        
        # My Actions (4)
        features.extend([1 if my_actions[a] else 0 for a in ACTIONS])
        
        # Op Actions (4)
        features.extend([1 if op_actions[a] else 0 for a in ACTIONS])
        
        # My Side Counts (7)
        features.extend([len(my_side[g]) for g in range(7)])
        
        # Op Side Counts (7)
        features.extend([len(op_side[g]) for g in range(7)])
        
        # My Discard Counts (7)
        counts = [0]*7
        for c in my_discard: counts[c] += 1
        features.extend(counts)
        
        # Op Discard Counts (7)
        counts = [0]*7
        for c in op_discard: counts[c] += 1
        features.extend(counts)
        
        # My Secret (8: 0-6 types, 7=None)
        secret_vec = [0]*8
        if my_secret is not None: secret_vec[my_secret] = 1
        else: secret_vec[7] = 1
        features.extend(secret_vec)
        
        # Pending Action (Type + Cards)
        # Type: None, Gift, Comp (3)
        # Cards: Counts (7)
        pending_vec = [0] * 10
        if self.pending_action:
            if self.pending_action['type'] == 'GIFT': pending_vec[0] = 1
            elif self.pending_action['type'] == 'COMPETITION': pending_vec[1] = 1
            
            # Cards in pending
            cards = []
            if 'cards' in self.pending_action: cards = self.pending_action['cards']
            if 'piles' in self.pending_action: 
                for p in self.pending_action['piles']: cards.extend(p)
            
            for c in cards: pending_vec[3+c] += 1
        else:
            pending_vec[2] = 1 # None
            
        features.extend(pending_vec)
        
        # Favor (7) - NEW
        favor_vec = []
        for g in range(7):
            f = self.favors[g]
            if f == -1: favor_vec.append(0)
            elif f == player_idx: favor_vec.append(1)
            else: favor_vec.append(-1)
        features.extend(favor_vec)
        
        return np.array(features, dtype=np.float32)

    def get_valid_actions(self, player_idx):
        # Returns a boolean mask of valid actions
        mask = [0] * ACTION_DIM
        
        if self.phase == 'ACTION':
            if self.turn != player_idx: return mask # No actions valid if not turn
            
            my_actions = self.p0_actions if player_idx == 0 else self.p1_actions
            my_hand = self.p0_hand if player_idx == 0 else self.p1_hand
            hand_len = len(my_hand)
            
            # Check each action type
            # Secret (0-5)
            if not my_actions['SECRET'] and hand_len >= 1:
                for i in range(hand_len): mask[i] = 1
                
            # Discard (6-20)
            if not my_actions['DISCARD'] and hand_len >= 2:
                for i in range(6, 21):
                    _, indices = ACTION_MAP[i]
                    if all(idx < hand_len for idx in indices): mask[i] = 1
                    
            # Gift (21-40)
            if not my_actions['GIFT'] and hand_len >= 3:
                for i in range(21, 41):
                    _, indices = ACTION_MAP[i]
                    if all(idx < hand_len for idx in indices): mask[i] = 1
                    
            # Competition (41-85)
            if not my_actions['COMPETITION'] and hand_len >= 4:
                for i in range(41, 86):
                    _, piles = ACTION_MAP[i]
                    all_indices = piles[0] + piles[1]
                    if all(idx < hand_len for idx in all_indices): mask[i] = 1
                    
        elif self.phase == 'RESPONSE':
            # Only the target player can act
            target = 1 - self.pending_action['source']
            if player_idx != target: return mask
            
            if self.pending_action['type'] == 'GIFT':
                for i in range(3): mask[86+i] = 1
            elif self.pending_action['type'] == 'COMPETITION':
                for i in range(2): mask[89+i] = 1
                
        return np.array(mask, dtype=np.bool_)

    def step(self, action_idx):
        player_idx = self.turn if self.phase == 'ACTION' else (1 - self.pending_action['source'])
        
        # Debug check
        if action_idx >= len(ACTION_MAP):
            print(f"Error: Invalid action_idx {action_idx}")
            return self.get_state(player_idx), 0, True, {}

        action_type, action_data = ACTION_MAP[action_idx]
        
        my_hand = self.p0_hand if player_idx == 0 else self.p1_hand
        
        # Debug for KeyError
        if action_type == 'COMPETITION' and len(my_hand) < 4:
            print(f"CRITICAL ERROR: Competition selected with hand size {len(my_hand)}")
            print(f"Player: {player_idx}, Phase: {self.phase}")
            print(f"Actions: {self.p0_actions if player_idx==0 else self.p1_actions}")
            print(f"Deck: {len(self.deck)}")
            # Force end episode
            return self.get_state(player_idx), 0, True, {}

        my_actions = self.p0_actions if player_idx == 0 else self.p1_actions
        my_side = self.p0_side if player_idx == 0 else self.p1_side
        my_discard = self.p0_discard if player_idx == 0 else self.p1_discard
        
        op_side = self.p1_side if player_idx == 0 else self.p0_side
        
        reward = 0
        done = False
        
        if self.phase == 'ACTION':
            # Execute Action
            if action_type == 'SECRET':
                idx = action_data[0]
                card = my_hand.pop(idx)
                if player_idx == 0: self.p0_secret = card
                else: self.p1_secret = card
                my_actions['SECRET'] = True
                self.turn = 1 - self.turn
                self.draw_card(self.turn)
                
            elif action_type == 'DISCARD':
                indices = sorted(action_data, reverse=True)
                for idx in indices:
                    my_discard.append(my_hand.pop(idx))
                my_actions['DISCARD'] = True
                self.turn = 1 - self.turn
                self.draw_card(self.turn)
                
            elif action_type == 'GIFT':
                indices = sorted(action_data, reverse=True)
                cards = []
                for idx in indices: cards.append(my_hand.pop(idx))
                self.pending_action = {'type': 'GIFT', 'source': player_idx, 'cards': cards}
                self.phase = 'RESPONSE'
                my_actions['GIFT'] = True
                
            elif action_type == 'COMPETITION':
                piles_indices = action_data
                flat_indices = sorted(piles_indices[0] + piles_indices[1], reverse=True)
                
                cards_map = {i: my_hand[i] for i in range(len(my_hand))}
                
                try:
                    pile1 = [cards_map[i] for i in piles_indices[0]]
                    pile2 = [cards_map[i] for i in piles_indices[1]]
                except KeyError as e:
                    print(f"KeyError in Competition: {e}")
                    print(f"Hand: {len(my_hand)}, Indices: {piles_indices}")
                    return self.get_state(player_idx), 0, True, {}

                for idx in flat_indices:
                    my_hand.pop(idx)
                    
                self.pending_action = {'type': 'COMPETITION', 'source': player_idx, 'piles': [pile1, pile2]}
                self.phase = 'RESPONSE'
                my_actions['COMPETITION'] = True

                
        elif self.phase == 'RESPONSE':
            source = self.pending_action['source']
            source_side = self.p0_side if source == 0 else self.p1_side
            
            if action_type == 'RESPONSE_GIFT':
                choice = action_data # 0, 1, 2
                cards = self.pending_action['cards']
                chosen = cards.pop(choice)
                my_side[chosen].append(chosen)
                for c in cards: source_side[c].append(c)
                
            elif action_type == 'RESPONSE_COMP':
                choice = action_data # 0, 1
                piles = self.pending_action['piles']
                chosen_pile = piles[choice]
                other_pile = piles[1-choice]
                
                for c in chosen_pile: my_side[c].append(c)
                for c in other_pile: source_side[c].append(c)
            
            self.pending_action = None
            self.phase = 'ACTION'
            self.turn = 1 - source # Turn goes to opponent of source? 
            # Rules: After response, the player who performed the action ends their turn.
            # So turn goes to the other player.
            # Wait, if P0 did Gift, P1 responds. Then P0's turn ends. P1 starts.
            # So turn = 1 - source. Correct.
            self.draw_card(self.turn)

        # Check Round End
        if all(self.p0_actions.values()) and all(self.p1_actions.values()):
            # Round End Logic
            
            # 1. Reveal Secrets
            if self.p0_secret is not None: self.p0_side[self.p0_secret].append(self.p0_secret)
            if self.p1_secret is not None: self.p1_side[self.p1_secret].append(self.p1_secret)
            
            # 2. Update Favors
            for g in range(7):
                p0_c = len(self.p0_side[g])
                p1_c = len(self.p1_side[g])
                if p0_c > p1_c: self.favors[g] = 0
                elif p1_c > p0_c: self.favors[g] = 1
                # Tie: Favor unchanged
            
            # 3. Check Win
            p0_score = self.calc_score(0)
            p1_score = self.calc_score(1)
            
            p0_win = False
            p1_win = False
            
            if p0_score['points'] >= 11 and p1_score['points'] < 11: p0_win = True
            elif p1_score['points'] >= 11 and p0_score['points'] < 11: p1_win = True
            elif p0_score['geishas'] >= 4 and p1_score['geishas'] < 4: p0_win = True
            elif p1_score['geishas'] >= 4 and p0_score['geishas'] < 4: p1_win = True
            elif p0_score['points'] >= 11 and p1_score['points'] >= 11:
                if p0_score['points'] > p1_score['points']: p0_win = True
                elif p1_score['points'] > p0_score['points']: p1_win = True
            elif p0_score['geishas'] >= 4 and p1_score['geishas'] >= 4:
                if p0_score['points'] > p1_score['points']: p0_win = True
                elif p1_score['points'] > p0_score['points']: p1_win = True
            
            if p0_win or p1_win:
                done = True
                if player_idx == 0: reward = 1 if p0_win else -1
                else: reward = 1 if p1_win else -1
            else:
                # No winner yet
                # Intermediate Reward for Favor Control
                # Calculate favor difference
                p0_favors = sum(1 for f in self.favors if f == 0)
                p1_favors = sum(1 for f in self.favors if f == 1)
                
                favor_diff = p0_favors - p1_favors
                if player_idx == 1: favor_diff = -favor_diff
                
                # Small reward for having more favors
                reward = favor_diff * 0.1
                
                if self.round >= 5:
                    done = True
                    # Draw - maybe small penalty or 0
                    reward += 0 
                else:
                    # Next Round
                    self.round += 1
                    self.start_round()
            
        return self.get_state(self.turn if self.phase == 'ACTION' else (1 - self.pending_action['source'])), reward, done, {}

    def draw_card(self, player_idx):
        # Only draw if deck not empty and hand < 6? 
        # Rules: Start with 6. Turn start -> Draw 1 -> 7. Action -> 6 or less.
        # Max 4 turns per player.
        if not self.deck: return
        # Check if player needs card
        # Actually, standard flow: P1 Draw -> Action. P2 Draw -> Action.
        # We only draw if it's the start of their turn action phase.
        
        target_hand = self.p0_hand if player_idx == 0 else self.p1_hand
        # If they have used all actions, don't draw
        target_actions = self.p0_actions if player_idx == 0 else self.p1_actions
        if all(target_actions.values()): return
        
        target_hand.append(self.deck.pop())

    def calculate_reward(self, player_idx):
        # This function is now only used for logging or auxiliary checks, 
        # as the main reward logic is inside step() for multi-round games.
        # But for compatibility with training loop which calls this at done:
        
        p0_score = self.calc_score(0)
        p1_score = self.calc_score(1)
        
        p0_win = False
        p1_win = False
        
        # ... (Same win logic as above) ...
        if p0_score['points'] >= 11 and p1_score['points'] < 11: p0_win = True
        elif p1_score['points'] >= 11 and p0_score['points'] < 11: p1_win = True
        elif p0_score['geishas'] >= 4 and p1_score['geishas'] < 4: p0_win = True
        elif p1_score['geishas'] >= 4 and p0_score['geishas'] < 4: p1_win = True
        elif p0_score['points'] >= 11 and p1_score['points'] >= 11:
            if p0_score['points'] > p1_score['points']: p0_win = True
            elif p1_score['points'] > p0_score['points']: p1_win = True
        elif p0_score['geishas'] >= 4 and p1_score['geishas'] >= 4:
            if p0_score['points'] > p1_score['points']: p0_win = True
            elif p1_score['points'] > p0_score['points']: p1_win = True
            
        if player_idx == 0: return 1 if p0_win else (-1 if p1_win else 0)
        else: return 1 if p1_win else (-1 if p0_win else 0)

    def calc_score(self, p_idx):
        geishas = 0
        points = 0
        for g in range(7):
            # Use self.favors!
            if self.favors[g] == p_idx:
                geishas += 1
                points += GEISHAS[g]
        return {'geishas': geishas, 'points': points}

# --- Neural Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, CONFIG['hidden_dim'])
        self.fc2 = nn.Linear(CONFIG['hidden_dim'], CONFIG['hidden_dim'])
        
        self.actor = nn.Linear(CONFIG['hidden_dim'], action_dim)
        self.critic = nn.Linear(CONFIG['hidden_dim'], 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x

    def act(self, state, mask):
        x = self.forward(state)
        action_logits = self.actor(x)
        
        # Apply mask
        # mask is already a tensor on device
        action_logits = action_logits.masked_fill(~mask, -1e9)
        
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action, action_logprob
    
    def evaluate(self, state, action):
        x = self.forward(state)
        action_logits = self.actor(x)
        state_values = self.critic(x)
        
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy

# --- Vectorized Environment ---
# --- Multiprocessing Vectorized Environment ---
import multiprocessing as mp

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'get_valid_actions':
                player_idx = env.turn if env.phase == 'ACTION' else (1 - env.pending_action['source'])
                mask = env.get_valid_actions(player_idx)
                remote.send(mask)
            elif cmd == 'get_current_player':
                p = env.turn if env.phase == 'ACTION' else (1 - env.pending_action['source'])
                remote.send(p)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('Worker interrupt')
    finally:
        remote.close()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VectorizedHanamikojiEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.waiting = False
        self.closed = False
        
        def make_env():
            return HanamikojiEnv()
            
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(make_env)))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_valid_actions(self):
        for remote in self.remotes:
            remote.send(('get_valid_actions', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_current_players(self):
        for remote in self.remotes:
            remote.send(('get_current_player', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

# --- PPO Agent (Updated for Batch) ---
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(CONFIG['device'])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=CONFIG['lr'])
        self.policy_old = ActorCritic(state_dim, action_dim).to(CONFIG['device'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state, mask):
        # state: (batch, state_dim)
        # mask: (batch, action_dim)
        state = torch.FloatTensor(state).to(CONFIG['device'])
        # Convert numpy bool array to torch tensor
        mask = torch.as_tensor(mask, dtype=torch.bool).to(CONFIG['device'])
        
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state, mask)
        
        return action.cpu().numpy(), action_logprob.cpu().numpy()
    
    def update(self, memory):
        # Flatten batched memory
        # memory.states is list of (batch, state_dim) tensors
        # memory.actions is list of (batch,) tensors
        
        old_states = torch.cat(memory.states, dim=0).to(CONFIG['device'])
        if len(old_states.shape) == 1:
             # This happens if state_dim is somehow lost or flattened incorrectly
             # We need to reshape to (-1, 68)
             # BUT wait, if we have 2026 states, 2026 * 68 = 137768
             # If old_states has 137768 elements, view(-1, 68) gives 2026 rows.
             # This matches tensor b (2026).
             # So state_values has 2026 rows.
             # But old_rewards has 4052 rows.
             # 4052 = 2 * 2026.
             # This means memory.rewards has 2x elements.
             # I suspect memory.states is NOT flattened?
             # memory.states is list of (68,) tensors.
             # cat -> (N*68,).
             # view -> (N, 68).
             # This path is correct.
             # So memory.states has N items.
             # memory.rewards has 2N items.
             # WHY?
             pass
             
        # Let's check memory.rewards construction again.
        # old_rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(CONFIG['device'])
        # If memory.rewards is list of floats.
        # Maybe I am adding lists to lists?
        # memory.rewards.extend(disc_rewards)
        # disc_rewards is list of floats.
        # This is correct.
        
        # Wait!
        # `discounted_reward = r + CONFIG['gamma'] * running_add`
        # `disc_rewards.insert(0, running_add)`
        # This is correct.
        
        # Is it possible that `memory` object is persisting across updates incorrectly?
        # `memory.clear()` clears everything.
        
        # Let's look at `memory.rewards` type.
        # It is list.
        
        # Is it possible `memory.states` is missing items?
        # `memory.states.extend(env_buffers[i]['states'])`
        # `env_buffers[i]['states']` is list of tensors.
        
        # Ah!
        # `env_buffers[i]['states'].append(torch.FloatTensor(states[i]))`
        # `states[i]` is (68,).
        
        # What if `states` variable in main loop is wrong?
        # `states = next_states`
        # `next_states` comes from `vec_env.step`.
        # `np.stack(next_states)`.
        # This is (64, 68).
        
        # I am completely baffled why rewards is 2x states.
        # Let's print the length of memory.states and memory.rewards in update.
        print(f"Update: Memory States {len(memory.states)}, Rewards {len(memory.rewards)}")
        
        old_states = torch.cat(memory.states, dim=0).to(CONFIG['device'])
        if len(old_states.shape) == 1:
             old_states = old_states.view(-1, 68)
        
        # Actions might be list of 0-D tensors (scalars)
        # We want to stack them into a 1D tensor
        old_actions = torch.stack(memory.actions).to(CONFIG['device'])
        
        old_logprobs = torch.stack(memory.logprobs).to(CONFIG['device'])
        
        # Flatten rewards list which might be nested or just long list
        # memory.rewards is a list of floats.
        old_rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(CONFIG['device'])
        
        # Debug shapes
        # print(f"States: {old_states.shape}, Actions: {old_actions.shape}, Rewards: {old_rewards.shape}")
        
        # Normalize rewards
        old_rewards = (old_rewards - old_rewards.mean()) / (old_rewards.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(CONFIG['k_epochs']):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Ensure rewards and state_values have same shape
            if state_values.dim() == 0: state_values = state_values.unsqueeze(0)
            
            ratios = torch.exp(logprobs - old_logprobs)
            
            advantages = old_rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-CONFIG['eps_clip'], 1+CONFIG['eps_clip']) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, old_rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean().item()
        
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(self.policy.state_dict())

# --- ActorCritic (Updated act for batch) ---
# Need to update ActorCritic.act to handle batch masking correctly if not already
# The previous implementation used masked_fill on logits, which works for batches too.
# But Categorical sampling needs to be checked.

    # In ActorCritic class (inside train_hanamikoji.py, need to ensure it's correct)
    # def act(self, state, mask):
    #     ...
    #     action = dist.sample() -> returns (batch_size,)
    #     ...

# --- Main Training Loop (Vectorized) ---
# --- Random Agent ---
class RandomAgent:
    def select_action(self, mask):
        valid_indices = np.where(mask == 1)[0]
        return np.random.choice(valid_indices)
class HeuristicTeacher:
    def get_action_probs(self, env, player_idx):
        mask = env.get_valid_actions(player_idx)
        valid_indices = np.where(mask == 1)[0]
        
        if len(valid_indices) == 0: return np.zeros(ACTION_DIM), 0
        if len(valid_indices) == 1: 
            probs = np.zeros(ACTION_DIM)
            probs[valid_indices[0]] = 1.0
            return probs, valid_indices[0]
            
        best_action = None
        best_score = -float('inf')
        
        for action_idx in valid_indices:
            score = self.evaluate_move(env, player_idx, action_idx)
            # Add small random noise to break ties
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best_action = action_idx
                
        probs = np.zeros(ACTION_DIM)
        probs[best_action] = 1.0
        return probs, best_action

    def evaluate_move(self, env, player_idx, action_idx):
        action_type, action_data = ACTION_MAP[action_idx]
        
        # Get current hand cards (values)
        my_hand = env.p0_hand if player_idx == 0 else env.p1_hand
        
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

# --- Main Training Loop (Self-Play) ---
def main():
    # Configuration
    NUM_ENVS = 64
    BATCH_SIZE = 2048 # Larger batch for PPO
    UPDATE_TIMESTEP = 2000 
    TRAIN_MODE = 'SELF_PLAY'
    
    vec_env = VectorizedHanamikojiEnv(NUM_ENVS)
    
    state_dim = 68
    action_dim = ACTION_DIM
    
    # Initialize PPO Agent
    ppo_agent = PPO(state_dim, action_dim)
    
    # Load pre-trained model if exists
    if os.path.exists(CONFIG['model_path']):
        try:
            ppo_agent.load(CONFIG['model_path'])
            print(f"Loaded pre-trained model from {CONFIG['model_path']}")
        except:
            print("Failed to load model, starting from scratch.")
    
    print(f"Starting PPO SELF-PLAY on {CONFIG['device']} with {NUM_ENVS} environments...")
    
    # Memory for PPO
    class Memory:
        def __init__(self):
            self.actions = []
            self.states = []
            self.logprobs = []
            self.rewards = []
            self.is_terminals = []
        
        def clear(self):
            del self.actions[:]
            del self.states[:]
            del self.logprobs[:]
            del self.rewards[:]
            del self.is_terminals[:]
            
    memory = Memory()
    
    # Track metrics
    total_timesteps = 0
    
    # For Self-Play, we need to store transitions for BOTH players.
    # But standard PPO usually trains one agent against a copy of itself or shared policy.
    # Here we use Shared Policy (Self-Play).
    # We need to be careful about rewards.
    # P0 gets reward R. P1 gets reward -R (Zero Sum).
    # We collect (s, a, r, s') for whoever acted.
    
    # To simplify, we can treat each step as a transition for the current player.
    # But we only get reward at the end of the game (or round).
    # So we need to store trajectories for each environment and assign rewards at the end.
    
    # Buffer for each env to store episode trajectory
    env_buffers = [{'states':[], 'actions':[], 'logprobs':[], 'rewards':[], 'dones':[]} for _ in range(NUM_ENVS)]
    
    states = vec_env.reset()
    
    try:
        # Run for 1,000,000 timesteps
        MAX_TIMESTEPS = 1000000
        pbar = tqdm(total=MAX_TIMESTEPS)
        
        while total_timesteps < MAX_TIMESTEPS:
            
            # 1. Select Action (Self-Play)
            # Both players use the same policy
            masks = vec_env.get_valid_actions()
            
            # PPO Select Action
            # states: (NUM_ENVS, 68)
            # masks: (NUM_ENVS, 91)
            actions, logprobs = ppo_agent.select_action(states, masks)
            
            # 2. Step
            next_states, rewards, dones, _ = vec_env.step(actions)
            
            # 3. Store transitions
            # We need to attribute reward to the player who acted?
            # In Hanamikoji, turns alternate.
            # If P0 acts, state transitions to S'. 
            # Reward is usually 0 until end.
            
            current_players = vec_env.get_current_players() # Who JUST acted? No, this returns who is ABOUT to act.
            # We need who acted to produce 'next_states'.
            # The 'states' and 'actions' we just used belong to the player who was active at 'states'.
            # So we need to know who was active at 'states'.
            # We can re-calculate or store it.
            # Let's assume we store it.
            
            # Actually, for PPO in self-play zero-sum:
            # We can just store (s, a, r) for the agent.
            # Since we share the network, every move is a training sample.
            # The only trick is the reward.
            # If P0 wins (+1), P1 loses (-1).
            # We need to propagate +1 to all P0 moves and -1 to all P1 moves in that episode.
            
            for i in range(NUM_ENVS):
                # Store current step
                env_buffers[i]['states'].append(torch.FloatTensor(states[i]))
                env_buffers[i]['actions'].append(torch.tensor(actions[i]))
                env_buffers[i]['logprobs'].append(torch.tensor(logprobs[i]))
                env_buffers[i]['dones'].append(dones[i])
                
                # We don't know the final reward yet, so we store placeholder or intermediate
                # We will backpropagate final reward when done.
                
                if dones[i]:
                    # Episode Done. Calculate Rewards.
                    # rewards[i] is the reward for P0 (from perspective of P0).
                    # If P0 won, r=1. If P1 won, r=-1.
                    
                    final_reward_p0 = rewards[i]
                    
                    # Backtrack through buffer and assign rewards
                    # We need to know who played each move.
                    # We can infer from state? Or just store player_idx.
                    # Let's assume we alternate, but some actions (Response) might break strict alternation?
                    # Hanamikoji is strict: P0 -> P1 -> P0 -> P1 ... 
                    # EXCEPT: Gift/Comp offer -> Opponent Response -> Original Player Turn Ends -> Opponent Turn.
                    # So: P0 Offer -> P1 Response -> P1 Turn Start.
                    # The sequence of moves:
                    # 1. P0 (Offer)
                    # 2. P1 (Response)
                    # 3. P1 (Offer)
                    # ...
                    
                    # We need to store player_idx in buffer.
                    # Let's re-implement buffer to store player_idx.
                    pass
            
            # Re-do loop with player tracking
            # We need player_idx for the state we just acted on.
            # We can get it from the envs BEFORE step, but we already stepped.
            # But we have `vec_env.get_current_players()` which gives current turn.
            # We called it before step? No.
            # Let's get it before step in next iteration.
            
            # Correct approach:
            # 1. Get current players
            # 2. Select actions
            # 3. Step
            # 4. Store (s, a, logp, player)
            
            # Since we are inside the loop, let's fix the flow.
            # We need to restart the loop structure.
            break # Break to restart with correct logic
            
    except:
        pass

    # --- Real Loop ---
    
    # Reset buffers
    env_buffers = [{'states':[], 'actions':[], 'logprobs':[], 'players':[]} for _ in range(NUM_ENVS)]
    states = vec_env.reset()
    
    pbar = tqdm(total=MAX_TIMESTEPS)
    
    while total_timesteps < MAX_TIMESTEPS:
        
        # 1. Get who is about to act
        active_players = vec_env.get_current_players()
        masks = vec_env.get_valid_actions()
        
        # 2. Select Action
        actions, logprobs = ppo_agent.select_action(states, masks)
        
        # 3. Step
        next_states, rewards, dones, _ = vec_env.step(actions)
        
        # 4. Store Data
        for i in range(NUM_ENVS):
            buf = env_buffers[i]
            buf['states'].append(torch.FloatTensor(states[i]))
            buf['actions'].append(torch.tensor(actions[i]))
            buf['logprobs'].append(torch.tensor(logprobs[i]))
            buf['players'].append(active_players[i])
            
            if dones[i]:
                # Episode finished
                # rewards[i] is P0's reward (1 or -1)
                r_p0 = rewards[i]
                
                # Construct discounted rewards
                # We need to assign r_p0 to P0's moves and -r_p0 to P1's moves
                
                ep_len = len(buf['states'])
                disc_rewards = []
                
                # We can use simple reward: Win=+1, Loss=-1 for every move
                # Or discounted. Let's use discounted.
                
                running_add_p0 = r_p0
                running_add_p1 = -r_p0
                
                # Iterate backwards
                ep_rewards = [0] * ep_len
                
                # We need to handle gamma carefully.
                # Usually we discount from the end.
                
                # Simple approach: Everyone gets the final reward discounted by distance
                for t in range(ep_len - 1, -1, -1):
                    p = buf['players'][t]
                    if p == 0:
                        ep_rewards[t] = running_add_p0
                        running_add_p0 = running_add_p0 * CONFIG['gamma']
                        # P1's reward decay? Or just independent?
                        # If P0 wins, P1 loses.
                        # P1's reward should also be discounted from end.
                        running_add_p1 = running_add_p1 * CONFIG['gamma'] 
                    else:
                        ep_rewards[t] = running_add_p1
                        running_add_p1 = running_add_p1 * CONFIG['gamma']
                        running_add_p0 = running_add_p0 * CONFIG['gamma']
                        
                # Add to global memory
                memory.states.extend(buf['states'])
                memory.actions.extend(buf['actions'])
                memory.logprobs.extend(buf['logprobs'])
                memory.rewards.extend(ep_rewards)
                memory.is_terminals.extend([False]*(ep_len-1) + [True])
                
                # Clear env buffer
                buf['states'] = []
                buf['actions'] = []
                buf['logprobs'] = []
                buf['players'] = []
                
        states = next_states
        total_timesteps += NUM_ENVS
        pbar.update(NUM_ENVS)
        
        # 5. Update PPO
        if len(memory.states) >= BATCH_SIZE:
            loss = ppo_agent.update(memory)
            memory.clear()
            
        # 6. Save
        if total_timesteps % 10000 == 0:
             ppo_agent.save(CONFIG['model_path'])
             
    pbar.close()
    ppo_agent.save(CONFIG['model_path'])
    print("Self-Play Training Complete.")

if __name__ == '__main__':
    main()
