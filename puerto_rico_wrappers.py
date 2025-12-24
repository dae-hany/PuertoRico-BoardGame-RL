
import gymnasium as gym
import numpy as np
import puerto_rico_constants as c
from sb3_contrib.common.wrappers import ActionMasker

class PuertoRicoSelfPlayWrapper(gym.Wrapper):
    """
    Wrapper for Self-Play in 2-Player Puerto Rico.
    
    Features:
    1. Canonical Observation: 
       - Always presents the 'Current Player' as Player 0 in the observation vector.
       - Swaps player data if the current player is Player 1.
    2. Reward Shaping:
       - Calculates intermediate VP gains.
       - Assigns Win/Loss rewards at game end.
    3. Action Masking Compatibility:
       - Exposes `action_masks` method for MaskablePPO.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.prev_scores = {0: 0, 1: 0}
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_scores = {0: 0, 1: 0}
        
        # Canonicalize Observation
        current_p_idx = self.env.game_state.current_player_idx
        obs = self._get_canonical_obs(obs, current_p_idx)
        
        return obs, info

    def step(self, action):
        # Who is acting?
        current_p_idx = self.env.game_state.current_player_idx
        
        # Execute
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate Reward
        # 1. Intermediate Reward (VP Delta)
        # We need to calculate ANY score change for the ACTING player.
        # Note: step() might have advanced the queue, so env.game_state.current_player_idx 
        # might now be different (Next Player).
        # We need to reward the player who JUST ACTED (`current_p_idx`).
        
        scores, tie_breakers = self.env._calculate_score()
        
        # Delta for the actor
        current_score = scores[current_p_idx]
        delta = current_score - self.prev_scores[current_p_idx]
        
        # Update prev scores
        self.prev_scores = scores
        
        # Shaping Reward (e.g., 0.01 per VP point gain to guide learning)
        reward += delta * 0.01 
        
        # 2. Terminal Reward (Win/Loss)
        if terminated:
            # Determine Winner
            p0_score = scores[0]
            p1_score = scores[1]
            
            winner = -1
            if p0_score > p1_score:
                winner = 0
            elif p1_score > p0_score:
                winner = 1
            else:
                # Tie Breaker
                if tie_breakers[0] > tie_breakers[1]:
                    winner = 0
                elif tie_breakers[1] > tie_breakers[0]:
                    winner = 1
                else:
                    winner = -1 # True Tie
            
            # Assign Reward to the ACTOR
            if winner == current_p_idx:
                reward += 1.0
            elif winner == -1:
                reward += 0.0 # Tie
            else:
                reward -= 1.0 # Lost
                
            info['winner'] = winner
            info['scores'] = scores
        
        # Canonicalize Observation for the NEXT player (who is about to act)
        next_p_idx = self.env.game_state.current_player_idx
        obs = self._get_canonical_obs(obs, next_p_idx)
        
        return obs, reward, terminated, truncated, info

    def _get_canonical_obs(self, obs, player_idx):
        """
        Transform observation so `player_idx` (Current Player) is always at index 0 of `players_vec`.
        """
        if player_idx == 0:
            return obs
            
        # If Player 1 is current, Swap P0 and P1 in 'players'
        # obs structure: 'global', 'players', 'market_plantations'
        
        new_obs = obs.copy()
        p_vec = new_obs['players'].copy()
        
        # Swap rows 0 and 1
        # P0 -> P1 slot, P1 -> P0 slot
        new_obs['players'] = np.flip(p_vec, axis=0)
        
        # Update Global Vector Details if they depend on absolute index
        # global[32] = Governor Index
        # global[33] = Current Player Index
        # global[34] = Colonist Ship
        
        g_vec = new_obs['global'].copy()
        
        # Governor Index: If it was 0, and I am 1. Relative?
        # Let's make it Relative Governor: 1 if Me, 0 if Opponent.
        gov_idx = g_vec[32]
        relative_gov = 1 if gov_idx == player_idx else 0
        g_vec[32] = relative_gov
        
        # Current Player Index: Always 0 in canonical view
        g_vec[33] = 0 
        
        new_obs['global'] = g_vec
        
        return new_obs

    def action_masks(self):
        # MaskablePPO uses this.
        # Must return mask for the observable state.
        # The env's get_action_mask() logic relies on `game_state.current_player_idx`.
        # Since `game_state` is the ground truth, we can just call env.get_action_mask().
        return self.env.get_action_mask()

