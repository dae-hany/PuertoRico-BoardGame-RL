
import gymnasium as gym
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
import puerto_rico_constants as c

def test_action_logic():
    print("Initialize Environment...")
    env = PuertoRicoEnv2P()
    obs, info = env.reset(seed=42)
    
    gs = env.game_state
    
    # 1. Initial State: Role Selection, Governor P0 starts
    print("\n--- Test 1: Initial State ---")
    print(f"Phase: {gs.phase} (Expected {c.PHASE_ROLE_SELECTION})")
    print(f"Current Player: {gs.current_player_idx} (Expected {gs.governor_idx})")
    assert gs.phase == c.PHASE_ROLE_SELECTION
    assert gs.current_player_idx == 0
    
    mask = env.get_action_mask()
    print(f"Settler Avail: {mask[c.ACTION_CHOOSE_ROLE_SETTLER]}")
    assert mask[c.ACTION_CHOOSE_ROLE_SETTLER] == 1
    
    # 2. P0 Picks Settler
    print("\n--- Test 2: P0 Picks Settler ---")
    obs, reward, term, trunc, info = env.step(c.ACTION_CHOOSE_ROLE_SETTLER)
    
    print(f"Phase: {gs.phase} (Expected {c.PHASE_SETTLER})")
    print(f"Current Player: {gs.current_player_idx} (Expected 0 - Queue starts with Selector)")
    assert gs.phase == c.PHASE_SETTLER
    assert gs.current_player_idx == 0
    
    mask = env.get_action_mask()
    print(f"Quarry Option: {mask[c.ACTION_SETTLER_TAKE_QUARRY]}")
    assert mask[c.ACTION_SETTLER_TAKE_QUARRY] == 1, "Selector should have privilege (Quarry)"
    
    # 3. P0 Takes Quarry
    print("\n--- Test 3: P0 Takes Quarry ---")
    # P0 Island Before
    print(f"P0 Island Slot 1: {gs.players[0].island[1]}") # Slot 0 has Corn/Indigo
    
    obs, reward, term, trunc, info = env.step(c.ACTION_SETTLER_TAKE_QUARRY)
    
    # P0 Island After (Should have Quarry in first empty slot)
    # Slot 0 is occupied by start tile. Slot 1 should be Quarry (ID 5).
    # Wait, simple loop finds first -1.
    print(f"P0 Island Slot 1: {gs.players[0].island[1]}")
    assert gs.players[0].island[1]['tile'] == c.PLANTATION_QUARRY
    
    print(f"Current Player: {gs.current_player_idx} (Expected 1)")
    assert gs.current_player_idx == 1
    
    mask = env.get_action_mask()
    print(f"P1 Quarry Option: {mask[c.ACTION_SETTLER_TAKE_QUARRY]}")
    assert mask[c.ACTION_SETTLER_TAKE_QUARRY] == 0, "Non-selector should NOT have privilege"
    print(f"P1 Market 0 Option: {mask[c.ACTION_SETTLER_TAKE_PLANTATION_0]}")
    assert mask[c.ACTION_SETTLER_TAKE_PLANTATION_0] == 1
    
    # 4. P1 Takes Plantation 0
    print("\n--- Test 4: P1 Takes Plantation 0 ---")
    # P1 Island Before
    print(f"P1 Island Slot 1: {gs.players[1].island[1]}")
    
    original_market_0 = gs.market_plantations[0]
    obs, reward, term, trunc, info = env.step(c.ACTION_SETTLER_TAKE_PLANTATION_0)
    
    # P1 Island After
    print(f"P1 Island Slot 1: {gs.players[1].island[1]}")
    assert gs.players[1].island[1]['tile'] == original_market_0
    
    print(f"Phase: {gs.phase} (Expected {c.PHASE_ROLE_SELECTION})")
    assert gs.phase == c.PHASE_ROLE_SELECTION
    
    # Turn Logic: Round 1, Turn 2. Gov (0) -> Opp (1).
    print(f"Current Player: {gs.current_player_idx} (Expected 1)")
    assert gs.current_player_idx == 1
    
    print(f"Roles Taken: {gs.roles_taken_count} (Expected 1)")
    
    mask = env.get_action_mask()
    print(f"Settler Avail: {mask[c.ACTION_CHOOSE_ROLE_SETTLER]}")
    assert mask[c.ACTION_CHOOSE_ROLE_SETTLER] == 0, "Settler should be taken"
    
    print("\nAll action logic tests passed successfully!")

if __name__ == "__main__":
    try:
        test_action_logic()
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        # Print Traceback manually if needed, but standard python will do it
        import traceback
        traceback.print_exc()
        sys.exit(1)
