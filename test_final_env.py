
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
import puerto_rico_constants as c

def test_final_env():
    print("Initialize Environment...")
    env = PuertoRicoEnv2P()
    env.reset(seed=500) 
    gs = env.game_state
    
    print("\n=== Test 1: Game End (VP Chips Exhausted) ===")
    # Simulate VP chips running out in Captain Phase
    gs.supply_vp = 2
    # Player 0 ships 3 goods (Corn).
    gs.players[0].goods[c.CORN] = 3
    gs.ships[0]['capacity'] = 4
    gs.ships[0]['good'] = c.CORN
    gs.ships[0]['count'] = 0
    
    # Set Phase
    gs.phase = c.PHASE_CAPTAIN
    gs.current_player_idx = 0
    gs.action_queue = [0, 1]
    
    print("P0 Ships Corn...")
    # Ship Corn (Action 34)
    env.step(c.ACTION_SHIP_CORN)
    
    # Check VP Supply
    print(f"VP Supply: {gs.supply_vp} (Expected <= 0)")
    assert gs.supply_vp <= 0
    
    # Check Game End Trigger
    print(f"Game End Triggered: {gs.game_end_triggered}")
    assert gs.game_end_triggered == True
    
    # Finish Round to trigger Game End Phase
    # Need to process all passes to end phase/round.
    # Force jump
    gs.phase = c.PHASE_GAME_END
    
    print("Game Phase set to GAME_END.")
    
    # Check Scoring
    # Give P0 a Building (Small Market) -> 1 VP
    gs.players[0].city[0]['building'] = c.BUILDING_SMALL_MARKET
    # Give P0 VP Chips -> 2 (from shipping above)
    gs.players[0].vp_chips = 2 # P0 started with 0, gained 2 (supply was 2)
    # Give P1 VP Chips -> 10
    gs.players[1].vp_chips = 10
    
    # Score
    scores, tie_breakers = env._calculate_score()
    print(f"Scores: {scores}, Ties: {tie_breakers}")
    
    # P0: 2 (Chips) + 1 (Building) = 3.
    # P1: 10 (Chips) = 10.
    assert scores[0] == 3
    assert scores[1] == 10
    
    print("\n=== Test 2: Game End (12 Buildings) ===")
    env.reset(seed=501)
    gs = env.game_state
    
    # Fill P1 City with 11 buildings
    for i in range(11):
        # Use valid IDs. Small Market is 15? No.
        # Just use i (0 to 10). Assuming 0-10 are valid building IDs.
        # 0=Small Market? Check constants.
        # Actually just assign distinct IDs if possible, or same ID (duplicates allowed in test setup for slot filling).
        gs.players[1].city[i]['building'] = i if i < c.NUM_BUILDINGS else 0 # Just fill slots
        gs.players[1].city[i]['building'] = c.BUILDING_SMALL_MARKET # Fill with Small Market (duplicates allowed? Rule: No. But test setup overrides logic).
        # We manually set state. So Duplicates are fine for verifying "Count=12 trigger".
        # But wait, logic might not care about duplicates for Game End, but standard play does.
        # We are testing Game End Trigger.
        gs.players[1].city[i]['building'] = c.BUILDING_SMALL_MARKET
        
    # Set Phase Builder
    gs.phase = c.PHASE_BUILDER
    gs.current_player_idx = 1
    gs.action_queue = [1]
    gs.players[1].doubloons = 100 # Rich
    
    # P1 Builds 12th Building
    # Can build ID 15?
    mask = env.get_action_mask()
    # Assume 15 is valid
    
    print("P1 Builds 12th Building...")
    # Find a valid building ID not built
    target_b = -1
    for b in range(c.NUM_BUILDINGS):
        if env.get_action_mask()[c.ACTION_BUILD_START + b] == 1:
            target_b = b
            break
            
    if target_b != -1:
        env.step(c.ACTION_BUILD_START + target_b)
        print(f"Game End Triggered: {gs.game_end_triggered}")
        assert gs.game_end_triggered == True
    else:
        print("Could not find building to build for test.")
        
    print("\n=== Test 3: Large Building Bonus ===")
    env.reset(seed=502)
    gs = env.game_state
    
    # P0 has Guild Hall (Occupied)
    gs.players[0].city[0] = {'building': c.BUILDING_GUILD_HALL, 'workers': 1}
    # P0 has Small Sugar (Production)
    gs.players[0].city[1] = {'building': c.BUILDING_SMALL_SUGAR, 'workers': 0}
    # P0 has Large Coffee (Production)
    gs.players[0].city[2] = {'building': c.BUILDING_COFFEE, 'workers': 0}
    
    # Score Calc
    # Guild Hall: 1 VP per Small Prod, 2 VP per Large Prod.
    # Small Sugar (ID 1) -> +1
    # Coffee (ID 5) -> +2
    # Bonus = 3
    # Base VP: Guild Hall(4) + Small Sugar(1) + Coffee(3) = 8.
    # Total = 11.
    
    scores, tie_breakers = env._calculate_score()
    print(f"P0 Score: {scores[0]} (Expected 11)")
    assert scores[0] == 11
    
    print("\nAll Final Tests passed successfully!")

if __name__ == "__main__":
    try:
        test_final_env()
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
