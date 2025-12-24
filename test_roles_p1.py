
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
import puerto_rico_constants as c

def test_roles():
    print("Initialize Environment...")
    env = PuertoRicoEnv2P()
    env.reset(seed=100) # Seed 100 for predictable logic
    gs = env.game_state
    
    print("\n=== Test 1: Prospector (P0) ===")
    initial_doubloons = gs.players[0].doubloons
    print(f"P0 Doubloons Before: {initial_doubloons}")
    
    # P0 picks Prospector
    env.step(c.ACTION_CHOOSE_ROLE_PROSPECTOR)
    
    print(f"P0 Doubloons After: {gs.players[0].doubloons}")
    assert gs.players[0].doubloons == initial_doubloons + 1
    
    # Check Phase: Should be Role Selection again (Prospector is instant)
    # But wait, logic says end_role_phase -> check round end. 
    # Roles count = 1.
    # Player Turn Check: Gov+RolesTaken%2 = (0+1)%2 = 1.
    print(f"Phase: {gs.phase} (Expected {c.PHASE_ROLE_SELECTION})")
    print(f"Current Player: {gs.current_player_idx} (Expected 1)")
    assert gs.phase == c.PHASE_ROLE_SELECTION
    assert gs.current_player_idx == 1
    
    print("\n=== Test 2: Settler (P1) ===")
    # P1 picks Settler
    # P1 has no privilege (Selector does, but Settler logic depends on who selected)
    # P1 (Selector) gets privilege.
    initial_island_p1 = gs.players[1].island[1]['tile'] # Should be -1
    initial_quarries = gs.supply_quarries
    
    print("P1 picks Settler...")
    env.step(c.ACTION_CHOOSE_ROLE_SETTLER)
    
    print(f"Phase: {gs.phase} (Expected {c.PHASE_SETTLER})")
    print(f"Privilege Check: {gs.current_role_privilege} (Expected True)")
    
    mask = env.get_action_mask()
    print(f"P1 Can Take Quarry: {mask[c.ACTION_SETTLER_TAKE_QUARRY]}")
    assert mask[c.ACTION_SETTLER_TAKE_QUARRY] == 1
    
    # P1 takes Quarry
    print("P1 takes Quarry...")
    env.step(c.ACTION_SETTLER_TAKE_QUARRY)
    
    assert gs.players[1].island[1]['tile'] == c.PLANTATION_QUARRY
    assert gs.supply_quarries == initial_quarries - 1
    
    # P0 Turn (Gov)
    print(f"Current Player: {gs.current_player_idx} (Expected 0)")
    
    # P0 takes first Market Plantation
    market_0 = gs.market_plantations[0]
    print(f"P0 takes Market[0] ({market_0})...")
    env.step(c.ACTION_SETTLER_TAKE_PLANTATION_0)
    
    # Check refill
    print(f"Market Size: {len(gs.market_plantations)} (Expected 3 refreshed)")
    assert len(gs.market_plantations) == 3
    assert gs.market_plantations[0] != market_0 # Likely different
    
    print("\n=== Test 3: Mayor (P0) ===")
    # Next turn: RolesTaken=2. (0+2)%2 = 0. P0 acts.
    print(f"Current Player: {gs.current_player_idx} (Expected 0)")
    
    # P0 picks Mayor
    # P0 gets +1 Privilege + (Ship/2 approx)
    # Ship starts with 2. 
    # Order: P0, P1, P0, P1...
    # P0 gets Privilege(1) + Ship(1) = 2.
    # P1 gets Ship(1) = 1.
    
    initial_workers_p0 = gs.players[0].san_juan_workers # likely 0
    
    print("P0 picks Mayor...")
    env.step(c.ACTION_CHOOSE_ROLE_MAYOR)
    
    # Check distribution
    print(f"Phase: {gs.phase} (Expected {c.PHASE_MAYOR})")
    
    # Note: Logic auto-lifts colonists to San Juan.
    # P0 starts with 1 Corn (0 workers).
    # So P0 San Juan = 0(lifted) + 2(new) = 2.
    print(f"P0 San Juan Workers: {gs.players[0].san_juan_workers} (Expected 2)")
    assert gs.players[0].san_juan_workers == 2
    
    # P0 must place. Can place on Corn (island[0]) or Quarry (island[1]? No P0 didn't take Quarry, P1 did).
    # P0: island[0] Corn, island[1] NewPlantation.
    # Can place on both.
    
    mask = env.get_action_mask()
    print(f"P0 Can Place Island 0: {mask[c.ACTION_MAYOR_PLACE_PLANTATION_0]}")
    assert mask[c.ACTION_MAYOR_PLACE_PLANTATION_0] == 1
    print(f"P0 Can Pass: {mask[c.ACTION_PASS]}")
    assert mask[c.ACTION_PASS] == 0 # Must place
    
    # Place on 0
    env.step(c.ACTION_MAYOR_PLACE_PLANTATION_0)
    print("Placed on 0.")
    print(f"P0 San Juan: {gs.players[0].san_juan_workers} (Expected 1)")
    
    # Place on 1
    env.step(c.ACTION_MAYOR_PLACE_PLANTATION_0 + 1)
    print("Placed on 1.")
    print(f"P0 San Juan: {gs.players[0].san_juan_workers} (Expected 0)")
    
    # Now must Pass
    mask = env.get_action_mask()
    print(f"P0 Can Place: {np.sum(mask[:-1])}") # All 0
    print(f"P0 Can Pass: {mask[c.ACTION_PASS]}")
    assert mask[c.ACTION_PASS] == 1
    
    print("P0 Passes...")
    env.step(c.ACTION_PASS)
    
    # P1 Turn
    print(f"Current Player: {gs.current_player_idx} (Expected 1)")
    # P1 San Juan = Lifted(0) + New(1) = 1.
    print(f"P1 San Juan: {gs.players[1].san_juan_workers} (Expected 1)")
    
    print("P1 Places on Island 0...")
    env.step(c.ACTION_MAYOR_PLACE_PLANTATION_0)
    
    print("P1 Passes...")
    env.step(c.ACTION_PASS)
    
    # Mayor End -> Refill Ship
    # Supply was 40. P0 took 1(Priv). Ship took 2. Supply = 37.
    # Refill = Max(Empty, 2). Current Empty: Many. 
    # Just check ship refreshed > 0.
    print(f"Colonist Ship: {gs.colonist_ship}")
    assert gs.colonist_ship >= 2
    
    print("\nAll Role tests passed successfully!")

if __name__ == "__main__":
    try:
        test_roles()
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
