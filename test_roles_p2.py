
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
import puerto_rico_constants as c

def test_roles_p2():
    print("Initialize Environment...")
    env = PuertoRicoEnv2P()
    env.reset(seed=200) 
    gs = env.game_state
    
    # Setup for Builder Test: Give P0 (Gov) money and Quarry
    # P0 starts with 3 Doubloons.
    # Let's give P0 a Quarry on Island[1] and 1 Worker on it.
    gs.players[0].island[1] = {'tile': c.PLANTATION_QUARRY, 'workers': 1}
    # And 2 more doubloons -> Total 5.
    gs.players[0].doubloons = 5
    print(f"P0 Setup: 5 Doubloons, 1 Occupied Quarry.")
    
    print("\n=== Test 1: Builder (P0) ===")
    # P0 picks Builder
    # Discount: 1 (Privilege) + 1 (Quarry) = 2.
    # Target: Small Market (Cost 1). Free?
    # Target: Construction Hut (Cost 2). Free?
    # Target: Large Market (Cost 5). Cost 3?
    
    env.step(c.ACTION_CHOOSE_ROLE_BUILDER)
    
    mask = env.get_action_mask()
    
    # Check Large Market (Cost 5)
    # 5 - 1(Priv) - 1(Quarry, Limit 2) = 3.
    # P0 has 5. Should be affordable.
    print(f"Can Build Large Market (Cost 5->3): {mask[c.ACTION_BUILD_START + c.BUILDING_LARGE_MARKET]}")
    assert mask[c.ACTION_BUILD_START + c.BUILDING_LARGE_MARKET] == 1
    
    # Check Factory (Cost 7)
    # 7 - 1 - 1 = 5. Affordable.
    print(f"Can Build Factory (Cost 7->5): {mask[c.ACTION_BUILD_START + c.BUILDING_FACTORY]}")
    assert mask[c.ACTION_BUILD_START + c.BUILDING_FACTORY] == 1
    
    # Check Guild Hall (Cost 10)
    # 10 - 1 - 1 = 8. Not Affordable (5 < 8).
    print(f"Can Build Guild Hall (Cost 10->8): {mask[c.ACTION_BUILD_START + c.BUILDING_GUILD_HALL]}")
    assert mask[c.ACTION_BUILD_START + c.BUILDING_GUILD_HALL] == 0
    
    # Build Large Market
    print("P0 building Large Market...")
    env.step(c.ACTION_BUILD_START + c.BUILDING_LARGE_MARKET)
    
    # Check P0 State
    # Cost 3 paid. Remainder 2.
    print(f"P0 Doubloons: {gs.players[0].doubloons} (Expected 2)")
    assert gs.players[0].doubloons == 2
    print(f"P0 City Slot 0: {gs.players[0].city[0]}")
    assert gs.players[0].city[0]['building'] == c.BUILDING_LARGE_MARKET
    
    # P1 Turn (Builder Phase, no privilege)
    # P1 has 3 Doubloons. No Quarry.
    # Can build Small Sugar (Cost 2)? Yes.
    # Can build Large Sugar (Cost 4)? No.
    mask = env.get_action_mask()
    print(f"P1 Can Build Small Sugar (Cost 2): {mask[c.ACTION_BUILD_START + c.BUILDING_SMALL_SUGAR]}")
    assert mask[c.ACTION_BUILD_START + c.BUILDING_SMALL_SUGAR] == 1
    print(f"P1 Can Build Large Sugar (Cost 4): {mask[c.ACTION_BUILD_START + c.BUILDING_LARGE_SUGAR]}")
    assert mask[c.ACTION_BUILD_START + c.BUILDING_LARGE_SUGAR] == 0
    
    print("P1 Passes...")
    env.step(c.ACTION_PASS)
    
    print("\n=== Test 2: Craftsman (P1) ===")
    # Next turn logic: RolesTaken=2. Gov(0)+2 = 2 % 2 = 0. P0 acts next?
    # Wait, my turn logic: (Governor + roles_taken_count) % 2.
    # Round 1:
    # 1. Gov(0) -> P0 (Selector). Builder.
    # 2. Roles=1. Gov(0)+1 = 1 -> P1 (Selector).
    # So P1 should be current player.
    print(f"Current Player: {gs.current_player_idx} (Expected 1)")
    
    # Setup for Production:
    # P1 has Corn (Start). Occupy it.
    gs.players[1].island[0]['workers'] = 1
    # P1 has Small Sugar (lets give it). Occupy it.
    gs.players[1].city[0] = {'building': c.BUILDING_SMALL_SUGAR, 'workers': 1}
    # P1 needs Sugar Plantation. Give one.
    gs.players[1].island[1] = {'tile': c.PLANTATION_SUGAR, 'workers': 1}
    
    # P0 has Indigo (Start). Give Corn to island[2] and occupy it.
    gs.players[0].island[2] = {'tile': c.PLANTATION_CORN, 'workers': 1}
    
    print("P1 Picks Craftsman...")
    env.step(c.ACTION_CHOOSE_ROLE_CRAFTSMAN)
    
    # Production should have happened.
    # P1 Produced: 1 Corn, 1 Sugar.
    # P0 Produced: 1 Corn (from island[2]).
    
    print(f"P1 Corn: {gs.players[1].goods[c.CORN]} (Expected 1)")
    print(f"P1 Sugar: {gs.players[1].goods[c.SUGAR]} (Expected 1)")
    print(f"P0 Corn: {gs.players[0].goods[c.CORN]} (Expected 1)")
    
    assert gs.players[1].goods[c.CORN] == 1
    assert gs.players[1].goods[c.SUGAR] == 1
    assert gs.players[0].goods[c.CORN] == 1
    
    # P1 Bonus Round
    # P1 produced Corn and Sugar. Can take either.
    mask = env.get_action_mask()
    print(f"P1 Bonus Corn Option: {mask[c.ACTION_CRAFTSMAN_BONUS_CORN]}")
    print(f"P1 Bonus Sugar Option: {mask[c.ACTION_CRAFTSMAN_BONUS_SUGAR]}")
    assert mask[c.ACTION_CRAFTSMAN_BONUS_CORN] == 1
    assert mask[c.ACTION_CRAFTSMAN_BONUS_SUGAR] == 1
    assert mask[c.ACTION_CRAFTSMAN_BONUS_TOBACCO] == 0
    
    print("P1 Takes Bonus Sugar...")
    env.step(c.ACTION_CRAFTSMAN_BONUS_SUGAR)
    
    print(f"P1 Sugar Final: {gs.players[1].goods[c.SUGAR]} (Expected 2)")
    assert gs.players[1].goods[c.SUGAR] == 2
    
    print("\nAll Builder/Craftsman tests passed successfully!")

if __name__ == "__main__":
    try:
        test_roles_p2()
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
