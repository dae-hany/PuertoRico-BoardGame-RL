
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
import puerto_rico_constants as c

def test_roles_p3():
    print("Initialize Environment...")
    env = PuertoRicoEnv2P()
    env.reset(seed=300) 
    gs = env.game_state
    
    print("\n=== Test 1: Trader (P0) ===")
    # Setup: Give P0 Corn and Sugar.
    gs.players[0].goods[c.CORN] = 1
    gs.players[0].goods[c.SUGAR] = 1
    # Give P1 Sugar.
    gs.players[1].goods[c.SUGAR] = 1
    
    # Initialize P0 Doubloons
    gs.players[0].doubloons = 0
    
    print("P0 Picks Trader...")
    env.step(c.ACTION_CHOOSE_ROLE_TRADER)
    
    # P0 should start. Can sell Corn or Sugar.
    mask = env.get_action_mask()
    print(f"P0 Can Sell Corn: {mask[c.ACTION_SELL_CORN]}")
    print(f"P0 Can Sell Sugar: {mask[c.ACTION_SELL_SUGAR]}")
    assert mask[c.ACTION_SELL_CORN] == 1
    assert mask[c.ACTION_SELL_SUGAR] == 1
    
    # P0 Sells Sugar
    # Price 2 + 1(Priv) = 3.
    print("P0 Sells Sugar...")
    env.step(c.ACTION_SELL_SUGAR)
    
    print(f"P0 Doubloons: {gs.players[0].doubloons} (Expected 3)")
    assert gs.players[0].doubloons == 3
    print(f"Trading House: {gs.trading_house}")
    assert gs.trading_house[0] == c.SUGAR
    
    # P1 Turn. Has Sugar.
    # House has Sugar. No Office.
    # P1 cannot sell Sugar.
    mask = env.get_action_mask()
    print(f"P1 Can Sell Sugar: {mask[c.ACTION_SELL_SUGAR]}")
    assert mask[c.ACTION_SELL_SUGAR] == 0
    print(f"P1 Can Pass: {mask[c.ACTION_PASS]}")
    assert mask[c.ACTION_PASS] == 1
    
    print("P1 Passes...")
    env.step(c.ACTION_PASS)
    
    # End of Trader. House should remain full? Or reset?
    # House resets ONLY if full. It has 1 item.
    assert gs.trading_house[0] == c.SUGAR
    
    print("\n=== Test 2: Captain (P1) ===")
    # Next Round logic: 
    # Round 1 Starts P0. Roles: 1(Trader).
    # Next P1 picks.
    print(f"Current Player: {gs.current_player_idx} (Expected 1)")
    
    # Setup for Captain:
    # P1 has Sugar (1).
    # P0 has Corn (1).
    # Give P0 more Corn (Total 5).
    gs.players[0].goods[c.CORN] = 5
    # Give P1 Coffee (1).
    gs.players[1].goods[c.COFFEE] = 1
    
    print("P1 Picks Captain...")
    env.step(c.ACTION_CHOOSE_ROLE_CAPTAIN)
    
    # Ship 1 (Cap 4), Ship 2 (Cap 6). Both Empty.
    # P1 (Captain, Privilege +1 VP).
    # P1 has Sugar(1), Coffee(1).
    # Can ship either.
    
    mask = env.get_action_mask()
    print(f"P1 Can Ship Sugar: {mask[c.ACTION_SHIP_SUGAR]}")
    print(f"P1 Can Ship Coffee: {mask[c.ACTION_SHIP_COFFEE]}")
    assert mask[c.ACTION_SHIP_SUGAR] == 1
    assert mask[c.ACTION_SHIP_COFFEE] == 1
    
    # P1 Ships Sugar.
    # VP: 1 (Good) + 1 (Priv) = 2.
    gs.players[1].vp_chips = 0
    print("P1 Ships Sugar...")
    env.step(c.ACTION_SHIP_SUGAR)
    
    print(f"P1 VP: {gs.players[1].vp_chips} (Expected 2)")
    assert gs.players[1].vp_chips == 2
    # Check Ship
    # Logic picks first empty ship? Ship 0 (Cap 4).
    print(f"Ship 0: {gs.ships[0]}")
    assert gs.ships[0]['good'] == c.SUGAR
    assert gs.ships[0]['count'] == 1
    
    # P0 Turn.
    # P0 has Corn(5).
    # Ship 0 has Sugar.
    # Ship 1 Empty.
    # Must ship Corn to Ship 1.
    print("P0 Turn...")
    mask = env.get_action_mask()
    print(f"P0 Can Ship Corn: {mask[c.ACTION_SHIP_CORN]}")
    assert mask[c.ACTION_SHIP_CORN] == 1
    
    print("P0 Ships Corn...")
    # Ship 1 Capacity 6. P0 has 5. Ships 5.
    env.step(c.ACTION_SHIP_CORN)
    
    print(f"Ship 1: {gs.ships[1]}")
    assert gs.ships[1]['good'] == c.CORN
    assert gs.ships[1]['count'] == 5
    assert gs.players[0].goods[c.CORN] == 0
    
    # P1 Turn Again (Cyclic).
    # P1 has Coffee(1).
    # Ship 0 (Sugar, 1/4).
    # Ship 1 (Corn, 5/6).
    # No ship for Coffee (Both occupied).
    # Must Pass?
    mask = env.get_action_mask()
    print(f"P1 Can Ship Coffee: {mask[c.ACTION_SHIP_COFFEE]}")
    assert mask[c.ACTION_SHIP_COFFEE] == 0
    print(f"P1 Can Pass: {mask[c.ACTION_PASS]}")
    assert mask[c.ACTION_PASS] == 1
    
    print("P1 Passes...")
    env.step(c.ACTION_PASS)
    
    # P0 Turn (Cyclic).
    # P0 has nothing. Must Pass.
    print("P0 Passes...")
    env.step(c.ACTION_PASS)
    
    # End of Captain.
    # Rotting Logic check?
    # P1 passed with Coffee(1). Allow keeping 1.
    print(f"P1 Coffee: {gs.players[1].goods[c.COFFEE]} (Expected 1)")
    assert gs.players[1].goods[c.COFFEE] == 1
    
    # Ship Clearing logic check
    # Ship 0 (1/4) -> Not Full -> Keeps Sugar.
    # Ship 1 (5/6) -> Not Full -> Keeps Corn.
    print(f"Ship 0 Goods: {gs.ships[0]['good']}")
    assert gs.ships[0]['good'] == c.SUGAR
    
    # Let's fill Ship 0 to test clearing.
    # Reset Environment? Or continue?
    # Continue.
    # Next Round...
    
    print("\nAll Trader/Captain tests passed successfully!")

if __name__ == "__main__":
    try:
        test_roles_p3()
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
