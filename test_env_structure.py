
import gymnasium as gym
import numpy as np
import sys
import os

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
import puerto_rico_constants as c

def test_env():
    print("Initialize Environment...")
    env = PuertoRicoEnv2P()
    
    print("Resetting Environment...")
    obs, info = env.reset()
    
    print("Checking Observation Structure...")
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert "global" in obs
    assert "players" in obs
    assert "market_plantations" in obs
    
    print(f"Global Shape: {obs['global'].shape}")
    assert obs['global'].shape == (35,), f"Expected (35,), got {obs['global'].shape}"
    
    print(f"Players Shape: {obs['players'].shape}")
    assert obs['players'].shape == (2, 56), f"Expected (2, 56), got {obs['players'].shape}"
    
    print(f"Market Shape: {obs['market_plantations'].shape}")
    assert obs['market_plantations'].shape == (3,), f"Expected (3,), got {obs['market_plantations'].shape}"
    
    # Check Initial Values
    print("Checking Initial Values...")
    
    # Players start with 3 Doubloons
    p0_doubloons = obs['players'][0][0]
    p1_doubloons = obs['players'][1][0]
    print(f"P0 Doubloons: {p0_doubloons}")
    print(f"P1 Doubloons: {p1_doubloons}")
    assert p0_doubloons == 3
    assert p1_doubloons == 3
    
    # Check Supply
    colonist_supply = obs['global'][0]
    print(f"Colonist Supply: {colonist_supply}")
    # 40 supply + 2 market = 42 total? 
    # Rulebook: "supply... 40", "market... 2".
    # Env init: supply=40. Correct.
    assert colonist_supply == 40
    
    # Check Goods Supply
    goods_supply = obs['global'][3:8]
    print(f"Goods Supply: {goods_supply}")
    assert np.array_equal(goods_supply, c.GOODS_SUPPLY)
    
    print("\nAll structure tests passed successfully!")

if __name__ == "__main__":
    try:
        test_env()
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
