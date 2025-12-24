
import gymnasium as gym
import os
import sys
import shutil
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
from puerto_rico_wrappers import PuertoRicoSelfPlayWrapper

def make_env():
    env = PuertoRicoEnv2P()
    env = PuertoRicoSelfPlayWrapper(env)
    env = ActionMasker(env, lambda env: env.action_masks())
    return env

def verify_train():
    print("Verifying Training Pipeline...")
    env = DummyVecEnv([make_env])
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=128, # Short buffer
        batch_size=32
    )
    
    print("Learning for 256 steps...")
    model.learn(total_timesteps=256)
    print("Training verified successfully.")
    
    # Cleanup logs if any
    if os.path.exists("./logs_verify/"):
        shutil.rmtree("./logs_verify/")

if __name__ == "__main__":
    verify_train()
