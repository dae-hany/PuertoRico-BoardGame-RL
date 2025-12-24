import gymnasium as gym
import os
import sys
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from puerto_rico_env import PuertoRicoEnv2P
from puerto_rico_wrappers import PuertoRicoSelfPlayWrapper

def make_env():
    env = PuertoRicoEnv2P()
    # 1. 먼저 SelfPlayWrapper로 감쌉니다.
    env = PuertoRicoSelfPlayWrapper(env)
    
    # 2. ActionMasker를 Monitor보다 먼저 적용하여 action_masks 메서드에 접근 가능하게 합니다.
    # env 객체가 여전히 action_masks를 가지고 있는 상태에서 감싸야 합니다.
    env = ActionMasker(env, lambda env: env.action_masks())
    
    # 3. 마지막으로 Monitor를 적용합니다.
    env = Monitor(env) 
    return env

def train():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 가급적 시드(seed)를 고정하여 재현성을 확보합니다.
    env = DummyVecEnv([make_env])
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log=log_dir
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./checkpoints/',
        name_prefix='ppo_puerto'
    )
    
    print("Starting Training...")
    try:
        model.learn(
            total_timesteps=1_000_000, 
            callback=checkpoint_callback,
            progress_bar=True # tqdm/rich 설치 확인됨
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        model.save("ppo_puerto_final")
        print("Model saved.")

if __name__ == "__main__":
    train()