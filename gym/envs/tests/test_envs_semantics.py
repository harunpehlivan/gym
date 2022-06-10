"""
Currently disabled since this was done in a very poor way
Hashed str representation of objects
"""


import json
import hashlib
import os

import pytest
from gym import spaces, logger
from gym.envs.tests.spec_list import spec_list

DATA_DIR = os.path.dirname(__file__)
ROLLOUT_STEPS = 100
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

ROLLOUT_FILE = os.path.join(DATA_DIR, 'rollout.json')

if not os.path.isfile(ROLLOUT_FILE):
    with open(ROLLOUT_FILE, "w") as outfile:
        json.dump({}, outfile, indent=2)

def hash_object(unhashed):
    return hashlib.sha256(str(unhashed).encode('utf-16')).hexdigest() # This is really bad, str could be same while values change

def generate_rollout_hash(spec):
    spaces.seed(0)
    env = spec.make()
    env.seed(0)

    observation_list = []
    action_list = []
    reward_list = []
    done_list = []

    total_steps = 0
    for episode in range(episodes):
        if total_steps >= ROLLOUT_STEPS: break
        observation = env.reset()

        for step in range(steps):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)

            action_list.append(action)
            observation_list.append(observation)
            reward_list.append(reward)
            done_list.append(done)

            total_steps += 1
            if total_steps >= ROLLOUT_STEPS: break

            if done: break

    observations_hash = hash_object(observation_list)
    actions_hash = hash_object(action_list)
    rewards_hash = hash_object(reward_list)
    dones_hash = hash_object(done_list)

    env.close()
    return observations_hash, actions_hash, rewards_hash, dones_hash

@pytest.mark.parametrize("spec", spec_list)
def test_env_semantics(spec):
    logger.warn("Skipping this test. Existing hashes were generated in a bad way")
    return
