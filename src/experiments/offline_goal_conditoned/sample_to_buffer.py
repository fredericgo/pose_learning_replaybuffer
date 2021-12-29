import argparse
import datetime
import pathlib
import gym
import numpy as np
import itertools
import torch
import logging

import envs
from offline_goal_conditioned.offline_goal_sac import OfflineGoalSAC
from offline_goal_conditioned.networks import GaussianPolicy
from offline_goal_conditioned.episodic_replay_buffer import EpisodicReplayBuffer

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_random",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, nargs="+", default=[400, 300], metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--traj_len', type=int, default=5000, 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_epochs', type=int, default=100, 
                    help='# of epochs')  
parser.add_argument('--policy_dir', type=str, default=None, 
                    help="model path")

args = parser.parse_args()

def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) 
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def sample_trajectory(env, policy):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    state = env.reset()
    for t in range(args.traj_len):
        states.append(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, action = policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        actions.append(action)
        
        state, r, _, _ = env.step(action)
        rewards.append(0.)
        next_states.append(state)
        done = t == args.traj_len - 1 
        dones.append(done)
        print(t)
    
    return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)
 
def main():
    # Environment
    # env sample dataset
    env = envs.create_env(args.env_name)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    num_features = env.num_features
    action_dim = env.action_space.shape[0]
    policy = GaussianPolicy(num_features, action_dim, args.hidden_size, env.action_space)
    load_policy(policy, args.policy_dir)

    memory = EpisodicReplayBuffer(args.replay_size, args.seed)
    
    logging.info(f'Starting sampling procedure form policy file {args.policy_dir}')
    for e in range(args.num_epochs):
        logging.info(f'episode {e}')
        states, actions, rewards, next_states, dones = sample_trajectory(env, policy)
        memory.add_episode(states, actions, rewards, next_states, dones)
    env.close()
    memory_file_name = f"buffer_{args.num_epochs}.torch"
    memory.save(memory_file_name)

if __name__ == "__main__":
    main()