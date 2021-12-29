import argparse
import datetime
import envs

import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

from goal_conditioned.goal_sac import GoalSAC

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_goal",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden_size', type=int, nargs="+", default=[400, 300], metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=20, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_epoch', type=int, default=50, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--checkpoint_interval', type=int, default=50, 
                    help='checkpoint training model every # steps')
parser.add_argument('--log_interval', type=int, default=10, 
                    help='checkpoint training model every # steps')
parser.add_argument('--eval_interval', type=int, default=100, 
                    help='checkpoint training model every # steps')

parser.add_argument('--traj_len', type=int, default=500, 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_epochs', type=int, default=50000, 
                    help='# of epochs')  
parser.add_argument('--k', type=int, default=4, 
                    help='# of epochs')  
parser.add_argument('--num_workers', type=int, default=1, 
                    help='# of epochs')  
parser.add_argument('--state_filter', type=list, default=[2, 3, 4, 5, 6],
                    help='state filter')

args = parser.parse_args()


def main():
    # Environment
    env = envs.create_goal_env(args.env_name)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = GoalSAC(env, args)
    agent.pretrain()

    env.close()

if __name__ == "__main__":
    main()