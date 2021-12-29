import argparse
import datetime
import envs

import numpy as np
import itertools
import pathlib
import imageio
import torch
from torch.utils.tensorboard import SummaryWriter

from offline_goal_conditioned.networks import GoalGaussianPolicy

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
parser.add_argument('--num_epochs', type=int, default=10, 
                    help='# of epochs')  
parser.add_argument('--k', type=int, default=4, 
                    help='# of epochs')  
parser.add_argument('--num_workers', type=int, default=1, 
                    help='# of epochs')  
parser.add_argument('--state_filter', type=list, default=[2, 3, 4, 5, 6],
                    help='state filter')
parser.add_argument('--init_env_noise', type=float, default=.01, 
                    help='checkpoint training model every # steps')
parser.add_argument('--model_dir', type=str, default=None, 
                    help="model path")

parser.add_argument('--video_file_name', type=str, default=None, 
                    help='video file name')
args = parser.parse_args()



def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def render_trajectory(policy, env, writer):

    state = env.reset()
    goal = env.sample_goal()

    done = False
    for t in range(args.traj_len):

        writer.append_data(env.render(mode="rgb_array"))
        #if done:
        #    state = env.reset()

        state = torch.FloatTensor(state)
        goal = torch.FloatTensor(goal)

        _, _, action = policy.sample(state[None], goal[None])
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        state, r, done, info = env.step(action)
    print(info)

def main():
    # Environment
    env = envs.create_goal_env(args.env_name, initial_noise=args.init_env_noise)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal_space.shape[0]
    policy = GoalGaussianPolicy(state_dim, action_dim, goal_dim, args.hidden_size, env.action_space)

    load_policy(policy, args.model_dir)
    
    if args.video_file_name:
        writer = imageio.get_writer(args.video_file_name, fps=30) 
    else:
        writer = None

    for _ in range(args.num_epochs):
        render_trajectory(policy, env, writer)
    env.close()

if __name__ == "__main__":
    main()