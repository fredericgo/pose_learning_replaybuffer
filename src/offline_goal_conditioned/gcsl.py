from functools import reduce
import os
import time
import datetime
import scipy

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from common.constants import EPS

from offline_goal_conditioned.utils import soft_update, hard_update
from offline_goal_conditioned.networks import GoalGaussianPolicy, GoalQNetwork, DeterministicPolicy
from offline_goal_conditioned.episodic_replay_buffer import EpisodicReplayBuffer


class GCSL(object):
    def __init__(self, env, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        self.epsilon = args.epsilon
        self.log_interval = args.log_interval
        self.checkpoint_interval = args.checkpoint_interval
        self.updates_per_step = args.updates_per_step
        self.pretrain_epochs = args.pretrain_epochs
        self.traj_len = args.traj_len
      
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.goal_dim = env.goal_space.shape[0]

        self.goal_filter = np.array(args.goal_filter)
        self.horizon = args.horizon

        self.memory = EpisodicReplayBuffer(args.replay_size, args.seed)
        if args.expert_file:
            self.memory.load(args.expert_file)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GoalGaussianPolicy(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size, env.action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        #Tesnorboard
        datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f'runs/offline_goal_conditioned/{datetime_st}_SL_{args.env_name}'
        self.writer = SummaryWriter(self.log_dir)
        self.save_log_file(args)
 
    def select_action(self, state, goal, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        goal = torch.FloatTensor(goal).to(self.device)

        if evaluate is False:
            action, _, _ = self.policy.sample(state, goal)
        else:
            _, _, action = self.policy.sample(state, goal)

        action = action.detach().cpu().numpy()[0]
        if not evaluate and np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        return action

    def update_parameters(self, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, goal_batch, _, _, _ = self.memory.sample(batch_size=batch_size, horizon=self.horizon)
        state_batch = torch.tensor(state_batch, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device)
        goal_batch = torch.tensor(goal_batch[:, self.goal_filter], device=self.device)

        pi, log_pi, mu = self.policy.sample(state_batch, goal_batch)

        policy_loss = F.mse_loss(pi, action_batch) # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return policy_loss.item()

    def evaluate(self, num=10, test=False):
        # sample initial and goal from buffer
        def rollout():
            rewards = []
            distances = []
            if not test:
                state = self.env.reset()
                state, action, goal, _, _, _ = self.memory.sample(1, horizon=self.horizon)
                state = state[0]
                goal  = goal[0, self.goal_filter]
                self.env.set_init_state_and_goal(state, goal)
            else:
                goal = self.env.sample_goal()
                state = self.env.reset()

            l = self.horizon if self.horizon else self.traj_len
            for t in range(l):
                action = self.select_action(state[None], goal[None], evaluate=True)
                
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                distances.append(info["goal_distance"])
                rewards.append(reward)
            return np.mean(rewards), distances
        rewards = []
        distances = []
        for _ in range(num):
            r, d = rollout()
            rewards.append(r)
            distances.append(d[-1])
        return np.mean(rewards), np.mean(distances)

    def save_log_file(self, args):
        with open(os.path.join(self.log_dir, 'log_info.txt'), 'w') as f:
            f.write(f"Run info:\n")
            f.write("-"*10 + "\n")

            for key, value in vars(args).items():
                f.write(f"{key}={value}\n")

            f.write("-"*10 + "\n")
            f.write(self.policy.__str__())
            f.write("\n")

            if args.seed is None:
                args.seed = np.random.randint(2**16-1)
                f.write(f"Setting random seed {args.seed}\n")
                
    # Save model parameters
    def save_model(self, dir_name):
        model_path = os.path.join(dir_name, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        actor_path = os.path.join(model_path, 'actor')
        print('Saving models to {}'.format(actor_path))
        torch.save(self.policy.state_dict(), actor_path)

    # Load model parameters
    def load_model(self, model_path):
        actor_path = os.path.join(model_path, 'actor')
        critic_path = os.path.join(model_path, 'critic')

        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    def sample_trajectory(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        dones = []

        done = False
        state = self.env.reset()
        state, action, goal, _, _, _ = self.memory.sample(1, horizon=self.horizon)
        state = state[0]
        goal  = goal[0, self.goal_filter]
        self.env.set_init_state_and_goal(state, goal)
        for t in range(self.traj_len):
            states.append(state)
            dones.append(done)
            if done:
                state = self.env.reset()
                goal = self.env.sample_goal()

            goals.append(goal)
            action = self.select_action(state[None], goal[None])
            
            actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            rewards.append(reward)
            next_states.append(next_state)
            
        return np.stack(states), np.array(actions),  np.stack(rewards), np.stack(next_states), np.stack(goals), np.stack(dones)

    def train(self):
        epoch = 0
        t0 = time.time()
        updates = 0

        while epoch < self.num_epochs:
            t0 = time.time()      

            if epoch > self.pretrain_epochs:
                states, actions, rewards, next_states, goals, dones = self.sample_trajectory()
                self.memory.add_episode(states, actions, rewards, next_states, dones)

            # update parameters
            for _ in range(self.updates_per_step):
                policy_loss = self.update_parameters(self.batch_size, updates)
                self.writer.add_scalar('pre_train/loss/policy', policy_loss, updates)
                updates += 1
            
            if epoch % self.log_interval == 0:
                reward, final_distance  = self.evaluate()
                self.writer.add_scalar('Train/avg_reward', reward, epoch)
                self.writer.add_scalar('Train/final_distance', final_distance, epoch)
                reward, final_distance  = self.evaluate(test=True)
                self.writer.add_scalar('Test/avg_reward', reward, epoch)
                self.writer.add_scalar('Test/final_distance', final_distance, epoch)

            if epoch % self.checkpoint_interval == 0:
                self.save_model(self.log_dir)
                print("----------------------------------------")
                print(f"Save Model: {epoch} episodes.")
                print("----------------------------------------")

            if epoch == self.num_epochs:
                break
          
            epoch += 1
            execution_time = time.time() - t0