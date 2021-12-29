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
from sklearn.neighbors import NearestNeighbors
from common.constants import EPS

from goal_conditioned.utils import soft_update, hard_update
from goal_conditioned.networks import GoalGaussianPolicy, GoalQNetwork, DeterministicPolicy
from goal_conditioned.replay_memory import ReplayMemory
from goal_conditioned.episodic_replay_buffer import EpisodicReplayBuffer


class GoalSAC(object):
    def __init__(self, env, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.pretrain_epochs = args.pretrain_epochs
        self.log_interval = args.log_interval
        self.checkpoint_interval = args.checkpoint_interval
        self.updates_per_step = args.updates_per_step

        self.horizon = args.horizon
        self.traj_len = args.traj_len

        self.k = args.k
        self.num_workers = args.num_workers

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.goal_dim = env.goal_space.shape[0]
        self.goal_filter = np.array(args.goal_filter)

        self.B = np.log(self.k) - scipy.special.digamma(self.k)
        self.G = scipy.special.gamma(self.goal_dim/2 + 1)

        self.expert_memory = EpisodicReplayBuffer(args.replay_size, args.seed)
        if args.expert_file:
            self.expert_memory.load(args.expert_file)

        self.memory = ReplayMemory(args.replay_size, args.seed)

        self.critic = GoalQNetwork(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = GoalQNetwork(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GoalGaussianPolicy(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size, env.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size, env.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        #Tesnorboard
        datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f'runs/goal_conditioned/{datetime_st}_SACMepol_pretrain_{args.env_name}'
        self.writer = SummaryWriter(self.log_dir)
        self.save_log_file(args)

        if args.model_dir:
            self.load_model(args.model_dir)
 
    def select_action(self, state, goal, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        goal = torch.FloatTensor(goal).to(self.device)

        if evaluate is False:
            action, _, _ = self.policy.sample(state, goal)
        else:
            _, _, action = self.policy.sample(state, goal)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, goal_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(-1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, goal_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, goal_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch, goal_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch, goal_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi, goal_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

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
        critic_path = os.path.join(model_path, 'critic')
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, model_path):
        actor_path = os.path.join(model_path, 'actor')
        critic_path = os.path.join(model_path, 'critic')

        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if os.path.exists(actor_path):
            self.policy.load_state_dict(torch.load(actor_path))
    
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))

    def sample_trajectory(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []

        done = False
        state, goal = self.expert_memory.sample_state_and_goal(1, horizon=self.horizon)
        state = state[0]
        goal  = goal[0, self.goal_filter]
        self.env.set_init_state_and_goal(state, goal)
        for t in range(self.traj_len):
            states.append(state)
            if done:
                state = self.env.reset()
                goal = self.env.sample_goal()

            goals.append(goal)
           
            #observation = self.env.observation(state)
            action = self.select_action(state[None], goal[None])
            
            actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            rewards.append(reward)
            next_states.append(next_state)
            
        return np.stack(states), np.array(actions),  np.stack(rewards), np.stack(next_states), np.stack(goals)

    def evaluate(self, num=10, test=False):
        # sample initial and goal from buffer
        def rollout():
            rewards = []
            distances = []
            if not test:
                #state, action, goal, _, _, _ = self.expert_memory.sample(1)
                state, goal = self.expert_memory.sample_state_and_goal(1)

                state = state[0]
                goal  = goal[0, self.goal_filter]            
                self.env.set_init_state_and_goal(state, goal)
            else:
                goal = self.env.sample_goal()
                state = self.env.reset()
          
            for t in range(self.traj_len):
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

    def compute_knn(self, states, k, num_workers):
        # Fit knn for the batch of collected particles
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto', n_jobs=num_workers)
        nbrs.fit(states)
        distances, indices = nbrs.kneighbors(states)
        return distances, indices
    
    def compute_entropy(self, states, k, num_workers):
        states = states[..., self.state_filter]
        distances, indices = self.compute_knn(states, k, num_workers)
        # Compute objective function
        # compute volume for each particle
        ns = states.shape[-1]

        volumes = (np.power(distances[:, self.k], ns) * np.power(np.pi, ns/2)) / self.G
        # compute entropy
        entropy = - np.mean(np.log((self.k / (self.traj_len * volumes + EPS)) + EPS)) + self.B
        return entropy

    def pretrain(self):
        epoch = 0
        t0 = time.time()
        updates = 0

        while epoch < self.pretrain_epochs:
            t0 = time.time()
            # Collect particles to optimize off policy
            states, actions, rewards, next_states, goals = self.sample_trajectory()
            entropy = self.compute_entropy(next_states, self.k, self.num_workers)

            reward = entropy 
            rewards.fill(reward)
            
            for i in range(self.traj_len):
                mask = 0. if i == (self.traj_len - 1) else float(1)
                self.memory.push(states[i], actions[i], rewards[i], next_states[i], goals[i], mask)
                
            if self.start_epoch < epoch:
                # update parameters
                for _ in range(self.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                        self.update_parameters(self.memory, self.batch_size, updates)
                    self.writer.add_scalar('pretrain_loss/critic_1', critic_1_loss, updates)
                    self.writer.add_scalar('pretrain_loss/critic_2', critic_2_loss, updates)
                    self.writer.add_scalar('pretrain_loss/policy', policy_loss, updates)
                    self.writer.add_scalar('pretrain_loss/entropy_loss', ent_loss, updates)
                    self.writer.add_scalar('pretrain_entropy_temprature/alpha', alpha, updates)
                    updates += 1

            if epoch % self.log_interval == 0:
                self.writer.add_scalar('pretrain_reward/train', reward, epoch)
            
            if epoch % self.checkpoint_interval == 0:
                pretrain_path = self.log_dir + "/pretrain/"
                self.save_model(self.log_dir )
                print("----------------------------------------")
                print(f"Save Model: {epoch} episodes.")
                print("----------------------------------------")

            epoch += 1
            execution_time = time.time() - t0

    def train(self):
        epoch = 0
        t0 = time.time()
        updates = 0

        while epoch < self.num_epochs:
            t0 = time.time()
            # Collect particles to optimize off policy
            states, actions, rewards, next_states, goals = self.sample_trajectory()
            
            for i in range(self.traj_len):
                mask = 0. if i == (self.traj_len - 1) else float(1)
                self.memory.push(states[i], actions[i], rewards[i], next_states[i], goals[i], mask)
                
            if self.start_epoch < epoch:
                # update parameters
                for _ in range(self.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                        self.update_parameters(self.memory, self.batch_size, updates)
                    self.writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    self.writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    self.writer.add_scalar('loss/policy', policy_loss, updates)
                    self.writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    self.writer.add_scalar('entropy_temprature/alpha', alpha, updates)
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

            epoch += 1
            execution_time = time.time() - t0