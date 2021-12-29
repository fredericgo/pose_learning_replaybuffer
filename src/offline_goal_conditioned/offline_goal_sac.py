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

def weighted_mse_loss(input, target, weight):
    weight /= weight.sum(0)
    return (weight * (input  ** 2)).mean()

class OfflineGoalSAC(object):
    def __init__(self, env, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
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

        expert_memory = EpisodicReplayBuffer(args.replay_size, args.seed)
        expert_memory.load(args.memory_file)
        self.expert_memory_dataset = expert_memory.as_dataset()

        self.pretraining = True
        self.memory = EpisodicReplayBuffer(args.replay_size, args.seed)

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
        self.log_dir = f'runs/offline_goal_conditioned/{datetime_st}_offlineSAC_{args.env_name}'
        self.writer = SummaryWriter(self.log_dir)
        self.save_log_file(args)
 
    def select_action(self, state, goal, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        goal = torch.FloatTensor(goal).to(self.device)

        if evaluate is False:
            action, _, _ = self.policy.sample(state, goal)
        else:
            _, _, action = self.policy.sample(state, goal)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, goal_batch, reward_batch, next_state_batch, mask_batch = self.sample(batch_size=batch_size)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, goal_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, goal_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch, goal_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        
        # negative examples
        low = self.env.action_space.low
        high = self.env.action_space.high
        with torch.no_grad():
            neg_action = torch.rand((batch_size, self.action_dim), device=self.device)
            neg_action *= torch.tensor(high - low, device=self.device)
            neg_action += torch.tensor(low, device=self.device)
        
            log_prob = self.policy.log_prob(state_batch, neg_action, goal_batch)
            prob = log_prob.exp()
            weight = prob / prob.sum(0)

        neg_qf_1, neg_qf_2 = self.critic(state_batch, neg_action, goal_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        neg_qf1_loss = (weight*(neg_qf_1**2)).mean()
        neg_qf2_loss = (weight*(neg_qf_2**2)).mean()
        qf_loss = qf1_loss + qf2_loss + neg_qf1_loss + neg_qf2_loss

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

    def evaluate(self, num=10, test=False):
        # sample initial and goal from buffer
        def rollout():
            rewards = []
            distances = []
            if not test:
                state = self.env.reset()
                state, action, goal, _, _, _ = self.sample(1)
                state = state.cpu().numpy()[0]
                goal  = goal.cpu().numpy()[0]
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
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    def sample(self, batch_size):
        if self.pretraining:
            return self.sample_from_expert_buffer(batch_size)
        return self.sample_from_buffer(batch_size)
    
    def sample_from_buffer(self, batch_size):
        state, action, goal, reward, next_state, mask = self.memory.sample(batch_size)
        
        state = torch.tensor(state, device=self.device)
        action = torch.tensor(action, device=self.device)
        goal = torch.tensor(goal, device=self.device)
        goal = goal[:, self.goal_filter]
        reward = torch.tensor(reward, device=self.device).unsqueeze(-1)
        next_state = torch.tensor(next_state, device=self.device)
        mask = torch.tensor(mask, device=self.device).unsqueeze(-1)
        return state, action, goal, reward, next_state, mask

    def sample_from_expert_buffer(self, batch_size):
        # sample episodes
        episode_idx = np.random.choice(np.arange(len(self.expert_memory_dataset)), batch_size)

        episode_len = self.expert_memory_dataset['state'].shape[1]
        frame_idx1 = np.random.randint(0, episode_len, batch_size)
        frame_idx2 = np.random.randint(0, episode_len, batch_size)

        state_idx = np.minimum(frame_idx1, frame_idx2)
        goal_idx = np.maximum(frame_idx1, frame_idx2)

        state = torch.tensor(self.expert_memory_dataset['state'][episode_idx, state_idx], device=self.device)
        action = torch.tensor(self.expert_memory_dataset['action'][episode_idx, state_idx], device=self.device)
        goal_state = torch.tensor(self.expert_memory_dataset['state'][episode_idx, goal_idx], device=self.device)
        goal = goal_state[:, self.goal_filter]
        next_state = torch.tensor(self.expert_memory_dataset['next_state'][episode_idx, state_idx], device=self.device)
        done = state.isclose(goal_state).all(dim=1).float().unsqueeze(-1)
        mask = 1 - done
        
        reward, _ = self.env.dense_reward(state.cpu().numpy(), goal.cpu().numpy())
        reward = torch.tensor(reward, device=self.device).float().unsqueeze(-1)
        return state, action, goal, reward, next_state, mask

    def sample_trajectory(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        dones = []

        done = False
        state = self.env.reset()
        goal = self.env.sample_goal()
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
            #states, actions, rewards, next_states, goals, dones = self.sample_trajectory()
            #self.memory.add_episode(states, actions, rewards, next_states, dones)
            # update parameters
            for _ in range(self.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                    self.update_parameters(self.batch_size, updates)
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

            if epoch == self.num_epochs:
                break
            if epoch == self.pretrain_epochs:
                self.pretraining = False
            epoch += 1
            execution_time = time.time() - t0