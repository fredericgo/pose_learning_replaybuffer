import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RNDNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_output):
        super(RNDNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_output)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class GoalQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_goals, hidden_dims):
        super(GoalQNetwork, self).__init__()

        # Q1 architecture
        num_features = num_inputs + num_actions + num_goals
        # Q1 architecture
        layers = []
        layers.extend((nn.Linear(num_features, hidden_dims[0]), nn.ReLU()))
        for i in range(len(hidden_dims) - 1):
            layers.extend((nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q1 = nn.Sequential(*layers)

        # Q2 architecture
        layers = []
        layers.extend((nn.Linear(num_features, hidden_dims[0]), nn.ReLU()))
        for i in range(len(hidden_dims) - 1):
            layers.extend((nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*layers)

        self.apply(weights_init_)

    def forward(self, state, action, goal):
        xu = torch.cat([state, action, goal], -1)
        
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2


class GoalGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, num_goals, hidden_sizes, action_space=None):
        super(GoalGaussianPolicy, self).__init__()
        self.activation = nn.ReLU

        num_features = num_inputs + num_goals

        layers = []
        layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
        for i in range(len(hidden_sizes) - 1):
            layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

        self.net = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, goal):
        input_ = torch.cat([state, goal], -1)
        z = self.net(input_)
        mean = self.mean_linear(z)
        log_std = self.log_std_linear(z)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def log_prob(self, state, action, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        
        normal = Normal(mean, std)

        y_t = (action - self.action_bias) / self.action_scale
        log_prob = normal.log_prob(y_t)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
          
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GoalGaussianPolicy, self).to(device)


class GaussianPolicy(nn.Module):
    def __init__(
        self, 
        num_inputs, 
        num_actions, 
        hidden_sizes, 
        action_space,
        log_std_init=-0.5):
        super(GaussianPolicy, self).__init__()
        self.activation = nn.ReLU
        
        layers = []
        layers.extend((nn.Linear(num_inputs, hidden_sizes[0]), self.activation()))
        for i in range(len(hidden_sizes) - 1):
            layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

        self.net = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], num_actions)

        self.initialize_weights()
        # action rescaling

        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.mean_linear.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, state):
        z = self.net(state)
        mean = self.mean_linear(z)
        log_std = self.log_std_linear(z)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
