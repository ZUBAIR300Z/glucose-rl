"""
SAC (Soft Actor-Critic) implemented from scratch in PyTorch.

This is the "prove I understand it" version: the same algorithm Stable-Baselines3
runs, written explicitly so every step is visible. It plugs into the exact same
environment as the SB3 version, so we can benchmark the two head-to-head.

Components:
    ReplayBuffer  - stores past transitions for off-policy learning
    Actor         - squashed-Gaussian policy network
    Critic        - a Q-network (state, action) -> scalar value
    SAC           - ties them together with the actor-critic update rules
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20, 2  # clamp policy std into a sane range


class ReplayBuffer:
    """A fixed-size circular buffer of (s, a, r, s', done) transitions.

    Off-policy RL learns from a big memory of past experience instead of only
    the latest episode. We store everything as numpy and convert a sampled
    minibatch to torch tensors on demand.
    """

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0      # where to write next (wraps around)
        self.size = 0     # how many valid entries we have

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.dones)


class Actor(nn.Module):
    """Squashed-Gaussian policy: state -> a distribution over actions.

    The net outputs a mean and log-std. We sample with the reparameterization
    trick, squash through tanh into (-1, 1), then rescale to the env's action
    range. We also return the log-probability (needed for the entropy term),
    with the tanh change-of-variables correction.
    """

    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        # Map tanh output (-1, 1) -> (low, high):  a = tanh(u) * scale + bias.
        # Stored as buffers so they move with .to(device) and get saved.
        act_low = np.asarray(act_low, dtype=np.float32)
        act_high = np.asarray(act_high, dtype=np.float32)
        self.register_buffer("act_scale", torch.tensor((act_high - act_low) / 2.0))
        self.register_buffer("act_bias", torch.tensor((act_high + act_low) / 2.0))

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        """Stochastic action + its log-prob (reparameterized, grads flow)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)

        u = normal.rsample()                      # reparameterized: u = mu + std * eps
        a_tanh = torch.tanh(u)
        action = a_tanh * self.act_scale + self.act_bias

        # log prob of the SQUASHED action = log N(u) - log|d a/d u|.
        # The correction term below is the numerically stable form of
        # sum_i log(1 - tanh^2(u_i)).
        logp = normal.log_prob(u).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return action, logp

    def act_deterministic(self, obs):
        """The mean action (no exploration noise) — used at evaluation time."""
        mu, _ = self.forward(obs)
        return torch.tanh(mu) * self.act_scale + self.act_bias


class Critic(nn.Module):
    """A Q-network: estimates expected return for a (state, action) pair."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class SAC:
    """Soft Actor-Critic agent: actor + twin critics + auto entropy tuning."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, device="cpu",
                 gamma=0.99, tau=0.005, lr=3e-4, hidden=256):
        self.device = torch.device(device)
        self.gamma = gamma            # discount factor
        self.tau = tau                # polyak averaging coefficient
        self.config = dict(obs_dim=obs_dim, act_dim=act_dim, act_low=act_low,
                           act_high=act_high, hidden=hidden)

        self.actor = Actor(obs_dim, act_dim, act_low, act_high, hidden).to(self.device)
        # Twin critics: two Q-nets; we use the smaller estimate to fight
        # the well-known overestimation bias of a single critic.
        self.q1 = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q2 = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q1_targ = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q2_targ = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        for p in (*self.q1_targ.parameters(), *self.q2_targ.parameters()):
            p.requires_grad = False   # targets are updated by polyak, not gradients

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # Automatic entropy temperature: learn alpha to hit a target entropy.
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.act_deterministic(obs_t)
            else:
                a, _ = self.actor.sample(obs_t)
        return a.cpu().numpy()[0]

    def update(self, batch):
        obs, act, rew, next_obs, done = batch

        # ---- 1. Critic update: regress Q toward the soft Bellman target ----
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            q_next = torch.min(self.q1_targ(next_obs, next_a),
                               self.q2_targ(next_obs, next_a)) - self.alpha * next_logp
            target = rew + self.gamma * (1.0 - done) * q_next
        q1_loss = F.mse_loss(self.q1(obs, act), target)
        q2_loss = F.mse_loss(self.q2(obs, act), target)
        q_loss = q1_loss + q2_loss
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ---- 2. Actor update: maximize Q while staying stochastic ----
        new_a, logp = self.actor.sample(obs)
        q_pi = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- 3. Temperature update: drive entropy toward the target ----
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- 4. Polyak update of the target critics ----
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {"q_loss": float(q_loss), "actor_loss": float(actor_loss),
                "alpha": float(self.alpha)}

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": self.config,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt["config"]
        agent = cls(c["obs_dim"], c["act_dim"], c["act_low"], c["act_high"],
                    device=device, hidden=c["hidden"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        return agent
