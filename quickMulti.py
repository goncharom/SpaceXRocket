from mlagents.envs import UnityEnvironment
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


total_rewards_to_plot = []
total_updates = []
total_means = []
total_value_loss = []
total_policy_loss = []
total_value_loss_means = []
steps = 30e6
agents = 8
batch_size =3100
epochs = 3
plot_points = 5 #number of updates between episode reward plot
lr = 2.5e-4

total_rewards_to_plot = []
total_updates = []
total_means = []
class actorCritic(nn.Module):
	def __init__(self):
		super(actorCritic, self).__init__()

		self.fc1 = nn.Linear(18, 32)
		self.fc2 = nn.Linear(32, 32)


		self.critic = nn.Linear(32, 1)
		self.actor = nn.Linear(32, 6)


	def forward(self, inputs): 
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		probs = F.softmax(self.actor(x), dim=1)
		value = self.critic(x)
		return probs, value


def gae (rewards, masks, values):

	gamma = 0.99
	lambd = 0.95
	T, W = rewards.shape
	real_values = np.zeros((T, W))
	advantages = np.zeros((T, W))
	adv_t = np.zeros((W, 1)).squeeze()
	for t in reversed(range(T)):
		delta = rewards[t] + values[t+1] * gamma*masks[t] - values[t]
		adv_t = delta + adv_t*gamma*lambd*masks[t]
		advantages[t] = adv_t
	real_values = values[:T] + advantages

	return advantages, real_values


def plotRewards(rewards):
	total_means.append(np.sum(rewards))
	total_rewards_to_plot.append(np.mean(total_means))
	plt.figure(2)
	plt.clf()
	plt.plot(total_rewards_to_plot, 'r')
	plt.plot(total_means)
	plt.savefig('rews.png')
	#plt.pause(0.001)
def plotValueLoss(valuesLoss):
	total_value_loss.append(float(valuesLoss))
	total_value_loss_means.append(np.mean(total_value_loss))
	plt.figure(1)
	plt.clf()
	plt.plot(total_value_loss, 'r')
	plt.pause(0.001)
def plotPolicyLoss(policyLoss):
	total_policy_loss.append(float(policyLoss))
	plt.figure(3)
	plt.clf()
	plt.plot(total_policy_loss)
	plt.pause(0.001)


class PPO():
	def __init__(self, lr, agents, env, info_):
		self.agents = agents
		self.env = env
		self.network = actorCritic()
		self.old_network = actorCritic()
		self.dic_placeholder = self.network.state_dict()
		self.old_network.load_state_dict(self.dic_placeholder)
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
		self.info_= info_
		#self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 48, gamma=0.96)
	def experience(self, steps):
		total_obs = np.zeros((steps, self.agents, 18))
		total_rewards = np.zeros((steps, self.agents))
		total_actions = np.zeros((steps, self.agents))
		total_values = np.zeros((steps+1, self.agents))
		masks = np.zeros((steps, self.agents))

		for step in range(steps):
			total_obs[step] = self.info_['RocketBrain'].vector_observations

			experience_probs, values = self.network(torch.from_numpy(self.info_['RocketBrain'].vector_observations).type(torch.FloatTensor))
			total_values[step] = values.view(-1).detach().numpy()
			m = Categorical(experience_probs)
			actions = m.sample()

			n_info = self.env.step(actions.numpy())
			dones = np.logical_not(n_info['RocketBrain'].local_done)*1
			total_rewards[step] = n_info['RocketBrain'].rewards
			total_actions[step] = actions.numpy()
			masks[step] = dones
			self.info_ = n_info

		_, values = self.network(torch.from_numpy(self.info_['RocketBrain'].vector_observations).type(torch.FloatTensor))
		total_values[steps] = values.view(-1).detach().numpy()
		advantage, real_values = gae(total_rewards, masks, total_values)
		advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
		plotRewards(total_rewards.reshape(steps*self.agents, -1))
		return(total_obs, total_values, total_rewards, total_actions, masks, advantage, real_values)

	def update(self, epochs, steps, total_obs, total_actions, advantage, real_values):

		total_obs_ = torch.from_numpy(total_obs).view(steps, -1, 1).type(torch.FloatTensor)
		total_actions = total_actions.reshape(steps, -1)
		advantage_ = torch.from_numpy(np.float64(advantage)).view(steps, -1).type(torch.FloatTensor)
		real_values_ = torch.from_numpy(np.float64(real_values)).view(steps, -1).type(torch.FloatTensor)
		
		for _ in range(epochs):
			indices = np.arange(steps)
			np.random.shuffle(indices)

			indices = indices.reshape(100, 248)
			for i in range(100):
				inds = indices[i, :]



				probs, values_to_backprop = self.network(total_obs_[inds].squeeze())
				m = Categorical(probs)

				actions_taken_prob = probs.squeeze()[np.arange(248), total_actions[inds].reshape(-1)]

				entropy = m.entropy()
				old_probs, _ = self.old_network(total_obs_[inds].squeeze())
				old_actions_taken_probs = old_probs.squeeze()[np.arange(248), total_actions[inds].reshape(-1)]
				old_probs.detach()
				ratios = (actions_taken_prob.squeeze())/(old_actions_taken_probs.squeeze() + 1e-5)
				surr1 = ratios * advantage_[inds].squeeze()
				surr2 = torch.clamp(ratios, min=(1.-.2), max=(1.+.2))*advantage_[inds].squeeze()
				policy_loss = -torch.min(surr1, surr2)

				value_loss = ((values_to_backprop-real_values_[inds])**2)

				total_loss = policy_loss.squeeze()+0.5*value_loss.squeeze()-0.001*entropy
				self.optimizer.zero_grad()
				total_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
				self.optimizer.step()
		#self.scheduler.step()
		self.old_network.load_state_dict(self.dic_placeholder)
		self.dic_placeholder = self.network.state_dict()
		return (value_loss.mean(), policy_loss.mean())

env = UnityEnvironment(file_name='SX.x86_64', worker_id=0, seed=1)
info_ = env.reset()
algo = PPO(lr, agents, env, info_)
algo.network.load_state_dict(torch.load('./modeloSX'))
for param in algo.network.parameters():
 	print(param.data)
iterations = steps/batch_size
for t in range(int(iterations)):
	total_obs, total_values, total_rewards, total_actions, masks, advantage, real_values = algo.experience(batch_size)

	
	#valueLoss, policyLoss = algo.update(epochs, batch_size*agents, total_obs, total_actions, advantage, real_values)

	#plotValueLoss(valueLoss)
	#plotPolicyLoss(policyLoss)
	#if (t%plot_points == 0) and (t != 0):
		#rewards_to_plot = algo.eval()
		#plotRewards(rewards_to_plot, t)

	torch.save(algo.network.state_dict(), './modeloSX')
	print(t/iterations)


