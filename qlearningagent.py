import random
from argparse import ArgumentParser
from pprint import pprint


def seed(seed):
	random.seed(seed)


class QLearningAgent():
	def __init__(self, epsilon=0.2, epsilon_decay=1, epsilon_min=0, alpha=0.5, alpha_decay=1, alpha_min=0.5, gamma=1, gamma_decay=1, gamma_min=0, number_of_states=100):
		self.q_table = {}
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.alpha = alpha
		self.alpha_decay = alpha_decay
		self.alpha_min = alpha_min
		self.gamma = gamma
		self.gamma_decay = gamma_decay
		self.gamma_min = gamma_min
		self.number_of_states = number_of_states

	def run_episode(self, env, render=True, silent=False):
		observation = env.reset()
		actions = 0
		episodic_reward = 0

		self.epsilon = self.hyperparameter(self.epsilon, self.epsilon_decay, self.epsilon_min)
		self.alpha = self.hyperparameter(self.alpha, self.alpha_decay, self.alpha_min)
		self.gamma = self.hyperparameter(self.gamma, self.gamma_decay, self.gamma_min)

		while True:
			if render:
				env.render()

			current_state = self.descretize(observation, env)

			action = self.choose_action(current_state, env)

			observation, reward, done, info = env.step(action)

			episodic_reward += reward

			new_state = self.descretize(observation, env)

			old_q = self.q(current_state, env, action=action)
			new_q = old_q + (self.alpha * (reward + (self.gamma * max(self.q(new_state, env))) - old_q))

			self.update_q(current_state, action, new_q)

			if done and not silent:
				if self.is_success(episodic_reward):
					print('SUCCESS! after {} timesteps, reward: {}'.format(actions, episodic_reward))
					break
				print('Episode Complete with reward: {}'.format(episodic_reward))
				break

			actions += 1

		return episodic_reward

	def is_success(self, total_reward):
		raise NotImplementedError()

	def hyperparameter(self, param, param_decay, param_min):
		return max([param_min, param * param_decay])

	def choose_action(self, state, env):
		if random.random() < self.epsilon:
			return env.action_space.sample()
		else:
			return self.best_action(self.q(state, env))

	def descretize(self, observation, env):
		highest = self.finite(env.observation_space.high)
		lowest = self.finite(env.observation_space.low)

		denominator = highest - lowest
		descretized = observation - lowest / denominator
		
		return tuple([int(self.number_of_states * num) for num in descretized])

	def best_action(self, actions):
		if self.all_equal(actions):
			return random.randint(0, len(actions) - 1)

		max_index = -1
		max_val = float('-inf')

		for index, num in enumerate(actions):
			if num > max_val:
				max_val = num
				max_index = index

		return max_index

	def q(self, state, env, action=None):
		if state not in self.q_table:
			self.q_table[state] = [0] * env.action_space.n

		return self.q_table[state][action] if action is not None else self.q_table[state]

	def update_q(self, state, action, new_q):
		self.q_table[state][action] = new_q

	def all_equal(self, actions):
		for i in range(len(actions) - 1):
			if actions[i] == actions[i + 1]:
				continue

			return False
		return True

	def finite(self, numbers, maximum=100000000, minimum=-100000000):
		for i in range(len(numbers)):
			numbers[i] = max([minimum, min([numbers[i], maximum])])

		return numbers


def arg_parser():
	arg_parser = ArgumentParser()

	arg_parser.add_argument('-e', '--epsilon', help='Sets exploration rate of Q-Learning Agent.', type=float, default=0.2)
	arg_parser.add_argument('-ed', '--epsilon-decay', help='Sets decay of exploration rate of Q-Learning Agent.', type=float, default=1)
	arg_parser.add_argument('-em', '--epsilon-min', help='Sets mininum value for exploration rate of Q-Learning Agent.', type=float, default=0)

	arg_parser.add_argument('-a', '--alpha', help='Sets learning rate of Q-Learning Agent.', type=float, default=0.5)
	arg_parser.add_argument('-ad', '--alpha-decay', help='Sets decay of learning rate of Q-Learning Agent.', type=float, default=1)
	arg_parser.add_argument('-am', '--alpha-min', help='Sets mininum value for learning rate of Q-Learning Agent.', type=float, default=0.5)

	arg_parser.add_argument('-g', '--gamma', help='Sets future reward discount of Q-Learning Agent.', type=float, default=1)
	arg_parser.add_argument('-gd', '--gamma-decay', help='Sets decay of future reward discount of Q-Learning Agent.', type=float, default=1)
	arg_parser.add_argument('-gm', '--gamma-min', help='Sets mininum value for future reward discount of Q-Learning Agent.', type=float, default=0)

	arg_parser.add_argument('-s', '--num-of-states', help='Limits the number of buckets a state can fit into.', type=int, default=100)
	arg_parser.add_argument('-ne', '--num-of-episodes', help='Number of episodes that will run', type=int, default=1000)

	arg_parser.add_argument('-r', '--render', help='Will render the graphics', default=False, action='store_true')

	return arg_parser


class NotImplementedError(Exception):
	pass


class Queue:
	def __init__(self, max_length=10000000):
		self.max_length = max_length
		self.queue = []

	def push(self, value):
		""" 
		Lazily adds value to the queue
		If max_length is reached, we remove from head of queue, then add to tail
		"""
		if self.size() >= self.max_length:
			self.pop()
		self.queue.append(value)

	def pop(self):
		""" Removes from the head of the queue """
		return self.queue.pop(0)

	def peek(self):
		""" Shows the value at the head of the queue, but doesn't remove it """
		return self.queue[0]

	def size(self):
		return len(self.queue)

	def is_empty(self):
		return len(self.queue) == 0

	def avg(self):
		return sum(self.queue) / self.size() if not self.is_empty() else 0

