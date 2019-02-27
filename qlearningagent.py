import random
from pprint import pprint

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
		random.seed(0)

	def run_episode(self, env, render=True, silent=False):
		observation = env.reset()
		actions = 0
		episodic_reward = 0

		self.epsilon = max([self.epsilon_min, self.epsilon * self.epsilon_decay])
		self.alpha = max([self.alpha_min, self.alpha * self.alpha_decay])
		self.gamma = max([self.gamma, self.gamma * self.gamma_decay])

		while True:
			if render:
				env.render()

			current_state = self.descretize(observation, env)

			action = self.choose_action(current_state, env)

			observation, reward, done, info = env.step(action)

			episodic_reward += reward

			new_state = self.descretize(observation, env)

			old_q = self.q(current_state, action, env)
			new_q = old_q + (self.alpha * (reward + (self.gamma * max(self.q(new_state, None, env))) - old_q))

			self.update_q(current_state, action, new_q)

			if done and not silent:
				if self.is_success(episodic_reward):
					print('SUCCESS! after {} timesteps, reward: {}'.format(actions, episodic_reward))
					break
				print('Episode Complete with reward: {}'.format(episodic_reward))
				break

			actions += 1

	def is_success(self, total_reward):
		raise NotImplementedError()

	def choose_action(self, state, env):
		if random.random() < self.epsilon:
			return env.action_space.sample()
		else:
			return self.best_action(self.q(state, None, env))

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

	def q(self, state, action, env):
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

class NotImplementedError(Exception):
	pass
