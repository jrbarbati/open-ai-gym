import gym
import sys
from q_learning_agent import *

class MountainCarAgent(QLearningAgent):
	def is_success(self, total_reward):
		return total_reward > -200


def main(args):
	env = gym.make('MountainCar-v0')
	mountain_car_agent = MountainCarAgent(epsilon=0.9, epsilon_decay=0.99, epsilon_min=0.05, alpha=0.9, alpha_decay=0.999, alpha_min=0.05, gamma=1)

	for episode in range(int(args[0])):
		print('Running {}/{}'.format(episode + 1, args[0]))
		mountain_car_agent.run_episode(env)


if __name__ == '__main__':
	main(sys.argv[1:])
