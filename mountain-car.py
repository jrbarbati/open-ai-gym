import gym
import sys
from qlearningagent import *

class MountainCarAgent(QLearningAgent):
	def is_success(self, total_reward):
		return total_reward > -200


def main(args):
	env = gym.make('MountainCar-v0')
	mountain_car_agent = MountainCarAgent(epsilon=0.3, alpha=0.5, gamma=0.95)

	for episode in range(int(args[0])):
		print('Running {}/{}'.format(episode + 1, args[0]))
		mountain_car_agent.run_episode(env)


if __name__ == '__main__':
	main(sys.argv[1:])
