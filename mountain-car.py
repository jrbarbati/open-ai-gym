import gym
import sys
from qlearningagent import *

class MountainCarAgent(QLearningAgent):
	def is_success(self, total_reward):
		return total_reward > -200


def main():
	args = arg_parser().parse_args()
	env = gym.make('MountainCar-v0')
	mountain_car_agent = MountainCarAgent(
		epsilon=args.epsilon, 
		epsilon_decay=args.epsilon_decay,
		epsilon_min=args.epsilon_min,
		alpha=args.alpha,
		alpha_decay=args.alpha_decay,
		alpha_min=args.alpha_min,
		gamma=args.gamma,
		gamma_decay=args.gamma_decay,
		gamma_min=args.gamma_min,
		number_of_states=args.num_of_states
	)

	rewards = Queue(max_length=100)

	seed(0)

	for episode in range(args.num_of_episodes):
		print('Running {}/{}'.format(episode + 1, args.num_of_episodes))
		reward = mountain_car_agent.run_episode(env, render=args.render)

		rewards.push(reward)
		print('Average Reward: {}'.format(rewards.avg()))

		if rewards.avg() >= -110.0:
			print('\n\n\n\t MountainCar-v0 solved in {} episodes\n\n'.format(episode + 1))
			return


if __name__ == '__main__':
	main()
