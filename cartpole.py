import gym
import sys
from qlearningagent import *

class CartpoleAgent(QLearningAgent):
	def is_success(self, total_reward):
		return total_reward >= 190


def main(args):
    env = gym.make('CartPole-v1')
    cartpole_agent = CartpoleAgent(epsilon=0.3, alpha=0.5, gamma=1)

    for episode in range(int(args[0])):
		print('Running {}/{}'.format(episode + 1, args[0]))
		cartpole_agent.run_episode(env)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as error:
        print(error)        
