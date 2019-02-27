import gym
import sys

def main():
    env = gym.make('CartPole-v1')


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as error:
        print(error)        
