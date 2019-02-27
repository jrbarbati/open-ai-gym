from gym import envs
from pprint import pprint

for env in envs.registry.all():
	print(env)