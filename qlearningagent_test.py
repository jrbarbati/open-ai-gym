import pytest
from .qlearningagent import *

###############################################################################
# Queue Tests

def test_queue_push():
	queue = Queue(max_length=2)

	queue.push(1)
	assert queue.size() == 1

	queue.push(2)
	assert queue.size() == 2

	queue.push(3)
	assert queue.size() == 2

	assert queue.queue[0] == 2
	assert queue.queue[1] == 3


def test_queue_pop():
	queue = Queue(max_length=2)

	queue.push(1)
	queue.push(2)

	assert queue.pop() == 1
	assert queue.size() == 1
	assert queue.pop() == 2
	assert queue.size() == 0


def test_queue_peek():
	queue = Queue(max_length=3)

	queue.push(1)
	assert queue.peek() == 1

	queue.push(2)
	assert queue.peek() == 1

###############################################################################
