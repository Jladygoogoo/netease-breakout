import logging

from threading import Thread, Lock
from multiprocessing import Process,Lock
from queue import Queue

class ThreadsGroup:
	def __init__(self, task, task_args=None, n_thread=10):
		self.task = task
		self.task_args = task_args
		self.n_thread = n_thread

		self.threads = []
		for i in range(n_thread):
			self.threads.append(Thread(target=task, args=(i+1, self.task_args)))

	def start(self):
		for t in self.threads:
			t.start()
		for t in self.threads:
			t.join()


class ProcessGroup:
	def __init__(self, task, task_args=None, n_procs=10):
		self.task = task
		self.task_args = task_args
		self.n_procs = n_procs

		self.procs = []
		for i in range(n_procs):
			self.procs.append(Process(target=task, args=(i+1, self.task_args)))

	def start(self):
		for t in self.procs:
			t.start()
		for t in self.procs:
			t.join()



def func(tid, args):
	print(args["a"])
	args["a"] += 1

if __name__ == '__main__':
	tg = ThreadsGroup(func, {"a":1})
	tg.start()
