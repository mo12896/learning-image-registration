
import time
import logging
import functools

log_time=False

def timer(func):
	@functools.wraps(func)
	def time_wrapper(*args, **kwargs):
		if log_time:
			tic = time.perf_counter()
			value = func(*args, **kwargs)
			toc = time.perf_counter()
			time = toc - tic
			logging.getLogger("TIME").info("Function %s takes %.5f s",
			                               func.__name__, time)
		else:
			value = func(*args, **kwargs)
		return value
	return time_wrapper
