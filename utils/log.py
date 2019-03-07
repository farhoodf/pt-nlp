class AverageMeter(object):
	"""Computes the average and stores the values"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.vals = []
		self.sum = 0
		self.count = 0

	def update(self, val):
		self.vals.append(val)
		self.count += 1
		self.sum += val

	def average(self):
		return self.sum / self.count

	def last(self):
		if len(self.vals) > 0:
			return self.vals[-1]
		else:
			return None 