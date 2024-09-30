import numpy as np

class RefLine(object):
	def __init__(self, id_num, x, y):
		self.id_num = id_num
		self.x = x
		self.y = y
		self.s = self.cur_accum_s()
		self.param_x, self.param_y, self.s2x, self.s2y = self.fit_poly()

	def cur_accum_s(self):
	    s = 0
	    accum_s = []
	    accum_s.append(s)
	    for i in range(len(self.x)):
	        if i >= 1:
	            s += np.sqrt((self.x[i] - self.x[i-1])**2 + (self.y[i] - self.y[i-1])**2)
	            accum_s.append(s)
	    return np.array(accum_s)

	def fit_poly(self):
		param_x = np.polyfit(self.s, self.x, 6)
		param_y = np.polyfit(self.s, self.y, 6)
		s2x = np.poly1d(param_x)
		s2y = np.poly1d(param_y)
		return param_x, param_y, s2x, s2y

	def get_id(self):
		return str(self.id_num)



