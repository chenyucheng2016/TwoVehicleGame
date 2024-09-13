import RefLine as rl
from scipy.spatial import KDTree

class RefLineGraph(object):
	def __init__(self, ref_line_arr):
		self.reflines = ref_line_arr
		self.ref_line_num = len(ref_line_arr)
		self.ref_line_graph = {}

	def buildRefLineGraph(self):
		for i in range(self.ref_line_num):
			self.ref_line_graph[str(self.reflines[i].id_num)] = []
			for j in range(self.ref_line_num):
				if j == i:
					continue
				curve1_segment, curve2_segment = self.intersect_segments(self.reflines[i], self.reflines[j])
				if len(curve1_segment) == 0 and len(curve2_segment) == 0:
					continue
				else:
					min_curve1 = curve1_segment[0]
					max_curve1 = curve1_segment[-1]
					min_curve2 = curve2_segment[0]
					max_curve2 = curve2_segment[-1]
					s1_min = self.reflines[i].s[min_curve1]
					s1_max = self.reflines[i].s[max_curve1]
					s2_min = self.reflines[j].s[min_curve2]
					s2_max = self.reflines[j].s[max_curve2]
					adj_info = {"child_id": str(str(self.reflines[j].id_num)), 
					            "s1_min": s1_min, 
					            "s1_max": s1_max, 
					            "s2_min": s2_min,
					            "s2_max": s1_max}
					self.ref_line_graph[str(self.reflines[i].id_num)].append(adj_info)

	def intersect_segments(self, ref_line1, ref_line2):
		curve1 = [ref_line1.x, ref_line1.y]
		curve2 = [ref_line2.x, ref_line2.y]
		
		tree1 = KDTree(curve1)
		tree2 = KDTree(curve2)
		threshold = 1.0

		close_pairs = tree1.query_ball_tree(tree2, threshold)

		curve1_indices = []
		curve2_indices = []

		for i, neighbors in enumerate(close_pairs):
			if neighbors:
				curve1_indices = list(set([i]) | set(curve1_indices))
				curve2_indices = list(set(neighbors) | set(curve2_indices))
		return sorted(curve1_indices), sorted(curve2_indices)


