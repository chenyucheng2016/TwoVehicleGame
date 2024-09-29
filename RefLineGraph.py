import RefLine as rl
from scipy.spatial import KDTree
import numpy as np

from itertools import product

class RefLineGraph(object):
	def __init__(self, ref_line_arr):
		self.reflines = ref_line_arr
		self.ref_line_num = len(ref_line_arr)
		self.ref_line_graph = {}
		self.ref_line_ids = []
		self.interactions = []
		self.decisions = ['overtake', 'follow']


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
					adj_info = {"child_id": str(self.reflines[j].id_num), 
					            "s_min_self": s1_min, 
					            "s_max_self": s1_max, 
					            "s_min_neighbor": s2_min,
					            "s_max_neighbor": s2_max}
					interaction = (str(self.reflines[i].id_num), str(self.reflines[j].id_num))
					interaction_reverse = (str(self.reflines[j].id_num), str(self.reflines[i].id_num))
					if interaction not in self.interactions and interaction_reverse not in self.interactions:
						self.interactions.append((interaction))
					self.ref_line_graph[str(self.reflines[i].id_num)].append(adj_info)
		self.ref_line_ids = list(self.ref_line_graph.keys())
	def generate_consistent_combinations(self):
	    # For each vehicle, determine how many other vehicles they interact with
	    interaction_counts = {vehicle: sum(1 for (v1, v2) in self.interactions if v1 == vehicle or v2 == vehicle) for vehicle in self.ref_line_ids}

	    # Generate all possible decision combinations for each vehicle based on its interactions
	    decision_combinations = {vehicle: list(product(self.decisions, repeat=interaction_counts[vehicle])) for vehicle in self.ref_line_ids}

	    consistent_combinations = []
	    
	    # Generate combinations for each vehicle based on their interaction count
	    for vehicle_decisions in product(*decision_combinations.values()):
	        # Build the decision dictionary for interacting vehicles
	        combination = {self.ref_line_ids[i]: {} for i in range(len(self.ref_line_ids))}
	        
	        idx_map = {v: 0 for v in self.ref_line_ids}  # Tracks how many decisions each vehicle has made

	        for i, vehicle in enumerate(self.ref_line_ids):
	            for (vehicle1, vehicle2) in self.interactions:
	                if vehicle == vehicle1:
	                    combination[vehicle1][vehicle2] = vehicle_decisions[i][idx_map[vehicle1]]
	                    idx_map[vehicle1] += 1
	                elif vehicle == vehicle2:
	                    combination[vehicle2][vehicle1] = vehicle_decisions[i][idx_map[vehicle2]]
	                    idx_map[vehicle2] += 1

	        # Check if the combination is consistent
	        if self.is_consistent(combination):
	            consistent_combinations.append(combination)
	    
	    return consistent_combinations

	def is_consistent(self, combination):
	    for (vehicle1, vehicle2) in self.interactions:
	        decision_1_to_2 = combination[vehicle1].get(vehicle2, None)
	        decision_2_to_1 = combination[vehicle2].get(vehicle1, None)
	        
	        # If there is an interaction between vehicle1 and vehicle2, check the consistency
	        if decision_1_to_2 == 'overtake' and decision_2_to_1 != 'follow':
	            return False
	        if decision_1_to_2 == 'follow' and decision_2_to_1 != 'overtake':
	            return False
	    return True


	def intersect_segments(self, ref_line1, ref_line2):
		curve1 = np.transpose(np.array([ref_line1.x.tolist(), ref_line1.y.tolist()]))
		curve2 = np.transpose(np.array([ref_line2.x.tolist(), ref_line2.y.tolist()]))
		
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


