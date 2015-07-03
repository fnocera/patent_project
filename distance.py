#file that makes database of distance metrics

import cPickle
import numpy as np
import pandas as pd
from collections import Counter
import random
from itertools import combinations
from time import time
from scipy import sparse
import igraph

file_open = open("org_list_of_lists.plk", 'rb')
org_list_of_lists = cPickle.load(file_open)

file_open = open("date_issue_array.plk", 'rb')
date_array = cPickle.load(file_open)

file_open = open("wku_subclass_matrix.plk", "rb")
subclass_matrix = cPickle.load(file_open)
subclass_matrix = subclass_matrix.tocsr()
#C = subclass_matrix

file_open = open("wku_org_matrix_final.plk", "rb")
org_matrix = cPickle.load(file_open)
org_matrix = org_matrix.tocsr()


#this is all for a given year
def format_data(target_year,C,date_array):
	patents = (date_array==target_year).nonzero()[0]
	selected_patents = C[patents,:]
	group_matrix = selected_patents.T*selected_patents
	zero = [0]*733
	visibility = group_matrix.diagonal()
	group_matrix.setdiag(zero)
	#making tuple list of non_zero indices 
	non_zeros = set()
	for item1, item2 in zip(group_matrix.nonzero()[0], group_matrix.nonzero()[1]):
		if item1 != item2:
			tupl = (item1, item2)
			other_tupl = (item2, item1)
			if other_tupl not in non_zeros and tupl not in non_zeros:
				non_zeros.add(tupl)
	g = igraph.Graph()
	g.add_vertices(sparse.csc_matrix.get_shape(group_matrix)[0])
	g.add_edges(non_zeros)
	return g, group_matrix

def distance_inverse(graph,list_of_pairs,group_matrix):
	tuples = list_of_pairs
	distance = []
	for tupl in tuples:
		a = tupl[0]
		b = tupl[1]
		val = graph.shortest_paths(a,b)
		value = val[0][0]
		#weight = graph[a,b]
		total_weight = 1.0*(group_matrix.sum())/2
		weight = group_matrix[a,b]
		weight_norm = 1.0*weight/total_weight
		if np.isinf(value)==True:
			inv_value = float(0)
		else:
			inv_value = weight_norm/value
		distance.append(inv_value)
	return distance


run = 0
industry = 'run' + str(run)
dist_cols = ["org_id","year","min_inv_distance","av_inv_distance"]
distance_df = pd.DataFrame(columns=dist_cols)
time_period = 34
start_year = 1976
end_year = start_year + time_period + 1

################### CHANGE ORG IDS AND NAME OF DISTANCE FILE
dist_df_name = "distance_df_" + industry + ".csv"
number_of_orgs = org_list_of_lists[run]

for year in range(start_year,end_year):
	#make the network
	toc = time()
	g_curr, matrix_out= format_data((year-1),subclass_matrix,date_array)
	mid = time()
	graph_time = mid - toc
	for org_id in number_of_orgs:
		patents = org_matrix[:,org_id].nonzero()[0]
		patent_in_org_year = patents[date_array[patents]==year]
		inv_distance_org = []
		for patent in patent_in_org_year:
			#get subclasses in patent and work out distance
			list_of_subclasses_in_patent = subclass_matrix[patent,:].nonzero()[1]
			all_tuples = combinations(list_of_subclasses_in_patent, 2)
			inv_distance = distance_inverse(g_curr,all_tuples,matrix_out)
			if len(np.array(inv_distance)) > 0:
				inv_distance = np.min(np.array(inv_distance))
				inv_distance_org.append(inv_distance)
		if len(np.array(inv_distance_org)) > 0:        
			min_org = np.min(np.array(inv_distance_org))
			average_min = np.mean(np.array(inv_distance_org))
			distance_df.loc[(len(distance_df)+1)] = (org_id, year,min_org, average_min)
	tic = time()
	actual_time = tic - toc
	print "graph time:", graph_time, "full time:", actual_time

distance_df.to_csv(dist_df_name)



