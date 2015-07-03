###File that makes dataframe of org and inventor data

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

file_open = open("wku_inventor_matrix_final.plk", "rb")
inventor_matrix = cPickle.load(file_open)
inventor_matrix = inventor_matrix.tocsr()


#############NEED TO PUT IN WHAT ORGS TO RUN AND NAMES OF DFS to SAVE!!!
run = 0
industry = 'run' + str(run)
org_df_name = "org_df" + "industry" +".csv"
inv_df_name = "inv_df" + "industry" +".csv"
org_df_pickle = "org_df" + "industry" + ".plk"
inv_df_pickle = "inv_df" + "industry" + ".plk"
yearly_df_nam = "yearly_df" + "industry" +".csv"
number_of_orgs = org_list_of_lists[run]
elapsed_time = []
time_period = 34
start_year = 1976
end_year = start_year + time_period + 1
save_list = np.arange(0,481,40)

#put the matrix making functions into a function to save memory
initialize_year = 1975
patents_start = (date_array==initialize_year).nonzero()[0]
subM = subclass_matrix[patents_start,:]

inv_cols = ["org_id","inventor_id","year","patent", "fract_inventor_already", "fract_new_to_inventor", "fract_other_inventor", "fract_remaining"]
inv_df = pd.DataFrame(columns=inv_cols)
org_cols = ["org_id","year", "patent", "fract_org_already", "fract_inv_contribution", "fract_remaining"]
org_df = pd.DataFrame(columns=org_cols)


for i in number_of_orgs:
	toc = time()
	org_id = i
	orgM = org_matrix[:,i]
	subM = subclass_matrix[patents_start,:]
	K_inv = (inventor_matrix[patents_start,:]).T*subM
	K_org = (org_matrix[patents_start,:]).T*subM
	K_org_current = set(K_org[org_id,:].nonzero()[1])
	patents = org_matrix[:,org_id].nonzero()[0]
	
	for year in range(start_year,end_year):
		K_org_current = set(K_org[org_id,:].nonzero()[1]) #not sure if I need this here
		patent_in_year = (date_array==year).nonzero()[0]
		patent_in_org_year = patents[date_array[patents]==year]

		for patent in patent_in_org_year:
			subclasses = set(subclass_matrix[patent,:].nonzero()[1])
			total_number = len(subclasses)
			#print total_number
			common = subclasses & K_org_current
			fraction_org = 1.0*len(common)/total_number 

			inventors_in_patent = inventor_matrix[patent,:].nonzero()[1]
			if len(inventors_in_patent) == 0:
				inventors = set()
				inventor_contribution = (subclasses - K_org_current) & inventors
				fraction_inv = 1.0*len(inventor_contribution)/total_number
				#print inventors_in_patent
				fraction_remaining = 1 - (fraction_org + fraction_inv)
				#print fraction_org, fraction_inv, fraction_remaining, year, patent
		
			elif len(inventors_in_patent) > 0:
				inventors = set((K_inv[inventors_in_patent,:]).nonzero()[1])#.sum(axis=0) REMOVED THIS BECAUSE NOT SURE WHY SUMMING!!!
				inventor_contribution = (subclasses - K_org_current) & inventors
				fraction_inv = 1.0*len(inventor_contribution)/total_number
				#print inventors_in_patent
				fraction_remaining = 1 - (fraction_org + fraction_inv)
				#print fraction_org, fraction_inv, fraction_remaining, year, patent
		
				for inventor in inventors_in_patent:
					invented = set(K_inv[inventor,:].nonzero()[1])
					fraction_inv_already = 1.0*len(subclasses & invented)/total_number
					fraction_new_to_inv = 1.0*len((subclasses - invented) & K_org_current)/total_number
					fraction_other_inv = 1.0*len((subclasses - invented - K_org_current) & inventors)/total_number
					other_fraction = 1 - (fraction_inv_already + fraction_new_to_inv + fraction_other_inv)

					inv_df.loc[(len(inv_df)+1)] = (org_id, inventor, year, patent, fraction_inv_already, fraction_new_to_inv, fraction_other_inv, other_fraction)#add row to dataframe, id, year, 4 fractions, org_id 
			
				#print inventor, year, patent, fraction_inv_already, fraction_new_to_inv, fraction_other_inv, other_fraction
			
			org_df.loc[(len(org_df)+1)] = (org_id, year, patent, fraction_org, fraction_inv, fraction_remaining)#add row for org for every patent with fractions

		subM = subclass_matrix[patent_in_year,:]
		K_inv = K_inv + (inventor_matrix[patent_in_year,:]).T*subM #CHECK THAT THIS IS DOING THE RIGHT THING
		K_org = K_org + (org_matrix[patent_in_year,:]).T*subM
	tic = time()
	
	if i in save_list:
		cPickle.dump(org_df,open(org_df_pickle,"wb"))
		cPickle.dump(inv_df,open(inv_df_pickle,"wb"))

	time_one_company = tic - toc
	print org_id, "time:",time_one_company
	elapsed_time.append(time_one_company)

org_df.to_csv(org_df_name)
inv_df.to_csv(inv_df_name)
cPickle.dump(org_df,open(org_df_pickle,"wb"))
cPickle.dump(inv_df,open(inv_df_pickle,"wb"))


###THIS IS TO LOOK AT THE DISTRIBUTIONS - PROBABLY HAVE TO AGGREGATE BY INDUSTRY INSTEAD
def make_org_yearly_df(org_df,inv_df,industry):
	number_of_orgs = list(org_df["org_id"].unique())
	yearly_df_name = "yearly_df" + industry + ".csv"
	time_period = 34
	start_year = 1976
	end_year = start_year + time_period + 1
	yearly_cols = ["org_id","year","fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching"]
	yearly_df = pd.DataFrame(columns=yearly_cols)
	for org_id in number_of_orgs:
		for year in range(start_year,end_year):
			#print year
			fract_org = org_df[(org_df["org_id"]==org_id) & (org_df["year"]==year)].mean()[3]
			fract_new = org_df[(org_df["org_id"]==org_id) & (org_df["year"]==year)].mean()[4]
			fract_inv_al = inv_df[(inv_df["org_id"]==org_id) & (inv_df["year"]==year)].mean()[4]
			fract_learn_from_org = inv_df[(inv_df["org_id"]==org_id) & (inv_df["year"]==year)].mean()[5]
			fract_learn_others = inv_df[(inv_df["org_id"]==org_id) & (inv_df["year"]==year)].mean()[6]
			fract_searching = inv_df[(inv_df["org_id"]==org_id) & (inv_df["year"]==year)].mean()[7]
			yearly_df.loc[(len(yearly_df)+1)] = (org_id, year, fract_org, fract_new, fract_inv_al, fract_learn_from_org, fract_learn_others,fract_searching)

	yearly_df.to_csv(yearly_df_name)
	return yearly_df

yearly_df = make_org_yearly_df(org_df,inv_df,industry)
yearly_df.to_csv(yearly_df_nam)


