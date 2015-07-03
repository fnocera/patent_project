''' Script to read mysql tables and get data into (wku_index, class_index) format and then into matrix 

'''
import MySQLdb
import cPickle
import numpy as np
from scipy import sparse
from scipy import array
import re


file_open = open("wku_to_index.plk", "rb") #LENGTH = 3463214
wku_to_index = cPickle.load(file_open)


file_open = open("org_to_id.plk", "rb") #LENGTH = 414351
org_to_id = cPickle.load(file_open)

file_open = open("class_to_id.plk", "rb") #LENGTH = 732
class_to_id = cPickle.load(file_open)

file_open = open("country_to_id.plk", "rb")
country_to_id = cPickle.load(file_open) 

file_open = open("org_to_country.plk", "rb")
org_to_country = cPickle.load(file_open)

file_open = open("subclass_to_id.plk", "rb")
subclass_to_id = cPickle.load(file_open)


def connect_sql(command):
	connect = MySQLdb.connect()
	cursor = connect.cursor()
	sql = command 
	cursor.execute(sql)
	data = cursor.fetchall()
	output_list = list(data)
	return output_list


#Wku_index to Class_id
sql = 'SELECT wku, main_class FROM fnocera.class;'
class_list = connect_sql(sql)
wku_class_id = []
for i in range(len(class_list)):
	wku = class_list[i][0]
	clas = class_list[i][1]
	wku_index = wku_to_index[wku]
	class_id = class_to_id[clas]
	tupl = (wku_index, class_id)
	wku_class_id.append(tupl)
print "length of wku and class to ids", len(wku_class_id)
wku_class_id.sort(key=lambda x: [x[0], x[1]])
wku_class_id_list = [list(t) for t in zip(*wku_class_id)]
cPickle.dump(wku_class_id_list, open("wku_class_id_list.plk", "wb"))

#Making the coo_matrix for wkus and classes
rows = np.array(wku_class_id_list[0])
columns = np.array(wku_class_id_list[1])

M_key = max(wku_to_index, key=wku_to_index.get)
N_key = max(class_to_id, key=class_to_id.get)
M = int(wku_to_index[M_key]) + 1
N = class_to_id[N_key]
N = int(N) + 1
print "M should be 3463214 but is", M, "N should be 732 but is", N
data = np.ones(len(rows))

wku_class_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(M, N))
cPickle.dump(wku_class_matrix, open("wku_class_matrix.plk", "wb"))


def cleaner(input_list):
	output_list = []
	for element in input_list:
		output_list.append(element[0])
	return output_list

#wku_index to subclass_id
cursor = connect.cursor()
sql = "SELECT main_class, sub_class FROM fnocera.sub_classes;"
cursor.execute(sql)
data = cursor.fetchall()
subclass_list = list(data)
cursor = connect.cursor()
sql = "SELECT wku FROM fnocera.sub_classes;"
cursor.execute(sql)
data = cursor.fetchall()
wku_list = cleaner(data)
connect.close()
wku_class_id = []
for i in range(len(subclass_list)):
	wku = wku_list[i]
	mainclas = subclass_list[i][0]
	subclas = subclass_list[i][1]
	clas = mainclas + "/" + subclas
	wku_index = wku_to_index[wku]
	class_id = subclass_to_id[clas]
	tupl = (wku_index, class_id)
	wku_class_id.append(tupl)
print "length of wku and class to ids", len(wku_class_id)
wku_class_id.sort(key=lambda x: [x[0], x[1]])
wku_class_id_list = [list(t) for t in zip(*wku_class_id)]
cPickle.dump(wku_class_id_list, open("wku_subclass_id_list.plk", "wb"))

#Making the coo_matrix for wkus and classes
rows = np.array(wku_class_id_list[0])
columns = np.array(wku_class_id_list[1])

M_key = max(wku_to_index, key=wku_to_index.get)
N_key = max(subclass_to_id, key=subclass_to_id.get)
M = int(wku_to_index[M_key]) + 1
N = subclass_to_id[N_key]
N = int(N) + 1
print "M should be 3463214 but is", M, "N should be 203996 but is", N
data = np.ones(len(rows))

wku_subclass_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(M, N))
cPickle.dump(wku_subclass_matrix, open("wku_subclass_matrix.plk", "wb"))


################## INVENTOR AND ORGANIZATION FROM disambiguated inventors Fleming 

file_open = open("inventor_to_id.plk", "rb")
inventor_to_id = cPickle.load(file_open)

file_open = open("patent_to_wku_dict.plk", "rb")
patent_to_wku = cPickle.load(file_open)

#Wku index to inventor_id
sql = 'SELECT Patent, Lastname, Firstname FROM fnocera.disamb_inventors;'
inv_list = connect_sql(sql)
wku_inv_id = []
for i in range(len(inv_list)):
	patent = inv_list[i][0]
	if re.match("^[0-9]",patent):
		if str(patent) in wku_to_index:
			mainclas = inv_list[i][1]
			subclas = inv_list[i][2]
			if mainclas.startswith('"') and mainclas.endswith('"'):
					mainclas = mainclas[1:-1]
			if subclas.startswith('"') and subclas.endswith('"'):   
					subclas = subclas[1:-1]
			inventor = mainclas + "/" + subclas
			wku_index = wku_to_index[str(patent)]
			inv_id = inventor_to_id[inventor]
			tupl = (wku_index, inv_id)
			wku_inv_id.append(tupl)
		elif str(patent[0]) == '0' and str(patent[1:]) in wku_to_index:
			mainclas = inv_list[i][1]
			subclas = inv_list[i][2]
			if mainclas.startswith('"') and mainclas.endswith('"'):
				mainclas = mainclas[1:-1]
			if subclas.startswith('"') and subclas.endswith('"'):   
				subclas = subclas[1:-1]
			inventor = mainclas + "/" + subclas
			wku_index = wku_to_index[str(patent[1:])]
			inv_id = inventor_to_id[inventor]
			tupl = (wku_index, inv_id)
			wku_inv_id.append(tupl)
	elif re.match("^R",patent):
		if str(patent) in wku_to_index:
			mainclas = inv_list[i][1]
			subclas = inv_list[i][2]
			if mainclas.startswith('"') and mainclas.endswith('"'):
					mainclas = mainclas[1:-1]
			if subclas.startswith('"') and subclas.endswith('"'):   
					subclas = subclas[1:-1]
			inventor = mainclas + "/" + subclas
			wku_index = wku_to_index[str(patent)]
			inv_id = inventor_to_id[inventor]
			tupl = (wku_index, inv_id)
			wku_inv_id.append(tupl)
	#else:
		#print patent, "not in patent_to_wku dictionary"
print "length of wku and inv to ids", len(wku_inv_id)
wku_inv_id.sort(key=lambda x: [x[0], x[1]])
wku_inv_id_list = [list(t) for t in zip(*wku_inv_id)]
cPickle.dump(wku_inv_id_list, open("wku_inventor_id_list.plk", "wb"))

#Making the coo_matrix for wkus and org
rows = np.array(wku_inv_id_list[0])
columns = np.array(wku_inv_id_list[1])

M_key = max(wku_to_index, key=wku_to_index.get)
N_key = max(inventor_to_id, key=inventor_to_id.get)
M = int(wku_to_index[M_key]) + 1
N = inventor_to_id[N_key]
N = int(N) + 1
print "M should be 3463214 but is", M, "N should be 1504927 but is", N
data = np.ones(len(rows))

wku_inv_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(M, N))
cPickle.dump(wku_inv_matrix, open("wku_inventor_matrix.plk", "wb"))

del inventor_to_id, wku_inv_matrix

file_open = open("org_to_id_NEW.plk", "rb")
org_to_id_NEW = cPickle.load(file_open)

#Wku index to org_id
sql = 'SELECT Patent, AsgNum FROM fnocera.disamb_inventors;'
org_list = connect_sql(sql)
wku_org_id = []
for i in range(len(org_list)):
	patent = org_list[i][0]
	org = org_list[i][1]
	if re.match("^[0-9]",patent):
		if str(patent) in wku_to_index:
			wku_index = wku_to_index[str(patent)]
			org_id = org_to_id_NEW[org]
			tupl = (wku_index, org_id)
			wku_org_id.append(tupl)
		elif str(patent[0]) == '0' and str(patent[1:]) in wku_to_index:
			wku_index = wku_to_index[str(patent[1:])]
			org_id = org_to_id_NEW[org]
			tupl = (wku_index, org_id)
			wku_org_id.append(tupl)
	elif re.match("^R",patent):
		if str(patent) in wku_to_index:
			wku_index = wku_to_index[str(patent)]
			org_id = org_to_id_NEW[org]
			tupl = (wku_index, org_id)
			wku_org_id.append(tupl)
	#else:
		#print patent, "not in patent_to_wku dictionary"
print "length of wku and org to ids", len(wku_org_id)
wku_org_id.sort(key=lambda x: [x[0], x[1]])
wku_org_id_list = [list(t) for t in zip(*wku_org_id)]
cPickle.dump(wku_org_id_list, open("wku_org_id_list.plk", "wb"))

#Making the coo_matrix for wkus and org
rows = np.array(wku_org_id_list[0])
columns = np.array(wku_org_id_list[1])

M_key = max(wku_to_index, key=wku_to_index.get)
N_key = max(org_to_id_NEW, key=org_to_id_NEW.get)
M = int(wku_to_index[M_key]) + 1
N = org_to_id_NEW[N_key]
N = int(N) + 1
print "M should be 3463214 but is", M, "N should be 267503 but is", N
data = np.ones(len(rows))

wku_org_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(M, N))
cPickle.dump(wku_org_matrix, open("wku_org_matrix.plk", "wb"))
