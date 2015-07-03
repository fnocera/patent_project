''' Makes dictionaries that need for the matrix, dates dictionary and dictionary mapping orgs to country
'''

import MySQLdb
import cPickle

#gets rid of annoying tuple format which comes out when request just one row
def cleaner(input_list):
	output_list = []
	for element in input_list:
		output_list.append(element[0])
	return output_list

def dictionary_maker(list_in):
	name_to_id = {}
	id_to_name = {}
	counter = 0
	for i in range(len(list_in)):
		name = list_in[i]
		if name.startswith('"') and name.endswith('"'):
    		name = name[1:-1]
		if name not in name_to_id:
			name_to_id[name] = counter 
			id_to_name[counter] = name
			counter += 1
	return name_to_id, id_to_name

def dictionary_maker2(list_in):
	import re
	name_to_id = {}
	id_to_name = {}
	counter = 0
	for i in range(len(list_in)):
		main_class = list_in[i][0]
		subclass = list_in[i][1]
		if main_class.startswith('"') and main_class.endswith('"'):
    		main_class = main_class[1:-1]
    	if subclass.startswith('"') and subclass.endswith('"'):
    		subclass = subclass[1:-1]
		name = main_class + '/' + subclass
		if name not in name_to_id:
			name_to_id[name] = counter 
			id_to_name[counter] = name
			counter += 1
	return name_to_id, id_to_name


#Makes index_to_wku and wku_to_index dictionaries and saves as plk
index_to_wku = []
wku_to_index = []
connect = MySQLdb.connect()
cursor = connect.cursor()
wku_selected = "SELECT * FROM fnocera.ID;" 
cursor.execute(wku_selected)
wku_selected = cursor.fetchall()
wku_list = cleaner(wku_selected)
#for wku in wku_selected:
#	wku_list.append(wku[0])
for i in range(len(wku_list)):
	index_to = (i, wku_list[i])
	wku_to = (wku_list[i], i)
	index_to_wku.append(index_to)
	wku_to_index.append(wku_to)
index_to_wku = dict(index_to_wku)
wku_to_index = dict(wku_to_index)
print "wku index length", len(index_to_wku), len(wku_to_index) #, index_to_wku[4], index_to_wku[100], index_to_wku[200] #LENGTH = 3463214
cPickle.dump(index_to_wku, open("index_to_wku.plk", "wb"))
cPickle.dump(wku_to_index, open("wku_to_index.plk", "wb"))


#Make class_to_id and id_to_class dictionaries and saves as plk
connect = MySQLdb.connect()
cursor = connect.cursor()
sql = 'SELECT main_class FROM fnocera.class;'
cursor.execute(sql)
classes = cursor.fetchall()
class_list = cleaner(classes)
class_to_id, id_to_class = dictionary_maker(class_list)
print "classes to id", len(class_to_id), len(id_to_class) #LENGTH = 732
cPickle.dump(class_to_id, open("class_to_id.plk", "wb"))
cPickle.dump(id_to_class, open("id_to_class.plk", "wb"))


#Make country_to_id and id_to_country dictionaries and saves as plk
connect = MySQLdb.connect()
cursor = connect.cursor()
sql = 'SELECT Country FROM fnocera.assignee;'
cursor.execute(sql)
countries = cursor.fetchall()
country_list = cleaner(countries)
country_to_id, id_to_country = dictionary_maker(country_list)
print "countries to id", len(country_to_id), len(id_to_country) #LENGTH = 907
cPickle.dump(country_to_id, open("country_to_id.plk", "wb"))
cPickle.dump(id_to_country, open("id_to_country.plk", "wb"))


#Make a date dictionary mapping wku to date
connect = MySQLdb.connect)
cursor = connect.cursor()
sql = 'SELECT wku, appl_date FROM fnocera.dates;'
cursor.execute(sql)
dates = cursor.fetchall()
dates_list = list(dates)
clean_dates = []
for i in range(len(dates_list)):
	wku = dates_list[i][0]
	date = dates_list[i][1]
	if len(date) == 8:
		year = date[0:4]
		mth = date[4:6]
		day = date[6:8]
		date = day+'-'+mth+'-'+year	
		#if year[0] > '2':
		#	print year
	elif len(date) == 10:
		year = date[0:4]
		mth = date[5:7]
		day = date[8:10]
		date = day+'-'+mth+'-'+year	
		#if year[0] > '2':
		#	print year
	tupl = (wku,date)
	clean_dates.append(tupl)
dates_dict = dict(clean_dates)
print "dates", len(dates_dict)
cPickle.dump(dates_dict, open("dates_dict.plk", "wb"))


#Makes an org_to_country dictionary
connect = MySQLdb.connect()
cursor = connect.cursor()
sql = 'SELECT OrgName, Country FROM fnocera.assignee;'
cursor.execute(sql)
request = cursor.fetchall()
country_list = list(request)
org_to_country = {}
for i in range(len(country_list)):
	org = country_list[i][0]
	country = country_list[i][1]
	if (org is not None) and (country is not None) and (org not in org_to_country):
		org_to_country[org] = country
print "org to country", len(org_to_country)
cPickle.dump(org_to_country, open("org_to_country.plk", "wb"))
'''

'''
#Make subclass_to_id and id_to_subclass dictionaries and saves as plk
connect = MySQLdb.connect()
cursor = connect.cursor()
sql = 'SELECT main_class, sub_class FROM fnocera.sub_classes;'
cursor.execute(sql)
classes = cursor.fetchall()
class_list = list(classes)
subclass_to_id, id_to_subclass = dictionary_maker2(class_list)
print "classes to id", len(subclass_to_id), len(id_to_subclass) #LENGTH = 203996
cPickle.dump(subclass_to_id, open("subclass_to_id.plk", "wb"))
cPickle.dump(id_to_subclass, open("id_to_subclass.plk", "wb"))

############### NEED TO MAKE DICTS and matrix for ORGS and INVENTOR 
#Make inventor to id and id to inventor
connect = MySQLdb.connect()
sql = 'SELECT Lastname, Firstname FROM fnocera.disamb_inventors;'
cursor.execute(sql)
names = cursor.fetchall()
name_list = list(names)
inventor_to_id, id_to_inventor = dictionary_maker2(name_list)
print "inventor to id", len(inventor_to_id), len(id_to_inventor) #LENGTH = 1504927
cPickle.dump(inventor_to_id, open("inventor_to_id.plk", "wb"))
cPickle.dump(id_to_inventor, open("id_to_inventor.plk", "wb"))


#Make org_to_id and id_to_org dictionaries and saves as plk
connect = MySQLdb.connect()
cursor = connect.cursor()
sql = 'SELECT AsgNum FROM fnocera.disamb_inventors;'
cursor.execute(sql)
orgs = cursor.fetchall()
org_list = cleaner(orgs)
org_to_id_NEW, id_to_org_NEW = dictionary_maker(org_list)
print "org to id", len(org_to_id_NEW), len(id_to_org_NEW) #, id_to_org[4], id_to_org[100], id_to_org[200] #LENGTH = 267503
cPickle.dump(org_to_id_NEW, open("org_to_id_NEW.plk", "wb"))
cPickle.dump(id_to_org_NEW, open("id_to_org_NEW.plk", "wb"))


