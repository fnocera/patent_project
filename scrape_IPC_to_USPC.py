###Script to scrape the IPC to USPC website based on industry lists

import numpy as np
import requests
import pandas as pd
import cPickle
from bs4 import BeautifulSoup


industries = pd.read_csv("industry.csv")
industry_names = industries["Field_en"].unique()

ind_dict = {}
for field in industry_names:
    list_field = []
    filtered = industries[(industries["Field_en"]==field)]
    codes = filtered["IPC_code"]
    for name in codes:
        code = name[0:4]
        code = code.lower()
        #print code
        list_field.append(code)
    ind_dict[field] = list_field

def get_classes_for_one_code(code):
    get_page = 'http://www.uspto.gov/web/patents/classification/international/ipc/ipc8/ipc_concordance/ipc8' + code + 'us.htm'
    #print get_page
    req = requests.get(get_page)
    if req.ok == False: 
        print "Error in request", code
        us_class = []
        doesnt_work_list = []
    else:
        src = req.text
        soup = BeautifulSoup(src)
        #print soup.prettify()
        us_class = []
        rows = soup.body.findAll('td',headers="usclas")
        for row in rows:
            element = row.string
            try:
                element = int(element)
                us_class.append(element)
            except (ValueError, TypeError):
                #doesnt_work_list.append(element)
                print "this is not working", element
    return us_class

def get_classes(code):
    get_page = 'http://www.uspto.gov/web/patents/classification/international/ipc/ipc8/ipc_concordance/ipc8' + code + 'us.htm'
    #print get_page
    req = requests.get(get_page)
    if req.ok == False: 
        print "Error in request", code
        us_class = []
    else:
        src = req.text
        soup = BeautifulSoup(src)
        #print soup.prettify()
        us_class = []
        rows = soup.body.findAll('td',headers="usclas")
        for row in rows:
            element = row.find('a')
            element = row.string
            #print element
            try:
                element = int(element)
                us_class.append(element)
            except (AttributeError, ValueError, TypeError):
                #print element
                print code
    return us_class


dict_ind_classes = {}
for key in ind_dict.keys():
    values = ind_dict[key]
    class_set = set()
    for value in values:
        classes = get_classes(value)
        for clas in classes:
            actual_classes = []
            if classes.count(clas) > 3:
                actual_classes.append(clas)
                class_set |= set(actual_classes)
    dict_ind_classes[key] = class_set
cPickle.dump(dict_ind_classes,open("dict_ind_classes.plk","wb"))




