
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import pandas as pd
from collections import Counter
import random
from itertools import combinations
from time import time
from scipy import sparse
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
#%matplotlib inline

#maybe try with 50 patents at least in industries, can try with median 
file_open = open("dict_ind_org_30.plk", "rb")
dict_ind_org = cPickle.load(file_open)
sectors = dict_ind_org.keys()

file_open = open("dict_ind_org_50.plk", "rb")
dict_ind_org = cPickle.load(file_open)
sectors = dict_ind_org.keys()

#might not need this!!
#file_open = open("org_list_of_lists.plk", 'rb')
#org_list_of_lists = cPickle.load(file_open)

cols = ["org_id","year","fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching"]
total_org_yearly_df = pd.DataFrame(columns=cols)
dist_cols = ["org_id","year","min_inv_distance","av_inv_distance"]
total_dist_df = pd.DataFrame(columns=dist_cols)

#open all dataframes and concatenate together 
for i in range(10):
    name = 'yearly_dfrun' + str(i) + '.csv'
    yearly_df = pd.DataFrame.from_csv(name)
    total_org_yearly_df = total_org_yearly_df.append(yearly_df)

# inner join the distance as well - BIGDATAFRAME
for i in range(10):
    name = "distance_df_run" + str(i) + ".csv"
    dist_df = pd.DataFrame.from_csv(name)
    total_dist_df = total_dist_df.append(dist_df)


learning_dist_df = pd.merge(total_org_yearly_df, total_dist_df, how='left', on=['year', 'org_id'])
learning_dist_df = learning_dist_df.dropna()
learning_dist_df["min_inv_distance"] = learning_dist_df["min_inv_distance"]*100000
learning_dist_df["av_inv_distance"] = learning_dist_df["av_inv_distance"]*100000


### Need to either ger rid of the av_inv_distance and min_inv_distance or scale them so that they are clear
#or separate plots? 
def make_industry_plot(mean_df,std_df,industry_name):
    fig,ax = plt.subplots()
    ax.errorbar(mean_df["year"],mean_df["fract_org"],yerr=std_orgs["fract_org"],fmt='-', ecolor='k', color='b')
    ax.errorbar(mean_df["year"],mean_df["fract_new_for_org"],yerr=std_orgs["fract_new_for_org"],fmt='-', ecolor='k', color='g')
    ax.errorbar(mean_df["year"],mean_df["fract_inv_already"],yerr=std_orgs["fract_inv_already"],fmt='-', ecolor='k', color='r')
    ax.errorbar(mean_df["year"],mean_df["fract_learning_from_org"],yerr=std_orgs["fract_learning_from_org"],fmt='-', ecolor='k', color='c')
    ax.errorbar(mean_df["year"],mean_df["fract_learning_others"],yerr=std_orgs["fract_learning_others"],fmt='-', ecolor='k', color='m')
    ax.errorbar(mean_df["year"],mean_df["fract_searching"],yerr=std_orgs["fract_searching"],fmt='-', ecolor='k', color='y')
    ax.errorbar(mean_df["year"],mean_df["min_inv_distance"],yerr=std_orgs["min_inv_distance"],fmt='-', ecolor='k', color='0.75')
    ax.errorbar(mean_df["year"],mean_df["av_inv_distance"],yerr=std_orgs["av_inv_distance"],fmt='-', ecolor='k', color='0.75')
    ax.set_xlabel('year')
    ax.set_ylabel('fraction')
    cols =  ["fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching","min_inv_distance","av_inv_distance"]
    ax.legend(cols, loc='best', fontsize = 'xx-small')
    name = "learning_" + industry_name + ".png"
    plt.title(name)
    plt.savefig(name)
    return

def make_industry_plot_no_dist(mean_df,std_df,industry_name):
    fig,ax = plt.subplots()
    ax.errorbar(mean_df["year"],mean_df["fract_org"],yerr=std_orgs["fract_org"],fmt='-', ecolor='k', color='b')
    ax.errorbar(mean_df["year"],mean_df["fract_new_for_org"],yerr=std_orgs["fract_new_for_org"],fmt='-', ecolor='k', color='g')
    ax.errorbar(mean_df["year"],mean_df["fract_inv_already"],yerr=std_orgs["fract_inv_already"],fmt='-', ecolor='k', color='r')
    ax.errorbar(mean_df["year"],mean_df["fract_learning_from_org"],yerr=std_orgs["fract_learning_from_org"],fmt='-', ecolor='k', color='c')
    ax.errorbar(mean_df["year"],mean_df["fract_learning_others"],yerr=std_orgs["fract_learning_others"],fmt='-', ecolor='k', color='m')
    ax.errorbar(mean_df["year"],mean_df["fract_searching"],yerr=std_orgs["fract_searching"],fmt='-', ecolor='k', color='y')
    ax.set_xlabel('year')
    ax.set_ylabel('fraction')
    cols =  ["fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching"]
    ax.legend(cols, loc='best', fontsize = 'xx-small')
    name = "learning_" + industry_name + ".png"
    plt.title(name)
    plt.savefig(name)
    return

def distance_plot(mean_df,std_df,industry_name):
    fig,ax = plt.subplots()
    ax.errorbar(mean_df["year"],mean_df["min_inv_distance"],yerr=std_orgs["min_inv_distance"],fmt='-', ecolor='k', color='0.75')
    ax.errorbar(mean_df["year"],mean_df["av_inv_distance"],yerr=std_orgs["av_inv_distance"],fmt='-', ecolor='k', color='k')
    ax.set_xlabel('year')
    ax.set_ylabel('inv distance')
    cols =  ["min_inv_distance","av_inv_distance"]
    ax.legend(cols, loc='best', fontsize = 'xx-small')
    name = "distance_" + industry_name + ".png"
    plt.title(name)
    plt.savefig(name)
    return

###THEN WILL NEED TO DO THIS FOR ALL INDUSTRIES 1) ADD IN NAMING THING THAT IS AUTOMATIC!! 2) FIGURE OUT HOW CAN MAKE DISTANCES LARGE SO CAN
# SEE THEM OTHERWISE ITS SILLY 
for sector in sectors:  
    splitted = sector.split()
    sector_name = splitted[0]
    list_of_orgs = list(dict_ind_org[sector])
    sector_cols = ["year","fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching","min_inv_distance","av_inv_distance"]
    sector_df = pd.DataFrame(columns=sector_cols)
    sector_df = sector_df.append(learning_dist_df.loc[learning_dist_df['org_id'].isin(list_of_orgs)])
    mean_orgs = sector_df.groupby("year").mean()
    mean_orgs["year"] = mean_orgs.index.values
    median_orgs = sector_df.groupby("year").median()
    median_orgs["year"] = median_orgs.index.values
    std_orgs = sector_df.groupby("year").std()
    std_orgs["year"] = std_orgs.index.values
    std_orgs.drop('org_id', axis=1, inplace=True)

    make_industry_plot_no_dist(mean_orgs,std_orgs,sector_name)

    distance_plot(median_orgs,std_orgs,sector_name)
    plt.close("all")
    



## Figure out how to do the regression for distance (or if dont have distance use one of the other metrics just to test)
def regression(data_to_fit,target):
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    X = sm.add_constant(data_to_fit)
    model = sm.OLS(target, X)
    results = model.fit()
    print(results.summary())
    return results

def GLM_poisson(data_to_fit,target):
    import statsmodels.api as sm
    X = sm.add_constant(data_to_fit)
    poisson_model = sm.GLM(target, X, family=sm.families.Poisson(link=sm.families.links.log))
    results = poisson_model.fit()
    print(results.summary())
    return results


#### WORK OUT HOW TO SAVE THE REGRESSION OUTPUT!!
def regression_to_dataframe(data_to_fit,target):
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    X = sm.add_constant(data_to_fit)
    model = sm.OLS(target, X)
    model_result = model.fit()
    print(model_result.summary())
    statistics = pd.Series({'r2': model_result.rsquared,
                  'adj_r2': model_result.rsquared_adj})
    # put them togher with the result for each term
    result_df = pd.DataFrame({'params': model_result.params,
                              'pvals': model_result.pvalues,
                              'std': model_result.bse,
                              'statistics': statistics})
    # add the complexive results for f-value and the total p-value
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue},
                              'pvals': {'_f_test': model_result.f_pvalue}})
    # merge them and unstack to obtain a hierarchically indexed series
    res_series = pd.concat([result_df, fisher_df]).unstack()
    return res_series.dropna()

def results_to_dataframe(model_result):
    statistics = pd.Series({'r2': model_result.rsquared,
                  'adj_r2': model_result.rsquared_adj})
    # put them togher with the result for each term
    result_df = pd.DataFrame({'params': model_result.params,
                              'pvals': model_result.pvalues,
                              'std': model_result.bse,
                              'statistics': statistics})
    # add the complexive results for f-value and the total p-value
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue},
                              'pvals': {'_f_test': model_result.f_pvalue}})
    # merge them and unstack to obtain a hierarchically indexed series
    res_series = pd.concat([result_df, fisher_df]).unstack()
    return res_series.dropna()


def missing_indicator(df, column_name):
    """ add a missing indicator for a feature to the dataframe, 1 if missing and 0 otherwise. """
    nul = df[[column_name]].isnull()
    nul = nul.applymap(lambda x: 1 if x else 0)
    name = column_name + "_missing"
    df[name] = nul
    return df

def run_missing_indicator(df, cols):
    for col in cols:
        df = missing_indicator(df,col)
    return df


def reg_to_df_line(model_result,df):
    result = list(model_result.params.values) + list(model_result.pvalues.values) + [model_result.rsquared] + [model_result.rsquared_adj] + list(model_result.bse.values)
    return result 

def poisson_to_df_line(model_result,df):
    line = list(model_result.params.values) + list(model_result.pvalues.values) + [model_result.deviance] + [model_result.aic] + [model_result.pearson_chi2] + [model_result.nobs] + [model_result.llf] + list(model_result.bse.values)
    return line 


cols = ["param_c","param_1", "param_2", "param_3", "param_4","pv_c","pv_1", "pv_2", "pv_3", "pv_4","deviance", "AIC", "pearson_chi","data_points","log_likelihood","std_c", "std_1", "std_2", "std_3", "std_4"]
poisson_df = pd.DataFrame(columns=cols)
for sector in sectors:  
    splitted = sector.split()
    sector_name = splitted[0]
    list_of_orgs = list(dict_ind_org[sector])
    sector_cols = ["year","fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching","min_inv_distance","av_inv_distance"]
    sector_df = pd.DataFrame(columns=sector_cols)
    sector_df = sector_df.append(learning_dist_df.loc[learning_dist_df['org_id'].isin(list_of_orgs)])
    df_to_fit = sector_df[["year","fract_org","fract_new_for_org","fract_learning_from_org"]]
    target = sector_df["av_inv_distance"]
    df_to_fit['year'] = df_to_fit['year'] - 1976
    scaler = StandardScaler()
    variables = scaler.fit_transform(df_to_fit)
    #results = regression(df_to_fit,target)
    results = GLM_poisson(variables,target)
    poisson_df.loc[sector_name] = poisson_to_df_line(results,poisson_df)

poisson_df.to_csv("distance_poisson_regression_results.csv")


#missing_indicator(sector_df,sector_scols)

####Count number of patents in industry per year ()
#this can only do with industry mean-ed data
file_open = open("wku_org_matrix_final.plk", "rb")
org_matrix = cPickle.load(file_open)
org_matrix = org_matrix.tocsr()

file_open = open("date_issue_array.plk", 'rb')
date_array = cPickle.load(file_open)

def no_of_patents_df(sector_df,org_matrix,date_array,list_of_orgs): 
    no_of_patents = pd.DataFrame(columns=[["year","no_of_patents"]])
    for year in sector_df['year'].unique():
        year = int(year)
        patents = np.unique(org_matrix[:,list_of_orgs].nonzero()[0])
        patent_in_year = len(patents[date_array[patents]==year])
        no_of_patents.loc[(len(no_of_patents)+1)] = (year,patent_in_year)
    return no_of_patents

def no_of_patents_per_org(sector_df,org_matrix,date_array,list_of_orgs): 
    no_of_patents = pd.DataFrame(columns=[["year","org_id","no_of_patents"]])
    for year in sorted(sector_df['year'].unique()):
        year = int(year)
        print year
        for org in list_of_orgs:
            patents = org_matrix[:,org].nonzero()[0]
            patent_in_year = len(patents[date_array[patents]==year])
            no_of_patents.loc[(len(no_of_patents)+1)] = (year,float(org),patent_in_year)
    return no_of_patents

from sklearn.preprocessing import StandardScaler
from time import time
cols = ["param_c","param_1", "param_2", "param_3", "param_4", "param_5", "pv_c","pv_1", "pv_2", "pv_3", "pv_4","pv_5","deviance", "AIC", "pearson_chi","data_points","log_likelihood","std_c", "std_1", "std_2", "std_3", "std_4", "std_5"]
poisson_df = pd.DataFrame(columns=cols)
for sector in sectors:
    count = 33  
    toc = time()
    splitted = sector.split()
    sector_name = splitted[0]
    list_of_orgs = list(dict_ind_org[sector])
    sector_cols = ["year","org_id","fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching","min_inv_distance","av_inv_distance"]
    sector_df = pd.DataFrame(columns=sector_cols)
    sector_df = sector_df.append(learning_dist_df.loc[learning_dist_df['org_id'].isin(list_of_orgs)])
    no_of_patents_company = no_of_patents_per_org(sector_df,org_matrix,date_array,list_of_orgs)
    data_nop_org = pd.merge(sector_df, no_of_patents_company, how='left', on=['org_id','year'])
    df_to_fit_nop = data_nop_org[["year","fract_org","fract_new_for_org","fract_learning_from_org","av_inv_distance"]]
    df_to_fit_nop['year'] = df_to_fit_nop['year'] - 1976
    scaler = StandardScaler()
    variables = scaler.fit_transform(df_to_fit_nop)
    target_nop = data_nop_org["no_of_patents"].astype(float)
    results = GLM_poisson(variables,target_nop)
    poisson_df.loc[sector_name] = poisson_to_df_line(results,poisson_df)
    tic = time()
    time_taken = tic - toc
    count -=1 
    print "time for " + sector_name +" "+ str(time_taken) + " " + str(count)


poisson_df.to_csv("no_patents_poisson_regression_results.csv")


#results_patents = regression(df_to_fit_nop,target_nop)
#results_series = results_to_dataframe(results_patents)

'''
#industry level
for sector in sectors:  
    splitted = sector.split()
    sector_name = splitted[0]
    list_of_orgs = list(dict_ind_org[sector])
    no_of_patents_industry = no_of_patents_df(sector_df,org_matrix,date_array,list_of_orgs)
    sector_cols = ["year","fract_org","fract_new_for_org","fract_inv_already","fract_learning_from_org","fract_learning_others","fract_searching","min_inv_distance","av_inv_distance"]
    sector_df = pd.DataFrame(columns=sector_cols)
    sector_df = sector_df.append(learning_dist_df.loc[learning_dist_df['org_id'].isin(list_of_orgs)])
    mean_orgs = sector_df.groupby("year").mean()
    mean_orgs["year"] = mean_orgs.index.values
    mean_orgs["av_inv_distance"] = mean_orgs["av_inv_distance"]
    mean_orgs["min_inv_distance"] = mean_orgs["min_inv_distance"]
    data_nop = pd.merge(mean_orgs, no_of_patents_industry, how='left', on=['year'])
    #df_to_fit_nop = data_nop[["year","fract_org","fract_new_for_org","fract_learning_others","fract_learning_from_org","fract_searching","min_inv_distance","av_inv_distance"]]
    df_to_fit_nop = data_nop[["year","fract_org","fract_new_for_org","fract_searching"]]
    target_nop = data_nop["no_of_patents"]
    #results_patents = regression(df_to_fit_nop,target_nop)
    #results_series = results_to_dataframe(results_patents)
    


    #SOMEHOW SAVE THE REGRESSION OUTPUT

#cols = ["param_0","param_1", "param_2", "param_3", "param_4", "param_5", "param_6","pv_0","pv_1", "pv_2", "pv_3", "pv_4", "pv_5", "pv_6", "rsqu", "adj_rsqu", "std_0","std_1", "std_2", "std_3", "std_4", "std_5", "std_6"]
#regression_df = pd.DataFrame(columns=cols)
'''



    #SOMEHOW SAVE THE REGRESSION OUTPUT

'''
1) run graphs with median and with mean for 50_patent industries (okay, basically same - just tiny bit clearer trend, use 50)
2) make graphs of distance for all industries (do for 30 if the 50 patents are basically the same) --> used 50 to do graphs, but maybe should have done the median, std big so no super clear trend 
(redo the graphs with the median maybe!!!)
3) re-scale distance for all regressions
Year make so that is year - earliest year....for all regressions --> ie work on the feature improvement thing 
4) check variable shape distance (do groupby maybe but after scale it so makes more sense) - is poisson as well!! 
5) check correlations between variables (learning metrics)
6) change regression to GLM Poisson 
7) figure out how to post results so that are understandable/can see what is going on....
8) Write up 

'''


'''
for sector in sectors:
    list_of_orgs = list(dict_ind_org[sector])
    sector_df = average_of_sector_inc_dist(list_of_orgs,total_org_yearly_df)
    mean_orgs = sector_df.groupby("year").mean()
    std_orgs = sector_df.groupby("year").std()

    
fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(x,y,yerr=list_of_std,fmt='-', color='b')
'''







