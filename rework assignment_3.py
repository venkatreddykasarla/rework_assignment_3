# loading the important libraries
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import seaborn as sns


# loading dataset

list_of_data = []
data_index = 0
with open('dataset/API_19_DS2_en_csv_v2_4028487.csv', 'r') as f:
    instance_reader = csv.reader(f, delimiter = ',')
    for csv_instance in instance_reader:
        if data_index >= 4:
            list_of_data.append(csv_instance)
        data_index += 1
        

main_df = pd.DataFrame(np.array(list_of_data)[1:, :], columns = np.array(list_of_data)[0])


# tidying the data
main_df[main_df == ''] = np.nan
main_df.iloc[:, 4:] = main_df.iloc[:, 4:].astype('float')


# selecting the indicators required for clustering:

## 1. Foreign direct investment, net inflows (% of GDP)
## 2. CO2 emissions (kg per PPP $ of GDP)

clus_datafram = main_df.loc[main_df['Indicator Name'].isin([
                                                            'Foreign direct investment, net inflows (% of GDP)',
                                                            'CO2 emissions (kg per PPP $ of GDP)'
                                                            ]), :]


# considering the features to be indicators for the year, 2018
country_2018_data = pd.pivot_table(data = clus_datafram,
                                   index = 'Country Name',
                                   columns = 'Indicator Name',
                                   values = '2018').dropna()

# selecting the positive foreign direct investments
country_2018_data = country_2018_data.loc[country_2018_data['Foreign direct investment, net inflows (% of GDP)'] > 0, :]


# min-max normalization module
def normalization_minmax(series_df):
    norm = MinMaxScaler()
    return norm.fit_transform(series_df)


# clustering min-max normalization
normalized_data = normalization_minmax(country_2018_data)
normalized_data = pd.DataFrame(normalized_data, 
                               index = list(country_2018_data.index),
                               columns = list(country_2018_data.columns))


# finding optimal number of clusters using Silhouette Scoring Method
silhouette_scores = []
for no_of_clusters in list(range(2,16)):
    cluster_modelling = KMeans(n_clusters = no_of_clusters, max_iter = 20, random_state = 4321).fit(normalized_data)
    silhouette_scores.append([no_of_clusters, silhouette_score(normalized_data, cluster_modelling.labels_)])

df_silhouette = pd.DataFrame(silhouette_scores, columns = ['Number of Clusters', 'Silhouette Score'])
sns.lineplot(x = 'Number of Clusters', y = 'Silhouette Score', data = df_silhouette)
sns.scatterplot(x = 'Number of Clusters', y = 'Silhouette Score', data = df_silhouette)
plt.show()


# 2 clusters are found to be as optimum
cluster_modelling = KMeans(n_clusters = 2, max_iter = 20, random_state = 4321)
cluster_modelling.fit(normalized_data)
cluster_preds = cluster_modelling.labels_
country_2018_data['Cluster'] = cluster_preds


# Boxplot Visualization of Foreign Direct Investment and CO2 Emissions for the 2 Clusters
plt.figure(figsize = (15, 5))

sns.set_style('whitegrid')
plt.subplot(1, 2, 1)
sns.boxplot(x = 'Cluster',
            y = 'Foreign direct investment, net inflows (% of GDP)',
            data = country_2018_data)
plt.subplot(1, 2, 2)
sns.boxplot(x = 'Cluster',
            y = 'CO2 emissions (kg per PPP $ of GDP)',
            data = country_2018_data)
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------#

# training the model using 2018 data by curve fitting to predict CO2 Emissions from Foreign Direct Investment
train_df = country_2018_data['Foreign direct investment, net inflows (% of GDP)']
train_Y = country_2018_data['CO2 emissions (kg per PPP $ of GDP)']

# model function definition
def curve_eq(x, a, b, c):
    return a*(x**(-2)) + (b/x) + c

# running the curve fit
popt, pcov = curve_fit(curve_eq, train_df, train_Y)


# utilizing the given err_ranges() for computing upper and lower confidence intervals
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


sigma = train_Y - curve_eq(train_df, *popt)
l, u = err_ranges(train_df.sort_values(), curve_eq, popt, sigma)
plt.figure(figsize = (15, 5))
plt.plot(train_df.sort_values(), l, 'r--', label = 'Lower Confidence Level')
plt.scatter(train_df, train_Y, label = 'original', marker = 'X')
plt.plot(train_df.sort_values(), curve_eq(train_df.sort_values(), *popt), color = 'red',
         label = 'fit: a=%3.2f, b=%3.2f, c=%3.2f' % tuple(popt))
plt.plot(train_df.sort_values(), u, 'r--', label = 'Upper Confidence Level')

plt.xlabel('Foreign direct investment, net inflows (% of GDP)')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP)')
plt.legend()
plt.title('2018')
plt.show()


# finding the C02 Emissions from Foreign Direct Investment for countries from each of the 2 clusters
cluster0_c = list(country_2018_data.loc[country_2018_data.Cluster == 0].index)[0]
print()
print('C02 emissions (kg per PPP $ of GDP) in', cluster0_c, curve_eq(train_df, *popt)[cluster0_c])
cluster1_c = list(country_2018_data.loc[country_2018_data.Cluster == 1].index)[0]
print()
print('C02 emissions (kg per PPP $ of GDP) in', cluster1_c, curve_eq(train_df, *popt)[cluster1_c])