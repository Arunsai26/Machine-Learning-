# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 20:56:11 2025

@author: Arunsaikoya
"""

import pandas as pd 

df = pd.read_csv(r"C:\Users\Arunsaikoya\Desktop\ml practice\clustering\Project\Data Set (2)\Data Set (5)\AirTraffic_Passenger_Statistics.csv")
df.head()
df.columns

from sqlalchemy import create_engine , text
from urllib.parse import quote 

# moving my data into local sql file 

user , pw , db = "root" , quote("Akmp@235") , "univ_db"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost:3306/{db}")
df.to_sql('airport_tbl',
          con = engine , 
          if_exists = 'replace',
          chunksize = 1000 ,
          index = False)

# Data preprocessing stage 

import dtale 

d = dtale.show(df , host = 'localhost' , port = 8000)
d.open_browser()

# checking for missing values 

null_values = df.isna().sum()
print("null values of the Airtraffic passenger dataset :", null_values)

df = df.dropna(subset = ["Operating Airline IATA Code"]) # drop all null values doesnt effect anything 

# Type casting 

df["Activity Period"] = pd.to_datetime(df["Activity Period"] , format = "%Y%m")
df.dtypes 
df.head()


# dropping the duplicates
df = df.drop_duplicates()

# checking dupilcates 

checking_duplicates = df.duplicated().sum()
print(checking_duplicates)

# One hot encoding in order to convert the categorical data to numerical data 

# before one hot enoding we shall drop Operating Airline IATA Code as it is a redundant column 

df = df.drop(columns = ["Operating Airline IATA Code"])

from sklearn.preprocessing import OneHotEncoder

columns = ["Operating Airline","GEO Region","Terminal","Boarding Area","Month"]

df_encoded = pd.get_dummies(df,columns = columns , drop_first = True , dtype = int )

df_encoded.shape

# Apply on Passenger Count 

from sklearn.preprocessing import MinMaxScaler

min_max = MinMaxScaler() # minmaxscaler function 

df_encoded[["Passenger Count", "Year"]] = min_max.fit_transform(df_encoded[["Passenger Count", "Year"]])
df_encoded["Passenger Count"].head()
df_encoded["Passenger Count","Year"].describe()
df_encoded["Passenger Count"].info()




# remove out_liers from passenger count 

Q1 = df_encoded["Passenger Count"].quantile(0.25)
Q3 = df_encoded["Passenger Count"].quantile(0.75)
IQR = Q3-Q1 


lower_bound = Q1-IQR*1.5
upper_bound = Q3+IQR*1.5

df_encoded["Passenger Count"] = np.where(df_encoded["Passenger Count"] > upper_bound, upper_bound,
                                         np.where(df_encoded["Passenger Count"] < lower_bound, 
                                                  lower_bound, 
                                                  df_encoded["Passenger Count"]))

df_encoded.shape

# Outlier checking 

import seaborn as sns 

sns.boxplot(df_encoded["Passenger Count"])


df_encoded[["Passenger Count", "Year"]] = min_max.fit_transform(df_encoded[["Passenger Count", "Year"]])
df_encoded[["Passenger Count","Year"]].describe()

df_encoded.to_sql(
    "processeddata_airport",
    con = engine , 
    if_exists = 'replace',
    chunksize = 10000 , 
    index = False
    )


df_encoded['Activity Period'] = df_encoded['Activity Period'].astype('int64')
from scipy.cluster.hierarchy import dendrogram,linkage

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt



# after encoding there are too many rows we shall consider 500 from df_encoded and select randomly 
sample_df = df_encoded.sample(500 , random_state = 42)
Z = linkage(sample_df , method = 'complete')
plt.figure(1,figsize = (16,8))
dendrogram(Z , truncate_mode="level",p=5)
plt.title("'Hierarchical Clustering Dendrogram")
plt.xlabel('index')
plt.ylabel('Euclidean')
plt.show()






# Aggregate dataset by airline (mean passenger count per airline)
airline_df = df.groupby("Operating Airline")["Passenger Count"].mean().reset_index()

# Linkage on aggregated data
Z = linkage(airline_df[["Passenger Count"]], method="complete")

# Plot dendrogram
plt.figure(figsize=(12,6))
dendrogram(Z, labels=airline_df["Operating Airline"].values, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (Airline Level)")
plt.xlabel("Airlines")
plt.ylabel("Euclidean Distance")
plt.show()


# apply clustering 

hc = AgglomerativeClustering(
    n_clusters=3,
    metric = 'euclidean',
    linkage = 'complete'
    
    
    )

clusters = hc.fit_predict(df_encoded)
clusters

df_encoded["cluster"] = clusters 

df_encoded["cluster"].value_counts()

df_encoded

df.head()
df["clusters"] = clusters
df["clusters"].value_counts()


# So i now i would like draw some insights from clustered 

cluster_summary = df.groupby(['clusters'])['Passenger Count'].sum()

cluster_summary


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(df_encoded , clusters)
silhouette
dbi = davies_bouldin_score(df_encoded , clusters)
dbi
chs = calinski_harabasz_score(df_encoded , clusters )
chs



from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV

# Custom scorer for clustering
def custom_scorer(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    if len(set(labels)) == 1:  # avoid trivial single-cluster case
        return -1
    return silhouette_score(X, labels)

# Parameter grid
param_grids = {
    'n_clusters' : [2,3,4,5],
    'metric' : ['euclidean','manhattan','cosine'],
    'linkage' : ['single','complete','average','ward']
}

# Agglomerative clustering base model
agg_clustering = AgglomerativeClustering()

# GridSearchCV
gridsearch = GridSearchCV(
    estimator=agg_clustering,
    param_grid=param_grids,
    scoring=custom_scorer,
    cv=5
)

gridsearch.fit(df_encoded)

print("Best Parameters:", gridsearch.best_params_)
print("Best Silhouette Score:", gridsearch.best_score_)

#hc_1 = gridsearch.fit(df_encoded)
#print("best paramaters :",hc_1.best_params_)
#print("best silhouette score is : ", hc_1.best_score_)

results = []

for n_clusters in [2,3,4,5]:
    for metric in ['euclidean','manhattan','cosine'] :
        for linkage in ['single','complete','average','ward']:
            model = AgglomerativeClustering(
                n_clusters = n_clusters , 
                metric = metric , 
                linkage = linkage
                )
            labels = model.fit_predict(df_encoded)
            if len(set(labels)) > 1:
                sil = silhouette_score(df_encoded , labels)
                dbi = davies_bouldin_score(df_encoded, labels)
                chs = calinski_harabasz_score(df_encoded, labels)
                results.append([n_clusters,metric,linkage,sil,dbi,chs])
                
results_df = pd.DataFrame(results , columns = ["Clusters","Affinity","Linkage","Silhouette","DBI","CHS"])            
print(results_df)
                   

