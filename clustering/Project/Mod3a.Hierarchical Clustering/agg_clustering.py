# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 20:56:11 2025
@author: Arunsaikoya
"""

# Import required library
import pandas as pd 

# Load the dataset into a DataFrame
df = pd.read_csv(r"C:\Users\Arunsaikoya\Desktop\ml practice\clustering\Project\Data Set (2)\Data Set (5)\AirTraffic_Passenger_Statistics.csv")
df.head()
df.columns   # Show first few rows and column names

# Import SQLAlchemy for saving data to MySQL
from sqlalchemy import create_engine , text
from urllib.parse import quote 

# Move dataset into local MySQL database
user , pw , db = "root" , quote("Akmp@235") , "univ_db"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost:3306/{db}")
df.to_sql('airport_tbl',
          con = engine , 
          if_exists = 'replace',
          chunksize = 1000 ,
          index = False)

# -------------------- Data Preprocessing Stage --------------------

# D-Tale (interactive EDA tool) for quick exploration
import dtale 
d = dtale.show(df , host = 'localhost' , port = 8000)
d.open_browser()

# Check for missing values
null_values = df.isna().sum()
print("null values of the Airtraffic passenger dataset :", null_values)

# Drop rows where "Operating Airline IATA Code" is missing
df = df.dropna(subset = ["Operating Airline IATA Code"])  

# Convert Activity Period (YYYYMM) to datetime
df["Activity Period"] = pd.to_datetime(df["Activity Period"] , format = "%Y%m")
df.dtypes 
df.head()

# Drop duplicate rows
df = df.drop_duplicates()

# Check number of duplicates remaining
checking_duplicates = df.duplicated().sum()
print(checking_duplicates)

# Drop redundant column before encoding
df = df.drop(columns = ["Operating Airline IATA Code"])

# One-Hot Encoding for categorical columns
from sklearn.preprocessing import OneHotEncoder
columns = ["Operating Airline","GEO Region","Terminal","Boarding Area","Month"]
df_encoded = pd.get_dummies(df,columns = columns , drop_first = True , dtype = int )
df_encoded.shape

# -------------------- Feature Scaling --------------------

from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()

# Scale Passenger Count and Year using MinMaxScaler
df_encoded[["Passenger Count", "Year"]] = min_max.fit_transform(df_encoded[["Passenger Count", "Year"]])
df_encoded["Passenger Count"].head()
df_encoded[["Passenger Count","Year"]].describe()
df_encoded[["Passenger Count"]].info()

# -------------------- Outlier Treatment --------------------

# Detect outliers using IQR method
Q1 = df_encoded["Passenger Count"].quantile(0.25)
Q3 = df_encoded["Passenger Count"].quantile(0.75)
IQR = Q3-Q1 

lower_bound = Q1-IQR*1.5
upper_bound = Q3+IQR*1.5

# Apply Winsorization (cap values beyond bounds)
import numpy as np
df_encoded["Passenger Count"] = np.where(df_encoded["Passenger Count"] > upper_bound, upper_bound,
                                         np.where(df_encoded["Passenger Count"] < lower_bound, 
                                                  lower_bound, 
                                                  df_encoded["Passenger Count"]))
df_encoded.shape

# Visualize outliers after treatment
import seaborn as sns 
sns.boxplot(df_encoded["Passenger Count"])

# Re-scale Passenger Count and Year after capping
df_encoded[["Passenger Count", "Year"]] = min_max.fit_transform(df_encoded[["Passenger Count", "Year"]])
df_encoded[["Passenger Count","Year"]].describe()

# Save processed data back to SQL
df_encoded.to_sql(
    "processeddata_airport",
    con = engine , 
    if_exists = 'replace',
    chunksize = 10000 , 
    index = False
    )

# Convert Activity Period into int64 (for compatibility in SQL)
df_encoded['Activity Period'] = df_encoded['Activity Period'].astype('int64')

# -------------------- Dendrogram Visualization --------------------

from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Take a sample of 500 rows for dendrogram (too many rows otherwise)
sample_df = df_encoded.sample(500 , random_state = 42)
Z = linkage(sample_df , method = 'complete')

# Plot truncated dendrogram
plt.figure(1,figsize = (16,8))
dendrogram(Z , truncate_mode="level",p=5)
plt.title("Hierarchical Clustering Dendrogram (Sample)")
plt.xlabel('index')
plt.ylabel('Euclidean Distance')
plt.show()

# -------------------- Airline-Level Aggregation --------------------

# Aggregate by airline (average passenger count)
airline_df = df.groupby("Operating Airline")["Passenger Count"].mean().reset_index()

# Perform hierarchical clustering on airline-level data
Z = linkage(airline_df[["Passenger Count"]], method="complete")

# Plot dendrogram at airline level
plt.figure(figsize=(12,6))
dendrogram(Z, labels=airline_df["Operating Airline"].values, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (Airline Level)")
plt.xlabel("Airlines")
plt.ylabel("Euclidean Distance")
plt.show()

# -------------------- Agglomerative Clustering --------------------

hc = AgglomerativeClustering(
    n_clusters=3,
    metric = 'euclidean',
    linkage = 'complete'
)

clusters = hc.fit_predict(df_encoded)
df_encoded["cluster"] = clusters 
df["clusters"] = clusters

# Cluster distribution
df_encoded["cluster"].value_counts()
df["clusters"].value_counts()

# -------------------- Cluster Insights --------------------

# Total passenger count per cluster
cluster_summary = df.groupby(['clusters'])['Passenger Count'].sum()
print(cluster_summary)

# -------------------- Cluster Evaluation --------------------

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(df_encoded , clusters)
dbi = davies_bouldin_score(df_encoded , clusters)
chs = calinski_harabasz_score(df_encoded , clusters )

print("Silhouette:", silhouette)
print("Davies-Bouldin:", dbi)
print("Calinski-Harabasz:", chs)

# -------------------- Grid Search for Hyperparameter Tuning --------------------

from sklearn.model_selection import GridSearchCV

# Custom scorer using silhouette score
def custom_scorer(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    if len(set(labels)) == 1:  
        return -1
    return silhouette_score(X, labels)

# Parameter grid
param_grids = {
    'n_clusters' : [2,3,4,5],
    'metric' : ['euclidean','manhattan','cosine'],
    'linkage' : ['single','complete','average','ward']
}

agg_clustering = AgglomerativeClustering()

# GridSearchCV with 5-fold CV
gridsearch = GridSearchCV(
    estimator=agg_clustering,
    param_grid=param_grids,
    scoring=custom_scorer,
    cv=5
)
gridsearch.fit(df_encoded)

print("Best Parameters:", gridsearch.best_params_)
print("Best Silhouette Score:", gridsearch.best_score_)

# -------------------- Manual Parameter Search --------------------

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

# Save results in DataFrame
results_df = pd.DataFrame(results , columns = ["Clusters","Affinity","Linkage","Silhouette","DBI","CHS"])            
print(results_df.sort_values("Silhouette",ascending=False).head())

# Save results to SQL
results_df.to_sql(
    'result_for_best_model',
    con = engine , 
    if_exists = 'replace',
    chunksize = 1000 , 
    index = False
)

# -------------------- Final Model with Best Params --------------------

model_2 = AgglomerativeClustering(
    n_clusters = 2 , 
    metric='euclidean',
    linkage='ward'
)

labels = model_2.fit_predict(df_encoded)
df["cluster_labels"] = labels

# Merge clusters into main df
df_conact = pd.concat([df,df["cluster_labels"]],axis = 1)
df_conact.head()

# Cluster summary by Passenger Count
df_summary = df.groupby(["cluster_labels"])["Passenger Count"].sum()
print(df_summary)

# Evaluate final model
model_2_silhouette = silhouette_score(df_encoded, df["cluster_labels"])
print("Final Silhouette Score:", model_2_silhouette)
