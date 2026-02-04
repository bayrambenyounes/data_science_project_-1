# data_science_project_-1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as ply

df=pd.read_csv("/content/context_master.csv")

df.head()



df.shap

df.info()

df.drop(columns="RANK",inplace=True)

df.isnull().sum()

missing_percentage=df.isnull().sum()/len(df)*100
columns_to_drop=missing_percentage[missing_percentage >50].index

df=df.drop(columns=columns_to_drop)

df.shape

df.duplicated().sum()

df.drop_duplicates().inplace=True

df.shape

df.head()

df

df=df.drop_duplicates()

df.shape

df.head()

df.dropna(subset=['REFEREE'], inplace=True)
df.shape



df.dropna()

import pandas as pd

df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
df.columns = df.columns.str.replace(r' ', '_', regex=True)
df.columns = df.columns.str.replace(r'__', '_', regex=True)
df.columns = df.columns.str.upper()
df.head()

df.head()

from sklearn.preprocessing import MinMaxScaler

column_scaled=["EXPERIENCE_YEARS","GAMES_OFFICIATED","TOTAL_POINTS_PER_GAME","CALLED_FOULS_PER_GAME"]

df_copy=df.copy()
scaler=MinMaxScaler()

df_copy[column_scaled]=scaler.fit_transform(df[column_scaled])

df_copy

df=df_copy

df.shape

df.head()


df['MASTER_BUILT_AT'] = pd.to_datetime(df['MASTER_BUILT_AT'])

print(df['MASTER_BUILT_AT'].dtype)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.hist(df["EXPERIENCE_YEARS"],bins=10,color='skyblue',edgecolor='black')
plt.title("bayram")
plt.xlabel("bayram")
plt.ylabel("bayram")
plt.grid(alpha=0.75,axis='y')

df["GENDER"]=df["GENDER"].replace("MALE","1")

df["GENDER"]=df["GENDER"].replace("FEMALE","0")

df.head()

df["SCRAPED_AT"]=pd.to_datetime(df["SCRAPED_AT"])

df

df.drop("CTX_TYPE",axis=1)

role_dummies = pd.get_dummies(df["ROLE"], prefix='ROLE', dtype=int)
df = pd.concat([df.drop("ROLE", axis=1), role_dummies], axis=1)
