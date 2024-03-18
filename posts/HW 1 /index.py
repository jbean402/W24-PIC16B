#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3 
import pandas as pd
import numpy as np 

import plotly.io as pio
pio.renderers.default="iframe"


# In[3]:


temps_df = pd.read_csv("temps.csv")
temps_df.shape


# In[4]:


temps_df.head()


# In[5]:


def prepare_df(df):
    df = df.set_index(keys=["ID", "Year"])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns = {"level_2"  : "Month" , 0 : "Temp"})
    df["Month"] = df["Month"].str[5:].astype(int)
    df["Temp"]  = df["Temp"] / 100
    return(df)


# In[6]:


# creating the connections 
conn = sqlite3.connect("climate-database.db") # temperature database


# In[7]:


temps_df = prepare_df(temps_df)
temps_df.head()


# In[8]:


temps_iter = pd.read_csv("temps.csv", chunksize = 100000)

for i, temps_df in enumerate(temps_iter): 
    df = prepare_df(temps_df)
    df.to_sql("temperatures", conn, if_exists="replace" if i == 0 else "append", index = False)


# In[9]:


stations = pd.read_csv("station-metadata.csv") 
stations.to_sql("stations", conn, if_exists = "replace", index = False)


# In[10]:


cursor = conn.cursor() 
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())


# In[11]:


# adding the countries table 
countries = pd.read_csv("countries.csv")
countries.to_sql("countries", conn, if_exists = "replace", index = False)


# In[12]:


countries.head()


# In[13]:


cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())


# In[14]:


cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")

for result in cursor.fetchall():
    print(result[0])


# In[15]:


# testing the query 
from climate_database import query_climate_database
import inspect
print(inspect.getsource(query_climate_database))


# In[16]:


india_df = query_climate_database(db_file = "climate_database.db",
                       country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)


# ## Part 2: Write a function called temperature_coefficient_plot(). This function should accept six explicit arguments, and an undetermined number of keyword arguments.
# 
#  db_file, country, year_begin, year_end, and month should be as in the previous part.
#     
# min_obs, the minimum required number of years of data for any given station. Only data for stations with at least min_obs years worth of data in the specified month should be plotted; the others should be filtered out. df.transform() plus filtering is a good way to achieve this task.
#     
# **kwargs, additional keyword arguments passed to px.scatter_mapbox(). These can be used to control the colormap used, the mapbox style, etc.
# 

# In[17]:


# more additional packages
import plotly.express as px
from sklearn.linear_model import LinearRegression
import datetime


# In[18]:


india_df.groupby(["NAME", "Month"])["Temp"].agg([np.mean, np.std, len])


# In[19]:


# finding the linear regression coefficient, taken from the lecture 
def coef(data_group):
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]
    
def temperature_coefficient_plot(db_file, country, year_begin, year_end, month, min_obs, **kwargs): 
    '''
    The output of this function should be an interactive geographic scatterplot, 
    constructed using Plotly Express, with a point for each station, such that the 
    color of the point reflects an estimate of the yearly change in temperature during 
    the specified month and time period at that station. 
    A reasonable way to do this is to compute the first coefficient of a linear regression model 
    at that station, as illustrated in the lecture where we used the .apply() method.
    '''
    
    # creating the dataframe
    df = query_climate_database(db_file, country, year_begin, year_end, month) 
    
    # Cleaning the dataframe 
    counts = df.groupby(["NAME", "Month"])["Year"].transform(len)
    df = df[counts >= min_obs]
    coefs = df.groupby(["NAME", "Month", "LATITUDE", "LONGITUDE"]).apply(coef) #find the estimated yearly change in temperature for each station
    coefs = coefs.round(3) # round data to 3 decimal places
    coefs = coefs.reset_index()
    coefs = coefs.rename(columns = {0 : "Estimated Yearly Change (C)"})
    
    title = "Estimates of Yearly Increase in Temperature in {a} for stations in {b}, years {c} - {d}"\
    .format(a=datetime.date(2021, month, 1).strftime('%B'), b=country, c=year_begin, d=year_end)
    fig = px.scatter_mapbox(coefs,
                            lat = "LATITUDE",
                            lon = "LONGITUDE",
                            hover_name = "NAME",
                            color = "Estimated Yearly Change (C)",
                            title = title,
                            **kwargs)
    return fig
    
    
    


# In[20]:


# testing the function out (given from the homework assignment) 

# assumes you have imported necessary packages
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot(db_file = "climate_database.db", country = "India", year_begin = 1980, year_end = 2020, month = 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()


# ## Part 3: Creating two more interesting figures
# Create at least one more SQL query function in climate_database.py and at least two more complex and interesting interactive data visualizations using the same data set. These plots must be of different types (e.g. line and bar, scatter and histogram, etc). The code to construct each visualization should be wrapped in functions, such that a user could create visualizations for different parts of the data by calling these functions with different arguments. At least one of these plots must involve multiple facets (i.e. multiple axes (in the sense of facets), each of which shows a subset of the data).
# 
# Alongside the plots, you should clearly state a question that the plot addresses, similar to the question that we posed in Part 3. The questions for your two additional plots should be meaningfully different from each other and from the Part 3 question. You will likely want to define different query functions for extracting data for these new visualizations.
# 
# It is not necessary to create geographic plots for this part. Scatterplots, histograms, and line plots (among other choices) are all appropriate. Please make sure that they are complex, engaging, professional, and targeted to the questions you posed. In other words, push yourself! Don’t hesitate to ask your peers or talk to me if you’re having trouble coming up with questions or identifying plots that might be suitable for addressing those questions.

# In[21]:


# Creating the query function 

cursor.execute("SELECT Name FROM countries") # trying to find the country names 
result = cursor.fetchall() 
result


# In[400]:


from climate_database import second_climate_database
print(inspect.getsource(second_climate_database))


# In[218]:


def second_climate_database(db_file, country1, country2, year_begin, year_end): 
    
    conn = sqlite3.connect(db_file) 
    
    query = f'''
            SELECT S.ID, S.name, C.name, T.year, T.month, T.temp
            FROM temperatures T 
            LEFT JOIN stations S on T.id = S.id
            LEFT JOIN countries C on SUBSTRING(T.id, 1, 2) = C.'FIPS 10-4' 
            
            WHERE T.year >= {year_begin} AND T.year <= {year_end} AND C.name == "{country1}"
            UNION 
            SELECT S.ID, S.name, C.name, T.year, T.month, T.temp
            FROM temperatures T 
            LEFT JOIN stations S on T.id = S.id
            LEFT JOIN countries C on SUBSTRING(T.id, 1, 2) = C.'FIPS 10-4' 
            
            WHERE T.year >= {year_begin} AND T.year <= {year_end} AND C.name == "{country2}"
            
            '''
    
    df = pd.read_sql_query(query, conn) 
    
    conn.close() 
    return df 


# In[270]:


df = second_climate_database(db_file = "climate_database.db", 
                           country1 = "Philippines", 
                           country2 = "Indonesia",
                           year_begin = 1970, 
                           year_end = 2020)
df


# ### Visualization 1: Comparing the highest and the minimum temperatures temerature in two different countries over the span of 70 years

# In[305]:


unique_items_counts = df['NAME'].value_counts()
print(unique_items_counts)


# In[306]:


df["rank"] = df.groupby(["NAME", "Month"])["Temp"].rank().astype(int)


# In[307]:


unique_items_counts = df['rank'].value_counts()
print(unique_items_counts)


# In[308]:


# data cleaning 
missing_values = df.isna().sum()
print("Number of missing values", missing_values)


# In[313]:


print("Maximum Temp per Year")
max_data = df.groupby(['Name', 'NAME', 'Year'])['Temp'].max()

print()
print("Minimum Temp per Year")
min_data = df.groupby(['Name', 'NAME', 'Year'])['Temp'].min()

# turning them into dataframes 
min_data_df = min_data.to_frame() 
display(min_data_df)

max_data_df = max_data.to_frame() 
display(max_data_df)


# In[321]:


print("Min temp mean")
# min data 
min_mean = min_data_df.groupby(['Year', 'Name'])['Temp'].mean()
print(min_mean)

print()
print("Max temp mean")
# max data 
max_mean = max_data_df.groupby(['Year', 'Name'])['Temp'].mean()
print(max_mean)


# In[354]:


# converting series to frame 
min_df = min_mean.to_frame()
max_df = max_mean.to_frame()

min_df = min_df.reset_index()
min_df

max_df = max_df.reset_index() 
max_df




# In[362]:


fig1 = px.line(min_df, x="Year", y="Min_Temp", color="Name", title="Comparing the Min Temperatures of the Philippines and Indonesia")
fig2 = px.line(max_df, x = "Year", y = "Temp", color="Name", title="Comparing the Max Temperatures of the Philippines and Indonesia")

fig1.show()
fig2.show()


# ### Visualization 2: Comparing the Average Temperatures of two different airports in the Philippines 

# In[393]:


df = second_climate_database(db_file = "climate_database.db", 
                           country1 = "Philippines", 
                           country2 = "United States",
                           year_begin = 2000, 
                           year_end = 2010)
df


# In[394]:


# filtering the united states data out since we only want to focus on the philippines
df_filtered = df[df['Name'] != 'United States']
df_filtered

# checking to see if there are any NAN values and making sure that they are dropped 
df = df_filtered 
df = df.dropna()

# now looking at the unique values 
unique_values = df['NAME'].unique()
unique_values


# In[396]:


specific_airports = ['MANILA_INT_AIRPORT', 'MACTAN_CEBU_INTL'] 

df_filtered = df_filtered[df_filtered['NAME'].isin(specific_airports)]
print(df_filtered)



# In[399]:


fig = px.box(df_filtered, x='Year', y='Temp', color='NAME')
fig.show()


# In[82]:


# closing the connection 
conn.close()

