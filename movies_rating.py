import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Importing Libraries for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_regression

# For feature engineering and model evaluation
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score

# For seasonal decomposition and plots
from matplotlib import dates as mdates
import matplotlib.ticker as ticker

# Other useful libraries
import warnings
warnings.filterwarnings('ignore')
movie = pd.read_csv('/kaggle/input/imdb-india-movies/IMDb Movies India.csv', encoding='ISO-8859-1')
movie.head()


movie.shape


movie.info

movie.describe(include='all')

movie.isnull().sum()

movie.drop_duplicates(subset='Name',inplace=True)
movie.duplicated().value_counts()

movie = movie.drop(movie.index[0]).reset_index(drop=True)
movie.head()



movie.isnull().sum().sort_values(ascending=False)/len(movie)


print("Number of rows:", movie.shape[0])
print("Number of columns:", movie.shape[1])


sns.heatmap(movie.isnull())


plt.title("Missing Values IMDb Indian Movie Dataset",
          fontsize=14,
          fontweight='bold')

# Title for x and y-axis labels with formatting
plt.xlabel("Number of Columns",
           fontweight='bold')
plt.ylabel("Number of Rows",
           fontweight='bold')
plt.show()

movie.drop_duplicates(inplace=True)
movie.shape

movie.head()


movie.isnull().sum()

missing_count = movie.isnull().sum().sort_values(ascending=False)
missing_percent = (round(movie.isnull().sum()/movie.isnull().count(), 4)*100).sort_values(ascending=False)
missing_data = pd.concat([missing_count, missing_percent],
                       axis=1,
                       keys=['missing_count', 'missing_percent'])
missing_data

movie.dropna(subset=['Rating'], inplace=True)
(round(movie.isnull().sum()/movie.isnull().count(), 4)*100).sort_values(ascending=False)


movie.isnull().sum()

movie.dropna(subset=['Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Genre'], inplace=True)

(round(movie.isnull().sum()/movie.isnull().count(), 4)*100).sort_values(ascending=False)

movie['Duration'] = movie['Duration'].astype(str)
movie['Duration'] = pd.to_numeric(movie['Duration'].str.replace(' min', ''), errors='coerce')
movie['Duration'].fillna(movie['Duration'].mean(), 
                         inplace=True)

movie.isnull().sum()


movie['Year'] = movie['Year'].apply(lambda x: x.split(')')[0])

year_lst = []
for val in movie['Year']:
    if len(val.split('(')) == 1:
        year_lst.append(val.split('(')[0])
    elif len(val.split('(')) > 1:
        year_lst.append(val.split('(')[1])
movie['Year'] = year_lst
# Check the data type of the 'Votes' column
print(movie['Votes'].dtype)

# If it's not already a string, convert it to string with commas
movie['Votes'] = movie['Votes'].astype(str)

# Replace commas and convert to int
movie['Votes'] = movie['Votes'].str.replace(',', '').astype(int)

# Check the data type after conversion
print(movie['Votes'].dtype)

movie.info()


movie['Year'].unique()


movie[['Rating', 'Duration', 'Votes']].describe()

top5_rating = movie[['Year', 'Rating']].sort_values(by = 'Rating',
                                                    ascending = True).head()
bars = top5_rating.plot(kind = 'bar',
                        x = 'Year',
                        y = 'Rating',
                        color = 'blue',
                        legend = None,
                        figsize = (8,7))
plt.xlabel('Year',
           fontsize = 14,
           fontweight = 'bold')
plt.ylabel('Rating',
           fontsize = 15,
           fontweight = 'bold')
plt.title('Top 5 Ratings from 2008-2020',
          fontsize = 18,
          fontweight = 'bold')
plt.xticks(fontweight = 'bold',
           rotation = 0)
plt.yticks(np.arange(0, 13, 2), fontweight='bold')

#Labelling Plot
for bar in bars.patches:
    plt.annotate(format(bar.get_height(), '.1f'),
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha='center',
                 va='center',
                 size=15,
                 xytext=(0, 8),
                 textcoords='offset points')

plt.show()

movie.drop('Duration', axis=1, inplace=True)
movie[movie['Votes']>10000]

movie.shape
movie_update = movie.drop(['Name'], axis=1)
movie_update.info()

X = movie_update.drop('Rating', axis=1)
Y = movie_update['Rating']

X.head()

actor1_encoding_map = movie_update.groupby('Actor 1').agg({'Rating': 'mean'}).to_dict()
actor2_encoding_map = movie_update.groupby('Actor 2').agg({'Rating': 'mean'}).to_dict()
actor3_encoding_map = movie_update.groupby('Actor 3').agg({'Rating': 'mean'}).to_dict()
director_encoding_map = movie_update.groupby('Director').agg({'Rating': 'mean'}).to_dict()
genre_encoding_map = movie_update.groupby('Genre').agg({'Rating': 'mean'}).to_dict()
movie_update['actor1_encoded'] = round(movie_update['Actor 1'].map(actor1_encoding_map['Rating']),1)
movie_update['actor2_encoded'] = round(movie_update['Actor 2'].map(actor2_encoding_map['Rating']),1)
movie_update['actor3_encoded'] = round(movie_update['Actor 3'].map(actor3_encoding_map['Rating']),1)
movie_update['director_encoded'] = round(movie_update['Director'].map(director_encoding_map['Rating']),1)
movie_update['genre_encoded'] = round(movie_update['Genre'].map(genre_encoding_map['Rating']),1)
movie_update.drop(['Actor 1', 'Actor 2', 'Actor 3', 'Director', 'Genre'], axis=1, inplace=True)
movie_update.head()


X= movie_update[['Year', 'Votes', 'genre_encoded', 'director_encoded', 'actor1_encoded', 'actor2_encoded', 'actor3_encoded']]
y= movie_update['Rating']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)




lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred= lr.predict(X_test)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred= rf.predict(X_test)
#For Linear Regression
print('Mean Squared error:', mean_squared_error(y_test, lr_pred))
print('Mean Absolute error:', mean_absolute_error(y_test, lr_pred))
print('R2 Score', r2_score(y_test, lr_pred))
print('This is the result by using Linear Regression')



print('Mean squared error: ',mean_squared_error(y_test, rf_pred))
print('Mean absolute error: ',mean_absolute_error(y_test, rf_pred))
print('R2 score: ',r2_score(y_test, rf_pred))
print('This is the result by using Random Forest Regressor')