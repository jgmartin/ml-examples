import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Perceptron, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv('./data/imdb-5000.csv')
movies.head()

### Prep the data ###
# Grab all of the numerical columns
movie_num = movies.select_dtypes(exclude=['object'])

# Remove nulls
movie_num = movie_num.fillna(value=0, axis=1)

# Scale the data with StandardScaler
# Estimators generally expect features to be centered around zero
X = movie_num.values
X_std = StandardScaler().fit_transform(X)





## Plot hexbin ###
# Density plot to help visualize the data

movies.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))
movies.plot(y= 'imdb_score', x ='movie_facebook_likes',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb Score and total Facebook likes',figsize=(12,8))
movies.plot(y= 'imdb_score', x ='cast_total_facebook_likes',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb Score and cast total Facebook likes',figsize=(12,8))





# ### Pearson Correlation heat map ###
# ### Used to visually signify the strength of correlations between columns

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))
plt.title('Pearson Correlation of Movie Features')
# Draw the heatmap using seaborn
labels = ['score', 'gross', 'total', 'cast', 'critic_rev', 'dur', 'user_votes', 'faces', 'user_rev', 'budget']
sns.heatmap(movie_num[['imdb_score',
                       'gross',
                       'movie_facebook_likes',
                       'cast_total_facebook_likes',
                       'num_critic_for_reviews',
                       'duration',
                       'num_voted_users',
                       'facenumber_in_poster',
                       'num_user_for_reviews',
                       'budget'
                       ]].astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True, xticklabels=labels, yticklabels=labels)





### Explained variance ###
### See: http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
### Helps to determine how many features are necessary to describe the variance in the data

# Calculating eigenvectors and eigenvalues of the covariance matrix
# For this purpose, eigenvalues explain the variance along a new feature axis defined by the eigenvectors
# It will let us know how many features we want to project down to in the next step
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

# Plot explained variances
plt.figure(figsize=(10, 5))
plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')




### Principal Component Analysis ###
### Dimensionality reduction via projection of the features to a lower dimension
### Purpose is to determine if there are any distinct clusters already present
### Clustering at this stage indicates that data can be linearly separated into groups for future use as features
pca = PCA(n_components=10)
x_10d = pca.fit_transform(X_std)
plt.figure(figsize = (9,7))
plt.scatter(x_10d[:,0],x_10d[:,1], c='goldenrod',alpha=0.5)
plt.ylim(-10,30)




### Visualization with K-Means clustering
## Set KMeans clustering for 3 clusters (a guess)
# kmeans = KMeans(n_clusters=3)
# # Compute cluster centers and predict cluster indices
# X_clustered = kmeans.fit_predict(x_10d)
# # Define our own color map
# label_colors = [
#     (1, 0, 0),
#     (0, 1, 0),
#     (0, 0, 1)
# ]
# # Plot the scatter digram
# plt.figure(figsize = (7,7))
# # Only plotting the first two PCA projections (of 10)
# plt.scatter(x_10d[:,0],x_10d[:,2], c=label_colors, cmap=plt.cm.RdYlGn, alpha=0.5)




# ### Automatically plot all of the other features in the frame (PCA projections)
# # Create a temp dataframe from our PCA projection data "x_10d"
# df = pd.DataFrame(x_10d)
# # only the first 5.. this gets expensive quickly
# df = df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
# df['X_cluster'] = X_clustered
# # Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data
# sns.pairplot(df, hue='X_cluster', palette='Dark2', size=2)



## Show the plots ###
# plt.show()





### Make a useful prediction!

# We will predict imdb_score
# Using a Lasso linear model.  It's a form of regression that tends to produce solutions with few variables.

# some_new_data = pd.DataFrame({
# 'num_critic_for_reviews' : [120, 1056],
# 'duration' : [120, 140],
# 'director_facebook_likes' : [1500, 15000],
# 'actor3_facebook_likes' : [30, 3000],
# 'actor1_facebook_likes' : [13000, 13000000],
# 'gross' : [12000000, 246000000],
# 'num_voted_users' : [1400, 28000],
# 'cast_total_facebook_likes' : [2450, 12000000],
# 'facenumber_in_poster' : [1, 3],
# 'num_user_for_reviews' : [213, 10423],
# 'budget' : [8000000, 180000000],
# 'title_year' : [2015, 2017],
# 'actor_2_facebook_likes' : [317, 43500],
# 'aspect_ratio' : [2.35, 2.35],
# 'movie_facebook_likes' : [9000, 22000000]
# })

### estimators

# Simply running the algorithm against all of the data as-is

# ridge = Ridge()
#
# msk = np.random.rand(len(df)) < 0.8
# train = movie_num[msk]
# test = movie_num[~msk]
#
# y_train = train.loc[: , 'imdb_score']
# y_test = test.loc[: , 'imdb_score']
# x_train = train.drop('imdb_score', axis=1)
# x_test = test.drop('imdb_score', axis=1)
#
#
# result = ridge.fit(x_train, y_train).predict(x_test)
# print(result)
# error = r2_score(y_test, result)
# print(error)


### Putting it all into a pipeline

# Running the algorithm with dimensionally reduced data

msk = np.random.rand(len(movie_num)) < 0.8
train = movie_num[msk]
test = movie_num[~msk]

prediction = 'gross'

y_train = train.loc[: , prediction]
y_test = test.loc[: , prediction]
x_train = train.drop(prediction, axis=1)
x_test = test.drop(prediction, axis=1)

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)

estimator = ElasticNet()
pipeline = Pipeline([("features", pca), ("estimator", estimator)])
result = pipeline.fit(x_train, y_train).predict(x_test)
print(result)
error = mean_absolute_error(y_test, result)
print(error)


### Running it with hand-selected data

# # critic_rev, user_votes, user_rev
#
# msk = np.random.rand(len(movie_num)) < 0.8
# train = movie_num[msk]
# test = movie_num[~msk]
#
# prediction = 'gross'
# print(train.info)
# y_train = train.loc[: , prediction]
# y_test = test.loc[: , prediction]
# x_train = train[['num_critic_for_reviews', 'num_voted_users', 'num_user_for_reviews']]
# x_test = test[['num_critic_for_reviews', 'num_voted_users', 'num_user_for_reviews']]
#
# estimator = Ridge()
# result = estimator.fit(x_train, y_train).predict(x_test)
# print(result)
# error = r2_score(y_test, result)
# print(error)

# pipeline = Pipeline([("features", pca), ("estimator", estimator)])
# result = pipeline.fit(x_train, y_train).predict(x_test)
# print(result)
# error = r2_score(y_test, result)
# print(error)
