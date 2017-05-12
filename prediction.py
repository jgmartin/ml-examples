import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

### We're given the task of predicting total charges resulting from inpatient procedures
### Data includes 12 features across 163k samples

### Load in the data
x = pd.read_csv('./data/inpatientCharges.csv')
x.head()

### We'll be predicting the average covered charges
predicted_value = ' Average Covered Charges '

# ### Transform all of the currency columns into floats
currency_columns = [
    ' Average Covered Charges ',
    ' Average Total Payments ',
    'Average Medicare Payments'
]
for column in currency_columns:
    x[[column]] = x[[column]].replace('[\$,]','',regex=True).astype(float)

y = x.loc[:, predicted_value]
x = x.drop(predicted_value, axis=1)

#
# encoder = OneHotEncoder()
# result = encoder.fit_transform(x[['Provider Id']])
#
# print(result.__class__.__name__)
# print(result.feature_indices)


### Set up a DataFrameMapper to perform all the necessary transformations per column
mapper = DataFrameMapper([
        ('Provider Name', FeatureHasher(input_type='string')),
        ('Provider City', FeatureHasher(input_type='string')),
        ('Provider State', FeatureHasher(n_features=50, input_type='string')),
        ('DRG Definition', FeatureHasher(input_type='string')),
        ('Hospital Referral Region Description', FeatureHasher(input_type='string')),
        ('Average Medicare Payments', StandardScaler()),
        (' Average Total Payments ', StandardScaler()),
        (' Total Discharges ', StandardScaler()),
        ('Provider Id', None),
        ('Provider Zip Code', None)
    ])


print('Tranforming...')
x = mapper.fit_transform(x)
print('Finished transforming')

print(x)

print("Splitting train/test sets...")
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2)
print("Finished splitting train/test sets")

# regressor = SGDRegressor() --> mae on the order of 1e16
regressor = Ridge(alpha=0.5) # --> mae ~21.5%
# regressor = SVR(kernel='linear')
# regressor = ElasticNet() # --> mae ~22.4%
# regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
# selector = SelectKBest()

# print('Selecting features...')
# x_train = selector.fit_transform(x_train, y_train)
# print(x_train)
# print('Finished selecting features')

print('Performing estimation...')
estimator = regressor.fit(x_train, y_train)
prediction = estimator.predict(x_test)
print('Finished performing estimation')

average = np.mean(y_test, axis=0)
print(average)
error = mean_absolute_error(y_test, prediction)
print(error)
percentage = (error / average) * 100
print(percentage)
