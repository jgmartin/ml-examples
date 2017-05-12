import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns

census = pd.read_csv('./data/adult_census.csv')
census.head()

# separate string a numerical columns
census_num = census.select_dtypes(exclude=['object']).fillna(value=0, axis=1)
census_str = census.select_dtypes(include=['object']).fillna(value='?', axis=1)

print(census_num.info(verbose=True))
print(census_str.info(verbose=True))

census = census_str # only care about string values now

### Split the data into training and test sets
msk = np.random.rand(len(census)) < 0.8
train = census[msk]
test = census[~msk]

prediction = 'income'

y_train = train.loc[: , prediction]
y_test = test.loc[: , prediction]
x_train = train.drop(prediction, axis=1)
x_test = test.drop(prediction, axis=1)

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)


### Set up a pipeline
# scaler = StandardScaler()
vect = CountVectorizer()
tfid = TfidfTransformer()
transformer = FunctionTransformer(lambda x: x.todense(), accept_sparse=True)
# pca = PCA(n_components=8)
bayes = GaussianNB()
svc = SVC()

pipe = Pipeline([('vect', vect), ('tfid', tfid), ('transformer', transformer), ('bayes', bayes)])
result = pipe.fit(x_train, y_train).predict(x_test)
print(result)
