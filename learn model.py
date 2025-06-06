import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('coords.csv')

x = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

pipeline = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models ={}
for algo, pipeline in pipeline.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

from sklearn.metrics import accuracy_score
import pickle

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

with open('body_language.pkl','wb') as f:
    pickle.dump(fit_models['rf'], f)