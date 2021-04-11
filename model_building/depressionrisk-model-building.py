import pandas as pd
survey_df = pd.read_csv('Final_DF.csv')

df = survey_df.copy()
target = 'RISKY_GROUP'
encode = ['BMI','DAYS_DRINK_12MONTHS']

#Separating X and Y
X = df.drop('RISKY_GROUP', axis=1)
Y = df['RISKY_GROUP']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X,Y)

# Saving the model
import pickle
pickle.dump(clf, open('depressionrisk_clf.pkl', 'wb'))