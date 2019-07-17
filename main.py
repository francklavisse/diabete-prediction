import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

df = pd.read_csv('diabetes.csv')

#print(df.head())

def display_hist():
    df.hist()
    plt.show()

def display_density_plots():
    plt.subplots(3, 3, figsize=(15,15))
    for idx, col in enumerate(df.columns):
        ax = plt.subplot(3, 3, idx + 1)
        ax.yaxis.set_ticklabels([])
        sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel=False,
            kde_kws={'linestyle':'-', 'color':'black', 'label':'No diabetes'}
        )
        sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel=False,
            kde_kws={'linestyle':'--', 'color':'black', 'label':'Diabetes'}
        )
        ax.set_title(col)
    plt.subplot(3,3,9).set_visible(False) # plot number 9 doesn't exist we only have 8 columns
    plt.show()

def print_missing_data():
    print("\nNumber of rows with 0 values for each variables")
    for col in df.columns:
        missing_rows = df.loc[df[col] == 0].shape[0]
        print(col + ": " + str(missing_rows))
    
def data_cleaning():
    print(df.isnull().any())
    print(df.describe())

    print_missing_data()
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols:
        df[col] = df[col].replace(0, np.nan)
    print_missing_data()

    for col in cols:
        df[col] = df[col].fillna(df[col].mean())

    print(df.describe())

def scale_data():
    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']
    return df_scaled    

data_cleaning()
df = scale_data()

X = df.loc[:, df.columns != 'Outcome'] # features 
y = df.loc[:, 'Outcome'] # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

#print(df.describe().loc[['mean', 'std', 'max']].round(2).abs())