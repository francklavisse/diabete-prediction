import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('diabetes.csv')

print(df.head())

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

def data_cleaning():
    print(df.isnull().any())
    print(df.describe())

    print("Number of rows with 0 values for each variables")
    for col in df.columns:
        missing_rows = df.loc[df[col] == 0].shape[0]
        print(col + ": " + str(missing_rows))


data_cleaning()