import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from keras.models import Sequential
from keras.layers import Dense
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
    #print(df.describe().loc[['mean', 'std', 'max']].round(2).abs())
    return df_scaled    

data_cleaning()
df = scale_data()

X = df.loc[:, df.columns != 'Outcome'] # features 
y = df.loc[:, 'Outcome'] # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=8)) # Relu: f(x) = max(0,x), treat negative values as 0 or return x 
    model.add(Dense(16, activation='relu'))  # Add an hidden

    # only 1 layer because we want a binary input
    # sigmoid: f(x) = 1 / (1 - e^-x) 
    # squashes the output between 0 and 1, if sig(x) < 0.5 it will be 0, if not 1
    model.add(Dense(1, activation='sigmoid')) 
    return model

def train_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # binary_crossentropy because we have a binary classification problem
    model.fit(X_train, y_train, epochs=200)


model = build_model()
train_model(model)

scores = model.evaluate(X_train, y_train)
print("Training accuracy: %.2f%%\n" % (scores[1] * 100))

scores = model.evaluate(X_test, y_test)
print("Testing accuracy: %.2f%%\n" % (scores[1] * 100))

y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)

# ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap='Blues')
# ax.set_xlabel("Prediction")
# ax.set_ylabel("Actual")
# plt.show()

y_test_pred_probs = model.predict(X_test)
# FPR -> False Positive Rate = (false positive) / (true negative + false positive)
# TPR -> True Positive Rate = (true positive) / (true positive + false negative)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)

plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--',color='black')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()