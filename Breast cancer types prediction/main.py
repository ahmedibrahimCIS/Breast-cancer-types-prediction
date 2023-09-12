import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

#load data from csv file
df = pd.read_csv("breast-cancer.csv")

#drop missing values
df.dropna(axis=1)

#drop duplicates
df.drop_duplicates(subset=None , keep="first" , inplace=False , ignore_index=False )

#visualizing the correlation of the data
sns.heatmap(df.corr(), annot = True , annot_kws = {"fontsize":8})
plt.show()

#convert diagnosis data  M to 0 and B to 1
df['diagnosis'].replace(to_replace="M" , value = 0 , inplace=True)
df['diagnosis'].replace(to_replace="B" , value = 1 , inplace=True)



#split the data set into independent x and dependant y
X = df.iloc[:,2:31].values   # attributes
y = df.iloc[:,1].values   # target variable

# split data set int 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state =0)

#scale the data (feature scaling/ Data Normalization)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# create a function for the models
def models(X_train, y_train):

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pre = knn.predict(X_test)
    cm = confusion_matrix(y_pre,y_test)
    acc = accuracy_score(y_pre, y_test)

    #Decision Tree
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)
    cm1 = confusion_matrix(y_predict, y_test)
    accuracy = accuracy_score(y_predict, y_test)

    #Naive Bayes
    NB = GaussianNB()
    NB.fit(X_train,y_train)
    y_pred = NB.predict(X_test)
    cm2 = confusion_matrix(y_pred, y_test)
    ac = accuracy_score(y_pred, y_test)

    # print the models' evaluation metrics
    print('knn Accuracy :', acc)
    print('knn confusion matrix :', cm)

    print('Decision Tree Accuracy :', accuracy)
    print('Decision Tree confusion matrix :', cm1)

    print('Naive Bayes Accuracy :',ac)
    print('Naive Bayes confusion matrix :', cm2)

# getting all the models
model = models(X_train, y_train)
print (model)

