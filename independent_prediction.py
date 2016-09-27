
#!/usr/local/bin/python

import csv as csv
import numpy as np
import pandas as pd
import pylab as P

from sklearn.ensemble import RandomForestClassifier

"""

For ease of tinkering, the original header is:
Survived is a boolean

    0               1           2       3       4       5       7       8       9       10          11      12
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

"""

def neural_network(train_file, test_file):

    #train_ids, train_data = random_forest_helper(train_file)
    #test_ids,  test_data  = random_forest_helper(test_file)

    #sanity check
    # this should really be an assert
    if np.isnan(test_data).any():
        print "contains a Nan"

    # Random forest -- Machine learning algorithm
    # Create random forest object
    forest = RandomForestClassifier(n_estimators = 100)

    #train the data
    forest = forest.fit(train_data[0::,1::] , train_data[0::,0])

    output = forest.predict(test_data).astype(int)

    prediction_file = open("randomForest_second.csv", 'wb')
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])
    
    #for i in output:
    #    prediction_file_object.writerow([ test_ids[i], output[i] ])
    #    print test_ids[i], output[i]

    prediction_file_object.writerows(zip(test_ids, output))

    
    prediction_file.close()


def random_forest_helper(csv_file):

    #instead of using a csv reader, we use a pandas dataframe
    df = pd.read_csv(csv_file, header=0)

    #for i in range(1,4):
    #   print i, len(df[ (df['Sex'] == "male") & (df['Pclass'] == i)])
        
    #df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )

    df['Gender'] = df['Sex'].map( { 'female': 0, 'male': 1}).astype(int)

    median_ages = np.zeros((2,3))

    #Change the NaN values for the median
    for i in range(0,2):
         for j in range(0,3):
              median_ages[i, j] = df[ (df['Gender'] == i) &\
                                      (df['Pclass'] == j+1) ]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    for i in range(0,2):
        for j in range(0,3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
                    'AgeFill'] = median_ages[i, j]

    #Fare manipulation goes here
    df['FareFill'] = df['Fare']
    df.loc[ df['FareFill'] == 0, 'FareFill' ] == .01


    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass

    #df['Age*Class'].dropna().hist(bins=16, range=(0,80), alpha = .5)
    #P.show()

    passenger_ids = df["PassengerId"].values

    df = df.drop( [ 'Fare', 'PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age' ] , axis = 1)

    print df.head

    return passenger_ids, df.values



"""
Starter code for a neural network prediction of titanic survivors.
I also plan on finishing the random forest predicition by taking fare's into account
"""
def random_forest_independent(train_file, test_file):

    train_ids, train_data = random_forest_helper(train_file)
    test_ids,  test_data  = random_forest_helper(test_file)

    #sanity check
    if np.isnan(test_data).any():
        print "Contains a Nan"

    # Random forest -- Machine learning algorithm
    # Create random forest object
    forest = RandomForestClassifier(n_estimators = 100)

    #train the data
    forest = forest.fit(train_data[0::,1::] , train_data[0::,0])

    output = forest.predict(test_data).astype(int)

    prediction_file = open("prediction_files/randomForest_with_fare.csv", 'wb')
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])
    
    #for i in output:
    #    prediction_file_object.writerow([ test_ids[i], output[i] ])
    #    print test_ids[i], output[i]

    prediction_file_object.writerows(zip(test_ids, output))

    
    prediction_file.close()


def main():

    random_forest_independent("train.csv", "test.csv")

    print "Nothing implmented yet"


if __name__== "__main__":
    main()
