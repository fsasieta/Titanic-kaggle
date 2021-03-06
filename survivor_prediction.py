#!/usr/local/bin/python

import csv as csv
import numpy as np
import pandas as pd
import pylab as P

from sklearn.ensemble import RandomForestClassifier

"""
Reads the name of the file into a np array
as tuple with the header of the list

For ease of tinkering, the original header is:
Survived is a boolean

    0               1           2       3       4       5       7       8       9       10          11      12
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


Note: all of these were done for learning purposes and were 
typed directly from kaggle examples
visit kaggle.com for the tutorial

"""

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

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass

    #df['Age*Class'].dropna().hist(bins=16, range=(0,80), alpha = .5)
    #P.show()

    passenger_ids = df["PassengerId"].values

    df = df.drop( [ 'Fare', 'PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age' ] , axis = 1)

    return passenger_ids, df.values


#I dropped the "embarked" and "fare"  info for now.
# haven't decided how to I think it will affect the model.
def random_forest(train_file, test_file):

    train_ids, train_data = random_forest_helper(train_file)
    test_ids,  test_data  = random_forest_helper(test_file)

    #sanity check
    if np.isnan(test_data).any():
        print "contains a Nan"

    # Random forest -- Machine learning algorithm
    # Create random forest object
    forest = RandomForestClassifier(n_estimators = 100)

    #train the data
    forest = forest.fit(train_data[0::,1::] , train_data[0::,0])

    output = forest.predict(test_data).astype(int)

    prediction_file = open("randomForest_first.csv", 'wb')
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])
    
    #for i in output:
    #    prediction_file_object.writerow([ test_ids[i], output[i] ])
    #    print test_ids[i], output[i]

    prediction_file_object.writerows(zip(test_ids, output))

    
    prediction_file.close()

def gender_class_and_ticket_price(header, data):

    fare_ceiling = 40
    #any fare greater than 40 will equal 39 now
    data[ data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

    fare_bracket_size = 10
    number_of_price_brackets = fare_ceiling / fare_bracket_size
    number_of_classes = len(np.unique(data[0::2]))

    survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

    for passenger_class in xrange(number_of_classes):
        for price_bracket in xrange(number_of_price_brackets):

            #creates two array of booleans
            women_only_stats = data[ (data[0::,4] == "female") \
                            #select only women
                        &   (data[0::,2].astype(np.float) == passenger_class  + 1) \
                            #of a specific passenger class
                        &   (data[0::,9].astype(np.float) >= price_bracket * fare_bracket_size) \
                            #make sure theyre ticket range
                        &   (data[0::,9].astype(np.float) <  (price_bracket + 1) * fare_bracket_size),1]
                            #put in the second column

            men_only_stats = data[ (data[0::,4] == "female") \
                            #select only men
                        &   (data[0::,2].astype(np.float) == passenger_class  + 1) \
                            #of a specific passenger class
                        &   (data[0::,9].astype(np.float) >= price_bracket * fare_bracket_size) \
                            #make sure theyre ticket range
                        &   (data[0::,9].astype(np.float) <  (price_bracket + 1) * fare_bracket_size),1]
                            #put in the second column

            survival_table[0, passenger_class, price_bracket] = np.mean(women_only_stats.astype(np.float))
            survival_table[1, passenger_class, price_bracket] = np.mean(men_only_stats.astype(np.float))

            survival_table[ survival_table != survival_table] = 0

    survival_table[ survival_table <   0.5 ] = 0        
    survival_table[ survival_table >=  0.5 ] = 1   

    #print survival_table
    
    #have yet to think of a good way to modularize this code
    test_file = open("test.csv", 'rb')
    test_file_object = csv.reader(test_file)
    test_file_header = test_file_object.next()

    prediction_file = open("genderclassticketmodel.csv", 'wb')
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])


    for passenger in test_file_object:

        for j in xrange(number_of_price_brackets):

            try:
                passenger[8] = float(passenger[8])
            except:
                bin_fare = 3 - float(passenger[1])
                break

            if passenger[8] > fare_ceiling:
                bin_fare = number_of_price_brackets - 1
                break

            if passenger[8] >= j * fare_bracket_size\
                and passenger[8] < (j+1) * fare_bracket_size:
                bin_fare = j
                break

        if passenger[3] == 'female':
            prediction_file_object.writerow([passenger[0], 
                                            "%d" % int( survival_table[0, float(passenger[1]) - 1, bin_fare])])
        else:
            
            prediction_file_object.writerow([passenger[0], 
                                            "%d" % int( survival_table[1, float(passenger[1]) - 1, bin_fare])])

    #close the files for good
    test_file.close()
    prediction_file.close()

    



def gender_prediction():

    test_file = open("test.csv", 'rb')
    test_file_object = csv.reader(test_file)

    test_file_header = test_file_object.next()

    prediction_file = open("genderbasedmodel.csv", 'wb')
    prediction_file_object = csv.writer(prediction_file)

    prediction_file_object.writerow(["PassengerId", "Survived"])

    for person in test_file_object:
        if person[4] == "female":
            prediction_file_object.writerow([person[0], '1'])
        else:
            prediction_file_object.writerow([person[0], '0'])

    test_file.close()
    prediction_file.close()


def read_file_into_array(name_of_csv_file):
    csv_file_as_string = csv.reader(open(name_of_csv_file, 'rb'))

    header = csv_file_as_string.next()

    data = []

    for row in csv_file_as_string:
        data.append(row)

    data = np.array(data)
    return header, data

def main():

    header, data = read_file_into_array("train.csv")

    # example of selecting a column
    #print data[0::,4]

    #get proportion of survivors:
    # 'Survived' is a 1 or 0  so ok to add them
    number_passengers = np.size(data[0::,1].astype(np.float))
    number_survived = np.sum(data[0::,1].astype(np.float))
    # we are dviding floats, so this is ok.
    proportion_survived = number_survived / number_passengers

    #creates array of booleans
    women_only_stats = data[0::,4] == "female"
    men_only_stats = data[0::,4] == "male"

    #gets column of Survived from specific rows, 
    men_onboard = data[men_only_stats, 1].astype(np.float)
    women_onboard = data[women_only_stats, 1].astype(np.float)

    men_percentage_survived   = np.sum(men_onboard) / np.size(men_onboard)
    women_percentage_survived = np.sum(women_onboard) / np.size(women_onboard)

    #print "Proportion of women survivors: {}".format(women_percentage_survived)
    #print "Proportion of men survivors:   {}".format(men_percentage_survived)

    #first prediction model only uses gender
    #gender_prediction()

    #gender_class_and_ticket_price(header, data)

    random_forest("train.csv", "test.csv")

    print "Done"

if __name__== "__main__":
    main()
