#!/usr/local/bin/python

import csv as csv
import numpy as np

"""
Reads the name of the file into a np array
as tuple with the header of the list


For ease of tinkering, the original header is:
Survived is a boolean

['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

"""

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

    #print men_only_stats
    #gets column of Survived from specific rows, 
    men_onboard = data[men_only_stats, 1].astype(np.float)
    women_onboard = data[women_only_stats, 1].astype(np.float)

    men_percentage_survived   = np.sum(men_onboard) / np.size(men_onboard)
    women_percentage_survived = np.sum(women_onboard) / np.size(women_onboard)

    #print "Proportion of women survivors: {}".format(women_percentage_survived)
    #print "Proportion of men survivors:   {}".format(men_percentage_survived)

    #for i in range(0, 10):
    #    print data[i]

    #gender_prediction()

if __name__== "__main__":
    main()
