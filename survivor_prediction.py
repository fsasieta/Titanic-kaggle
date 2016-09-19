#!/usr/local/bin/python

import csv as csv
import numpy as numpy

"""
Reads the name of the file into a numpy array
as tuple with the header of the list
"""
def read_file_into_array(name_of_csv_file):
    csv_file_as_string = csv.reader(open(name_of_csv_file, 'r'))

    header = csv_file_as_string.next()

    data = []

    for row in csv_file_as_string:
        data.append(row)

    data = numpy.array(data)
    return header, data

def main():

    header, data = read_file_into_array("train.csv")

    print data



if __name__== "__main__":
    main()
