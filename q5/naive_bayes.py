#-------------------------------------------------------------------------
# AUTHOR: Tomes, Christopher
# FILENAME: naive_bayes.py
# SPECIFICATION: This program reads in file weather_training.csv  and fits the data. A test set is then run and instances that are predicted at a confidence level above .75 is printed to screen. 
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
mapping = {'Sunny': 1, 'Overcast': 2, 'Rain': 3, 'Hot': 1, 'Mild': 2, 'Cool': 3, 'High': 1, 'Normal': 2, 'Strong': 1, 'Weak': 2, 'Yes': 1, 'No':2}

# split on columns
feature_indices = [ 1, 2, 3, 4]
label_index = 5
X=[]

Y=[]
with open('weather_training.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Skip the header row
    header_row = next(csv_reader)


#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# extract the features and label for each row
    for row in csv_reader:
        # Extract the features (i.e., all but the last column)
        features = [mapping[row[i]] if row[i] in mapping else float(row[i]) for i in feature_indices]
        # Append the features to X
        X.append(features)

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

        Y.append(mapping[row[label_index]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
test_samples = []
confidence = .75
with open('weather_test.csv', 'r') as csv_file:
    # Create a CSV reader object
    test_reader = csv.reader(csv_file)

#printing the header os the solution
#--> add your Python code here

    header_row = next(test_reader)
    print(header_row)

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

    for row in test_reader:
        # Extract the features (i.e., all but the last column)
        new_row = row
        features = [mapping[row[i]] if row[i] in mapping else float(row[i]) for i in feature_indices]
        prediction = clf.predict_proba([features])[0]
        if prediction[0] >= confidence:
            new_row[-1] = 'Yes'
            print(str(new_row) + ' Confidence: ' + str(round(prediction[0],2)))
        elif prediction[1] >= confidence:
            new_row[-1] = 'No'
            print(str(new_row) + ' Confidence: ' + str(round(prediction[1],2)))

print('Done.')
