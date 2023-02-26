#-------------------------------------------------------------------------
# AUTHOR: Tomes, Christopher
# FILENAME: decision_tree2.py
# SPECIFICATION: this program reads in 3 different training sets and builds decisions trees via sklearn. 
#              # The trees are then tested and averaged over 10 runs and the average accuracy is printed at end.

# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5/2 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#create a dictionary to map the potential categorical vals to integers.
featureToInt = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Hypermetrope': 2, 'Yes': 1, 'No': 2, 'Normal': 1, 'Reduced': 2}



for ds in dataSets:
    #append path to CSV file and replaced ds in open func with var string_path
    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

    #map the dbTraining vals to featureToInt vals
    for i in range(len(dbTraining)):
        for j in range(len(dbTraining[i])):
            dbTraining[i][j] = featureToInt[dbTraining[i][j]]

    #partition X        
    X = [row[:-1] for row in dbTraining]

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]

    Y = [row[-1] for row in dbTraining]

    #running sum of accuracy
    average_accuracy = 0
    test_runs = 10

    #loop your training and test tasks 10 times here
    for i in range (test_runs):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       dbTest = []
       
       with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)


       #transform categorical data
       for i in range(len(dbTest)):
            for j in range(len(dbTest[i])):
                dbTest[i][j] = featureToInt[dbTest[i][j]]
       
       #maintain a sum of incorrect predictions
       incorrect = 0

       for data in dbTest:
       
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label

           #Line 75 is where I pre-transformed the features.

           #test each sample
           testX = data[:-1]
           trueY = data[-1]
           class_predicted = clf.predict([testX])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.

           #if not the same increment our sum
           if trueY - class_predicted != 0:
                incorrect+=1
    #find the average of this model during the 10 runs (training and test set)
       average_accuracy+= (1 - incorrect/len(dbTest))/test_runs


    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2

    print('Final accuracy when training on ' + ds + ': {:.4f}'.format(average_accuracy))

