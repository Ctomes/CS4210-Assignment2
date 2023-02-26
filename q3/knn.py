#-------------------------------------------------------------------------
# AUTHOR: Tomes, Christopher
# FILENAME: knn.py
# SPECIFICATION: this program reads in a file named binary_pts.csv and fits it to a knn algorithm. It then outputs the LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv



db = []
classToNum = {'+': 1,'-': 2}

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

error_rate = 0
#loop your data to allow each instance to be your test set
for pt in db:

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = []
    Y= []
    for row in db:
        if row != pt:
            X.append(row[:-1])
            Y.append(row[-1])

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            X[i][j] = float(X[i][j])



    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here

    #Y = [row[-1] for row in db]

    for i, val in enumerate(Y):
        Y[i] = float(classToNum[val])

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = pt

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)


    #use your test sample in this iteration to make the class prediction. For instance:
       #--> add your Python code here
    test_pt = [[float(pt[0]),float(pt[1])]]
    true_class = classToNum[pt[2]]
    class_predicted = clf.predict(test_pt)[0]

    #--> add your Python code here

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if true_class - class_predicted != 0:
        #print('FAIL')
        error_rate +=1
    else:
        #print("PASS")
        error_rate +=0
#print the error rate
#--> add your Python code here
print('Error rate: ' +str(error_rate/len(db)))






