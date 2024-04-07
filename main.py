
import numpy as np
from math import log


#Load the training file and the test file
test5 = np.loadtxt("test5.txt", dtype=int, delimiter=' ')
test10 = np.loadtxt("test10.txt", dtype=int, delimiter=' ')
test20 = np.loadtxt("test20.txt", dtype=int, delimiter=' ')
testing = None

#This function processes a dataset from a file, converting user-movie rating data into a structured format and saving it to a new file.
#Each row represents a user and each column represnts a 
def processData(filename):
    users = {}
    max_movie_id = 0  # Track the highest movie ID
    with open(filename, 'r') as file:
        for line in file:
            user, movie, rating = map(int, line.split())
            if user not in users:
                users[user] = {}
            users[user][movie - 1] = rating
            if movie > max_movie_id:
                max_movie_id = movie

    # Create a matrix from the users dictionary
    num_users = max(users.keys())
    trained = np.zeros((num_users, max_movie_id), dtype=int)
    for user, movies in users.items():
        for movie, rating in movies.items():
            trained[user - 1, movie] = rating

    return trained



    
#Initializes and populates a testing matrix with user-movie ratings from the three test datasets based on the provided test number.
def createTesting(test_num):
    global testing
    testing = np.zeros([100,1000], dtype=int)
    if test_num == 5:
        test_arr = test5
    elif test_num == 10:
        test_arr = test10
    else:
        test_arr = test20
    first_item = test_arr[0][0]
    for row in test_arr:
        testing[row[0]-first_item, row[1]-1] = row[2] 



#Apply the Inverse User Frequency to Trained Data
def cosine_similarity(vector1, vector2):
    mask = np.logical_and(vector1, vector2)
    a = vector1[mask]
    b = vector2[mask]
    if a.size <=1:
        return a.size
    else:
        return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))
#Function used to complete IUF to apply to Pearson method
def computeIUF():
    global trained
    m = 200
    for col in range(trained.shape[1]):
        nz = np.count_nonzero(trained[:,col])
        if nz!=0:
            trained[:,col] = trained[:,col]* (log(m)-log(nz))
#Function to calcualte Pearson similarity
def pearson_similarity(vector1, vector2):
    mask = np.logical_and(vector1, vector2)
    a = vector1[mask]
    b = vector2[mask]
    if a.size <= 1:
        return a.size
    else:
        avg1 = np.average(a)
        avg2 = np.average(b)
        a = a-avg1
        b = b-avg2
        denom = (np.linalg.norm(a)*np.linalg.norm(b))
        if denom==0:
            return 0
        weight =  round(np.sum(a*b)/denom, 10)
        # To run basic pearson leave this return commented
        return weight 
        '''
        To apply case modification uncomment this part
        
        p=2.5 #higher makes worse
        new_weight = weight * (abs(weight)**(p-1))
        return new_weight
        '''
       
#Function to calculate item based Collaborative Filtering
def itemBased(item1, item2):
    s1 = item1.size 
    s2 = item2.size
    if s1>s2:
        item1 = item1[:s2]
    else:
        item2 = item2[:s1]
    if s1 <=1:
        return s1
    else:
        avg1 = np.average(item1)
        avg2 = np.average(item2)
        item1 = item1-avg1
        item2 = item2-avg2
        denom = (np.linalg.norm(item1)*np.linalg.norm(item2))
        if denom==0:
            return 0
        return round(np.sum(item1*item2)/denom,10)

#Function to predict the Cosine Similarity, option to add a K value
def predictCosine(userID, movieID, k=35):
    neighbors = np.zeros([200,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for row in trained:
        if row[movieID]==0:
            continue
        else:
            sim = cosine_similarity(row,testing[userID]) 
            if sim>0:
                neighbors[filled][0] = sim
                neighbors[filled][1] = row[movieID]
                filled+=1
    sorted_knn = (neighbors[np.argsort(neighbors[:,0])])[::-1]
    if k==None:
        knns = sorted_knn[:filled-1] 
    else:
        knns = sorted_knn[:k] 
    numer = np.sum(knns[:,0]*knns[:,1])
    denom = np.sum(knns[:,0])  
    if denom>0:
        prediction = round(numer/denom)
        if prediction>5:
            return 5
        return prediction
    else:
        return 3
#Function to Predict Pearson Similarity, Option to add K value and case based 
def predictPearsonMethods(userID, movieID, k=None):
    neighbors = np.zeros([200,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for row in trained:
        if row[movieID]==0:
            continue
        else:
            sim = pearson_similarity(testing[userID], row) 
            if abs(sim)>1:
                print(sim)
            if sim!=0:
                neighbors[filled][0] = sim 
                neighbors[filled][1] = row[movieID] 
                filled+=1
    sorted_knn = (neighbors[np.argsort(abs(neighbors[:,0]))])[::-1]
    if k==None:
        knns = sorted_knn[:filled-1] 
    else:
        knns = sorted_knn[:k] 
    avg_u = np.average(knns[:,1])
    numer = np.sum(knns[:,0]*(knns[:,1]-avg_u))
    denom = np.sum(abs(knns[:,0])) 

    ra = np.sum(testing[userID])/np.count_nonzero(testing[userID])
    if denom!=0:
        prediction = round(ra+(numer/denom))
        if prediction>5:
            return 5
        if prediction<=0:
            return 1
        return prediction
    else:
        return 3
#Function to Predict Item Based Collaborative Filtering
def predictItemBasedCF(userID, movieID):
    neighbors = np.zeros([1000,2],dtype=np.float32)
    filled = 0
    userID = (userID-1)%100
    movieID = movieID-1
    for i in range(trained.shape[1]):
        col = trained[:,i]
        if col[userID]==0:
            continue
        else:
            sim = itemBased(col,testing[:,movieID]) 
            if sim>0:
                neighbors[filled][0] = sim
                neighbors[filled][1] = col[userID]
                filled+=1
    sorted_knn = (neighbors[np.argsort(neighbors[:,0])])[::-1]
    knns = sorted_knn[:filled-1] 
    numer = np.sum(knns[:,0]*knns[:,1])
    denom = np.sum(knns[:,0]) 
    if denom>0:
        prediction = round(numer/denom)
        if prediction>5:
            return 5
        return prediction
    else:
        return 3
#Fucntion to predict my custom algorithm
def predictCustom(userID, movieID, userWeight, itemWeight, popularityWeight, k=None):
    userBasedPrediction = predictPearsonMethods(userID, movieID, k)
    itemBasedPrediction = predictItemBasedCF(userID, movieID, k)
    
    popularityScore = np.count_nonzero(trained[:, movieID-1]) / np.max(np.count_nonzero(trained, axis=0))
    
    finalPrediction = (userWeight * userBasedPrediction + 
                      itemWeight * itemBasedPrediction + 
                      popularityWeight * popularityScore * 5) / (userWeight + itemWeight + popularityWeight)

    return max(1, min(round(finalPrediction), 5))
#Function to run Cosine Tests
def runCosine():
    global test5, test10, test20
    createTesting(5)
    test5 = test5[np.any(test5 == 0, axis=1)]
    for row in test5:
        row[2] = predictCosine(row[0],row[1])
    np.savetxt("result5.txt", test5,fmt='%d', delimiter=' ')
    createTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictCosine(row[0],row[1])
    np.savetxt("result10.txt", test10,fmt='%d', delimiter=' ')
    createTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictCosine(row[0],row[1])
    np.savetxt("result20.txt", test20,fmt='%d', delimiter=' ')

def runPerson():
    global test5, test10, test20
   #computeIUF()  Uncomment this in order to apply IUF to the pearson method before calculatomgh
    createTesting(5) 
    test5 = test5[np.any(test5 == 0, axis=1)]
    for row in test5:
        row[2] = predictPearsonMethods(row[0],row[1])
    print(test5)
    np.savetxt("result5pcm.txt", test5,fmt='%d', delimiter=' ')
    createTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictPearsonMethods(row[0],row[1])
    print(test10)
    np.savetxt("result10pcm.txt", test10,fmt='%d', delimiter=' ')
    createTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictPearsonMethods(row[0],row[1])
    print(test20)
    np.savetxt("result20pcm.txt", test20,fmt='%d', delimiter=' ')

def runItemBased():
    global test5, test10, test20
    createTesting(5)
    test5 = test5[np.any(test5 == 0, axis=1)]
    for row in test5:
        row[2] = predictItemBasedCF(row[0],row[1])
    print(test5)
    np.savetxt("result5i.txt", test5,fmt='%d', delimiter=' ')
    createTesting(10)
    test10 = test10[np.any(test10 == 0, axis=1)]
    for row in test10:
        row[2] = predictItemBasedCF(row[0],row[1])
    print(test10)
    np.savetxt("result10i.txt", test10,fmt='%d', delimiter=' ')
    createTesting(20)
    test20 = test20[np.any(test20 == 0, axis=1)]
    for row in test20:
        row[2] = predictItemBasedCF(row[0],row[1])
    print(test20)
    np.savetxt("result20i.txt", test20,fmt='%d', delimiter=' ')



def runCustom():
    global test5, test10, test20
    userWeight = 1
    itemWeight = 1
    popularityWeight = 0.5  
    createTesting(5)
    for row in test5:
        row[2] = predictCustom(row[0], row[1], userWeight, itemWeight, popularityWeight)
    print(test5)
    np.savetxt("resultcustom.txt", test5, fmt='%d', delimiter=' ')
    createTesting(10)
    for row in test10:
        row[2] = predictCustom(row[0], row[1], userWeight, itemWeight, popularityWeight)
    print(test10)
    np.savetxt("result10custom.txt", test10, fmt='%d', delimiter=' ')
    createTesting(20)
    for row in test20:
        row[2] = predictCustom(row[0], row[1], userWeight, itemWeight, popularityWeight)
    print(test20)
    np.savetxt("result20custom.txt", test20, fmt='%d', delimiter=' ')


def main():
    global trained
    print("Select the recommendation algorithm to run:")
    print("1: Basic Cosine User-Based Collaborative Filtering")
    print("2: Basic Pearson User-Based Collaborative Filtering")
    print("3: Cosine Item-Based Collaborative Filtering")
    print("4: Custom Hybrid Method")

    # Loading the data only once here
    trained = processData("train.txt")

    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        runCosine()
    elif choice == '2':
        runPerson()
    elif choice == '3':
        runItemBased()
    elif choice == '4':
        runCustom()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
