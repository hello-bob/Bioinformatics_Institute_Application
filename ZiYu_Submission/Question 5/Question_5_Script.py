"""Importing libraries"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools

"""Get current working directory"""
wd = os.getcwd() #need to set wd as the file right above the level of the individual questions


"""Defining function to create square grid (matrix) for inserting beads """
def create_grid(row, col):
    # np.zeros creates an array of zeroes formatted as strings
    matrix = np.zeros(row * col, "str").reshape(row, col)  # Zero np-array reshaped into specified dimensions

    # np.pad pads the square grid with zeroes on all 4 sides. Required so that the algorithm can "scan"
    # the square grid properly.
    matrix = np.pad(matrix, pad_width=1, mode='constant')

    # returns created matrix
    return matrix


"""Creation of list of coordinates for square grid which will be used to define where on the square grid
to put the beads in"""
row = 64
col = 64

#Returns all possible permutations of coordinates. From a list of values, select 2 values which are different.
#Also, coordinates will start from 1 for both row and col index because will use zero padding later
coordinates = itertools.permutations(range(1,row+1),2)
coordinates = list(coordinates) #convert to list

#Because coordinates were created using permutations without replacement, need to fill in the gaps
for i in range(1,row+1):
    coordinates.append((i,i)) #Will fill (1,1), (2,2), ....

#convert list to pandas because it's easier to sort
coordinates = pd.DataFrame(coordinates, columns = ["row","col"])

#sort by row, followed by column so algorithm later will scan in proper order (Left to right, top to bottom)
coordinates = coordinates.sort_values(['row', 'col'])

#convert back to numpy
coordinates = coordinates.to_numpy()

#coordinates #check


""" 
Main algorithm: A brief summary of what it should do is to scan the entire matrix (non-padded parts) from left 
to right, top to bottom. First coordinate selected is the (1,1) coordinate. A random bead will be selected from a 
"bag of beads" and will be put into the selected coordinate.

Based on that coordinate, beads on its left and top (i.e. the coordinates that have been scanned before), 
should not be the same of the chosen bead. If there is a bead of the same colour nearby, insert a different colour. 
Only when there are no other options, insert the only colour left. Penalty count will increase by 1 should this 
happen. 

After running several iterations, select the grid which resulted in the lowest penalty. 
"""

list_of_matrices = [] #list of matrices tested
penalty_list = [] #list of penalties tested

#iterates whole algorithm several times
for k in range(1000):
    """Initialising values"""
    #bag of beads for algorithm to pick beads from. This way of defining the beads will simulate the way beads
    #will be selected from in reality. I wanted to created an algorithm which will use the colour of higher numbers
    #as much as possible to prevent them from aggregating at the end (expensive penalty).
    bag_of_beads = list('R'*139 + 'B'*1451 + 'G'*977 + 'W'*1072 + 'Y'*457)
    matrix = create_grid(row,col) #using function to create padded matrix
    penalty = 0 #penalty counter to identify how many conflicting colours there are

    #for each coordinate in the square grid (except the padding), select a bead to put in.
    for i,j in coordinates: #will pull the row (i) and column (j) coordinates.
        #use of np.random.choice because imbalanced dataset used. This will dictate the colour of the bead to be
        #inserted into square grid. Because blue beads are of highest frequency in the bag of beads, will be
        #most likely to be chosen. Strive towards equilibrium of percentages of the different coloured marbles
        cur_bead = np.random.choice(bag_of_beads) #picks one bead from the list of beads

        left = matrix[i][j-1] #store values left of the selected coordinate
        up = matrix[i-1][j] #store values above of the selected coordinate

        #if any of the beads match the ones on its left or top, change bead.
        if (left == cur_bead) or (up == cur_bead):
            #this checks if there are any other colours to pick which is different from current bead.
            if (len([i for i in bag_of_beads if i != cur_bead]) > 0):
                #choose another bead from list of beads which is different from bead at the top of selected
                #coordinates. In a better algorithm, it would also exclude the bead colour on the left of
                #selected coordinates.
                cur_bead = np.random.choice([i for i in bag_of_beads if (i != up)])
                #insert bead into selected coordinate of square matrix
                matrix[i][j] = cur_bead
            else: #else if there are no other options, just have to use the current bead chosen initially
                matrix[i][j] = cur_bead
                penalty += 1 #because if there's a match without another bead to use, will definitely have neighbours
                #with same coloured beads.

        #else if current bead is different from left or top bead, insert current bead into selected coordinate
        else:
            matrix[i][j] = cur_bead

        #remove bead from bag of beads. Will adjust probability of the bead to be drawn from bag of beads
        bag_of_beads.remove(cur_bead)

    list_of_matrices.append(matrix) #append matrix to list of matrices
    penalty_list.append(penalty) #append penalty to list of penalties
    print(k) #check iterations happening


"""Converting list of matrices and penalties into a dataframe for sorting and choosing the one with the least
penalty"""
#zip function aligns 2 lists of equal lengths for easy conversion into dataframe
matrices_and_penalty = pd.DataFrame(zip(list_of_matrices, penalty_list), columns = ["matrices","penalty"])

#sorts values in combined dataframe according to penalty, from lowest to highest. inplace = to sort in place.
matrices_and_penalty.sort_values("penalty", inplace = True)
matrices_and_penalty.reset_index(inplace = True)

#extracting the matrix with the least penalty, which should be the first one of the sorted dataframe
best_matrix = matrices_and_penalty.loc[0,"matrices"]
print("Minimum penalty =", matrices_and_penalty.loc[0,"penalty"])


"""convert to pandas dataframe so i can remove the padding easily."""
best_matrix = pd.DataFrame(best_matrix)

#coordinates of non-padded square grid, converted into numpy array
output_matrix = np.array(best_matrix.iloc[1:row+1,1:col+1])

#check if padding was removed successfully
output_matrix


"""Creation of output file for submission"""
np.savetxt(r"Question 5/output question_5_2.txt", output_matrix, fmt = "%s")
