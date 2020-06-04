"""Importing libraries"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""Get current working directory"""
wd = os.getcwd() #need to set wd as the file right above the level of the individual questions

"""Initialisation of known values"""
row = 9 #Number of rows in matrix
col = 9 #Number of columns in matrix

#Creates list of all possible movement i.e. spaces between all numbers between columns and rows.
directions = list("R"*(col-1) + "D"*(row-1))

"""
Because the matrix has repeating values within each row, I figured that it would make sense to just use an
algorithm on column values, and allow the algorithm to extract values for calculating the sum from movement. 
Also, only two movements,"Right" and "Down", are allowed. Hence, every "Right" movement on a 2D matrix would be 
the same as "staying" on a 1D array. Likewise, every "Down movement" would be the same as moving to the next 
larger integer on a 1D array. 


The idea behind my algorithm was to shuffle the values of possible movement between the matrix, and then feed 
that sequence of directions to the algorithm so it knows which values to extract and add it to the sum. 

"""

# defining what numbers we want to find
target_number_list = [65, 72, 90, 110]
# initialising of list to store the correct dirrections which will allow us to get the desired sums
answers_list = []

# while loop so that the algorithm will keep trying different permutations of the directions until the correct
# set of directions allows us to get the target numbers we want to find. More precisely, it will continue
# until there are no more numbers in the target_number_list, which means that all directions to
# all target numbers have been found
while len(target_number_list) != 0:
    # iterates over a shrinking list of target numbers. Shrinking because when a number is found, that number is
    # removed from the target_number_list
    for target_number in target_number_list:
        position_value = 1  # starting position value will always be 1
        total_sum = position_value  # initialising total_sum container
        np.random.shuffle(directions)  # shuffling the order of directions within the list

        # this for loop is for identifying the next value which the algorithm will add to the sum. Will iterate
        # through the list to ensure that the full number of steps necessary will be taken, and in order
        for move in directions:
            # if the movement is shown to be "R", the current position value to add to the sum is the same as the
            # previous position value
            if move == "R":
                position_value = position_value  # identifier of current position value
                total_sum += position_value  # track total sum across all moved positions
                print(total_sum)

            # if the movement is shown to be "D", the current position value will have increased by 1,
            # and the current position value value will be added to the sum
            else:
                position_value += 1  # identifier of current position value
                total_sum += position_value  # track total sum across all moved positions
        #        print(position_value)
        #        print(total_sum)

        # this is to check if the given list of directions gave a final sum which is present in the target_number_list
        # only if the number is present, will the number be removed.
        if total_sum in target_number_list:  # to create loop-breaker when the target number is achieved
            # .join will concatenate all values in iterable without spacing. This line is to format the answer nicely
            formatted_answer = str(total_sum) + " " + "".join(directions)
            # append to list of answers for easy submission later opn
            answers_list.append(formatted_answer)
            # remove the total sum which was identified to be within the target_number_list
            target_number_list.remove(total_sum)
#         print(total_sum)


print(answers_list)

# Creation of output file
answers_list = np.array(answers_list)
file = np.savetxt(r"Question 1/output_file_1a", answers_list, fmt = "%s")

