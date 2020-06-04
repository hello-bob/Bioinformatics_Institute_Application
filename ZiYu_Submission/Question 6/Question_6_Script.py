"""Importing libraries"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


"""Get current working directory"""
wd = os.getcwd() #need to set wd as the file right above the level of the individual questions

"""Loading polygon coordinates and points coordinates"""
q6_wd = os.path.join(wd, "Question 6")

#Polygon coordinates
poly_coord = np.loadtxt(q6_wd + "/input_question_6_polygon")
print(poly_coord)

#Points coordinates
points_coord = np.loadtxt(q6_wd + "/input_question_6_points")
print(points_coord)


"""Visualising polygon"""
coord = poly_coord.tolist() #formating polygon coordinates into list-form
coord.append(coord[0]) #Append first coordinates to end of polygon coordinates to ensure closed-loop polygon

#* unpacks values within list, zip to return a list of tuples, and return x and y values of all tuples accordingly
x_values, y_values = zip(*coord)

plt.plot(x_values,y_values) #Plotting y values again x values
plt.title("Polygon")
plt.show()


"""Creating Shapely polygon object"""
#What shapely.geometry.Polygon does is that it creates a Polygon object which covers an area within a Cartesian
#plane. It is possible to then see if there are any specified points which overlap with this area. Uses point-set
#topology.
poly = Polygon(coord)

#creates list of output: the coordinates of a point, and whether the point lies inside or outside the polygon
output = []
for coords in points_coord:
    x, y = coords #extract x and y values of each coordinate point
    point = Point(x,y) #Convert x and y point into a shapely.geometry.Point object
    within_poly = point.within(poly) #check for any overlaps of objects. If there is an overlap, will return True
    if within_poly == True: #if true, then concatenate the coordinates, and indicate that it lies "Inside"
        evaluation = str(int(x)) + " " + str(int(y)) + " Inside"
        output.append(evaluation) #append coordinates and location to output list
    else: #if true, then concatenate the coordinates, and indicate that it lies "Outside"
        evaluation = str(int(x)) + " " + str(int(y)) + " Outside"
        output.append(evaluation) #append coordinates and location to output list

"""Creation of output file for submission"""
output = np.array(output) #convert to numpy array and save to text file for submission
np.savetxt(r"Question 6/output_question_6.txt", output, fmt="%s")


