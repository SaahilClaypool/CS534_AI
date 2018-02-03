# The problem

For this problem, you will use hill climbing and genetic algorithms to determine the ideal location of industry, commerce, and residential sections a city.  You will input a map with the following symbols:

- X:  former toxic waste site.  Industrial zones within 2 tiles take a penalty of -10.  Commercial and residential zones within 2 tiles take a penalty of -20.  You cannot build directly on a toxic waste site.
- S:  scenic view.  Residential zones within 2 tiles gain a bonus of 10 points.  If you wish, you can build on a scenic site but it destroys the view.  
- 0...9:  how difficult it is to build on that square.  You will receive a penalty of that many points to put anything on that square.  

You will have to place industrial, residential, and commercial tiles on the terrain.  
Industrial tiles benefit from being near other industry.  For each industrial tile within 2, there is a bonus of 3 points.
Commercial sites benefit from being near residential tiles.  For each residential tile within 3 squares, there is a bonus of 5 points.  However, commercial sites do not like competition.  For each commercial site with 2 squares, there is a penalty of 5 points.
Residential sites do not like being near industrial sites.  For each industrial site within 3 squares there is a penalty of 5 points.  However, for each commercial site with 3 squares there is a bonus of 5 points.

Note that all distances uses the Manhattan distance approach.  So distance is computed in terms of moves left/right and up/down, not diagonal movement.
Approaches
You will use hill climbing with restarts and genetic algorithms for this assignment.  When your hill climbing restarts, it may not modify the map!  It can only change where it initially places the different zones to begin the hill climbing process.  

Your genetic algorithms will have to come up with a representation, selection and crossover methods, and must make use of culling, elitism, and mutation.
Program behavior
Your program will read in a file where the first 3 lines are the number of industrial, commercial, and residential locations (respectively).  The remainder of the file is a rectangular map of the terrain to place the town.  Your program should then run for approximately 10 seconds and output the following to a file:
The score for this map
At what time that score was first achieved.
The map, with the various industrial, commercial, and residential sites marked.  
Writeup
You should analyze your program’s behavior to answer the following questions:
Explain how your genetic algorithm works.  You must describe your selection, crossover, elitism, culling, and mutation approaches.  
Create a graph of program performance vs. time.  Run your program 10 times and plot how well hill climbing and genetic algorithms perform after 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9 and 10 seconds.  If you only had 0.25 seconds to make a decision, which technique would you use?  Does your answer change if you have 10 seconds? 
How do elitism and culling affect performance?  
How did you perform selection and crossover for your population?  Find some known method for doing selection, other than the one described in class.  Explain what you did.  

## What to hand in:
You should submit your assignment on Canvas.  
Submit one file for part 1, and one file for part 2.
Include instructions for how to run your code.  If the TA needs help from you or is confused, you will lose points.
The writeups from part 1 and 2.  

## Hints
There is a moderate amount of analysis and writeup for this assignment -- do not forget that component.
Your program should run for the specified length of time.  Note the person grading your program may have a different hardware setup, so do not have a fixed number of iterations before the program terminates.  Instead use calls to determine the amount of elapsed time.  We are not worried about perfect precision -- if your program terminates after 10.2 seconds, that is ok.  Anything in the range of 9.5 seconds to 10.5 seconds is fine.  
