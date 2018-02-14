# Assignment 1

Authors: 
- Sola Shirai
- Myles Spencer
- Saahil Claypool

https://docs.google.com/document/d/1bndQf-ySCUv1R2x8La5auF2iPS_mbqfMGvmfEbAAGsI/edit

Writeup:
https://docs.google.com/document/d/1quOORXXEwJMkznCSqbIppvPW8D_YsTYuZqWHLj_RMw0/edit

## Organization

The project is split into two subfolders for part 1 and part 2.

## Part 1

TODO

## Part 2

The code is split between three files, UrbanParser.py, 
HillClimb.py, and GeneticUrbanPlanner.py. 

The UrbanParser contains the shared code for parsing a textfile, printing, 
and scoring a urban plan. The two other files contain the algorithm implementation. 

### Running: 
```sh 
cd 2_part

## Run Hill Climb
## Requires python 3.6
## python HillClimb.py <filname> <seconds>
python HillClimb.py sample1.txt 10

## Run GeneticAlgorithm
## python GeneticUrbanPlanner.py <filname> <seconds>
python GeneticUrbanPlanner.py sample1.txt 10
```

### Output: 

The program outputs look like the following: 

```sh 
Starting Genetic Algorithm urban planner.
Running on file:  .\sample1.txt
Running for time:  5.0

Best Score:  14
Time Best Score achieved:  0
Final Generation:  1346
Characteristic String: XROONOOCONOOOI
Score: 14
Best board: Industrial: 1
Commercial: 1
Residential: 1
Height: 3 Width: 4
[<Toxic: r:0 c:0 cost:-1>, <Resident: r:0 c:1 cost: 1>, <Basic: r:0 c:2 cost: 2>, <Basic: r:0 c:3 cost: 4>]
[<Basic: r:1 c:0 cost: 3>, <Basic: r:1 c:1 cost: 4>, <Commericial: r:1 c:2 cost: 0>, <Basic: r:1 c:3 cost: 3>]
[<Basic: r:2 c:0 cost: 6>, <Basic: r:2 c:1 cost: 0>, <Basic: r:2 c:2 cost: 2>, <Industrial: r:2 c:3 cost: 3>]
```


The board peices are slightly verbose, but contains the same information as a string board, just with actual names. 
The score for each peice is the cost to build over that location (-1 for locations that are toxic). The genetic
alrogithm also prints the characteristic string, explained more in the writeup. 

