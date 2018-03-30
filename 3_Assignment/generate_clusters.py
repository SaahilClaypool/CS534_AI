import random
import csv

def make_cluster(x=0,y=0,vx=1, vy=1, n=10):
    """
    return n points normally distributed with given x and y center and variance
    """
    points = []
    for i in range(n):

        x = random.normalvariate(x, vx)
        y = random.normalvariate(y, vy)
        points.append((x,y))
    
    return points

def make_clusters(clusters=3, minx=0, maxx=10, minv=.1, maxv=.2, minPoints=15, maxPoints=30):
    points = []
    for i in range(clusters):
        x = random.random() * (maxx - minx) + minx
        y = random.random() * (maxx - minx) + minx
        vx = random.random() * (maxv - minv) + minv
        n = random.randint(minPoints, maxPoints)

        clust = make_cluster(x,y,vx,vx, n)
        points += clust
    
    return points


def write_clusters(filename="custom_sample.csv", number=3):
    with open(filename,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',', lineterminator='\n')
        csvWriter.writerows(make_clusters(number) )


def write_given_clusters(filename="custom_sample.csv", clusters=[]):
    with open(filename,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',', lineterminator='\n')
        csvWriter.writerows(clusters)

write_given_clusters("sample3.csv", make_clusters(clusters=3, minx = 0, maxx = 10, minv = .1, maxv = .2, minPoints = 100, maxPoints = 120))