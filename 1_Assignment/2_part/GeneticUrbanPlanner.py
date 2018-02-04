"""
Use genetic algorithm approach to output a solution for urban planning.
"""
import sys
import time
from typing import List
from UrbanParser import *

def main():
    max_population = 100
    elite_count = 10
    cull_count = 10
    mutation_probability = 0.01
    runtime_seconds = 1.0

    #for arg in sys.argv[1:]:
        #TODO actually take input file arument

    main_board = Board.read_from_file("./sampleInput.txt")
    pop = Population(max_population, elite_count, cull_count, mutation_probability)
    time_end = time.time() + runtime_seconds
    update_generations(pop, time_end)
    #return the best individual after generation updating is completed
    best_ind = pop.select_best_individual()

def cross_individuals(p1: 'Individual', p2: 'Individual', mp: int) -> 'Individual':
    return None
    #TODO perform crossover/mutation on two parents to produce a child.

def update_generations(population: 'Population', time_end: float):
    count = 0
    while time.time() < time_end:
        if count < population.max_population:
            population.select_pair_and_reproduce()
            count += 1
        else:
            population.current_generation = population.next_generation
            population.next_generation = []
            count = 0

class Individual():
    def __init__(self, c: str, b: Board):
        # the characteristic will be a string representing the board for this individual
        # String encoding the board will be as follows:
        # X: Toxic, S: Scene, R: Residential, C: Commercial, I: Industrial, S
        # Otherwise we assume the tile is empty, i.e. containing an int between 0..9
        self.characteristic = c
        self.score = decode_and_score(c, b)

    def __str__(self):
        content_str = "Characteristic String: "+self.characteristic+"\nScore: "+self.score
        return content_str

    @staticmethod
    def decode_and_score(characteristic: str, main_board: Board) -> int:
        r = 0
        c = 0
        tiles = []
        cur_row = []
        for letter in characteristic:
            #the letter N will encode a new row
            if letter == "N":
                r += 1
                c = 0
                tiles.append(cur_row)
                cur_row = []
            else:
                #Otherwise figure out what tile should be placed here
                cur_tile = main_board.current_state[r][c]
                cur_name = cur_tile.typename
                cur_cost = cur_tile.cost
                if cur_name != "Toxic":
                    if letter == "R":
                        cur_row.append(Resident(r, c, cur_cost))
                    elif letter == "C":
                        cur_row.append(Commerical(r, c, cur_cost))
                    elif letter == "I":
                        cur_row.append(Individual(r, c, cur_cost))
                    elif letter == "S":
                        cur_row.append(Scene(r, c))
                    else: #only other possibility is an empty basic tile
                        cur_row.append(Basic(r, c, cur_cost))
                else:
                    cur_row.append(Toxic(r, c))
                c += 1
        #return score of a dummy board created using the given tiles
        return Board(0, 0, 0, c, r, tiles).score()


class Population():
    def __init__(self, m: int, e: int, c: int, p: int):
        self.max_population = m
        #current and next generation will be kept separate, and once the next generation is filled we will swap
        #current for next, and set next to be empty for the process to repeat
        self.current_generation = Sequence['Individual']
        self.next_generation = Sequence['Individual']
        self.elite_count = e
        self.cull_count = c
        self.mutation_probability = p

    def cull_the_weak(self):
        return None
        #TODO: cull the weakest cull_count individuals from current generation

    def preserve_elites(self):
        return None
        #TODO: pass on the top elite_count individuals from current to next generation

    def select_pair_and_reproduce(self):
        #TODO: select two individuals and pass to the cross_individuals function, then add to next generation
        # TODO: determine how to do selection, bias towards more fit individuals
        return None

    def select_best_individual(self) -> Individual:
        return self.current_generation[0]


if __name__ == "__main__":
    main()