"""
Use genetic algorithm approach to output a solution for urban planning.
"""
import sys
import re
import time
import random
from typing import List
from UrbanParser import *

def main():
    max_population = 60
    elite_count = 15
    cull_count = 20
    mutation_probability = 0.1
    runtime_seconds = 10.0

    #for arg in sys.argv[1:]:
        #TODO actually take input file arument

    main_board = Board.read_from_file("./biggerSampleInput.txt")
    pop = Population(max_population, elite_count, cull_count, mutation_probability, 0, main_board, [], [])
    #start timing
    time_end = time.time() + runtime_seconds
    #seed initial generation
    seed_generation(pop)
    pop.finalize_generation()
    #update generations while timeer is running
    update_generations(pop, time_end)
    #return the best individual after generation updating is completed
    best_ind = pop.select_best_individual()
    print("Generation: "+str(pop.gen_number))
    print(best_ind)
    print("Best board: "+str(decode_board(best_ind.characteristic, main_board)))

def update_generations(population: 'Population', time_end: float):
    while time.time() < time_end:
        if len(population.next_generation) < population.max_population:
            population.select_pair_and_reproduce()
        else:
            population.finalize_generation()

def seed_generation(p: 'Population'):
    #initial seeding of population
    b = p.main_board
    count = 0
    while count < p.max_population:
        r_c = b.residential
        c_c = b.commercial
        i_c = b.industrial
        new_state = encode_board(b.current_state)
        encoded_len = len(new_state)-1
        while r_c+c_c+i_c > 0:
            i = random.randint(0, encoded_len)
            select_tile = new_state[i]
            if select_tile.typename not in ["Toxic", "Resident", "Commercial", "Industrial"]:
                if r_c > 0:
                    new_state[i] = Resident(select_tile.r, select_tile.c, select_tile.cost)
                    r_c -= 1
                elif c_c > 0:
                    new_state[i] = Commercial(select_tile.r, select_tile.c, select_tile.cost)
                    c_c -= 1
                else:
                    new_state[i] = Industrial(select_tile.r, select_tile.c, select_tile.cost)
                    i_c -= 1
        p.add_to_current_generation(Individual(new_state, b))
        count += 1

def encode_board(b: Sequence[Sequence['Tile']]) -> Sequence['Tile']:
    encode = lambda l: [item for sublist in l for item in sublist]
    encoded_b = encode(b)
    return encoded_b


class Individual():
    def __init__(self, c: Sequence['Tile'], b: Board):
        # the characteristic will be a string representing the board for this individual
        # String encoding the board will be as follows:
        # X: Toxic, S: Scene, R: Residential, C: Commercial, I: Industrial, S
        # Otherwise we assume the tile is empty, i.e. containing an int between 0..9
        self.characteristic = c
        self.score = decode_and_score(c, b)

    def __str__(self):
        content_str = "Characteristic String: "+str(self.characteristic)+"\nScore: "+str(self.score)
        return content_str

def decode_and_score(characteristic: Sequence['Tile'], main_board: 'Board') -> int:
    #return score of a dummy board created using the given tiles
    return decode_board(characteristic, main_board).score()

def decode_board(characteristic: Sequence['Tile'], main_board: 'Board') -> 'Board':
    r = main_board.height
    c = main_board.width
    tiles = []
    cur_row = []
    count = 0
    for tile in characteristic:
        cur_row.append(tile)
        count += 1
        if count == c:
            tiles.append(cur_row)
            count = 0
            cur_row = []
    return Board(main_board.industrial, main_board.commercial, main_board.residential, c, r, tiles)

class Population():
    def __init__(self, m: int, e: int, c: int, p: int,  g: int, b: Board,
                 current_gen: Sequence['Individual'], next_gen: Sequence['Individual']):
        self.gen_number = g
        self.main_board = b
        self.max_population = m
        #current and next generation will be kept separate, and once the next generation is filled we will swap
        #current for next, and set next to be empty for the process to repeat
        self.current_generation = current_gen
        self.next_generation = next_gen
        self.elite_count = e
        self.cull_count = c
        self.mutation_probability = p
        self.sum_score = 0 #sum_score will be used for weighed selection of individuals

    def cull_the_weak(self):
        self.current_generation = self.current_generation[0:self.max_population-self.cull_count-1]

    def preserve_elites(self):
        self.next_generation.extend(self.current_generation[0:self.elite_count-1])

    def select_pair_and_reproduce(self):
        #use fitness proportionate selection
        r1 = random.uniform(0, 1) * self.sum_score
        i1 = self.current_generation[0]
        for i in self.current_generation:
            r1 -= i.score
            if r1 < 0:
                i1 = i
                break
        r2 = random.uniform(0, 1) * self.sum_score
        i2 = self.current_generation[1]
        for i in self.current_generation:
            r2 -= i.score
            if r2 < 0:
                #if selected individual is same as i1, choose the previous one
                i2 = i
                if i1 == i2:
                    id2 = self.current_generation.index(i)
                    if id2 > 0:
                        i2 = self.current_generation[id2-1]
                    else:
                        i2 = self.current_generation[id2+1]
        self.cross_individuals(i1, i2)

    def cross_individuals(self, p1: 'Individual', p2: 'Individual'):
        p1_c = p1.characteristic
        p2_c = p2.characteristic
        p1_len = len(p1_c)
        #number of each type of character within p1_c will be used for repair purposes
        #X and N should not have to be checked since crossover wont affect them
        C_count = p1_c.count(Commercial.__class__)
        I_count = p1_c.count(Industrial.__class__)
        R_count = p1_c.count(Resident.__class__)
        O_count = p1_c.count(Basic.__class__) + p1_c.count(Scene.__class__)
        #two point crossover will be used
        cross_start = random.randint(0, p1_len-2)
        cross_end = random.randint(cross_start, p1_len-1)
        #create crossover children
        new_c1 = p1_c[0:cross_start]+p2_c[cross_start: cross_end]+p1_c[cross_end: p1_len]
        new_c2 = p2_c[0:cross_start]+p1_c[cross_start: cross_end]+p2_c[cross_end: p1_len]
        #repair crossover result
        new_c1 = self.repair_and_mutate(new_c1, C_count, I_count, R_count, O_count)
        new_c2 = self.repair_and_mutate(new_c2, C_count, I_count, R_count, O_count)
        #create chlidren
        child_1 = Individual(new_c1, self.main_board)
        child_2 = Individual(new_c2, self.main_board)
        #insert (ordered) to list
        self.add_to_next_generation(child_1)
        self.add_to_next_generation(child_2)

    def add_to_next_generation(self, child: 'Individual'):
        if len(self.next_generation) > 0:
            for i in range(0, len(self.next_generation)):
                if child.score > self.next_generation[i].score:
                    self.next_generation.insert(i, child)
                    return
            self.next_generation.append(child)
        else:
            self.next_generation.append(child)

    def add_to_current_generation(self, child: 'Individual'):
        if len(self.current_generation) > 0:
            for ind in self.current_generation:
                if child.score > ind.score:
                    self.current_generation.insert(self.current_generation.index(ind), child)
                    return
            self.current_generation.append(child)
        else:
            self.current_generation.append(child)

    def repair_and_mutate(self, c: Sequence['Tile'], Cc: int, Ic: int, Rc: int, Oc: int) -> str:
        #repair will operate by checking the occurences of C,I,R, and O in the string
        #excess occurences of C,I, and R will be converted into Os. excess Os will then be
        #converted into C,I, and R as necessary
        #work with the string as a list to make replacing easier
        Cdif = c.count(Commercial.__class__)-Cc
        C_indices = [i for i, x in enumerate(c) if x.__class__ == Commercial.__class__]
        Idif = c.count(Industrial.__class__)-Ic
        I_indices = [i for i, x in enumerate(c) if x.__class__ == Industrial.__class__]
        Rdif = c.count(Resident.__class__)-Rc
        R_indices = [i for i, x in enumerate(c) if x.__class__ == Resident.__class__]
        #remove excess C/I/R
        while Cdif > 0:
            i = random.choice(C_indices)
            loc_r = i.r
            loc_c = i.c
            loc_flat = self.main_board.width*loc_r+loc_c
            c[loc_flat] = self.main_board.current_state[loc_r][loc_c]
            C_indices.remove(i)
            Cdif -= 1
        while Idif > 0:
            i = random.choice(I_indices)
            loc_r = i.r
            loc_c = i.c
            loc_flat = self.main_board.width*loc_r+loc_c
            c[loc_flat] = self.main_board.current_state[loc_r][loc_c]
            Idif -= 1
        while Rdif > 0:
            i = random.choice(R_indices)
            loc_r = i.r
            loc_c = i.c
            loc_flat = self.main_board.width*loc_r+loc_c
            c[loc_flat] = self.main_board.current_state[loc_r][loc_c]
            Rdif -= 1
        #excess C/I/R are now excess Open/Scenes
        Odif = c.count(Basic.__class__)+c.count(Scene.__class__)-Oc
        O_indices = [i for i, x in enumerate(c) if x.__class__ == Basic.__class__ or x.__class__ == Scene.__class__]
        while Odif > 0:
            while Cdif < 0:
                i = random.choice(O_indices)
                c[i] = Commercial(i.r, i.c, i.cost)
                O_indices.remove(i)
                Cdif += 1
                Odif -= 1
            while Idif < 0:
                i = random.choice(O_indices)
                c[i] = Industrial(i.r, i.c, i.cost)
                O_indices.remove(i)
                Idif += 1
                Odif -= 1
            while Rdif < 0:
                i = random.choice(O_indices)
                c[i] = Resident(i.r, i.c, i.cost)
                O_indices.remove(i)
                Rdif += 1
                Odif -= 1
        #mutation will swap two elements, if those elements aren't N or X
        if random.uniform(0, 1) < self.mutation_probability:
            swap1 = random.randint(0, len(c)-1)
            while c[swap1].__class__ == Toxic.__class__:
                swap1 = random.randint(0, len(c)-1)
            swap2 = random.randint(0, len(c) - 1)
            while c[swap1].__class__ == Toxic.__class__:
                swap2 = random.randint(0, len(c) - 1)
            temp = c[swap1]
            c[swap1] = c[swap2]
            c[swap2] = temp
        return c


    def finalize_generation(self):
        self.gen_number += 1
        if len(self.next_generation) > 0:
            self.current_generation = self.next_generation
        self.next_generation = []
        self.preserve_elites()
        self.cull_the_weak()
        self.sum_score = sum(i.score for i in self.current_generation)

    def select_best_individual(self) -> Individual:
        #generation list is sorted by fitness
        return self.current_generation[0]


if __name__ == "__main__":
    main()