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
    #start timing
    time_end = time.time() + runtime_seconds
    pop = Population(max_population, elite_count, cull_count, mutation_probability, 0, main_board, [], [], time.time())
    #seed initial generation
    seed_generation(pop)
    pop.finalize_generation()
    #update generations while timeer is running
    update_generations(pop, time_end)
    #return the best individual after generation updating is completed
    best_ind = pop.select_best_individual()
    print("Generation: ", pop.gen_number)
    print("Time achieved: ", pop.top_achieved_time)
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
            if new_state[i] == "O":
                if r_c > 0:
                    new_state = new_state[:i] + "R" + new_state[i+1:]
                    r_c -= 1
                elif c_c > 0:
                    new_state = new_state[:i] + "C" + new_state[i+1:]
                    c_c -= 1
                else:
                    new_state = new_state[:i] + "I" + new_state[i+1:]
                    i_c -= 1
        p.add_to_current_generation(Individual(new_state, b))
        count += 1

def encode_board(b: Sequence[Sequence['Tile']]) -> str:
    encoded_b = ""
    for row in b:
        for col in row:
            t = col.typename
            if t == "Industrial":
                encoded_b += "I"
            elif t == "Commercial":
                encoded_b += "C"
            elif t == "Resident":
                encoded_b += "R"
            elif t == "Toxic":
                encoded_b += "X"
            elif t == "Scene":
                encoded_b += "O"
            else:
                #basic
                encoded_b += "O"
        encoded_b += "N" #add newline encoding
    encoded_b = encoded_b[:-1] #remove the last letter, excess "N"
    return encoded_b


class Individual():
    def __init__(self, c: str, b: Board):
        # the characteristic will be a string representing the board for this individual
        # String encoding the board will be as follows:
        # X: Toxic, S: Scene, R: Residential, C: Commercial, I: Industrial, S
        # Otherwise we assume the tile is empty, i.e. containing an int between 0..9
        self.characteristic = c
        self.score = decode_and_score(c, b)

    def __str__(self):
        content_str = "Characteristic String: "+self.characteristic+"\nScore: "+str(self.score)
        return content_str

def decode_and_score(characteristic: str, main_board: 'Board') -> int:
    #return score of a dummy board created using the given tiles
    return decode_board(characteristic, main_board).score()

def decode_board(characteristic: str, main_board: 'Board') -> 'Board':
    r = 0
    c = 0
    tiles = []
    cur_row = []
    for letter in characteristic:
        # the letter N will encode a new row
        if letter == "N":
            r += 1
            c = 0
            tiles.append(cur_row)
            cur_row = []
        else:
            # Otherwise figure out what tile should be placed here
            cur_tile = main_board.current_state[r][c]
            cur_name = cur_tile.typename
            cur_cost = cur_tile.cost
            if cur_name != "Toxic":
                if letter == "R":
                    cur_row.append(Resident(r, c, cur_cost))
                elif letter == "C":
                    cur_row.append(Commercial(r, c, cur_cost))
                elif letter == "I":
                    cur_row.append(Industrial(r, c, cur_cost))
                elif cur_name == "Scene":
                    cur_row.append(Scene(r, c))
                else:  # only other possibility is an empty basic tile
                    cur_row.append(Basic(r, c, cur_cost))
            else:
                cur_row.append(Toxic(r, c))
            c += 1
    tiles.append(cur_row)
    r += 1
    return Board(main_board.industrial, main_board.commercial, main_board.residential, c, r, tiles)

class Population():
    def __init__(self, m: int, e: int, c: int, p: int,  g: int, b: Board,
                 current_gen: Sequence['Individual'], next_gen: Sequence['Individual'], time_start: float):
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
        self.time_start = time_start
        self.top_achieved_time = 0

    def cull_the_weak(self):
        self.current_generation = self.current_generation[0:self.max_population-self.cull_count-1]

    def preserve_elites(self):
        self.next_generation.extend(self.current_generation[0:self.elite_count-1])

    def select_pair_and_reproduce(self):
        #use fitness proportionate selection
        r1 = random.uniform(0, 1) * self.sum_score
        i1 = None
        for i in self.current_generation:
            r1 -= i.score
            if r1 < 0:
                i1 = i
                break
        if i1 is None:
            i1 = self.current_generation[0]
        r2 = random.uniform(0, 1) * self.sum_score
        i2 = None
        for i in self.current_generation:
            r2 -= i.score
            if r2 < 0:
                #if selected individual is same as i1, choose the previous one
                if i != i1:
                    i2 = i
                    break
                else:
                    id2 = self.current_generation.index(i)
                    if id2 != 0:
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
        C_count = p1_c.count("C")
        I_count = p1_c.count("I")
        R_count = p1_c.count("R")
        O_count = p1_c.count("O")
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
                    if i == 0:
                        self.top_achieved_time = time.time() - self.time_start
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

    def repair_and_mutate(self, c: str, Cc: int, Ic: int, Rc: int, Oc: int) -> str:
        #repair will operate by checking the occurences of C,I,R, and O in the string
        #excess occurences of C,I, and R will be converted into Os. excess Os will then be
        #converted into C,I, and R as necessary
        #work with the string as a list to make replacing easier
        Cdif = c.count("C")-Cc
        C_indices = [m.start() for m in re.finditer("C", c)]
        Idif = c.count("I")-Ic
        I_indices = [m.start() for m in re.finditer("I", c)]
        Rdif = c.count("R")-Rc
        R_indices = [m.start() for m in re.finditer("R", c)]
        #remove excess C/I/R
        while Cdif > 0:
            i = random.choice(C_indices)
            c = c[:i] + "O" + c[i+1:]
            C_indices.remove(i)
            Cdif -= 1
        while Idif > 0:
            i = random.choice(I_indices)
            c = c[:i] + "O" + c[i+1:]
            I_indices.remove(i)
            Idif -= 1
        while Rdif > 0:
            i = random.choice(R_indices)
            c = c[:i] + "O" + c[i+1:]
            R_indices.remove(i)
            Rdif -= 1
        #excess C/I/R are now excess Os
        Odif = c.count("O")-Oc
        O_indices = [m.start() for m in re.finditer("O", c)]
        while Odif > 0:
            while Cdif < 0:
                i = random.choice(O_indices)
                c = c[:i] + "C" + c[i + 1:]
                O_indices.remove(i)
                Cdif += 1
                Odif -= 1
            while Idif < 0:
                i = random.choice(O_indices)
                c = c[:i] + "I" + c[i+1:]
                O_indices.remove(i)
                Idif += 1
                Odif -= 1
            while Rdif < 0:
                i = random.choice(O_indices)
                c = c[:i] + "R" + c[i + 1:]
                O_indices.remove(i)
                Rdif += 1
                Odif -= 1
        #mutation will swap two elements, if those elements aren't N or X
        clen = len(c)-1
        if random.uniform(0, 1) < self.mutation_probability:
            swap1 = random.randint(0, clen)
            while c[swap1] == "N" or c[swap1] == "X":
                swap1 = random.randint(0, clen)
            swap2 = random.randint(0, clen)
            while c[swap2] == "N" or c[swap2] == "X":
                swap2 = random.randint(0, clen)
            temp1 = c[swap1]
            temp2 = c[swap2]
            c = c[:swap1] + temp2 + c[swap1+1:]
            c = c[:swap2] + temp1 + c[swap2+1:]
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