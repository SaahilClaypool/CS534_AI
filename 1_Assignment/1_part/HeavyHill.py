import random
import heapq
from queue import PriorityQueue
import time
from typing import Sequence, Mapping
from copy import deepcopy
import argparse
class Board:
    """
    array of
    """
    def __init__(self, board: [], prev_cost: int, added_cost = 0, rand=False):
        self.size = len(board)
        self.prev_cost = prev_cost
        self.added_cost = added_cost
        if(rand):
            self.board = Board.generate(self.size)
        else:
            self.board = list(board)

    def __str__(self):
        st = "[" +  ", ".join(map(str, self.board)) + "]"
        st += "\n"
        st += "moved peice cost: " + str(self.prev_cost)
        st += "\n"
        st += "total cost: " + str(self.cost())
        st += "\n"
        st += "number of attacking queens: " + str(self.attacking_pairs())
        st += "\n"
        return st
    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        return self.board == value.board

    def __hash__(self):
        tens = 1
        s = 0
        for i in self.board:
            s += 0 * tens
            tens *= 10
        return s

    def __ne__(self, other):
        return not(self == other)

    def __lt__(self, other):
        return self.cost() < other.cost()

    @staticmethod
    def generate(size: int):
        """
        return random array
        """
        ar = []
        for i in range(size):
            ar.append(random.randint(0, size - 1))
        return ar

    def attacking_pairs(self):
        cnt = 0
        for c1 in range(len(self.board)):
            # calc horizontal
            for c2 in range(len(self.board)):
                if (c1 == c2):
                    continue
                elif (self.board[c1] == self.board[c2]):
                    cnt += 1
                elif (abs(self.board[c1] - self.board[c2]) == abs(c1 - c2)) : #if difference in column = difference in rows then on diag
                    cnt += 1

        if (cnt == 0):
            return 0
        else:
            return cnt / 2
    def calculate_heuristic(self):
        cnt = self.attacking_pairs()
        if (cnt == 0):
            return 0
        else:
            return 10 + cnt

    def cost(self) -> int:
        return self.prev_cost + self.calculate_heuristic()
        # return self.calculate_heuristic()

    def calc_next(self) -> Sequence['Board']:
        moves = []
        for c in range(self.size):
            for r in range(self.size):
                if (r == self.board[c]): # didn't move
                    continue
                new_board_ar : Sequence[int] = deepcopy(self.board)
                new_board_ar[c] = r
                added_cost = (self.board[c] - r) ** 2 + 10
                moves.append(Board(new_board_ar, added_cost + self.prev_cost, added_cost))

        return moves

    def a_star(self):
        explored = [] # already explored
        todo : PriorityQueue = PriorityQueue()
        todo.put(self)
        camefrom : Mapping[Board, Board] = {} # best previous for given state
        prevcost : Mapping[Board, int] = {} # cost to get to given state
        hcost: Mapping[Board, int] = {} # cost to get to given state

        prevcost[self] = 0 # start with zero
        hcost[self] = self.calculate_heuristic()

        while (todo):
            # cur = heapq.heappop(todo)
            cur = todo.get()
            if (cur.calculate_heuristic() == 0):
                return cur, camefrom, len(explored)

            explored.append(cur)
            neighbors = cur.calc_next()
            for n in neighbors:
                if (n in explored):
                    continue

                if (n not in todo.queue):
                    todo.put(n)

                cost_there = prevcost[cur] + n.added_cost

                if (n in prevcost.keys() and \
                    cost_there >= prevcost[n]):
                    continue

                camefrom[n] = cur
                prevcost[n] = cost_there
                hcost[n] = prevcost[n] + n.calculate_heuristic()




    def climb(self):
        start_time = time.time()
        best = self
        best_score = self.calculate_heuristic()
        best_chain = [self]

        cur_best = best
        cur_best_score = best.calculate_heuristic()
        cur_chain = [cur_best]

        while(time.time() - start_time < 10 and\
                not cur_best.calculate_heuristic() == 0):
            prev_best = cur_best
            next_moves = best.calc_next()
            random.shuffle(next_moves)
            next_moves.append(prev_best)
            for m in next_moves:
                if(m.calculate_heuristic() < cur_best_score or\
                    m.calculate_heuristic() == cur_best_score and m.prev_cost < cur_best.prev_cost):
                    cur_best_score = m.calculate_heuristic()
                    cur_best = m
                    cur_chain.append(cur_best)

                if (m.calculate_heuristic() < best_score or\
                    m.calculate_heuristic() == best_score and m.prev_cost < best.prev_cost):
                    best_score = m.calculate_heuristic()
                    best = m
                    best_chain = cur_chain

            if (cur_best == prev_best):
                cur_best = Board(self.board, 0, 0, True)
                cur_best_score = cur_best.calculate_heuristic()
                cur_chain = []



        return best, best_chain, time.time() - start_time



def climb(b: Board):
    print(b)
    best, chain, t = b.climb()

    print("Best Board: ")
    print(best)
    print("Chain: ")
    for i in chain:
        print("-----------------\n")
        print(i)
        print("-----------------\n")
    print("Completed in : ", t, "seconds")

def astar(b):
    print(b)
    s = time.time()
    best, camefrom, expanded = b.a_star()
    print("Steps in reverse:")
    print("-------------\n")
    cur = best
    steps = []
    while(cur in camefrom.keys()):
        steps.append(cur)
        cur = camefrom[cur]
    i = 0
    steps.append(b)
    steps.reverse()
    for step in steps:
        print("step {}: {}".format(i, step))
        i += 1
    
    print("Finished in {}".format(time.time() - s))
    print("Expanded {} nodes".format(expanded))



def main():
    parser = argparse.ArgumentParser(description='Simulates the Heavy Queens Problem.')
    parser.add_argument('N', help='The number of Queens', type=int)
    parser.add_argument('algorithm', help='Which algorithm to use: 1 - A*, 2 - Hill Climb', type=int)
    args = parser.parse_args()
    b = Board([i for i in range(args.N)], 0, 0, True)
    if args.algorithm == 1:
        astar(b)
    elif args.algorithm == 0:
        climb(b)
    else:
        print('Error: algorithm argument must be a 1 or a 0!')
if __name__ == "__main__":
    main()
