import random
import time
from typing import Sequence
from copy import deepcopy
class Board:
    """
    array of 
    """
    def __init__(self, board: [], prev_cost: int, rand=False):
        self.size = len(board)
        self.prev_cost = prev_cost
        if(rand):
            self.board = Board.generate(self.size)
        else:
            self.board = list(board)
    
    def __str__(self):
        st = "[" +  ", ".join(map(str, self.board)) + "]" 
        st += "\n"
        st += "cost to move: " + str(self.prev_cost)
        st += "\n"
        st += "number of attacking queens: " + str(self.attacking_pairs())
        st += "\n"
        return st
    def __repr__(self):
        return self.__str__()
    
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
    
    def calc_next(self) -> Sequence['Board']:
        moves = []
        for c in range(self.size):
            for r in range(self.size):
                if (r == self.board[c]): # didn't move
                    continue
                new_board_ar : Sequence[int] = deepcopy(self.board)
                new_board_ar[c] = r
                added_cost = (self.board[c] - r) ** 2 + 10
                moves.append(Board(new_board_ar, added_cost + self.prev_cost))
        
        return moves
        
    def climb(self): 
        start_time = time.time()
        best = self
        best_score = self.cost()
        best_chain = [self]

        cur_best = best
        cur_best_score = best.cost()
        cur_chain = [cur_best]

        while(time.time() - start_time < 10):
            prev_best = cur_best
            next_moves = best.calc_next()
            random.shuffle(next_moves)
            next_moves.append(prev_best)
            for m in next_moves: 
                if(m.cost() < cur_best_score):
                    cur_best_score = m.cost()
                    cur_best = m
                    cur_chain.append(cur_best)
                if (m.cost() < best_score):
                    best_score = m.cost()
                    best = m
                    best_chain = cur_chain
            
            if (cur_best == prev_best):
                cur_best = Board(self.board, 0, True)
                cost_from_orig = 0
                for r in range(self.size):
                    cost_from_orig += 10 + (self.board[r] - cur_best.board[r]) ** 2
                cur_best.prev_cost = cost_from_orig
                cur_best_score = cur_best.cost()
                cur_chain = []

                
        
        return best, best_chain, time.time() - start_time
                


def main():
    b = Board([i for i in range(16)], 0, True)
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
if __name__ == "__main__":
    main()