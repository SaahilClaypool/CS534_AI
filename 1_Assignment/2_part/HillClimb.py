from UrbanParser import *
from typing import Type, Sequence, Tuple
from copy import deepcopy
import random

class HillClimb(object):
    def __init__(self, board: 'Board'):
        self.original_state = board
        self.current_state = deepcopy(board)
        self.init_board()
    
    def __str__(self):
        return self.current_state.__str__()
    
    def __repr__(self):
        return self.current_state.__repr__()
    
    def init_board(self):
        # randomly place all tiles to start hillclimb
        # note: when moving off tile, set it to old value in original board
        for _ in range(self.original_state.commercial):

            spot = self.randspot()
            new_spot = Commercial(spot.r, spot.c, spot.cost)
            self.set_spot(self.current_state, new_spot)
        
        for _ in range(self.original_state.industrial):
            spot = self.randspot()
            new_spot = Industrial(spot.r, spot.c, spot.cost)
            self.set_spot(self.current_state, new_spot)
        
        for _ in range(self.original_state.residential):
            spot = self.randspot()
            new_spot = Resident(spot.r, spot.c, spot.cost)
            self.set_spot(self.current_state, new_spot)
            
    
    def move(self, board: Board, tile: 'Tile', r: int, c: int):
        # reset original spot
        board.current_state[tile.r][tile.c] = \
            self.original_state.current_state[tile.r][tile.c]
        to_spot = self.current_state.current_state[r][c]
        # get new sppot
        new_spot = Tile.make_tile(tile.typename, r,c, to_spot.cost)
        # set the new spot 
        self.set_spot(board, new_spot)
        

    def set_spot(self, board: Board, spot: 'Tile'):
        board.current_state[spot.r][spot.c] = spot

    def possible_moves(self):
        spots = []
        for r in self.current_state.current_state:
            for c in r:
                if (c.typename == "Scene" or \
                    c.typename == "Basic"): # only build on scene or basic
                    spots.append(c)
        return spots


    def randspot(self):
        return random.choice(self.possible_moves())

        
    def get_moveable(self):
        for r in self.current_state.current_state:
            for c in r:
                if (c.typename == "Industrial" or c.typename == "Commercial" or\
                    c.typename == "Resident"):
                    yield c
    
    def climb(self, restarts: int =0) -> Tuple[Board, int]:
        best_board, best_score = self.next_state()
        for i in range(restarts):
            print("Climb : ", i)
            # reset to original state
            self.current_state = deepcopy(self.original_state)
            self.init_board()
            new_state, new_score = self.next_state()
            if (new_score > best_score):
                print("New High Score: ", new_score)
                best_board = deepcopy(new_state)
                best_score = new_score
            else:
                print("Not a new high score", new_score)
            print("End Climb : ", i, "\n\n")
        # Update state because it will have been modified by next_state
        self.current_state = deepcopy(best_board)
        return (best_board, best_score)

    def next_state(self) -> Tuple[Board, int]:
        """ next_states: get the sequence of possible next states by moving one peice
        """
        # for each building, calculate all next moves
        moveable_tiles = list(self.get_moveable())
        # create a new board for each 
        best_board = self.current_state
        best_score = self.current_state.score()
        for tile in moveable_tiles:
            possible_moves = self.possible_moves()
            for move in possible_moves:
                new_board = deepcopy(self.current_state)
                self.move(new_board, tile, move.r, move.c)
                new_score = new_board.score()
                if(new_score > best_score):
                    best_score = new_score
                    best_board = new_board
        
        old_score = self.current_state.score()
        if(best_score != old_score):
            self.current_state = best_board
            print("CLIMBING!! from ", self.current_state.score())
            best_board, best_score = self.next_state()

        return best_board, best_score

def main():
    print ("hello world")
    board : Board = Board.read_from_file("sample2.txt")
    # print(board)

    alg : HillClimb = HillClimb(board)

    # print(alg)

    board, score = alg.climb(2)
    print("Final of ", score)
    print("Checking score: ", board.score())
    print("board\n", board)

    ## algorithm

if __name__ == "__main__":
    main()
