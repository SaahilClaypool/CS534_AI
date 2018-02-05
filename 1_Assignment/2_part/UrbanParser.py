"""
Creates funcitons to parse a text file for urban planning
"""
from typing import Type, Sequence


class Board(object):

    def __init__(self, industrial: int, commercial: int, residential: int,
                 width: int, height: int, current_state: Sequence[Sequence['Tile']]):
        self.industrial = industrial
        self.commercial = commercial
        self.residential = residential
        self.current_state = current_state 
        self.width = width
        self.height = height
    
    def __str__(self):
        buildstr = "Industrial: {}\nCommercial: {}\nResidential: {}\n"\
            .format(self.industrial, self.commercial, self.residential)\
            + "Height: {} Width: {}\n".format(self.height, self.width)
        for r in self.current_state:
            buildstr += str(r) + "\n"
        return buildstr
    
    def score(self) -> int:
        sumscore = 0
        for r in self.current_state:
            for c in r:
                sumscore += c.score(self)
        return sumscore 

    @staticmethod
    def read_from_file(filename: str) -> 'Board':
        with open(filename) as infile:
            ## parse meta
            indust = int(infile.readline())
            commer = int(infile.readline())
            resid = int(infile.readline())
            ## now  read each line
            r = 0
            c = 0
            tiles = []
            for line in infile:
                c = 0
                cur_row = []
                for letter in line.strip().split(" "):
                    cur_row.append(Tile.tile_from_str(letter, r, c))
                    c += 1
                tiles.append(cur_row)
                r += 1
            
            return Board(indust, commer, resid, c, r, tiles)

    @staticmethod
    def main():
        board = Board.read_from_file("./biggerSampleInput.txt")
        print(board)
        industrial = Industrial(2,2,0)
        res = Resident(2,2,0)
        comm = Commercial(2,2,0)

        print("I score: ", industrial.score(board))
        print("res score: ", res.score(board))
        print("comm score: ", comm.score(board))


class Tile(object):
    def __init__(self, typename: str, r: int, c: int, cost: int):
        self.typename = typename
        self.r = r
        self.c = c
        self.cost = cost
    
    def __str__(self):
        return "<{}: r:{} c:{} cost:{:2}>"\
            .format(self.typename, self.r, self.c, self.cost)
    
    def __repr__(self):
        return self.__str__()

    def dist(self, other: 'Tile') -> int:
        return abs(self.r - other.r) + abs(self.c - other.c)

    def score(self, board: 'Board') -> int:
        return 0

    def tiles_within(self, board: 'Board', distance: int, typename: str = "*"):
        close = []
        for r in board.current_state:
            for c in r:
                if (self.dist(c) <= distance and\
                    (typename == "*" or typename == c.typename)):
                    close.append(c)
        return close
    
    @staticmethod
    def tile_from_str(inp: str, r: int, c: int):
        if (inp is "S"):
            return Scene(r, c)
        elif (inp is "X"):
            return Toxic(r, c)
        else:
            return Basic(r, c, int(inp))
    
    @staticmethod
    def make_tile(typename: str, r: int, c: int, cost: int):
        if (typename == "Scene"):
            return Scene(r,c)
        elif (typename == "Toxic"):
            return Toxic(r,c)
        elif (typename == "Basic"):
            return Basic(r,c,cost)
        elif (typename == "Resident"):
            return Resident(r,c,cost)
        elif (typename == "Commercial"):
            return Commercial(r,c,cost)
        elif (typename == "Industrial"):
            return Industrial(r,c,cost)


class Scene(Tile):
    def __init__(self, r: int, c: int):
        Tile.__init__(self, "Scene", r, c, 0)


class Toxic(Tile):
    def __init__(self, r: int, c: int):
        Tile.__init__(self, "Toxic", r, c, -1)


class Basic(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Basic", r, c, cost)
    
class Resident(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Resident", r, c, cost)

    def score(self, board: 'Board') -> int:
        toxic_tiles = self.tiles_within(board, 2, "Toxic")
        toxic_score = +10 * len(toxic_tiles)

        scenic_tiles = self.tiles_within(board, 2, "Scene")
        scenic_score = -10 * len(scenic_tiles)

        industrial_tiles = self.tiles_within(board, 3, "Industrial")
        industrial_score = -5 * len(industrial_tiles)

        commericial_tiles = self.tiles_within(board, 3, "Commercial")
        commericial_score = -5 * len(commericial_tiles)

        return toxic_score + scenic_score + industrial_score + commericial_score - self.cost

class Commercial(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Commericial", r, c, cost)
    
    def score(self, board: 'Board') -> int:
        resident_tiles = self.tiles_within(board, 3, "Resident")
        resident_score = 5 * len(resident_tiles)

        commercial_tiles = self.tiles_within(board, 2, "Commercial")
        competition_score = -5 * len(commercial_tiles)
        return resident_score + competition_score - self.cost


class Industrial(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Industrial", r, c, cost)
    
    def score(self, board: 'Board') -> int:
        toxic_tiles = self.tiles_within(board, 2, "Toxic")
        toxic_score = -10 * len(toxic_tiles)

        industrial_tiles = self.tiles_within(board, 2, "Industrial")
        indust_score = len(industrial_tiles) * 3

        return toxic_score - self.cost + indust_score

if __name__ == "__main__":
    Board.main()
