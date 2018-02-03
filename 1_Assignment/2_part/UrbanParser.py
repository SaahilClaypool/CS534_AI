"""
Creates funcitons to parse a text file for urban planning
"""
from typing import Type, Sequence


class Board(object):

    def __init__(self, industrial: int, commerical: int, residential: int,
                 width: int, height: int, current_state: Sequence[Sequence['Tile']]):
        self.industrial = industrial
        self.commerical = commerical
        self.residential = residential
        self.current_state = current_state
    
    def __str__(self):
        buildstr = "Industrial: {}\nCommerical: {}\nResidential: {}\n"\
            .format(self.industrial, self.commerical, self.residential)
        for r in self.current_state:
            buildstr += str(r) + "\n"
        return buildstr

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
        board = Board.read_from_file("./sampleInput.txt")
        print(board)


class Tile(object):
    def __init__(self, typename: str, r: int, c: int):
        self.typename = typename
        self.r = r
        self.c = c
    
    def __str__(self):
        return "<{}: {} {}>"\
            .format(self.typename, self.r, self.c)
    
    def __repr__(self):
        return self.__str__()

    def dist(self, other: 'Tile') -> int:
        return abs(self.r - other.r) + abs(self.c - other.c)

    def score(self, board: 'Board') -> int:
        return 0

    @staticmethod
    def tile_from_str(inp: str, r: int, c: int):
        if (inp is "S"):
            return Scene(r, c)
        elif (inp is "X"):
            return Toxic(r, c)
        else:
            return Basic(r, c, int(inp))


class Scene(Tile):
    def __init__(self, r: int, c: int):
        Tile.__init__(self, "Scene", r, c)


class Toxic(Tile):
    def __init__(self, r: int, c: int):
        Tile.__init__(self, "Toxic", r, c)


class Basic(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Basic", r, c)
    
class Resident(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Resident", r, c)

class Commerical(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Commericial", r, c)

class Industrial(Tile):
    def __init__(self, r: int, c: int, cost: int):
        Tile.__init__(self, "Industrial", r, c)

if __name__ == "__main__":
    Board.main()
