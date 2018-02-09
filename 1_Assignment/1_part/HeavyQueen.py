import argparse, random

class Board:
    def __init__(self, size: int):
        self.size = size
        self.board = []

    def __str__(self):
        board_string = str(self.board) + '\n\n'
        for i in range(self.size):
            for j in range(self.size):
                if self.board[j] == i:
                    board_string += 'Q '
                else:
                    board_string += '* '
            board_string += '\n'

        return board_string

    def generate(self):
        for i in range(self.size):
            self.board.append(random.randint(0, self.size - 1))


parser = argparse.ArgumentParser(description='Simulates the Heavy Queens Problem.')
parser.add_argument('N', help='The number of Queens', type=int)
parser.add_argument('algorithm', help='Which algorithm to use: 1 - A*, 2 - Hill Climb', type=int)
args = parser.parse_args()

board = Board(args.N)
board.generate();
print(board)

if args.algorithm == 1:
    pass
elif args.algorithm == 0:
    pass
else:
    print('Error: algorithm argument must be a 1 or a 0!')
