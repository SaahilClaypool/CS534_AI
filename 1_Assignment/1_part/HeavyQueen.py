import argparse, random

class Board:
    def __init__(self, size: int, board = []):
        self.size = size
        self.board = board

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
        self.board = []
        for i in range(self.size):
            self.board.append(random.randint(0, self.size - 1))

    def calculate_column_heuristic(self, column: int):
        count = 0;
        for i in range(self.size):
            if self.board[column] == self.board[i] or abs(column - i) == abs(self.board[column] - self.board[i]):
                count += 1

        return count - 1

    def calculate_heuristic(self):
        count = 0
        for i in range(self.size):
            count += self.calculate_column_heuristic(i)

        if count > 0:
            return count + 10

        return 0

    def move(self, column, row):
        if column < self.size and row < self.size:
            self.board[column] = row

    def copy(self):
        return Board(self.size, list(self.board))

    def generate_moves(self):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[j] != i:
                    board = self.copy()
                    cost = (abs(board.board[j] - i) ** 2) + 10
                    board.move(j, i)
                    move = Move(board, cost)
                    moves.append(move)
        return moves



class Move:
    def __init__(self, board: Board, cost: int):
        self.board = board
        self.cost = cost

    def __str__(self):
        return str(self.board) + '\nCost: ' + str(self.cost)

    def total_cost():
        return self.cost + self.board.calculate_heuristic()





parser = argparse.ArgumentParser(description='Simulates the Heavy Queens Problem.')
parser.add_argument('N', help='The number of Queens', type=int)
parser.add_argument('algorithm', help='Which algorithm to use: 1 - A*, 2 - Hill Climb', type=int)
args = parser.parse_args()

board = Board(args.N)
board.generate()
print(board)
print(board.calculate_heuristic())
moves = board.generate_moves()
for i in range(len(moves)):
    print(moves[i])

if args.algorithm == 1:
    pass
elif args.algorithm == 0:
    pass
else:
    print('Error: algorithm argument must be a 1 or a 0!')
