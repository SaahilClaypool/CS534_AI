import argparse, random

class Board:
    def __init__(self, size: int, board = []):
        self.size = size
        self.board = board
        self.heuristic = -1
        if len(board) > 0:
            self.calculate_heuristic()

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

        self.calculate_heuristic()

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
            self.heuristic = count + 10
        else:
            self.heuristic = 0

    def modify(self, column, row):
        board = list(self.board)
        if column < self.size and row < self.size:
            board[column] = row

        return Board(self.size, board)

    def generate_moves(self, parent = None):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[j] != i:
                    cost = (abs(self.board[j] - i) ** 2) + 10
                    board = self.modify(j, i)
                    move = Move(board, cost, parent)
                    moves.append(move)
        return moves


class Move:
    def __init__(self, board: Board, cost: int, parent = None):
        self.board = board
        self.cost = cost
        self.parent = parent
        self.total_cost = self.cost
        if self.parent != None:
            self.total_cost += self.parent.total_cost
        self.heuristic_cost = self.total_cost + self.board.heuristic

    def __str__(self):
        return str(self.board) + '\nCost: ' + str(self.total_cost) + ' - Heuristic Cost: ' + str(self.heuristic_cost)

    def __lt__(self, other):
        if hasattr(other, 'heuristic_cost'):
            return self.heuristic_cost < other.heuristic_cost

    def __gt__(self, other):
        if hasattr(other, 'heuristic_cost'):
            return self.heuristic_cost > other.heuristic_cost

    def __eq__(self, other):
        if hasattr(other, 'heuristic_cost'):
            return self.heuristic_cost == other.heuristic_cost
        else:
            return False

    def __le__(self, other):
        if hasattr(other, 'heuristic_cost'):
            return self.heuristic_cost <= other.heuristic_cost

    def __ge__(self, other):
        if hasattr(other, 'heuristic_cost'):
            return self.heuristic_cost >= other.heuristic_cost

    def __ne__(self, other):
        if hasattr(other, 'heuristic_cost'):
            return self.heuristic_cost != other.heuristic_cost
        else:
            return True


def A_star(board: Board):
    if board.heuristic != 0:
        moves = board.generate_moves(Move(board, 0))
        while moves[0].board.heuristic > 0:
            moves = moves + moves[0].board.generate_moves(moves[0])
            holder = moves[0]
            del moves[0]
            moves.sort()
            # for i in range(5):
            #     print(moves[i])
            # input()
            # print('\n\n\n\n##################################')

        nodes_traversed = 0
        node = moves[0]
        print('\n\n\n\n##################################')
        while node != None:
            print(node)
            node = node.parent
            nodes_traversed += 1
        print('\n\n\n\n##################################')
        print(moves[0])
        print('It took ' + str(nodes_traversed) + ' moves')
    else:
        print('\n\n\n\n##################################')
        print('The board is requires no modification')
        print(board)



parser = argparse.ArgumentParser(description='Simulates the Heavy Queens Problem.')
parser.add_argument('N', help='The number of Queens', type=int)
parser.add_argument('algorithm', help='Which algorithm to use: 1 - A*, 2 - Hill Climb', type=int)
args = parser.parse_args()

board = Board(args.N)
board.generate()
print(board)
print(board.heuristic)
# moves = board.generate_moves()
# moves.sort()
# for i in range(len(moves)):
#     print(moves[i])

if args.algorithm == 1:
    A_star(board)
elif args.algorithm == 0:
    pass
else:
    print('Error: algorithm argument must be a 1 or a 0!')
