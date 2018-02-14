import argparse, random, time, sys

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

    def __eq__(self, other):
        if hasattr(other, 'board'):
            return self.board == other.board
        else:
            return False

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
        return '\n' + str(self.board) + '\nCost: ' + str(self.heuristic_cost)

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
    start_time = time.time()
    if board.heuristic != 0:
        moves = board.generate_moves(Move(board, 0))
        # moves.sort()
        nodes_expanded = 1
        best_move = moves[0]
        best_index = 0
        current_best = sys.maxsize
        for j in range(len(moves)):
            if moves[j].heuristic_cost < current_best:
                current_best = moves[j].heuristic_cost
                best_move = moves[j]
                best_index = j

        while best_move.board.heuristic > 0:
            moves = moves + best_move.board.generate_moves(best_move)
            # del moves[0]
            # moves.sort()
            nodes_expanded += 1
            del moves[best_index]
            current_best = sys.maxsize
            for j in range(len(moves)):
                if moves[j].heuristic_cost < current_best:
                    current_best = moves[j].heuristic_cost
                    best_move = moves[j]
                    best_index = j

        nodes_traversed = 0
        node = best_move
        print('\n##################################')
        print('Steps (in reverse order):')
        print('##################################')
        while node != None:
            print(node)
            node = node.parent
            nodes_traversed += 1
        print('\n##################################')
        print('Results:')
        print('##################################')
        print(best_move)
        print('Moves: ' + str(nodes_traversed))
        print('Nodes Expanded: ' + str(nodes_expanded))
        print('Time Taken: ' + str(time.time() - start_time) + ' secs')
    else:
        print('\n##################################')
        print('Results:')
        print('##################################\n')
        print('The board requires no modification')
        print(board)

def hill_climbing(board):
    start_time = time.time()
    if board.heuristic != 0:
        best_iteration = None
        for i in range(10):
            if i > 0:
                board.generate()
            sequence = [Move(board, 0)]
            start = time.time()
            while sequence[-1].board.heuristic > 0 and time.time() - start < 10:
                moves = sequence[-1].board.generate_moves()
                best_move = sequence[-1]
                current_best = sys.maxsize
                for j in range(len(moves)):
                    if moves[j].heuristic_cost < current_best:
                        current_best = moves[j].heuristic_cost
                        best_move = moves[j]

                if sequence[-1].board == best_move.board:
                    break
                sequence.append(best_move)
                print(best_move)
            if (sequence[-1].board.heuristic == 0 and not best_iteration) or \
            (sequence[-1].board.heuristic == 0 and len(sequence) < len(best_iteration)):
                best_iteration = sequence
        print(best_iteration)

        if best_iteration and best_iteration[0].board.heuristic == 0:
            print('\n##################################')
            print('Steps:')
            print('##################################')
            for i in range(len(best_iteration)):
                print(best_iteration[i])

        print('\n##################################')
        print('Results:')
        print('##################################')
        if best_iteration and best_iteration[0].board.heuristic == 0:
            print(best_iteration[0])
            print('Moves: ' + str(least_moves))
        else:
            print('No solution found')

        print('Time Taken: ' + str(time.time() - start_time) + ' secs')
    else:
        print('\n##################################')
        print('Results:')
        print('##################################\n')
        print('The board requires no modification')
        print(board)


parser = argparse.ArgumentParser(description='Simulates the Heavy Queens Problem.')
parser.add_argument('N', help='The number of Queens', type=int)
parser.add_argument('algorithm', help='Which algorithm to use: 1 - A*, 2 - Hill Climb', type=int)
args = parser.parse_args()

board = Board(args.N)
board.generate()
print('\n##################################')
print('Initial Board Configuration:')
print('##################################\n')
print(board)

if args.algorithm == 1:
    A_star(board)
elif args.algorithm == 0:
    hill_climbing(board)
else:
    print('Error: algorithm argument must be a 1 or a 0!')
