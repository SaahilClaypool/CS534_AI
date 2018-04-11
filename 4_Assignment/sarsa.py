from random import *
import csv

def load_board(filename: str = "./input_hw4.csv"):
    board = []
    with open(filename) as f: 
        reader = csv.reader(f)
        for row in reader:
            board.append(row)
    return board


def initialize_board():
    board = [['' for i in range(6)] for j in range(6)]

    board[2][2] = 'P'
    board[3][2] = 'P'
    board[1][3] = 'P'
    board[5][3] = 'P'
    board[2][4] = 'P'
    board[3][4] = 'P'
    board[4][4] = 'P'

    board[2][3] = 'G'

    return board


def best_move(utilities, x, y):
    max_utility = utilities[x][y][0]
    move = 'up'

    if utilities[x][y][1] > max_utility:
        max_utility = utilities[x][y][1]
        move = 'right'

    if utilities[x][y][2] > max_utility:
        max_utility = utilities[x][y][2]
        move = 'down'

    if utilities[x][y][3] > max_utility:
        max_utility = utilities[x][y][3]
        move = 'left'

    if utilities[x][y][4] > max_utility:
        max_utility = utilities[x][y][4]
        move = 'give-up'

    return move


def train(board, goal_reward, pit_reward, move_reward, give_up_reward, epsilon, num_trials):
    moves = ['up', 'right', 'down', 'left', 'give-up']
    board_width = len(board)
    board_height = len(board[0])
    utilities = [[[99 for i in range(len(moves))] for j in range(board_height)] for k in range(board_width)]
    for trial_num in range(num_trials):
        x = randrange(board_width)
        y = randrange(board_height)
        move = ''
        prev_x = 0
        prev_y = 0
        prev_move = ''
        trial_complete = False

        while not trial_complete:
            prev_x = x
            prev_y = y
            prev_move = move

            if random() < epsilon:
                move = moves[randrange(5)]
            else:
                move = best_move(utilities, x, y)

            modifier = 'none'
            modifier_likeliness = random()
            if modifier_likeliness >= 0.9:
                modifier = 'double'
            elif modifier_likeliness < 0.9 and modifier_likeliness >= 0.8:
                modifier = 'left'
            elif modifier_likeliness < 0.8 and modifier_likeliness >= 0.7:
                modifier = 'right' 


    return utilities


def main():
    goal_reward = 5
    pit_reward = -5
    move_reward = -0.1
    give_up_reward = -3
    epsilon = 0.1
    num_trials = 10

    board = initialize_board()
    trained_utilities = train(board, goal_reward, pit_reward, move_reward, give_up_reward, epsilon, num_trials)


if __name__ == "__main__":
    main()
