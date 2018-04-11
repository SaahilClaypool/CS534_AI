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
    board = [['X' for i in range(7)] for j in range(6)]

    board[2][2] = 'P'
    board[2][3] = 'P'
    board[3][1] = 'P'
    board[3][5] = 'P'
    board[4][2] = 'P'
    board[4][3] = 'P'
    board[4][4] = 'P'

    board[3][2] = 'G'

    return board


def best_move(utilities, y, x):
    max_utility = utilities[y][x][0]
    move = 'up'

    if utilities[y][x][1] > max_utility:
        max_utility = utilities[y][x][1]
        move = 'right'

    if utilities[y][x][2] > max_utility:
        max_utility = utilities[y][x][2]
        move = 'down'

    if utilities[y][x][3] > max_utility:
        max_utility = utilities[y][x][3]
        move = 'left'

    if utilities[y][x][4] > max_utility:
        max_utility = utilities[y][x][4]
        move = 'give-up'

    return move


def train(board, goal_reward, pit_reward, move_reward, give_up_reward, epsilon, num_trials):
    moves = ['up', 'right', 'down', 'left', 'give-up']
    board_width = len(board[0])
    board_height = len(board)
    utilities = [[[99 for i in range(len(moves))] for j in range(board_width)] for k in range(board_height)]

    for trial_num in range(num_trials):
        x = randrange(board_width)
        y = randrange(board_height)
        while board[y][x] == "G" or board[y][x] == "P":
            x = randrange(board_width)
            y = randrange(board_height)
        move = ''
        prev_x = x
        prev_y = y
        prev_move = ''
        trial_complete = False
        reward = 0

        while not trial_complete:
            prev_x = x
            prev_y = y
            prev_move = move

            if random() < epsilon:
                move = moves[randrange(5)]
            else:
                move = best_move(utilities, y, x)

            if move == 'give-up':
                reward = give_up_reward
                trial_complete = True

            modifier = 'none'
            modifier_likeliness = random()
            if modifier_likeliness >= 0.9:
                modifier = 'double'
            elif modifier_likeliness >= 0.8:
                modifier = 'left'
            elif modifier_likeliness >= 0.7:
                modifier = 'right' 

            # stupid and ugly, but whatever
            if modifier == 'left':
                if move == 'up':
                    move = 'left'
                elif move == 'right':
                    move = 'up'
                elif move == 'down':
                    move = 'right'
                else:
                    move = 'down'
            elif modifier == 'right':
                if move == 'up':
                    move = 'right'
                elif move == 'right':
                    move = 'down'
                elif move == 'down':
                    move = 'left'
                else:
                    move = 'up'

            #TODO double movement not implemented here
            # perform movement
            if move == 'up':
                if y != 0:
                    y -= 1
            elif move == 'right':
                if x != board_width - 1:
                    x += 1
            elif move == 'down':
                if y != board_height - 1:
                    y += 1
            else:
                if x != 0:
                    x -= 1

            # check current state after move
            if board[y][x] == 'P':
                reward = pit_reward
                trial_complete = True
            elif board[y][x] == 'G':
                reward = goal_reward
                trial_complete = True
            else:
                reward = move_reward

    for y in range(board_height):
        print(board[y][:])
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
