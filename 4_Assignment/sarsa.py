from random import *
import matplotlib.pyplot as plt
import csv
import sys

def load_board(filename: str = "./input_hw4.csv"):
    board = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            board.append(row)
    return board

def initialize_board():
    board = [['' for i in range(7)] for j in range(6)]

    board[2][2] = 'P'
    board[2][3] = 'P'
    board[3][1] = 'P'
    board[3][5] = 'P'
    board[4][2] = 'P'
    board[4][3] = 'P'
    board[4][4] = 'P'

    board[3][2] = 'O'

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
    alpha = 0.5
    gamma = 0.1
    moves = ['up', 'right', 'down', 'left', 'give-up']
    moves_smol = ['^', '>', 'v', '<', 'G']
    board_width = len(board[0])
    board_height = len(board)
    utilities = [[[0 for i in range(len(moves))] for j in range(board_width)] for k in range(board_height)]

    rewards = []
    trial_num = 0
    for trial_num in range(num_trials):
        x = randrange(board_width)
        y = randrange(board_height)
        while board[y][x] == 'O' or board[y][x] == "P":
            x = randrange(board_width)
            y = randrange(board_height)
        move = ''
        prev_x = x
        prev_y = y
        prev_move = ''
        trial_complete = False
        reward = 0

        first_move = True
        while not trial_complete:
            prev_x = x
            prev_y = y
            prev_move = move

            if random() < epsilon:
                move = moves[randrange(5)]
            else:
                move = best_move(utilities, y, x)

            modifier = 'none'
            modifier_likeliness = random()
            if modifier_likeliness >= 0.9:
                modifier = 'double'
            elif modifier_likeliness >= 0.8:
                modifier = 'left'
            elif modifier_likeliness >= 0.7:
                modifier = 'right'

            # stupid and ugly, but whatever
            mod_move = move
            if modifier == 'left':
                if move == 'up':
                    mod_move = 'left'
                elif move == 'right':
                    mod_move = 'up'
                elif move == 'down':
                    mod_move = 'right'
                else:
                    mod_move = 'down'
            elif modifier == 'right':
                if move == 'up':
                    mod_move = 'right'
                elif move == 'right':
                    mod_move = 'down'
                elif move == 'down':
                    mod_move = 'left'
                else:
                    mod_move = 'up'
            if move == 'give-up':
                mod_move = 'give-up'

            x, y, reward, trial_complete = move_fun(mod_move, x, y, board, board_width, \
                                            board_height, give_up_reward, pit_reward, \
                                            goal_reward, move_reward, reward)
            if(modifier == 'double' and not trial_complete):
                x, y, reward, trial_complete = move_fun(mod_move, x, y, board, board_width, \
                                                board_height, give_up_reward, pit_reward, \
                                                goal_reward, move_reward, reward)
            # reward += cur_reward
            move_index = 0
            prev_move_index = 0
            for i in range(len(moves)):
                if moves[i] == move:
                    move_index = i
                if moves[i] == prev_move:
                    prev_move_index = i
            if not first_move:
                utilities[prev_y][prev_x][prev_move_index] += alpha * (reward + gamma*utilities[y][x][move_index] - utilities[prev_y][prev_x][prev_move_index])

            first_move = False

        # end trial
        utilities[y][x][move_index] += alpha * reward
        trial_num += 1
        rewards.append(reward)

    for y in range(board_height):
        for x in range(board_width):
            if board[y][x] != 'O' and board[y][x] != 'P':
                maxind = utilities[y][x].index(max(utilities[y][x]))
                print(moves_smol[maxind], end = " ")
            else:
                print(board[y][x], end = " ")
        print("")
    return utilities, rewards

def move_fun(move, x, y, board, board_width, board_height, \
         give_up_reward, pit_reward, goal_reward, move_reward, reward):
    """
    returns:
        x, y, trial_complete, reward
    """
    trial_complete = False
    reward = 0
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
    elif move == 'left':
        if x != 0:
            x -= 1
    elif move == 'give-up':
        reward = give_up_reward
        trial_complete = True

    # check current state after move
    if board[y][x] == 'P':
        reward = pit_reward
        trial_complete = True
    elif board[y][x] == 'O':
        reward = goal_reward
        trial_complete = True

    if move != 'give-up':
        reward += move_reward
    return x, y, reward, trial_complete
    # uh oh SpaghettiOs

def main():
    goal_reward = float(sys.argv[1])
    pit_reward = float(sys.argv[2])
    move_reward = float(sys.argv[3])
    give_up_reward = float(sys.argv[4])
    num_trials = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    print(goal_reward, pit_reward, give_up_reward, num_trials)

    # board = initialize_board()
    board = load_board()
    trained_utilities, rewards = train(board, goal_reward, pit_reward, move_reward, give_up_reward, epsilon, num_trials)

    # plot(rewards)

def plot(rewards):
    offset = 0
    points = []
    group_size = 100
    while(offset + group_size < len(rewards)):
        points.append(sum(rewards[offset: offset + group_size] ) / group_size)
        offset += group_size
    plt.plot(points)
    plt.show()


if __name__ == "__main__":
    main()
