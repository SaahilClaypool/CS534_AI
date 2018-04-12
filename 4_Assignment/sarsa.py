from random import *
import matplotlib.pyplot as plt
import csv
import sys


def load_board(filename: str = "./input_hw4.csv"):
    """
    Set up initial board based on input file
    """
    board = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            board.append(row)
    return board


def best_move(utilities, y, x):
    """
    Compute the best move to take for a given row/column in from the current utilities
    """
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
        move = 'give-up'

    return move


def train(board, goal_reward, pit_reward, move_reward, give_up_reward, epsilon, num_trials):
    """
    Train an agent using SARSA to generate utilities for a given board with various settings.

    Output is a utility matrix where each [row][col] has an associated list of expected reward for each move.
    """
    # step size
    alpha = 0.1
    # influence of next state/action's utility
    gamma = 0.9

    moves = ['up', 'right', 'down', 'left', 'give-up']
    moves_smol = ['^', '>', 'v', '<', 'G']

    # initialize utility values
    board_width = len(board[0])
    board_height = len(board)
    utilities = [[[0 for i in range(len(moves))] for j in range(board_width)] for k in range(board_height)]
    for row in range(board_height):
        for col in range(board_width):
            if board[row][col] == 'O':
                utilities[row][col] = [goal_reward for i in range(len(moves))]
            elif board[row][col] == 'P':
                utilities[row][col] = [pit_reward for i in range(len(moves))]

    # rewards will be used to track the average reward from the trials - mostly for visualization & analysis
    rewards = []

    # run through trials
    for trial_num in range(num_trials):
        # slowly decrease epsilon so that the agent doesn't keep exploring forever
        decreasing_epsilon = epsilon * ((num_trials - trial_num) / num_trials)

        # randomly intialize initial x y locatoin
        x = randrange(board_width)
        y = randrange(board_height)
        while board[y][x] == 'O' or board[y][x] == "P":
            x = randrange(board_width)
            y = randrange(board_height)
        move = ''
        new_x = x
        new_y = y
        trial_complete = False
        reward = 0
        running_reward = 0
        max_moves = 100

        first_move = True
        # run trial until goal reached, pit reached, given up, or maximum moves has been reached
        while not trial_complete and max_moves > 0:
            max_moves -= 1
            # keep track of previous positions/rewards
            prev_reward = reward
            prev_x = x
            prev_y = y
            x = new_x
            y = new_y
            prev_move = move

            if random() < decreasing_epsilon:
                # random exploration
                move = moves[randrange(5)]
            else:
                # choose current best move
                move = best_move(utilities, y, x)

            # check by random chance whether or not to cause the agent to take a modified move
            # 10% chance each of turning right, left, or taking a double move
            modifier = 'none'
            modifier_likeliness = random()
            if modifier_likeliness >= 0.9:
                modifier = 'double'
            elif modifier_likeliness >= 0.8:
                modifier = 'left'
            elif modifier_likeliness >= 0.7:
                modifier = 'right'

            # modify the original chosen movement, if applicable.
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

            # apply movement for the chosen (and possibly modified) movement to get the new position and reward
            new_x, new_y, reward, trial_complete = move_fun(mod_move, x, y, board, board_width,
                                                            board_height, give_up_reward, pit_reward,
                                                            goal_reward, move_reward, reward)
            # special case if a double move modifier was applied
            if modifier == 'double' and not trial_complete:
                new_x, new_y, reward, trial_complete = move_fun(mod_move, new_x, new_y, board, board_width,
                                                                board_height, give_up_reward, pit_reward,
                                                                goal_reward, move_reward, reward)
            # add on to the running reward, for the case of giving up
            running_reward += reward

            move_index = 0
            prev_move_index = 0
            # get the index of current and previous move
            for i in range(len(moves)):
                if moves[i] == move:
                    move_index = i
                if moves[i] == prev_move:
                    prev_move_index = i
            if not first_move:
                # for the previous state/action, update based on the reward it got and the next state/action's utility
                utilities[prev_y][prev_x][prev_move_index] += alpha * (
                        prev_reward + gamma * utilities[y][x][move_index] -
                        utilities[prev_y][prev_x][prev_move_index])
            first_move = False

        # end trial
        move_index = 0
        for i in range(len(moves)):
            if moves[i] == move:
                move_index = i
        # final update for moves that ended the trial, i.e. gave up, got to goal, or went into pit
        if move == 'give-up':
            utilities[y][x][4] += alpha * ((1 + gamma) * running_reward - utilities[y][x][4])
        else:
            utilities[y][x][move_index] += alpha * ((1 + gamma) * reward - utilities[y][x][move_index])
        rewards.append(reward)

    # Agent finished training: print out results
    print("Computed best moves for the agent:")
    for y in range(board_height):
        for x in range(board_width):
            if board[y][x] != 'O' and board[y][x] != 'P':
                maxind = utilities[y][x].index(max(utilities[y][x]))
                print(moves_smol[maxind], end=" ")
            else:
                print(board[y][x], end=" ")
        print("")
    print("Expected rewards for moves:")
    for y in range(board_height):
        for x in range(board_width):
            print("{:+.6f}".format(max(utilities[y][x])), end=" ")
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


def main():
    # get parameters from input
    goal_reward = float(sys.argv[1])
    pit_reward = float(sys.argv[2])
    move_reward = float(sys.argv[3])
    give_up_reward = float(sys.argv[4])
    num_trials = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    print("Goal reward:",goal_reward, " Pit reward:", pit_reward, " Move reward:", move_reward, " Giveup reward:", give_up_reward)
    print("Running on ", num_trials, " trials with initial epsilon ", epsilon)

    board = load_board()
    trained_utilities, rewards = train(board, goal_reward, pit_reward, move_reward, give_up_reward, epsilon, num_trials)

    plot(rewards)


def plot(rewards):
    offset = 0
    points = []
    group_size = 20
    while (offset + group_size < len(rewards)):
        points.append(sum(rewards[offset: offset + group_size]) / group_size)
        offset += group_size
    plt.plot(points)
    plt.show()


if __name__ == "__main__":
    main()
