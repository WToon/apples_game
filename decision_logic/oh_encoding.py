import numpy as np
import time


def encode_state(player, players="", apples="", board_size=(15,15)):
    """
    One-Hot encode the state as retrieved from the game to input for CNN
    :param players: list of other players 'location' and 'orientation'
    :param apples: list of apple locations
    :return: np.mat with one-hot encoded values
    """
    player_location = players[player-1].get('location')
    dx = 8-player_location[0]
    dy = 8-player_location[1]
    # One-Hot mapping dict
    oh_mapping = {'empty':  np.array([1, 0, 0, 0, 0, 0]),
                  'apple':  np.array([0, 1, 0, 0, 0, 0]),
                  'up':     np.array([0, 0, 1, 0, 0, 0]),
                  'down':   np.array([0, 0, 0, 1, 0, 0]),
                  'left':   np.array([0, 0, 0, 0, 1, 0]),
                  'right':  np.array([0, 0, 0, 0, 0, 1])}

    # Initialise an empty board_state
    board_state = [[oh_mapping["empty"] for i in range(board_size[0])] for i in range(board_size[1])]

    # Load apples into board
    for location in apples:
        location[0] = (location[0]+dx)%15
        location[1] = (location[1]+dy)%15
        board_state[location[0]][location[1]] = oh_mapping["apple"]

    # Load other players into board
    for worm in players:
        location = worm["location"]
        if location == ["?","?"]: continue
        else:
          location[0] = (location[0] + dx)%15
          location[1] = (location[1] + dy)%15
          board_state[location[0]][location[1]] = oh_mapping[worm["orientation"]]
    return board_state

