def get_greedy_decision(player, players, apples):
    # Player position
    position = players[player-1]["location"]
    # Player orientation
    orientation = players[player-1]["orientation"]
    print(position)
    if len(apples) < 1:
        return 'move'
    for apple in apples:
        closest = None
        distance = 1000
        for apple in apples:
            d = abs(position[0]-apple[0])+abs(position[1]-apple[1])
            if d < distance:
                closest = apple
                distance = d
    if orientation == 'right':
        if apple[0] > position[0]:
            return 'move'
        else:
            if apple[1] > position[1]:
                return 'right'
            else:
                return 'left'
    if orientation == 'left':
        if apple[0] < position[0]:
            return 'move'
        else:
            if apple[1] < position[1]:
                return 'right'
            else:
                return 'left'
    if orientation == 'up':
        if apple[1] < position[1]:
            return 'move'
        else:
            if apple[0] > position[0]:
                return 'right'
            else:
                return 'left'
    if orientation == 'down':
        if apple[1] > position[1]:
            return 'move'
        else:
            if apple[0] < position[0]:
                return 'right'
            else:
                return 'left'
