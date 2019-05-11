import random

class GreedyAgent:
    """
    A greedy agent collects the nearest apple
    """

    def __init__(self):
        self.goal = None
        self.dt_goal = 50000

    def next_action(self, player, players, apples):
        position = players[player-1]["location"]
        orientation = players[player-1]["orientation"]

        if self.goal == position:
            self.goal = None

        self._verify_goal(position, apples)

        if len(apples) < 1:
            return self._empty_move(orientation)

        nm = self._route_to_goal(position, orientation)
        return nm

    @staticmethod
    def _empty_move(orientation):
        """
        :return: If agent not horizontally oriented - 'left'
                 Else - 'move'
        """
        if orientation == 'up' or orientation == 'down':
            return 'left'
        else:
            return random.choices(population=['move','right'], weights=[0.95,0.05])[0]

    def _verify_goal(self, position, apples):
        closest_apple, distance = self._get_closest_apple(position, apples)
        if self.goal is None:
            self.goal, self.dt_goal = closest_apple, distance
        elif self.goal != closest_apple and self.dt_goal > distance:
            self.goal, self.dt_goal = closest_apple, distance

    def _get_closest_apple(self, position, apples):
        rdt = 50000
        rap = None
        for apple in apples:
            dt,_ = self._calc_torus_distance(position, apple)
            if dt < rdt:
                rdt = dt
                rap = apple
        return rap, rdt

    @staticmethod
    def _calc_torus_distance(start, end):
        dx = abs(end[0]-start[0])
        dy = abs(end[1]-start[1])
        _x = 0; _y = 0
        if dx > 18:
            dx = 36-dx
            _x = 1
        if dy > 8:
            dy = 16-dy
            _y = 1
        return dx+dy, (_x,_y)

    def _route_to_goal(self, position, orientation):
        """
        Return the next move on the route to the goal
        :param position: Current position
        :param orientation: Current orientation
        :return: One of 'move', 'left', 'right'
        """
        _, (_x,_y) = self._calc_torus_distance(position, self.goal)
        move = None

        if orientation == 'up':
            if self.goal[1] > position[1] and _y > 0:
                move = 'move'
            elif self.goal[1] < position[1] and _y < 1:
                move = 'move'
            elif self.goal[0] > position[0]:
                if _x > 0:
                    move = 'left'
                else:
                    move = 'right'
            else:
                if _x > 0:
                    move = 'right'
                else:
                    move = 'left'

        if orientation == 'down':
            if self.goal[1] < position[1] and _y > 0:
                move = 'move'
            elif self.goal[1] > position[1] and _y < 1:
                move = 'move'
            elif self.goal[0] > position[0]:
                if _x > 0:
                    move = 'right'
                else:
                    move = 'left'
            else:
                if _x > 0:
                    move = 'left'
                else:
                    move = 'right'

        if orientation == 'right':
            if self.goal[0] < position[0] and _x > 0:
                move = 'move'
            elif self.goal[0] > position[0] and _x < 1:
                move = 'move'
            elif self.goal[1] > position[1]:
                if _y > 0:
                    move = 'left'
                else:
                    move = 'right'
            else:
                if _y > 0:
                    move = 'right'
                else:
                    move = 'left'

        if orientation == 'left':
            if self.goal[0] > position[0] and _x > 0:
                move = 'move'
            elif self.goal[0] < position[0] and _x < 1:
                move = 'move'
            elif self.goal[1] > position[1]:
                if _y > 0:
                    move = 'right'
                else:
                    move = 'left'
            else:
                if _y > 0:
                    move = 'left'
                else:
                    move = 'right'

        return move

    def save(self): pass