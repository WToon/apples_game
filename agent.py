#!/usr/bin/env python3
# encoding: utf-8
"""
agent.py
Template for the Machine Learning Project course at KU Leuven (2017-2018)
of Hendrik Blockeel and Wannes Meert.
Copyright (c) 2018 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio
import websockets
import json
import decision_logic.greedy as greedy
import decision_logic.random_agent as random
import decision_logic.dqn as dqn
import numpy as np


logger = logging.getLogger(__name__)
games = {}
agentclass = None
EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 750
agenttype = "greedy"

ORIENTATION = ["up","left","right","down"]
class Agent:
    def __init__(self, player):
        self.player = {player}
        self.ended = False
        self.agent = dqn.DQNAgent()
        self.loaded = False

    def register_action(self, players, apples):
        self.players = players
        self.apples = apples

    def next_action(self, player):
      if not self.loaded:
        self.agent.load('decision_logic/model_output/dqn_agent/Weights')
      nm = self.agent.next_action(player,self.players,self.apples,False)
        # if agenttype == 'greedy':
        #     nm = greedy.get_greedy_decision(player, self.players, self.apples)
        # elif agenttype == 'random':
        #     nm = random.get_random_decision(player, self.players, self.apples)
      return nm

    def end_game(self):
        self.ended = True


## MAIN EVENT LOOP
async def handler(websocket, path):
    logger.info("Start listening")
    try:
        async for msg in websocket:
            logger.info("< {}".format(msg))
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            game = msg["game"]
            answer = None

            # Initialize game
            if msg["type"] == "start":
                games[msg["game"]] = agentclass(msg["player"])
                # The first player generates a starting move
                if msg["player"] == 1:
                    # Start the game
                    games[game].register_action(msg["players"], msg["apples"])
                    nm = games[game].next_action(1)
                    if nm is None:
                        # Game over
                        logger.info("Generation of start move failed")
                        continue
                    answer = {
                        'type': 'action',
                        'action': nm,
                    }
                else:
                    # Other players wait for their turn
                    answer = None

            # Respond to actions
            elif msg["type"] == "action":
                # It is this agents turn TODO support multiple players in same agent
                if msg["nextplayer"] in games[game].player:
                    games[game].register_action(msg["players"], msg["apples"])
                    nm = games[game].next_action(msg["nextplayer"])
                    answer = {
                        'type': 'action',
                        'action': nm
                    }
                else:
                    answer = None

            elif msg["type"] == "end":
                # End the game
                games[msg["game"]].end_game()
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                await websocket.send(json.dumps(answer))
                logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


## COMMAND LINE INTERFACE

def main(argv=None):
    global agentclass, agenttype
    types = ['random', 'greedy']
    quiet = 0; verbose = 0

    parser = argparse.ArgumentParser(description='Start agent to play the Apples game')
    parser.add_argument('--verbose', '-v', action='count', default=verbose, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=quiet, help='Quiet output')
    parser.add_argument('--agenttype', '-a', action='store', default=1, help='Agent decision type')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentclass = Agent
    agenttype = types[int(args.agenttype)]
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
