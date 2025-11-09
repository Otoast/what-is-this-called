import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

import math
import agent_logic
import copy

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "DaPunisher"
AGENT_NAME = "HurtPeopleHurtPeople"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)
    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opponent_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        my_color = 'R' if player_number == 1 else 'B'
        opponent_color = 'B' if player_number == 1 else 'R'

        boosts_remaining = my_agent.boosts_remaining
        opponent_boosts_remaining = opponent_agent.boosts_remaining

    converted_board = agent_logic._convert_board(state)

    best_value = -math.inf
    best_move = None

    for move_name, direction in agent_logic.MOVES.items():
        my_head = my_agent.trail[-1]
        opponent_head = opponent_agent.trail[-1]
        print(f"Trying move: {move_name}")

        # Skip boost if you have no boosts left
        if ":BOOST" in move_name and boosts_remaining <= 0:
            print("Skipping boost (no boosts left)")
            continue

        # attempt_move now handles both normal and boost moves correctly
        new_grid, my_new_head = agent_logic.attempt_move(converted_board, my_head, my_color, direction)
        if my_new_head is None:
            print("Invalid move")
            continue

        new_my_boosts_remaining = boosts_remaining - 1 if ":BOOST" in move_name else boosts_remaining

        value = agent_logic.minimax(
            new_grid, my_new_head, opponent_head, my_color, opponent_color,
            depth=1, alpha=-math.inf, beta=math.inf, maximizing=False,
            red_boosts_remaining=new_my_boosts_remaining, blue_boosts_remaining=opponent_boosts_remaining
        )

        if value > best_value:
            best_value = value
            best_move = move_name

    print(f"Selected move: {best_move}")
    return jsonify({"move": best_move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
