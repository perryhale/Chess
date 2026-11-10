import pickle
import sys
import time

import chess
import jax
import jax.numpy as jnp

from library.data import board_to_vec837, board_to_mask4672, idx4672_to_move
from library.model import dense_neural_network

# unpack args
AUTO = "--auto" in sys.argv


if __name__ == "__main__":
	
	# load checkpoint
	with open("train_supervised.cache.pkl", 'rb') as f:
		CFG = pickle.load(f)
	policy_fn = jax.jit(lambda p,x,m: jnp.where(m, -jnp.inf, dense_neural_network(p, x)))
	board = chess.Board()
	
	# play game
	turn = True
	while not board.is_game_over():
		print(board, '\n')
		if board.turn == chess.WHITE and not AUTO:
			try:
				move = board.parse_san(input("White: "))
				board.push(move)
				turn = not turn
				print()
			except ValueError:
				print("Illegal move.\n")
		else:
			time.sleep(0.1 if AUTO else 0.5)
			x = jnp.asarray(board_to_vec837(board), dtype=jnp.float32)
			mask = jnp.asarray(board_to_mask4672(board))
			move = idx4672_to_move(int(policy_fn(CFG.policy, x, mask).argmax()), board)
			print(f"{'White' if turn else 'Black'}: {board.san(move)}\n")
			board.push(move)
			turn = not turn
	
	print(board.result())
