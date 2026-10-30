import time
import chess.pgn

T0 = time.time()

def count_moves(pgn_path):
	count = 0
	print(f"[{time.time()-T0:.2f}s] {count} ")
	with open(pgn_path, encoding="utf-8") as pgn:
		while True:
			game = chess.pgn.read_game(pgn)
			if game is None:
				break
			for move in game.mainline_moves():
				count += 1
				if count % 1_000 == 0:
					print(f"[{time.time()-T0:.2f}s] {count}")

num_moves = count_moves("data/lichess_db_standard_rated_2017-03.pgn")
print(f"[{time.time()-T0:.2f}s] {num_moves=}")
