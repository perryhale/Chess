import time
import sys

import numpy as np
import chess

from library.data import (
	move_to_idx4672,
	idx4672_to_move,
	board_to_vec837,
	vec837_to_board,
	stream_pgn_to_supervised,
)


def test_action_encoder_decoder_roundtrip(verbose=False):
	
	board = chess.Board()
	move = chess.Move.from_uci("b2b4")
	move_enc = move_to_idx4672(move)
	move_dec = idx4672_to_move(move_enc, board)
	move_differ = move.uci() != move_dec.uci()
	
	if verbose:
		print(f"\"{move.uci()}\":{move_enc}:\"{move_dec.uci()}\"")
		print(f"{move_differ=}")
	if move_differ:
		raise AssertionError(f"{move_differ=}")


def test_action_encoder_decoder_implicit_promotion(verbose=False):
	
	board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
	move = chess.Move.from_uci("a7a8")
	move_enc = move_to_idx4672(move)
	move_dec = idx4672_to_move(move_enc, board) ###! decoder handles implicit promotion
	move_promoted = move_dec.uci() == "a7a8q"
	
	if verbose:
		print(f"({move.uci()}, {board.fen()})->{move_dec.uci()}")
		print(f"{move_promoted=}")
	if not move_promoted:
		raise AssertionError(f"{move_promoted=}, {move_dec.uci()=}")


def test_state_encoder_decoder_roundtrip(verbose=False):
	
	board = chess.Board()
	board_enc = board_to_vec837(board)
	board_dec = vec837_to_board(board_enc)
	board_differ = board.fen() != board_dec.fen()
	
	if verbose:
		print(f"{board}")
		print(f"{''.join(['|' if bool(s) else '.' for s in board_enc])}")
		print(f"{board_dec}")
		print(f"{board_differ=}")
	if board_differ:
		raise AssertionError(f"{board_differ=}")


def test_data_loader_block_uniqueness(verbose=False):
	
	lichess_pgn = open("data/lichess_db_standard_rated_2017-03.pgn", "r", encoding="utf-8")
	data_x, data_y = stream_pgn_to_supervised(lichess_pgn, max_moves=50) # ; import pdb; pdb.set_trace() ###! numpy array takes up more memory than lists
	data_x1, data_y1 = stream_pgn_to_supervised(lichess_pgn, max_moves=50)
	lichess_pgn.close()
	
	data_x_differ = bool(np.any(data_x != data_x1))
	data_y_differ = bool(np.any(data_y != data_y1))
	
	if verbose:
		print(f"{data_x.shape=}, {data_x.dtype=}")
		print(f"{data_y.shape=}, {data_y.dtype=}")
		print(f"{data_x1.shape=}, {data_x1.dtype=}")
		print(f"{data_y1.shape=}, {data_y1.dtype=}")
		print(f"{data_x_differ=}")
		print(f"{data_y_differ=}")
	if not (data_x_differ or data_y_differ or data_yh_differ):
		raise AssertionError(f"{data_x_differ=}, {data_y_differ=}, {data_yh_differ=}")


def test_data_loader_mem_safety(verbose=False):
	lichess_pgn = open("data/lichess_db_standard_rated_2017-03.pgn", "r", encoding="utf-8")
	try:
		_ = stream_pgn_to_supervised(lichess_pgn, max_memory=0.3, debug=10_000 if verbose else -1, start_time=T0)
		raise AssertionError(f"stream_pgn_to_supervised unsafe memory") ###! assumes PGN larger than memory
	except RuntimeError as e:
		if verbose:
			print(e)
	finally:
		lichess_pgn.close()


if __name__ == "__main__":
	
	T0 = time.time()
	print(f"[{time.time()-T0:.2f}s]")
	
	verbose = len(sys.argv) > 1 and sys.argv[1] == "--verbose"
	
	for test_fn in [
		test_action_encoder_decoder_roundtrip,
		test_action_encoder_decoder_implicit_promotion,
		test_state_encoder_decoder_roundtrip,
		test_data_loader_block_uniqueness,
		test_data_loader_mem_safety
	]:
		_pass = False
		try:
			test_fn(verbose=verbose)
			_pass = True
		except AssertionError as e:
			print(e)
		print(f"[{time.time()-T0:.2f}s] {test_fn}: {'pass' if _pass else 'fail'}")
