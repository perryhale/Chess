# core
import time
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import deque

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# environment
import chess # python-chess
import chess.pgn

# logging
from tqdm import tqdm

"""

Environment <--> Model
______________________

chess.Board <--> E837       | state encoding
chess.Move  <--> E4672      | action encoding
chess.Board ---> M4672      | legal action mask

Data loader
______________________

|                       |   - all (state, action) pairs
|       |       |       |   - load blocks into memory
| | | | | | | | | | | | |   - yield batches from memory

1. open pgn textio stream
2. stream #max_moves (state+mask, action) pairs into memory
3. yield batches from memory
4. go to (2) when no more new batches and not stopping condition

"""


### utility

# type: (int, int) -> jax.Array
def one_hot(class_idx, n_classes):
	x = jnp.zeros(n_classes, dtype=jnp.bool_)
	x = x.at[class_idx].set(True)
	return x

# type: (Any, bool, str) -> tqdm.tqdm
def tqdm_option(_iter, _bool, desc=None):
	return tqdm(_iter, desc=desc) if _bool else _iter


### AlphaZero format action encoder / decoder (python-chess<-->python)

# define inter/+cardinal directions, (8*7) 0-55
E4672_DIR = [
	(0, 1),   # N
	(1, 1),   # NE
	(1, 0),   # E
	(1, -1),  # SE
	(0, -1),  # S
	(-1, -1), # SW
	(-1, 0),  # W
	(-1, 1),  # NW
]

# define knight translations, (8) 56-63
E4672_KNIGHT = [
	(1, 2),   # up right
	(2, 1),   # right up
	(2, -1),  # right down
	(1, -2),  # down right
	(-1, -2), # down left
	(-2, -1), # left down
	(-2, 1),  # left up
	(-1, 2),  # up left
]

# define underpromotions, directional: (3*3) 64-72
E4672_UNDER_PROM = [chess.KNIGHT, chess.BISHOP, chess.ROOK] # [1, 2, 3]


# type: (chess.Move) -> int
def move_to_idx4672(move):
	
	# determine to/from file/rank (col/row) indices
	r0, f0 = divmod(move.from_square, 8)
	r1, f1 = divmod(move.to_square, 8)
	df = f1 - f0
	dr = r1 - r0
	
	# handle promotion move offset
	if move.promotion and move.promotion in E4672_UNDER_PROM:
		df_signed = df if dr > 0 else -df # if black pawn flip file sign
		class_offset = 64 + (df_signed + 1) * 3 + (move.promotion - chess.KNIGHT)
	
	# handle knight move offset
	elif (df, dr) in E4672_KNIGHT:
		class_offset = 56 + E4672_KNIGHT.index((df, dr))
	
	# handle standard move offset
	else:
		distance = max(abs(df), abs(dr))
		direction = (np.sign(df), np.sign(dr))
		class_offset = E4672_DIR.index(direction) * 7 + distance - 1
	
	# determine class index and return
	return (r0 * 8 + f0) * 73 + class_offset


# type: (int, chess.Board) -> chess.Move
def idx4672_to_move(idx4672, board):
	
	# derive from-rank/from-file/action
	from_square, action = divmod(idx4672, 73)
	r0, f0 = divmod(from_square, 8)
	
	# initialise promotion class (int)
	promotion_class = None # no promotion
	
	# decode regular inter/+cardinal translation
	if action < 56:
		dir_idx, distance = divmod(action, 7)
		df, dr = E4672_DIR[dir_idx]
		df *= distance + 1
		dr *= distance + 1
	
	# decode knight translation
	elif action < 64:
		df, dr = E4672_KNIGHT[action - 56]
	
	# decode under-promotion translation and character
	else:
		dir_class, piece_idx = divmod(action - 64, 3)
		dr = 1 if r0 == 6 else -1 # up if white, down if black
		df = (dir_class - 1) * dr # right if white, left if black
		promotion_class = E4672_UNDER_PROM[piece_idx]
	
	# derive to-rank/to-file
	r1 = r0 + dr
	f1 = f0 + df
	to_square = r1 * 8 + f1
	
	# handle implicit promotion (directly from state encoding)
	is_pawn = board.piece_at(from_square).piece_type == chess.PAWN
	is_prom = r1 in (0, 7)
	is_impl = promotion_class == None
	if is_pawn and is_prom and is_impl:
		promotion_class = chess.QUEEN
	
	# construct chess.Move and return
	return chess.Move(from_square, to_square, promotion=promotion_class)


### state encoder / decoder / action mask (python-chess<-->numpy)

# type: (chess.Board) -> np.ndarray
def board_to_vec837(board):
	
	# init array
	vec837 = np.zeros((837), dtype=np.bool_)
	
	# init constants
	board_offset = 64 * 12
	
	# set position state OHE vectors: (classes 0-5:white, 6-11:black) [0:768] +768
	for pos_idx, piece in board.piece_map().items():
		class_idx = piece.piece_type - 1 # classes: [pawn, knight, bishop, rook, queen, king]
		if not piece.color:
			class_idx += 6
		vec837[pos_idx * 12 + class_idx] = True
	
	# set turn state: (values F:black, T:white) [768:769] +1
	vec837[board_offset + 0] = board.turn
	
	# set castling states: (values F:unavailable, T:available) [769:773] +4
	vec837[board_offset + 1] = board.has_kingside_castling_rights(chess.WHITE)
	vec837[board_offset + 2] = board.has_queenside_castling_rights(chess.WHITE)
	vec837[board_offset + 3] = board.has_kingside_castling_rights(chess.BLACK)
	vec837[board_offset + 4] = board.has_queenside_castling_rights(chess.BLACK)
	
	# set en-passant states: (values 0:unavailable, 1:available) [773:837] +64
	if board.ep_square is not None:
		vec837[board_offset + 5 + board.ep_square] = True
	
	return vec837


# type: (np.ndarray) -> chess.Board
def vec837_to_board(vec837):
	
	# init empty board
	board = chess.Board(None)
	board_offset = 64 * 12
	
	# decode and set pieces
	squares_ohe = vec837[:board_offset].reshape(64, 12)
	squares_class = np.argmax(squares_ohe, axis=-1) ###! (valid for) bool
	squares_full = np.take_along_axis(squares_ohe, squares_class[..., None], axis=-1).squeeze(-1) ###! bool
	squares_type = (squares_class % 6) + 1
	squares_color = squares_class < 6
	for sq, (f, t, c) in enumerate(zip(squares_full, squares_type, squares_color)):
		if bool(f):
			board.set_piece_at(sq, chess.Piece(int(t), bool(c)))
	
	# decode turn
	board.turn = bool(vec837[board_offset + 0])
	
	# decode castling rights
	castling_rights = chess.BB_EMPTY ###! 4 bit int
	
	for color, kingside_right_idx, queenside_right_idx in [
		(chess.WHITE, board_offset + 1, board_offset + 2),
		(chess.BLACK, board_offset + 3, board_offset + 4),
	]:
		
		# get home rank and king square (check for king)
		color_rank = 0 if color == chess.WHITE else 7
		king_square = board.king(color)
		if king_square is None:
			continue
		
		# get rooks on home rank
		rook_squares = board.pieces(chess.ROOK, color) & chess.BB_RANKS[color_rank] # bitwise and
		
		for right_idx, is_kingside in [
			(kingside_right_idx, True),
			(queenside_right_idx, False),
		]:
			if not vec837[right_idx]: # (check right)
				continue
			
			# get rooks to right of king if kingside else left of king if queenside
			rook_candidates = [sq for sq in rook_squares if (sq % 8 > king_square % 8) == is_kingside]
			
			# update castling rights with bitboard corresponding to rightmost rook of kingside else leftmost rook
			if rook_candidates:
				rook_sq = max(rook_candidates) if is_kingside else min(rook_candidates)
				castling_rights |= chess.BB_SQUARES[rook_sq] # bitwise or
	
	board.castling_rights = castling_rights
	
	# decode en-passant square
	ep_squares = vec837[board_offset + 5:]
	ep_square = np.argmax(ep_squares)
	board.ep_square = int(ep_square) if ep_squares[ep_square] else None
	
	return board


# type: (np.ndarray) -> np.ndarray
def board_to_mask4672(board):
	mask = np.ones(4672, dtype=np.bool_)
	for move in board.legal_moves:
		mask[move_to_idx4672(move)] = False
	return mask


### data loader (python-->jax)

# type: (TextIO, int, float, int, float) -> tuple[jax.Array, jax.Array]
def stream_pgn_to_supervised_mono(pgn_textio, max_moves=-1, max_memory=0.85, trace=-1, start_time=None):
	states = []
	actions = []
	move_count = 0
	max_memory_percent = 100 * max_memory
	if start_time is None:
		start_time = time.time()
	
	# stream up to #max_games games from lichess PGN file
	while max_moves < 0 or move_count < max_moves:
		
		# mem safety
		memory_percent = psutil.virtual_memory().percent
		if max_memory_percent >= 0 and memory_percent > max_memory_percent:
			raise RuntimeError(f"memory_percent > max_memory_percent, {memory_percent} > {max_memory_percent}")
		
		# initialise game (check EOF)
		game = chess.pgn.read_game(pgn_textio)
		if game is None:
			break
		board = game.board()
		
		# encode all states and moves from game (check max_moves)
		for move in game.mainline_moves():
			if max_moves >= 0 and move_count >= max_moves:
				break
			
			###! pre-compute mask and pack into input
			x = jnp.concatenate([board_to_vec837(board), board_to_mask4672(board)])
			y = move_to_idx4672(move)
			
			states.append(x)
			actions.append(y)
			
			board.push(move)
			move_count += 1
			
			if trace > 0 and move_count % trace == 0:
				print(f"[{time.time()-start_time:.2f}s] Load {move_count} ({memory_percent}%)")
	
	# construct jax arrays
	data_x = jnp.asarray(states, dtype=jnp.bool_)
	data_y = jnp.asarray(actions, dtype=jnp.int32)
	
	return data_x, data_y


# type: (str, list[chess.Move]) -> tuple[np.ndarray, np.ndarray]
def _encode_game_fn(fen, moves):
	board = chess.Board(fen) ###! send fen string, avoids board.copy()
	data_x = np.empty((len(moves), 5509), dtype=np.bool_)
	data_y = np.empty(len(moves), dtype=np.int32)
	for idx, move in enumerate(moves):
		data_x[idx, :837] = board_to_vec837(board)
		data_x[idx, 837:] = board_to_mask4672(board)
		data_y[idx] = move_to_idx4672(move)
		board.push(move)
	return data_x, data_y


# type: (TextIO, int, float, int, float | None, int, int) -> tuple[jax.Array, jax.Array]
def stream_pgn_to_supervised(pgn_textio, max_moves=-1, max_memory=0.85, trace=-1, start_time=None, num_workers=4, max_queue_per_worker=2):
	
	# init structures for requested block or unbounded moves
	data_x = np.empty((max(0, max_moves), 5509), dtype=np.bool_)
	data_y = np.empty(max(0, max_moves), dtype=np.int32)
	chunks_x, chunks_y = [], []
	
	# counters
	move_count = 0
	move_trace = trace
	
	# constants
	max_memory_percent = 100 * max_memory
	if start_time is None:
		start_time = time.time()
	
	# type: () ~> tuple[str, list[chess.Move]]
	def producer():
		schedule_count = 0
		while max_moves < 0 or schedule_count < max_moves:
			
			# mem safety
			memory_percent = psutil.virtual_memory().percent
			if max_memory_percent >= 0 and memory_percent > max_memory_percent:
				raise RuntimeError(f"memory_percent > max_memory_percent, {memory_percent} > {max_memory_percent}")
			
			# initialise game (check EOF)
			game = chess.pgn.read_game(pgn_textio)
			if game is None:
				break
			
			# filter games by ELO rating
			###! https://georgelichess.com/chess-rating-percentiles
			elo_threshold = 1800 # top quartile (25% players)
			###! if both players are below threshold continue in loop
			if max(int(game.headers.get('WhiteElo', 0)), int(game.headers.get('BlackElo', 0))) < elo_threshold:
				continue
			
			# extract moves under max_moves
			moves = list(game.mainline_moves())
			if max_moves >= 0:
				moves = moves[:max_moves-schedule_count]
			schedule_count += len(moves)
			
			# yield available moves
			if moves:
				yield game.board().fen(), moves
	
	# type: (np.ndarray, np.ndarray, int, int) -> tuple[int, int]
	def consume(xs, ys, _move_count, _move_trace):
		
		# insert encoder results
		start_move_count = _move_count
		stop_move_count = _move_count + xs.shape[0]
		if max_moves < 0:
			chunks_x.append(xs)
			chunks_y.append(ys)
		else:
			data_x[start_move_count:stop_move_count] = xs
			data_y[start_move_count:stop_move_count] = ys
		
		# trace
		if trace > 0 and stop_move_count >= _move_trace:
			print(f"[{time.time()-start_time:.2f}s] Load {stop_move_count} ({psutil.virtual_memory().percent}%)")
			_move_trace = (stop_move_count // trace + 1) * trace
		
		return stop_move_count, _move_trace
	
	# monoprocess conditionally
	if num_workers == 1 or max_moves == 0:
		for fen, moves in producer():
			_ = consume(_pgn_encode_game(fen, moves))
	
	# else multiprocess
	else:
		###! futures handles worker exception/error propagation
		with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as pool:
			pending = deque()
			max_pending = num_workers * max_queue_per_worker
			try:
				for fen, moves in producer():
					pending.append(pool.submit(_encode_game_fn, fen, moves))
					if len(pending) >= max_pending:
						move_count, move_trace = consume(*pending.popleft().result(), move_count, move_trace)
				while pending:
						move_count, move_trace = consume(*pending.popleft().result(), move_count, move_trace)
			finally:
				for future in pending:
					future.cancel()
	
	# construct jax arrays
	if max_moves < 0 and chunks_x:
		data_x = np.concatenate(chunks_x)
		data_y = np.concatenate(chunks_y)
	data_x = jnp.asarray(data_x[:move_count])
	data_y = jnp.asarray(data_y[:move_count])
	
	return data_x, data_y


# type: (jax.Array, jax.Array, int, bool) ~> tuple[jax.Array, jax.Array]
def batch_generator(data_x, data_y, batch_size, shuffle_key=None):
	
	# assertions
	assert (len(data_x)==len(data_y))
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else jnp.array(range(n_samples))
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			batch_y = data_y[batch_idx]
			yield batch_x, batch_y
