# core
import time
import psutil

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# environment
import chess # python-chess
import chess.pgn

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
		dir_class, piece_idx = divmod(channel - 64, 3)
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
def stream_pgn_to_supervised(pgn_textio, max_moves=-1, max_memory=0.85, debug=-1, start_time=None):
	states = []
	actions = []
	move_counter = 0
	max_memory_percent = 100 * max_memory
	if start_time is None:
		start_time = time.time()
	
	# stream up to #max_games games from lichess PGN file
	while max_moves < 0 or move_counter < max_moves:
		
		# mem safety
		memory_percent = psutil.virtual_memory().percent
		if memory_percent > max_memory_percent:
			raise RuntimeError(f"memory_percent > max_memory_percent, {memory_percent} > {max_memory_percent}")
		
		# initialise game (check EOF)
		game = chess.pgn.read_game(pgn_textio)
		if game is None:
			break
		board = game.board()
		
		# encode all states and moves from game (check max_moves)
		for move in game.mainline_moves():
			if max_moves < 0 or move_counter < max_moves:
				
				###! pre-compute mask and pack into input
				x = jnp.concatenate([board_to_vec837(board), board_to_mask4672(board)])
				y = move_to_idx4672(move)
				
				states.append(x)
				actions.append(y)
				
				board.push(move)
				move_counter += 1
				
				if debug > 0 and move_counter % debug == 0:
					print(f"[{time.time()-start_time:.2f}s] load {move_counter} ({memory_percent}%)")
	
	# construct jax arrays
	data_x = jnp.asarray(states, dtype=jnp.bool_)
	data_y = jnp.asarray(actions, dtype=jnp.int32)
	
	return data_x, data_y


###! multithreading (GIL limited 100% total util)

# import threading
# import queue

# # type: (TextIO, int, float, int, float, int, int) -> tuple[jax.Array, jax.Array]
# def stream_pgn_to_supervised(
		# pgn_textio,
		# max_moves,
		# max_memory=0.85,
		# debug=-1,
		# start_time=None,
		# num_workers=4,
		# max_queue=1000
	# ):
	
	# # pre-allocate arrays
	# assert max_moves > 0 ###! add dynamic array resizing for unbounded streaming
	# data_x = np.empty((max_moves, 5509), dtype=np.bool_)
	# data_y = np.empty((max_moves,), dtype=np.int32)
	
	# # initialise task queue
	# data_queue = queue.Queue(maxsize=max_queue)
	# move_counter = 0
	
	# # mem-safety threshold
	# max_memory_percent = 100 * max_memory
	
	# # logging
	# if start_time is None:
		# start_time = time.time()
	# idle_stats = {i: {"empty_gets": 0, "idle_seconds": 0.0} for i in range(num_workers)}
	
	# # start worker threads
	# def worker(worker_id):
		# while True:
			# try:
				# task = data_queue.get_nowait()
			# except queue.Empty:
				# idle_stats[worker_id]["empty_gets"] += 1
				# start_time = time.time()
				# task = data_queue.get()
				# stop_time = time.time()
				# idle_stats[worker_id]["idle_seconds"] += stop_time - start_time

			# try:
				# task = data_queue.get()
				# if task is None: # stop condition
					# return
				# idx, board, move = task
				# data_x[idx, :837] = board_to_vec837(board)
				# data_x[idx, 837:] = board_to_mask4672(board)
				# data_y[idx] = move_to_idx4672(move)
			# finally:
				# data_queue.task_done()
	
	# workers = [threading.Thread(target=worker, args=(i,), daemon=False) for i in range(num_workers)]
	# for w in workers:
		# w.start()
	
	# # stream up to #max_games games from lichess PGN file
	# while max_moves < 0 or move_counter < max_moves:
			
			# # mem safety
			# memory_percent = psutil.virtual_memory().percent
			# if memory_percent > max_memory_percent:
				# raise RuntimeError(f"memory_percent > max_memory_percent, {memory_percent} > {max_memory_percent}")
			
			# # initialise game (check EOF)
			# game = chess.pgn.read_game(pgn_textio)
			# if game is None:
				# break
			# board = game.board()
			
			# # encode all states and moves from game (check max_moves)
			# for move in game.mainline_moves():
				# if max_moves < 0 or move_counter < max_moves:
					# data_queue.put((move_counter, board.copy(), move))
					
					# board.push(move)
					# move_counter += 1
					
					# if debug > 0 and move_counter % debug == 0:
						# print(f"[{time.time()-start_time:.2f}s] load {move_counter} ({memory_percent}%)")
	
	# # block until all tasks completed
	# for _ in workers:
		# data_queue.put(None)
	# data_queue.join()
	# for w in workers:
		# w.join()
	
	# if debug > 0:
		# print(f"{idle_stats=}")
	
	# # construct jax arrays
	# data_x = jnp.asarray(data_x[:move_counter], dtype=jnp.bool_)
	# data_y = jnp.asarray(data_y[:move_counter], dtype=jnp.int32)
	
	# return data_x, data_y


###! multiprocessing (buggy and no performance gain)
###! + __main__ gate in caller

# import multiprocessing
# from multiprocessing.shared_memory import SharedMemory

# def _pgn_encode_worker(producer_task_queue, consumer_error_queue, shm_data_x_name, shm_data_y_name, max_moves):
	
	# # attach to shared memory buffers
	# shm_data_x = SharedMemory(name=shm_data_x_name)
	# shm_data_y = SharedMemory(name=shm_data_y_name)
	# data_x = np.ndarray((max_moves, 5509), dtype=np.bool_, buffer=shm_data_x.buf)
	# data_y = np.ndarray((max_moves,), dtype=np.int32, buffer=shm_data_y.buf)
	
	# # poll the task queue
	# try:
		# while True:
			# task = producer_task_queue.get()
			# try:
				# if task is None: # stop sentinel
					# return
				
				# # write to shared memory
				# idx, board, move = task
				# data_x[idx, :837] = board_to_vec837(board)
				# data_x[idx, 837:] = board_to_mask4672(board)
				# data_y[idx] = move_to_idx4672(move)
			
			# except Exception as e:
				# consumer_error_queue.put(e)
			# finally:
				# producer_task_queue.task_done()
	# finally:
		# shm_data_x.close()
		# shm_data_y.close()


# def stream_pgn_to_supervised(
		# pgn_textio,
		# max_moves=100_000_000,
		# max_memory=0.85,
		# debug=-1,
		# start_time=None,
		# num_workers=4,
		# max_queue=100,
	# ):
	
	# assert max_moves > 0
	
	# # init producer states
	# move_counter = 0
	# max_memory_percent = 100 * max_memory
	# if start_time is None:
		# start_time = time.time()
	
	# # init shared memory
	# data_x_shm = SharedMemory(create=True, size=max_moves * 5509 * np.dtype(np.bool_).itemsize)
	# data_y_shm = SharedMemory(create=True, size=max_moves * np.dtype(np.int32).itemsize)
	# data_x = np.ndarray((max_moves, 5509), dtype=np.bool_, buffer=data_x_shm.buf)
	# data_y = np.ndarray((max_moves,), dtype=np.int32, buffer=data_y_shm.buf)
	
	# # init producer/consumer processes
	# process_context = multiprocessing.get_context("spawn")
	# producer_error = None
	# producer_task_queue = process_context.JoinableQueue(maxsize=max_queue)
	# consumer_error_queue = process_context.Queue()
	# consumers = [process_context.Process(target=_pgn_encode_worker, args=(
		# producer_task_queue,
		# consumer_error_queue,
		# data_x_shm.name,
		# data_y_shm.name,
		# max_moves
	# )) for _ in range(num_workers)]
	
	# try:
		
		# # start consumers
		# for w in consumers:
			# w.start()
		
		# # start producer
		# while move_counter < max_moves:
			
			# # memory-safety
			# memory_percent = psutil.virtual_memory().percent
			# if memory_percent > max_memory_percent:
				# raise RuntimeError(f"memory_percent > max_memory: {memory_percent} > {100 * max_memory}")
			
			# # initialise game (check EOF)
			# game = chess.pgn.read_game(pgn_textio)
			# if game is None:
				# break
			# board = game.board()
			
			# # queue all states and moves from game (check max_moves)
			# for move in game.mainline_moves():
				# if move_counter >= max_moves:
					# break
				# producer_task_queue.put((move_counter, board.copy(), move))
				# board.push(move)
				# move_counter += 1
				# if debug > 0 and move_counter % debug == 0:
					# print(f"[{time.time() - start_time:.2f}s] load {move_counter} ({memory_percent}%)")
		
		# # block until all consumers completed
		# for _ in consumers:
			# producer_task_queue.put(None)
		# producer_task_queue.join()
		# for w in consumers:
			# w.join()
		
		# # propagate consumer errors
		# if not consumer_error_queue.empty():
			# raise consumer_error_queue.get()
		
		# # propagate non-zero consumer exit codes
		# if [w.exitcode for w in consumers if w.exitcode != 0]:
			# raise RuntimeError(f"Non-zero consumer exit.")
		
		# # construct jax arrays independently of shmem
		# data_x_jax = jnp.asarray(np.array(data_x[:move_counter], copy=True), dtype=jnp.bool_)
		# data_y_jax = jnp.asarray(np.array(data_y[:move_counter], copy=True), dtype=jnp.int32)
	
	# # always close shmem
	# finally:
		# producer_task_queue.close()
		# consumer_error_queue.close()
		# data_x_shm.close()
		# data_y_shm.close()
		# data_x_shm.unlink()
		# data_y_shm.unlink()
	
	# return data_x_jax, data_y_jax


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
