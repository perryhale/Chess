from types import SimpleNamespace
import sys
import time
import pprint
import pickle

import jax
import jax.numpy as jnp
import optax

from library.data import stream_pgn_to_supervised
from library.model import init_dense_neural_network, dense_neural_network, optimise_params


try:
	
	# unpack args
	NO_CACHE = len(sys.argv) > 1 and sys.argv[1] == "--no-cache"
	USE_TQDM = len(sys.argv) > 2 and sys.argv[2] == "--use-tqdm"
	
	# try cache
	if NO_CACHE:
		raise FileNotFoundError("Using --no-cache")
	with open(f"{__file__.replace('.py','')}.cache.pkl", "rb") as f:
		CFG = pickle.load(f)

except FileNotFoundError:
	
	# init default
	CFG = SimpleNamespace(
		pgn_path = "data/lichess_db_standard_rated_2017-03.pgn",
		batch_size = 64,
		train_block_count = 25,
		train_block_size = 160_000,
		val_block_size = 10_000,
		test_block_size = 40_000,
		rng_seed = 9,
		hidden_layers = [1024, 1024],
		policy = None,
		history = None
	)


if __name__ == "__main__":
	
	# start timer
	start_time = time.time()
	print(f"[Elapsed time: {time.time()-start_time:.2f}s]")
	
	# trace config to stdout
	pprint.pp(vars(CFG))
	
	# init RNG
	k0 = jax.random.key(CFG.rng_seed)
	
	# init model
	policy = CFG.policy if CFG.policy else init_dense_neural_network(k0, [837, *CFG.hidden_layers, 4672])
	policy_fn = jax.vmap(lambda p,x: jnp.where(x[837:], -jnp.inf, dense_neural_network(p, x[:837].astype(jnp.float32))), in_axes=(None, 0))
	loss_fn = lambda p,xs,ys: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_fn(p, xs), ys))
	
	# open data stream
	lichess_pgn = open(CFG.pgn_path, 'r', encoding='utf-8')
	
	# ensure data stream resource closure
	try:
		
		# load test/val blocks
		test_x, test_y = stream_pgn_to_supervised(lichess_pgn, max_moves=CFG.test_block_size, debug=CFG.test_block_size, start_time=start_time)
		val_x, val_y = stream_pgn_to_supervised(lichess_pgn, max_moves=CFG.val_block_size, debug=CFG.val_block_size, start_time=start_time)
		
		# determine step counts
		test_steps = int(jnp.ceil(CFG.test_block_size / CFG.batch_size))
		val_steps = int(jnp.ceil(CFG.val_block_size / CFG.batch_size))
		train_steps = int(jnp.ceil(CFG.train_block_size / CFG.batch_size))
		
		# online training loop
		if not CFG.policy:
			
			# init optimiser
			opt = optax.adamw(1e-4)
			opt_state = opt.init(policy)
			
			# jit compile optimiser core
			@jax.jit
			def opt_fn(_opt_state, _policy, xs, ys):
				loss, grad = jax.value_and_grad(loss_fn)(_policy, xs, ys)
				update, _opt_state = opt.update(grad, _opt_state, _policy)
				_policy = optax.apply_updates(_policy, update)
				return loss, _opt_state, _policy
			
			for i in range(CFG.train_block_count):
				
				# load train block (check EOF)
				train_x, train_y = stream_pgn_to_supervised(lichess_pgn, max_moves=CFG.train_block_size, debug=CFG.train_block_size, start_time=start_time,
					max_memory=-1 ###! disable mem safety
				)
				if train_x.shape[0] == 0:
					break
				
				k0, k1 = jax.random.split(k0)
				policy, _history = optimise_params(
					k1,
					policy,
					loss_fn,
					opt_fn,
					opt_state,
					train_data=(train_x, train_y, train_steps),
					val_data=(val_x, val_y, val_steps),
					batch_size=CFG.batch_size,
					start_time=start_time,
					start_epoch=i,
					use_tqdm=USE_TQDM
				)
				
				del train_x
				del train_y
				
				if not CFG.history:
					CFG.history = _history
				else:
					CFG.history['batch_loss'].extend(_history['batch_loss'])
					CFG.history['train_loss'].extend(_history['train_loss'])
					CFG.history['val_loss'].extend(_history['val_loss'])
			
			# update cache
			CFG.policy = policy
			with open(f"{__file__.replace('.py','')}.cache.pkl", 'wb') as f:
				pickle.dump(CFG, f)
	
	finally:
		
		# close data stream
		lichess_pgn.close()
	
	###! evaluation
	# test_loss =
	
	###! plotting
	# plt.
