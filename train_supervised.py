from types import SimpleNamespace
import sys
import time
import pprint
import pickle

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from library.data import tqdm_option, stream_pgn_to_supervised, batch_generator
from library.model import init_dense_neural_network, dense_neural_network, fit

	
# unpack args
NO_CACHE = "--no-cache" in sys.argv
USE_TQDM = "--use-tqdm" in sys.argv
SAVE_FIG = "--save-fig" in sys.argv

# artefact paths
CACHE_PATH = __file__.replace('.py','.cache.pkl')
FIG_PATH = __file__.replace('.py','.png')

try:
	
	# try cache
	if NO_CACHE:
		raise FileNotFoundError("Using --no-cache")
	with open(CACHE_PATH, "rb") as f:
		CFG = pickle.load(f)

except FileNotFoundError:
	
	# init default
	CFG = SimpleNamespace(
		pgn_path = "data/lichess_db_standard_rated_2017-03.pgn",
		batch_size = 8,#4,
		train_block_count = 20,#0,
		train_block_size = 160,#_000,
		val_block_size = 10,#_000,
		test_block_size = 40,#_000,
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
	policy_fn = jax.jit(jax.vmap(
		lambda p,x: jnp.where(x[837:], -jnp.inf, dense_neural_network(p, x[:837].astype(jnp.float32))),
		in_axes=(None, 0)
	))
	loss_fn = jax.jit(
		lambda p,xs,ys: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(policy_fn(p, xs), ys))
	)
	
	# open data stream
	lichess_pgn = open(CFG.pgn_path, 'r', encoding='utf-8')
	
	# ensure data stream resource closure
	try:
		
		# load test block
		test_x, test_y = stream_pgn_to_supervised(lichess_pgn, max_moves=CFG.test_block_size, trace=CFG.test_block_size, start_time=start_time)
		
		# determine step counts
		test_steps = int(jnp.ceil(CFG.test_block_size / CFG.batch_size))
		val_steps = int(jnp.ceil(CFG.val_block_size / CFG.batch_size))
		train_steps = int(jnp.ceil(CFG.train_block_size / CFG.batch_size))
		
		# online training loop
		if not CFG.policy:
			
			# load val block
			val_x, val_y = stream_pgn_to_supervised(lichess_pgn, max_moves=CFG.val_block_size, trace=CFG.val_block_size, start_time=start_time)
			
			# init optimiser
			opt = optax.adamw(1e-4)
			opt_state = opt.init(policy) ###! resume requires storing opt_state, k0, and lichess_pgn.tell() (for .seek(p)) and epoch
			
			# jit compile optimiser core
			@jax.jit
			def opt_fn(_opt_state, _policy, xs, ys):
				loss, grad = jax.value_and_grad(loss_fn)(_policy, xs, ys)
				update, _opt_state = opt.update(grad, _opt_state, _policy)
				_policy = optax.apply_updates(_policy, update)
				return loss, _opt_state, _policy
			
			# init checkpoint best-validation-loss threshold
			best_val_loss = float('inf')
			
			for i in range(CFG.train_block_count):
				
				# load train block (check EOF)
				train_x, train_y = stream_pgn_to_supervised(lichess_pgn, max_moves=CFG.train_block_size, trace=CFG.train_block_size, start_time=start_time,
					max_memory=-1 ###! disable mem safety
				)
				if train_x.shape[0] == 0:
					break
				
				k0, k1 = jax.random.split(k0)
				policy, opt_state, _history = fit(
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
				val_loss = CFG.history['val_loss'][-1]
				if val_loss < best_val_loss:
					print(f"[{time.time()-start_time:.2f}s] val_loss < best_val_loss, {val_loss:.4f} < {best_val_loss:.4f}")
					best_val_loss = val_loss
					with open(CACHE_PATH, 'wb') as f:
						CFG.policy = policy
						pickle.dump(CFG, f)
	
	finally:
		
		# close data stream
		lichess_pgn.close()
	
	# test set evaluation
	if not 'test_loss' in CFG.history:
		
		test_generator = batch_generator(test_x, test_y, CFG.batch_size)
		test_loss = 0.0
		test_accuracy = 0.
		
		for _ in tqdm_option(range(test_steps), USE_TQDM, desc="Test"):
			batch_x, batch_y = next(test_generator)
			test_yh = policy_fn(policy, batch_x).argmax(axis=-1)
			test_loss += float(loss_fn(policy, batch_x, batch_y)) ###! compute yh twice
			test_accuracy += float((test_yh == batch_y).sum())
		
		test_loss /= test_steps
		test_accuracy /= CFG.test_block_size
		
		CFG.history['test_loss'] = [test_loss]
		CFG.history['test_accuracy'] = [test_accuracy]
		
		with open(CACHE_PATH, 'wb') as f:
			pickle.dump(CFG, f)
	
	# plot history
	batch_loss = CFG.history["batch_loss"]
	train_loss = CFG.history["train_loss"]
	val_loss = CFG.history["val_loss"]
	test_loss = CFG.history["test_loss"][-1]
	test_accuracy = CFG.history["test_accuracy"][-1] * 100
	batch_axis = list(range(1, len(batch_loss) + 1))
	epoch_axis = batch_axis[train_steps-1::train_steps]
	
	plt.plot(batch_axis, batch_loss, color="purple", label="batch_loss")
	plt.plot(epoch_axis, train_loss, color="blue", label="train_loss")
	plt.plot(epoch_axis, val_loss, color="red", label="val_loss")
	plt.scatter(batch_axis[-1], test_loss, color="green", marker="x", label="test_loss")
	plt.gca().text(
		0.5, 0.98,
		f"Test accuracy: {test_accuracy:.1f}%",
		transform=plt.gca().transAxes,
		ha="center",
		va="top",
		color="white",
		bbox=dict(facecolor="black", edgecolor="black", boxstyle="square,pad=0.4"),
	)
	
	plt.grid()
	plt.ylabel("Loss")
	plt.xlabel("Train step")
	plt.legend()
	plt.tight_layout()
	if SAVE_FIG:
		plt.savefig(FIG_PATH)
	else:
		plt.show()
