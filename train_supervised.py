import time

import jax
import jax.numpy as jnp
import optax

from library.data import stream_pgn_to_supervised
from library.model import init_dense_neural_network, dense_neural_network, fit

# set constants
BATCH_SIZE = 256
TRAIN_BLOCK_COUNT = 20
TRAIN_BLOCK_SIZE = 50_000
VAL_BLOCK_SIZE = 8192
RNG_SEED = 9

# start timer
T0 = time.time()
print(f"[{time.time() - T0:.2f}s]")

###! try cache
try:
	raise NotImplementedError()

except:
	
	lichess_pgn = open("data/lichess_db_standard_rated_2017-03.pgn", "r", encoding="utf-8")
	
	k0 = jax.random.key(RNG_SEED)
	
	params = init_dense_neural_network(k0, [837, 1024, 1024, 4672])
	model = jax.vmap(lambda p, x: jnp.where(x[837:], -jnp.inf, dense_neural_network(p, x[:837].astype(jnp.float32))), in_axes=(None, 0))
	loss = lambda p, xs, ys: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(model(p, xs), ys))
	
	val_x, val_y = stream_pgn_to_supervised(lichess_pgn, max_moves=VAL_BLOCK_SIZE, debug=VAL_BLOCK_SIZE, start_time=T0)
	val_steps = int(jnp.ceil(VAL_BLOCK_SIZE / BATCH_SIZE))
	train_steps = int(jnp.ceil(TRAIN_BLOCK_SIZE / BATCH_SIZE))
	
	for i in range(TRAIN_BLOCK_COUNT):
		train_x, train_y = stream_pgn_to_supervised(lichess_pgn, max_moves=TRAIN_BLOCK_SIZE, debug=TRAIN_BLOCK_SIZE, start_time=T0)
		
		k0, k1 = jax.random.split(k0)
		params, history = fit(
			k1,
			params,
			loss,
			(train_x, train_y, train_steps),
			val_data=(val_x, val_y, val_steps),
			batch_size=BATCH_SIZE,
			epochs=1,
			opt=optax.adamw(1e-5),
			start_time=T0,
			start_epoch=i,
			#use_tqdm=True
		)
		
		del train_x
		del train_y
	
	lichess_pgn.close()
	
	###! write cache
