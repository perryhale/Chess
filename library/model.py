import time

import jax
import jax.numpy as jnp
import optax

from tqdm import tqdm

from .data import batch_generator

"""

________________________________________________________
init   |  U(-gs,+gs)           |  Glorot uniform
model  |  MLP: R^N -> R^K      |  MLP(p,x)->yh
loss   |  CCE: R^K x R^K -> R  |  CCE(yh,y)->l
optim  |  ~SGD: Θ x B -> Θ     |  OPT(s,p,x,y)->l,sh,ph
metric |  M: R^K x R^K -> R    |  M(yh,y)->m
________________________________________________________

initialiser
 - Glorot uniform

model
 - masked Multi-Layer Perceptron (LReLU)

loss
 - softmax categorical cross entropy with int labels

optimiser
 - any SGD

metrics
 - categorical accuracy
 - param count

"""


### model

# type: (int, list[int], int, int) -> jax.Array
def init_glorot_uniform(rng_key, shape, fan_in, fan_out):
	"""
	# shape inZ+ (*)
	# fan_in, fan_out inZ+
	# w inR shape
	"""
	limit = jnp.sqrt(6) / jnp.sqrt(fan_in+fan_out)
	w = jax.random.uniform(rng_key, shape, minval=-limit, maxval=limit)
	return w


# type: (int, list[int]) -> list[tuple[jax.Array, jax.Array]]
def init_dense_neural_network(rng_key, layers):
	
	# split RNG key
	rng_key, k1 = jax.random.split(rng_key)
	
	# initialise input layer
	w_list = [init_glorot_uniform(k1, (layers[1], layers[0]), layers[0], layers[1])]
	b_list = [jnp.zeros((layers[1], 1))]
	
	# initialise hidden layers
	for i in range(1, len(layers)-1):
		k1, k2 = jax.random.split(k1)
		w_list.append(init_glorot_uniform(k2, (layers[i+1], layers[i]), layers[i], layers[i+1]))
		b_list.append(jnp.zeros((layers[i+1], 1)))
	
	return list(map(tuple, zip(w_list, b_list)))


# type: (list[tuple[jax.Array, jax.Array]], jax.Array, Callable[[jax.Array]]) -> jax.Array
def dense_neural_network(params, x, ha=jax.nn.leaky_relu):
	"""
	# x inR (in, 1)
	# z inR (out, 1)
	"""
	
	# shape-safe I/O activations
	h = x.reshape(-1, 1)
	
	# hidden activations
	for w,b in params[:-1]:
		h = ha(jnp.dot(w, h) + b)
	
	# output activation
	wn, bn = params[-1]
	z = jnp.dot(wn, h) + bn
	
	return z.reshape(-1)


### optimiser

# type:
def optimise_params(
		rng_key,              # jax.Array
		params,               # PyTree[jax.Array]
		loss_fn,              # Callable[["PyTree[jax.Array]", jax.Array, jax.Array], float]
		opt_fn,               # Callable[[optax.OptState, "PyTree[jax.Array]", jax.Array, jax.Array], tuple[jax.Array, optax.OptState, "PyTree[jax.Array]"]]
		opt_state,            # optax.OptState
		train_data,           # tuple[jax.Array, jax.Array, int]
		
		val_data=None,        # tuple[jax.Array, jax.Array, int] | None
		batch_size=64,        # int
		epochs=1,             # int
		
		start_time=None,      # float | None
		start_epoch=0,        # int
		use_tqdm=False        # bool
	):                        # -> tuple[PyTree[jax.Array], Dict[str:list[float]]]
	"""
	Iteratively optimise `params` for `loss_fn` and `train_data` using stochastic gradient descent.
	`train_data=(data_x, data_y, train_steps=data_len/batch_size)`. `val_data` can be provided in the same format for evaluation _after_ every epoch.
	`batch_size:int`, `epochs:int` are provided as defaulting call arguments.
	Other optional args for reporting.
	"""
	
	# initialise
	train_x, train_y, train_steps = train_data
	history = {'batch_loss':[], 'train_loss':[]}
	if val_data is not None:
		val_x, val_y, val_steps = val_data
		val_generator = batch_generator(val_x, val_y, batch_size)
		history.update({'val_loss':[]})
	if start_time is None:
		start_time = time.time()
	def tqdm_option(_iter, _bool, desc=None):
		return tqdm(_iter, desc=desc) if _bool else _iter
	
	# run training loop
	###! opt: prevent accel-->host writeback between core steps
	for i in range(epochs):
		
		# shuffle train batches
		rng_key, subkey = jax.random.split(rng_key, 2)
		train_generator = batch_generator(train_x, train_y, batch_size, shuffle_key=subkey)
		
		# iterate optimiser
		train_loss = 0. ###! accumulate on host
		for j in tqdm_option(range(train_steps), use_tqdm, desc="Train"):
			batch_loss, opt_state, params = opt_fn(opt_state, params, *next(train_generator)) ###! generator defined jax->jax (accel->accel)
			batch_loss = float(batch_loss) ###! implicit accel->host
			train_loss += batch_loss / train_steps ###! O(N) denominator
			history['batch_loss'].append(batch_loss)
		history['train_loss'].append(train_loss)
		
		# determine validation loss
		if val_data is not None:
			val_loss = 0. ###! accumulate on host
			for _ in tqdm_option(range(val_steps), use_tqdm, desc="Val"):
				val_loss += float(loss_fn(params, *next(val_generator))) / val_steps ###! implicit accel->host, O(N) denominator
			history['val_loss'].append(val_loss)
		
		# trace to stdout
		print(
			f"[{time.time()-start_time:.2f}s]"
			+ f" epoch={start_epoch+i+1},"
			+ f" train_loss={train_loss:.4f},"
			+ f" val_loss={val_loss:.4f}" if val_data is not None else ""
		)
	
	return params, history


### metrics

# type: (jax.Array, jax.Array) -> float
def accuracy_score(yh, y):
	"""
	# yh, y inR (*, n_classes)
	# accuracy inR
	"""
	yhc = jnp.argmax(yh, axis=-1)
	yc = jnp.argmax(y, axis=-1)
	accuracy = jnp.mean(jnp.array(yhc==yc, dtype=jnp.float32))
	return float(accuracy)


# type: (PyTree[jax.Array]) -> int
def count_params(params):
	if isinstance(params, jnp.ndarray):
		return jnp.prod(jnp.array(params.shape))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([count_params(item) for item in params]))
	return 0
