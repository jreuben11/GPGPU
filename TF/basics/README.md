# 1. Basics
### Tensors
- `tf.Tensor`: `shape`, `dtype`
- `tf.constant`
- math ops (+, *, @), `tf.transpose`
- `tf.concat`
- `tf.nn.softmax`  , `tf.reduce_sum`
- `tf.config.list_physical_devices('GPU')`
### Variables
- `tf.Variable`: `assign`, `assign_add`, `numpy`
### Automatic Differentiation
- `tf.GradientTape`: `gradient`
### Graphs & `tf.function`
- `@tf.function`
### Modules, Layers and Models
- `tf.Module` -> `tf.keras.layers.Layer` and `tf.keras.Model`: `compile`, `fit` 
- ` tf.saved_model`: `save`, `load`
- `tf.train.Checkpoint`
### Training Loops
- `tf.linspace`
- `tf.cast`
- `tf.random.normal`
- `tf.keras.layers.Dense` , `tf.nn.relu`
- `tf.squeeze`
- `tf.optimizers.SGD`: `apply_gradients`
- `tf.keras.losses.MSE`


# 2. Introduction to Tensors
### basics
- `tf.ones`, `tf.zeros`
- `tf.add`, `tf.multiply`, `tf.matmul`
- `tf.reduce_max`, `tf.argmax`, `tf.nn.softmax`
### shapes
- `tf.Tensor`: `shape`, `dtype`, `ndim`
- ` tf.TensorShape`: `as_list`
- `tf.size`
### single/multi-axis Indexing
### manipulating shapes
-  `tf.reshape`
### DTypes
- `tf.dtypes.DType`
### Broadcasting
- ` tf.broadcast_to`
### conversion
- `tf.convert_to_tensor`
- `tf.register_tensor_conversion_function`
### ragged tensors
- `tf.ragged.RaggedTensor`
- `tf.ragged.constant`
### string tensors
- `tf.strings`: `split`, `to_number`
### sparse tensors
- `tf.sparse.SparseTensor` 
