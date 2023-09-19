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

# 3. Introduction to Variables
### create a variable
- `tf.Variable`: `assign`, `assign_add`, `assign_sub`, `numpy`
### lifecycles, naming and watching
- properties: `name`, `trainable`
### placing variables and tensors
- `with tf.device('CPU:0')`
- `tf.config.set_soft_device_placement`

# 4. Introduction to gradients and automatic differentiation
### Computing gradients
### Gradient tapes
- `tf.GradientTape`: `gradient`
- properties: `persistent`
- `tf.nest`
### Gradients with respect to a model
-  `Module.trainable_variables`
### Controlling what the tape watches
- ` GradientTape.watched_variables`, `watch_accessed_variables`
- ` GradientTape.watch`
- `tf.math.sin`
### Intermediate results
### Notes on performance
### Gradients of non-scalar targets
- `tf.nn.sigmoid`
### Control flow
### Getting a gradient of None
### Zeros instead of None
- `tf.UnconnectedGradients.ZERO`


# 5. Introduction to graphs and `tf.function`
### overview
- `tf.Graph`, `tf.Operation`, `tf.Tensor`
### Taking advantage of graphs
- `@tf.function`
- `tf.autograph.to_code`
```python
tf_simple_relu = tf.function(simple_relu)
print(tf.autograph.to_code(simple_relu))
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())
```
- `ConcreteFunction`
### Using tf.function
- `tf.config.run_functions_eagerly`
- `tf.print`
- `tf.debugging`
### Seeing the speed-up
### when is a function tracing
