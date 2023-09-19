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
