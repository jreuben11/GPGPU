# 1. The Sequential model
- `keras.Sequential`: `add`, `pop`
- `keras.layers.Dense`
### When to use a Sequential model
### Creating a Sequential model
### Specifying the input shape in advance
- `tf.keras.Model`: `layers`, `weights`, `summary`
- `keras.Input`
```python
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))
```
### A common debugging workflow: `add()` + `summary()`
- `layers.Conv2D`, `layers.MaxPooling2D`, `layers.GlobalMaxPooling2D`
### What to do once you have a model
### Feature extraction with a Sequential model
- `tf.keras.layers.Layer`: `input`, `activation`, `output`
```python
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)
```
### Transfer learning with a Sequential model
- `tf.keras.layers.Layer`: `trainable`
- `tf.keras.Model`: `load_weights`
- `keras.applications.Xception`



# 2. The Functional API
### Introduction
```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```
### Training, evaluation, and inference
- `keras.datasets.mnist.load_data`
- `keras.losses.SparseCategoricalCrossentropy`
- `keras.optimizers.RMSprop`
- `model.compile(loss, optimizer, metrics)`
- `history = model.fit(x_train, y_train, batch_size, epochs, validation_split)`
- `metrics = model.evaluate(x_test, y_test)`
### Save and serialize
- `tf.keras.Model`: `save`
- `keras.models.load_model`
### Use the same graph of layers to define multiple models
- autoencoder: `Conv2DTranspose`, `UpSampling2D`
### All models are callable, just like layers
```python
encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
...
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")

x = layers.Reshape((4, 4, 1))(encoder_output)
...
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
```
### Manipulate complex graph topologies
- Models with multiple inputs and outputs
    - `layers.Embedding`, `layers.LSTM`, `layers.concatenate`
    - `keras.utils.plot_model`
    - `keras.losses.BinaryCrossentropy`, `CategoricalCrossentropy`
- A toy ResNet model
    - `keras.datasets.cifar10.load_data`
### Shared layers
- `layers.Embedding`
### Extract and reuse nodes in the graph of layers
- `tf.keras.applications.VGG19`
```python
features_list = [layer.output for layer in vgg19.layers]
```
### Extend the API using custom layers
- `layers.Layer`: `__init__`, `build`, `add_weight`, `get_config`, `from_config`
### When to use the functional API
- strengths: Less verbose, Model validation while defining its connectivity graph, plottable and inspectable, can be serialized or cloned
- weaknesses: does not support dynamic architectures
### Mix-and-match API styles
- `tf.stack`


# 3. Training and evaluation with the built-in methods
- `tf.keras.Model`: `fit`, `evaluate`, `predict`
### API overview: a first end-to-end example
- `tf.data.Dataset` - `keras.datasets.mnist.load_data`
### The compile() method: specifying a loss, metrics, and an optimizer
- Many built-in optimizers, losses, and metrics are available
    - Optimizers: `SGD`, `RMSprop`, `Adam` etc
    - Losses: `MeanSquaredError`, `KLDivergence`, `CosineSimilarity` etc
    - Metrics: `AUC`, `Precision`, `Recall` etc.
- Custom losses `tf.keras.losses.Loss`: `__init__`, `call(self, y_true, y_pred)`
- Custom metrics `tf.keras.metrics.Metric`: `__init__`, `update_state`, `result`, `reset_state`
- Handling losses and metrics that don't fit the standard signature: `layer/model.add_loss(loss_tensor)`, `layer/model.add_metric(metric_tensor, name, aggregation)`
- Automatically setting apart a validation holdout set `model.fit(x_train, y_train, batch_size, validation_split, epochs)`
### Training & evaluation from `tf.Dataset`
- You can pass a `Dataset` instance directly to the `fit` / `evaluate` / `predict`
- `tf.data.Dataset.from_tensor_slices((x_train, y_train))`
- `tf.Dataset`: `shuffle`, `batch`
- `model.fit(train_dataset, epochs, steps_per_epoch)`
- Using a validation dataset `model.fit(train_dataset, epochs, validation_data, validation_steps)`
### Using a `keras.utils.Sequence` object as input
- `__getitem__`, `__len__`
- `Model.fit`: `shuffle=True` arg 
### Using sample weighting and class weighting
- `Model.fit`: `class_weight` / `sample_weight` args
- data iterator: Yield `(input_batch, label_batch, sample_weight_batch)` tuples
### Passing data to multi-input, multi-output models
- `Model.compile`: `loss`, `metrics` args can accept lists, dicts; `loss_weights` arg
### Using callbacks
- pass to `Model.fit`
- `keras.callbacks`: `EarlyStopping`, `ModelCheckpoint`, `TensorBoard`, `CSVLogger`
- Writing your own callback -  `keras.callbacks.Callback`: `on_train_begin`, `on_batch_end`
### Using learning rate schedules
-  `keras.optimizers.schedules`: `ExponentialDecay`, `PiecewiseConstantDecay`, `PolynomialDecay`, `InverseTimeDecay`
- `Optimizer`: `learning_rate` arg
- `ReduceLROnPlateau` callback - implement a dynamic learning rate schedule
### Visualizing loss and metrics during training
 - `keras.callbacks.TensorBoard` - specify `log_dir`, frequencies



# 4. Making new Layers and Models via subclassing
 ### The Layer class: the combination of state (weights) and some computation
 - `keras.layers.Layer`
 - `tf.random_normal_initializer`, `tf.zeros_initializer`
 ```python
 class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
 ```
 ### Layers can have non-trainable weights
 - `tf.Variable(initial_value, trainable)`
 ### Best practice: deferring weight creation until the shape of the inputs is known
 ```
 def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1]),initializer, trainable)
 ```
 - `tf.init_scope`
 ### Layers are recursively composable
 ### custom loss layers
 - `Layer.add_loss`, `layer.losses`
 - `keras.layers.Dense` `kernel_regularizer` 
 - `tf.keras.regularizers.l2`
 - `model.compile`: `loss` arg
 ### custom metrics
 - `Layer.add_metric`, `layer.metrics`
 - `keras.metrics.BinaryAccuracy`
 ### optionally enable serialization on your layers
 - `keras.layers.Layer`: `get_config`, `from_config`
 ### Privileged `training` argument in the `call()` method
 - pass from `fit` to control different behaviour in training / inference
 - eg `BatchNormalization`, `Dropout`
 ### Privileged `mask` argument in the `call()` method
 - RNN: skip timesteps
 - eg `Embedding(mask_zero)`, `Masking`
 ### The `Model` class
 - `fit`, `evaluate`, `predict`
 - `layers`
 - `save`, `save_weights`
 ### Putting it all together: an end-to-end example
 - VAE
 ### Beyond object-oriented development: the Functional API



 # 5. Save and load Keras models
 - `SavedModel` format
 ### How to save and load a model
 - `model.save(save_format)` or `tf.keras.models.save_model`, `keras.models.load_model`
 - What the SavedModel folder contains: assets  keras_metadata.pb  saved_model.pb  variables
 - How SavedModel handles custom objects: define `get_config` & `from_config`
 - Configuring the SavedModel: `model.save(save_traces)`
 - `save_format`: 
    - `'keras'` - default -> .keras file
    - `'tf'` -> HDF5 file 
    - `'h5'` -> .h5 file
 ### custom objects
 -  `get_config` & `from_config`
 - `keras.saving`: `serialize_keras_object`, `deserialize_keras_object`
 - Registering custom objects `@keras.saving.register_keras_serializable` - `package` arg
 - Using a custom object scope: `keras.saving.custom_object_scope`
 ### Saving the architecture
 - `keras.models.clone_model`
 - `tf.keras.models`: `model_to_json` / `model_from_json`
 - Loading the TensorFlow graph only: `tf.saved_model.load`
 - Registering the custom object (internals)
    - `tf.keras.layers.serialize`
    - `tf.keras.utils.custom_object_scope` or `tf.keras.utils.CustomObjectScope`
    - `tf.keras.utils.register_keras_serializable`
- In-memory model cloning:  `tf.keras.models.clone_model`
### Saving & loading only the model's weights values
- APIs for in-memory weight transfer: `tf.keras.layers.Layer`: `get_weights` / `set_weights`
- APIs for saving weights to disk & loading them back: `model.save_weights` / `load_weights`
- TF Checkpoint format: `tf.train.Checkpoint`, `tf.train.load_checkpoint`
    - eg `tf.keras.layers.Dense` contains two weights: `kernel` and `bias`
    - Transfer learning example:
    ```python
    pretrained = keras.Model(model.inputs, model.layers[-1].input, name="pretrained_model")
    ```
### Exporting
- `model.export`, `artifact.serve`, `keras.export.ExportArchive`
- Customizing export artifacts with `ExportArchive`: `track`, `add_endpoint`, `write_out`, `add_variable_collection`
- `keras.layers.MultiHeadAttention`
### Handling custom objects
- Defining the config methods `config.update`
- How custom objects are serialized `keras.layers.serialize`
- `keras.regularizers.L1L2(l1, l2)`




# 6. Working with preprocessing layers
### Available preprocessing
- `tf.keras.layers`:
- text: `TextVectorization`
- numerical: `Normalization`, `Discretization`
- categorical: `CategoryEncoding`, `Hashing`, `StringLookup`, `IntegerLookup`
- image: `Resizing`, `Rescaling`, `CenterCrop`
- image data augmentation: `RandomCrop`, `RandomFlip`, `RandomTranslation`, `RandomRotation`, `RandomZoom`, `RandomHeight`, `RandomWidth`, `RandomContrast`
### `layer.adapt` method
- `output_mode="one_hot/multi-hot/tf-idf"`
### Preprocessing data before the model or inside the model
-  `tf.data.Dataset.prefetch(tf.data.AUTOTUNE)`
### Benefits of doing preprocessing inside the model at inference time
### Quick recipes
- Image data augmentation: `keras.datasets.cifar10.load_data`, `keras.applications.ResNet50`
- Normalizing numerical features
- Encoding string categorical features via one-hot encoding
- Encoding integer categorical features via one-hot encoding
- Applying the hashing trick to an integer categorical feature
- Encoding text as a sequence of token indices
- Encoding text as a dense matrix of ngrams with multi-hot encoding
- Encoding text as a dense matrix of ngrams with TF-IDF weighting
### Important gotchas
- Working with lookup layers with very large vocabularies
