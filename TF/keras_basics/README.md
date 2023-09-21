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
