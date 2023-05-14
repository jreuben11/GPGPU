1. [quickstart](quickstart.ipynb)
   - Multiplying Matrices
   - Using `jit()` to speed up functions
   - Taking derivatives with `grad()`
   - Auto-vectorization with `vmap()`
   - JAX API: 
     - `grad`, `jit`, `vmap`, `jacfwd` / `jacrev` / `hessian`, `pmap`, `device_put`
     - `lax`:`psum`
     - `jax.numpy`: `dot`, `tanh`, `sum`, `exp`, `vjp` / `jvp`, `stack`, `mean`, `where`, `arange`, `eye`
     - `jax.random`: `PRNGKey`, `normal`


2. [Thinking in JAX](thinking_in_jax.ipynb)
   - JAX vs. NumPy
   - NumPy, lax & XLA: JAX API layering
   - To JIT or not to JIT
   - JIT mechanics: tracing and static variables
   - Static vs Traced Operations
   - ipython:
     ```python
     # Configure ipython to hide long tracebacks.
     import sys
     ipython = get_ipython()

     def minimal_traceback(*args, **kwargs):
       etype, value, tb = sys.exc_info()
       value.__cause__ = None  # suppress chained exceptions
       stb = ipython.InteractiveTB.structured_traceback(etype, value, tb)
       del stb[3:-1]
       return ipython._showtraceback(etype, value, stb)

     ipython.showtraceback = minimal_traceback
     ``` 
   -JAX API:
     - `lax`: `add`, `conv_general_dilated`
     - `make_jaxpr`
     - `jax.numpy`: `add`, `convolve`, `float32`, `array`

3. [THe sharp bits](Common_Gotchas_in_JAX.ipynb)
   - 🔪 Pure functions - `lax.fori_loop`, `lax.cond`, `lax.scan`
   - 🔪 In-Place Updates - `at`, `set`
   - 🔪 Out-of-Bounds Indexing
   - 🔪 Non-array inputs: NumPy vs. JAX
   - 🔪 Random Numbers - `jax.random`: `PRNGKey`,`normal`, `split`
   - 🔪 Control Flow - `f = jit(f, static_argnums=(0,))`
   - 🔪 NaNs = `config.update / parse_flags_with_absl`
   - 🔪 Double (64bit) precision `config.update("jax_enable_x64", True)`

4. 101 Tutorials
   1. [JAX As Accelerated NumPy](101_tutorials/01-jax-basics.ipynb)
      -  Getting started with JAX numpy - `jnp.arange`, `jnp.dot`, `block_until_ready`
      -  JAX first transformation: `jax.grad`
      -  Value and Grad `jax.value_and_grad`
      -  Auxiliary data `has_aux`
      -  Differences from NumPy `at`, `set`
      -  Your first JAX training loop: 1. sample data, 2. model, 3. loss, 4. update, 5. training loop :)
   2. [Just In Time Compilation with JAX](101_tutorials/02-jitting.ipynb)
      - How JAX transforms work `jax.make_jaxpr`
      - JIT compiling a function `jax.jit` , `jnp.where`
      - Why can’t we just JIT everything? ` jax.lax.cond`,  `static_argnums` / `static_argnames`, `functools.partial`
   3. [Automatic Vectorization in JAX](101_tutorials/03-vectorization.ipynb)
      - Manual Vectorization `jnp.stack`
      - Automatic Vectorization `jax.vmap` with `in_axes` and `out_axes`, ` jnp.transpose`
      - Combining transformations 
   4. [Advanced Automatic Differentiation in JAX](101_tutorials/04-advanced-autodiff.ipynb)
      - Higher-order derivatives ` jax.grad`, `jax.jacfwd` and `jax.jacrev`
      - Higher order optimization -> Model-Agnostic Meta-Learning (**MAML**)
      - Stopping gradients -> **TD(0)** (temporal difference) RL update `jax.lax.stop_gradient`
      - Straight-through estimator using stop_gradient - **???**
      - Per-example gradients ` jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))`