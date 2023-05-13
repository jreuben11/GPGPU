1. [quickstart](quickstart.ipynb)
### JAX API
- grad
- jit
- vmap
- jacfwd, jacrev
- hessian
- random
- pmap
- lax.psum
- device_put
### jax.numpy
- dot
- tanh
- sum
- exp
- vjp, jvp
- stack
- mean
- where
- arange
- eye
### jax.random
- PRNGKey
- normal

2. [Thinking in JAX](thinking_in_jax.ipynb)
### ipython
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
### JAX API
- lax.add
- lax.conv_general_dilated
- make_jaxpr
### jax.numpy
- add
- convolve
- float32
- array