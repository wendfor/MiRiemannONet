import numpy as np
import jax.numpy as jnp
from jax.example_libraries import optimizers

opt_init, opt_update, get_params = optimizers.adam(0.1)

a = [1,2,3]
b,c,d = a
print(c)



