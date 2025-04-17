import jax 
import jax.numpy as jnp

class LayerNorm():

    def __init__(self,
                 dimInput   : int,
                 eplison    = 1e-6):

        self.gamma      = jnp.ones(dimInput)
        self.beta       = jnp.zeros(dimInput)
        self.eplison    = eplison
    
    def GetParameters(self) -> dict:

        return {
            "gamma" : self.gamma,
            "beta" : self.beta
        }
    
    def SetParameters(self, parameters:dict):

        self.gamma = parameters["gamma"]
        self.beta  = parameters["beta"]

    def __call__(self,
                 x:jnp.ndarray) -> jnp.ndarray:
        
        mean    = jnp.mean(x, axis = -1, keepdims = True)
        var     = jnp.var(x, axis = -1, keepdims = True)
        return self.gamma * (x - mean) / jnp.sqrt(var + self.eplison) + self.beta