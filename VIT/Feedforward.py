import jax
import jax.numpy as jnp

class Feedforward():

    def __init__(self,
                  inputDim      : int,
                  hiddenDim     : int,
                  PRNGKey       : jax.Array):
        
        PRNGKey1, PRNGKey2, PRNGKey3, PRNGKey4  = jax.random.split(PRNGKey, 4)
        
        scale               = jnp.sqrt(1. /(inputDim + hiddenDim))
        self.W1            = jax.random.normal(PRNGKey1, (inputDim, hiddenDim)) * scale 
        self.W2            = jax.random.normal(PRNGKey2, (hiddenDim, inputDim)) * scale

    def GetParameters(self) -> dict:

        return {

            "W1" : self.W1,
            "W2" : self.W2
        }
    
    def SetParameters(self, parameters:dict):

        self.W1 = parameters["W1"]
        self.W2 = parameters["W2"]

    def __call__(self,
                 x: jnp.ndarray) -> jnp.ndarray:
        
        return jnp.einsum('bsh,hi->bsi', jax.nn.gelu(jnp.einsum('bsi,ih->bsh', x, self.W1)), self.W2)
