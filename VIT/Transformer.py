from MultiheadAttention import MultiheadAttention
from Feedforward import Feedforward
from LayerNorm import LayerNorm

import jax
import jax.numpy as jnp

class Transformer():

    def __init__(self, 
                 dimEmbedding: int, 
                 dimHidden: int, 
                 h: int, 
                 PRNGKey: jax.Array):

        self.mha                = MultiheadAttention((None, None, dimEmbedding), h, PRNGKey)
        self.linear             = Feedforward(dimEmbedding, dimHidden, PRNGKey)
        self.norm1              = LayerNorm(dimEmbedding)
        self.norm2              = LayerNorm(dimEmbedding)

    def GetParameters(self) -> dict:

        return {
            "MHA"           : self.mha.GetParameters(),
            "Linear"        : self.linear.GetParameters(),
            "LayerNorm1"    : self.norm1.GetParameters(),
            "LayerNorm2"    : self.norm2.GetParameters()
        }
    
    def SetParameters(self, parameters:dict) -> dict:

        self.mha.SetParameters(parameters["MHA"])
        self.linear.SetParameters(parameters["Linear"])
        self.norm1.SetParameters(parameters["LayerNorm1"])
        self.norm2.SetParameters(parameters["LayerNorm2"])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        # normalize input matrix
        normalized      = self.norm1(x)
        # attend input matrix
        attended_output = self.mha(normalized)
        # residual connection
        x               = x + attended_output  
        # normalize residual
        normalized      = self.norm2(x)
        # pass through linear layer
        ffOut           = self.linear(normalized)
        # residual connection
        x               = x + ffOut
        return x