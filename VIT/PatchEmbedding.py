import jax
import jax.numpy as jnp

class PatchEmbedding():

    def __init__(self, patchSize: int, embedSize : int, dimImage: tuple, PRNGKey: jax.Array):

        assert len(dimImage) == 4, "Image must have dimensions, (batch_size, C, H, W)"
        
        self.B, self.C, self.H, self.W  = dimImage
        self.P      = patchSize
        self.dE     = embedSize

        assert self.H % self.P == 0, "Image height must be divisible by patch size"
        assert self.W % self.P == 0, "Image widht must be divisible by patch size"

        self.NW, self.NH = self.W // self.P, self.H // self.P

        PRNGKeyW, PRNGKeyB, PRNGKeyClassToken, PRNGKeyPositionalEmbedding = \
            jax.random.split(PRNGKey, 4)
        
        scale = jnp.sqrt(1. / self.dE)
        
        # Weights & Biases Initialization
        self.Weights  = jax.random.normal(PRNGKeyW, (self.P * self.P * self.C,self.dE)) * scale
        # Class token Initialization
        self.WeightsClassToken = jax.random.normal(PRNGKeyClassToken, (1, 1, self.dE)) * scale
        # Positional embedding
        self.WeightsPositionalEmbedding = jax.random.normal(PRNGKeyPositionalEmbedding, 
                                                     (1, self.NH * self.NW + 1, self.dE)) * scale

    def GetParameters(self) -> dict:
        
        return {
            "Weights" : self.Weights,
            "WeightsClassToken" : self.WeightsClassToken,
            "WeightsPositionalEmbedding" : self.WeightsPositionalEmbedding
        }

    def SetParameters(self, parameters: dict):
        
        self.Weights                        = parameters["Weights"]
        self.WeightsClassToken              = parameters["WeightsClassToken"]
        self.WeightsPositionalEmbedding     = parameters["WeightsPositionalEmbedding"]

    def embed(self, w: jnp.ndarray, image: jnp.ndarray):
        self.B = image.shape[0]
        # image -> (B, H, W, C) -> (B, NH, P, NW, P, C))
        x = image.reshape(self.B, self.NH, self.P, self.NW, self.P, self.C)
        # x -> (B, NH, P, NW, P, C) -> (B, NH, NW, P, P, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        # x -> (B, NH, NW, P, P, C) -> (B, NH*NW, P*P*C)
        patchedImage = x.reshape(self.B, self.NH*self.NW, self.P * self.P*self.C)
        # patchedImage -> (B, NH*NW, P*P*C) @ (P*P*C, E) -> (B, NH*NW, E)
        projectedImage = jnp.einsum("bnx,xd->bnd", patchedImage, w)
        # Add class Token
        projectedImage = jnp.concatenate([jnp.tile(self.WeightsClassToken, (self.B, 1, 1)), projectedImage], 
                                         axis=1)
        # Add Positional Embedding and Return
        return projectedImage + self.WeightsPositionalEmbedding

