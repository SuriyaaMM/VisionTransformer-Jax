import jax
import jax.numpy as jnp

class MultiheadAttention:
    """
    @params
    dimEmbedding (tuple), (batch_size, sequence_length, embedding_dimension_length)
    h (int), number of transformer heads
    """
    def __init__(self,
                 dimEmbedding   : tuple,
                 h              : int,
                 PRNGKey        : jax.Array,
                 debug          = True):

        # embedding_dimension_length should be divisible by num heads
        assert dimEmbedding[2] % h == 0, \
            "Unsymmetrical head count for Multi-Head Attention!"

        PRNGKeyQuery, PRNGKeyKey, PRNGKeyValue, PRNGKeyOut = \
            jax.random.split(PRNGKey, 4)

        # dModel is the tuple itself
        self.dModel             = dimEmbedding
        # dE is the embedding dimension
        self.dE                 = dimEmbedding[2]
        # dK is Key matrix dimensions
        self.dK                 = self.dE // h
        # dV is value matrix dimensions
        self.dV                 = self.dE // h
        # h is num heads of transformer
        self.h                  = h

        # Glorot initialization Scale Factor
        scale = jnp.sqrt(2.0 / self.dK)

        # Initialization of Query, Key, Value and Output Weight's
        self.wQuery     = jax.random.normal(PRNGKeyQuery,  (self.dE, self.h, self.dK)) * scale
        self.wKey       = jax.random.normal(PRNGKeyKey,    (self.dE, self.h, self.dK)) * scale
        self.wValue     = jax.random.normal(PRNGKeyValue,  (self.dE, self.h, self.dV)) * scale
        self.wOut       = jax.random.normal(PRNGKeyOut,    (self.h * self.dV, self.dE)) * scale
        self.PRNGKeys   = jax.random.split(PRNGKey, h)

        # Debug Variables
        self.Debug                          = debug
        self.DebugAttentionWeights          = None
        self.DebugMultiHeadAttentionWeights = None

    def GetParameters(self) -> dict:
        return {
            "wQuery"    : self.wQuery,
            "wKey"      : self.wKey,
            "wValue"    : self.wValue,
            "wOut"      : self.wOut
        }
    
    def SetParameters(self, parameters: dict):
        
        self.wQuery     = parameters["wQuery"]
        self.wKey       = parameters["wKey"]
        self.wValue     = parameters["wValue"]
        self.wOut       = parameters["wOut"]

    def __call__(self, 
                embeddingMatrix: jnp.ndarray) -> jnp.ndarray:
        
        batchSize, sequenceLength, _ = embeddingMatrix.shape

        # Calculate Q, K and V
        Q = jnp.einsum("bse,ehq->bshq", embeddingMatrix, self.wQuery)
        K = jnp.einsum("bse,ehk->bshk", embeddingMatrix, self.wKey)
        V = jnp.einsum("bse,ehv->bshv", embeddingMatrix, self.wValue)
        
        # Reshape them into (batch_size, h, sequence_length, dim per head) (bhsq), (bhsv)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))        

        # Calculate attenion score
        attentionScores     = jnp.einsum("bhsq,bhtk->bhst", Q, K) / jnp.sqrt(self.dK)
        # Calculate attention weights
        attentionWeights    = jax.nn.softmax(attentionScores, axis=-1)
        # Calculate head output
        headOutputs         = attentionWeights @ V 
        # Concatenate heads
        concatenatedHeads   = jnp.transpose(headOutputs, (0, 2, 1, 3))
        # Reshape Concatenated heads
        concatenatedHeads   = concatenatedHeads.reshape(batchSize, sequenceLength, self.h * self.dV)
        # Calculate output
        multiheadAttention = jnp.einsum("bsd,dm->bsm", concatenatedHeads, self.wOut)
        
        self.DebugAttentionWeights          = attentionWeights
        self.DebugMultiHeadAttentionWeights = multiheadAttention
        return multiheadAttention

        