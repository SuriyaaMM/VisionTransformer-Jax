import jax
import jax.numpy as jnp

from VIT.Transformer        import Transformer
from VIT.PatchEmbedding     import PatchEmbedding

class VisionTransformer():

    def __init__(self, 
                dimImage    : tuple, 
                patchSize   : int,
                embedSize   : int, 
                numHidden   : int,
                h           : int, 
                numLayers   : int,
                numClasses  : int, 
                PRNGKey     : jax.Array,
                report      = True):
        
        PRNGKeyEmbedder, PRNGKeyClassifierWeight, PRNGKeyClassifierBias = jax.random.split(PRNGKey, 3)

        PRNGKeysTransformer = jax.random.split(PRNGKey, numLayers)

        scale = jnp.sqrt(1. / embedSize)

        # Initialize patch embedder
        self.embedder = PatchEmbedding(patchSize, embedSize, dimImage, PRNGKeyEmbedder)
        # Initialize transformer blocks
        self.transformerBlocks = [\
            Transformer(embedSize, numHidden, h, PRNGKeysTransformer[i]) for i in range(numLayers)]
        # Classifier Weight
        self.classifierWeight = jax.random.normal(PRNGKeyClassifierWeight, (embedSize, numClasses)) * scale
        # Classifier Bias
        self.classifierBias   = jax.random.normal(PRNGKeyClassifierBias, (numClasses,)) * scale

        # Debug Variables
        self.report                     = report
        self.training                   = True
        self.DebugAttentionWeights      = []
        self.DebugPositionalEmbeddings  = None
    
    def GetParameters(self):

        return {
            "Embedder"      : self.embedder.GetParameters(),
            "Classifier"    : {

                "Weight"  : self.classifierWeight,
                "Bias"    : self.classifierBias
            },

            "Transformers" : [block.GetParameters() for block in self.transformerBlocks]
        }
    
    def SetParameters(self, params):

        self.embedder.SetParameters(params["Embedder"])

        self.classifierWeight   = params["Classifier"]["Weight"]
        self.classifierBias     = params["Classifier"]["Bias"]

        for i, block in enumerate(self.transformerBlocks):

            block.SetParameters(parameters = params["Transformers"][i])

    def forward(self, image: jnp.ndarray):
        
        # Embed the Image as Patches
        embeddedImage = self.embedder.embed(self.embedder.Weights, image)

        # Debug Only
        self.DebugPositionalEmbeddings = self.embedder.WeightsPositionalEmbedding
        self.DebugAttentionWeights = []

        # Apply Attention using Transformers
        for i, transformerBlock in enumerate(self.transformerBlocks):
            projectedImage = transformerBlock(embeddedImage)

            if not self.training:
                self.DebugAttentionWeights.append(transformerBlock.mha.DebugAttentionWeights)

        predictedClassToken     = projectedImage[:, 0]
        logits                  = jnp.dot(predictedClassToken, self.classifierWeight) + self.classifierBias
        return logits


