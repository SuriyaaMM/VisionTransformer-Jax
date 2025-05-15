import wandb
import jax
import optax
import pickle
import jax.numpy            as jnp
import numpy                as np
import seaborn              as sns
import matplotlib.pyplot    as plt

from torchvision.datasets.mnist import MNIST
from torch.utils.data           import DataLoader, Subset
from torchvision.transforms     import Compose, ToTensor

from VisionTransformer import VisionTransformer

def CrossEntropyLoss(logits, labels):

    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def FNLoss(params : dict, model : VisionTransformer, images : jnp.ndarray, labels : jnp.ndarray):

    model.SetParameters(params)
    logits = model.forward(images)
    return CrossEntropyLoss(logits, labels)

def TrainWandB():

    # Initialize wandb
    wandb.init(
        project="visiontransformer-jax",
        config={
            "learning_rate" : 1e-5,
            "batch_size"    : 8,
            "epochs"        : 50,
            "patch_size"    : 7,
            "embed_size"    : 64,
            "num_hidden"    : 128,
            "num_heads"     : 4,
            "num_layers"    : 3,
            "architecture"  : "Vision Transformer"
        }
    )
    
    # Dataset setup
    datasetPath = "./dataset"
    trainDataset   = MNIST(datasetPath, download=True, transform=ToTensor())
    indices         = np.random.choice(len(trainDataset), 500, replace=False)
    smallDataset   = Subset(trainDataset, indices)
    trainLoader    = DataLoader(smallDataset, batch_size=wandb.config.batch_size, shuffle=True)
    
    # Get sample batch size
    sampleBatch, _  = next(iter(trainLoader))
    sampleNP        = sampleBatch.numpy()
    
    # Initialize Model
    vit = VisionTransformer(
        sampleNP.shape,
        patchSize=wandb.config.patch_size,
        embedSize=wandb.config.embed_size,
        numHidden=wandb.config.num_hidden,
        h=wandb.config.num_heads,
        numLayers=wandb.config.num_layers,
        numClasses=10,
        PRNGKey=jax.random.PRNGKey(42)
    )
    
    # Initialize optimizer
    params      = vit.GetParameters()
    optimizer   = optax.adam(learning_rate=wandb.config.learning_rate)
    opt_state   = optimizer.init(params)
    
    # Training loop
    for epoch in range(wandb.config.epochs):

        epochLoss = 0
        correct = 0
        total = 0
        
        for step, (images, labels) in enumerate(trainLoader):
            imagesJax = jnp.array(images.numpy())
            labelsJax = jnp.array(labels.numpy())
            
            # Forward and backward pass
            loss, grads         = jax.value_and_grad(FNLoss)(params, vit, imagesJax, labelsJax)
            updates, opt_state  = optimizer.update(grads, opt_state, params)
            params              = optax.apply_updates(params, updates)
            vit.SetParameters(params)
            
            # Calculate metrics
            logits          = vit.forward(imagesJax)
            preds           = jnp.argmax(logits, axis=-1)
            batchCorrect    = jnp.sum(preds == labelsJax)
            correct         += batchCorrect
            total           += labelsJax.shape[0]
            epochLoss       += loss
            
            # Log batch metrics
            wandb.log({
                "batch_loss": loss,
                "batch_accuracy": float(batchCorrect) / labelsJax.shape[0],
                "grad_norm": optax.global_norm(grads)
            })
        
        # Log epoch metrics
        epoch_accuracy = float(correct) / total

        wandb.log({
            "epoch"     : epoch,
            "loss"      : epochLoss / (step + 1),
            "accuracy"  : epoch_accuracy
        })
        
        # Save model checkpoint
        if epoch % 10 == 0:

            checkpointFile = f"checkpoints/vit_epoch_{epoch}.pkl"

            with open(checkpointFile, "wb") as f:
                pickle.dump(params, f)

            # Log model to W&B
            artifact = wandb.Artifact(f"model-epoch-{epoch}", type="model")
            artifact.add_file(checkpointFile)
            wandb.log_artifact(artifact)
            
            # Log attention visualizations
            if not vit.training:
                for layeridx, attentionWeights in enumerate(vit.DebugAttentionWeights):
                    for head_idx in range(attentionWeights.shape[1]):
                        fig = plt.figure(figsize=(7, 7))
                        sns.heatmap(attentionWeights[0, head_idx], cmap="viridis", square=True)
                        plt.title(f"Layer {layeridx+1}, Head {head_idx+1}")
                        wandb.log({f"attention_l{layeridx+1}_h{head_idx+1}": wandb.Image(fig)})
                        plt.close(fig)
    
    # Save final model
    final_checkpoint = "model/trained_params.pkl"

    with open(final_checkpoint, "wb") as f:
        pickle.dump(params, f)
    
    # Create final model artifact
    artifact = wandb.Artifact("vit-final-model", type="model")
    artifact.add_file(final_checkpoint)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    return vit, params

if __name__ == "__main__":
    TrainWandB()