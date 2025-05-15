import os
os.environ["JAX_BACKEND"] = "cuda"

import jax
import optax
import jax.numpy    as jnp
import numpy        as np

import matplotlib.pyplot    as plt
import seaborn              as sns

from torchvision.datasets.mnist     import MNIST
from torchvision.transforms         import ToTensor, Compose
from torch.utils.data               import DataLoader, Subset

from VIT.VisionTransformer import VisionTransformer

print(f"Jax Backend: {jax.default_backend()}")

def VisualizeAttention(attentionWeights : jnp.ndarray, epoch: int):

    for i, attentionWeight in enumerate(attentionWeights):
        B, h, T, _ = attentionWeight.shape 
        fig, axs = plt.subplots(1, h, figsize=(4 * h, 4))   
        if h == 1:
            axs = [axs] 
        for head in range(h):
            sns.heatmap(
                attentionWeight[0, head, :, :], 
                cmap="viridis", 
                square=True, 
                cbar=True, 
                ax=axs[head], fmt=".2f"
            )
            axs[head].set_title(f"Layer {i+1}, Head {head+1}")
            axs[head].set_xlabel("Key Token Index")
            axs[head].set_ylabel("Query Token Index")   
            plt.tight_layout()
            plt.savefig(f"./report/attention_layer_{i+1}_epoch{epoch}.png")


def CrossEntropyLoss(logits, labels):

    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def FNLoss(params : dict, model : VisionTransformer, images : jnp.ndarray, labels : jnp.ndarray):

    model.SetParameters(params)
    logits = model.forward(images)
    return CrossEntropyLoss(logits, labels)

# Create a smaller subset of the training data
def create_subset_dataset(dataset, num_samples=1000, num_classes=10):
    # Get indices for each class
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Select balanced samples from each class
    samples_per_class = num_samples // num_classes
    subset_indices = []
    for class_idx in range(num_classes):
        if len(class_indices[class_idx]) > samples_per_class:
            subset_indices.extend(class_indices[class_idx][:samples_per_class])
        else:
            subset_indices.extend(class_indices[class_idx])
    
    # Create a subset dataset
    from torch.utils.data import Subset
    return Subset(dataset, subset_indices)

def Train():

    # Path for dataset
    datasetPath = "./dataset"

    learning_rate   = 3e-4
    batch_size      = 8
    num_epochs      = 10

    __Debug = False
    
    # Load training dataset
    train_dataset = MNIST(datasetPath, download=True, transform=ToTensor())
    
    # Create a smaller subset (500 examples)
    indices         = np.random.choice(len(train_dataset), 500, replace=False)
    small_dataset   = Subset(train_dataset, indices)
    
    # Create dataloader with our consistent batch size
    train_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    
    # Get a sample batch for initialization
    sample_batch, _ = next(iter(train_loader))
    sample_np = sample_batch.numpy()
    
    # Initialize Vision Transformer
    vit = VisionTransformer(
        sample_np.shape,  
        patchSize=7,
        embedSize=64,    
        numHidden=128,    
        h=4,              
        numLayers=3,      
        numClasses=10,    # MNIST has 10 classes
        PRNGKey=jax.random.PRNGKey(42),
        report=False
    )

    # Set to Training Mode
    vit.training = True
    
    # Initialize parameters and optimizer
    params      = vit.GetParameters()
    optimizer   = optax.adam(learning_rate=learning_rate)
    opt_state   = optimizer.init(params)
    
    # Training loop
    for epoch in range(num_epochs):

        totalLoss = 0
        correct = 0
        total = 0
        
        for step, (images, labels) in enumerate(train_loader):
                
            images_jax = jnp.array(images.numpy())
            labels_jax = jnp.array(labels.numpy())
            
            # Compute loss and gradients
            loss, grads = jax.value_and_grad(FNLoss)(params, vit, images_jax, labels_jax)
            
            # Get gradient norm for monitoring
            gradNorm = optax.global_norm(grads)
            
            # Update parameters
            updates, opt_state  = optimizer.update(grads, opt_state, params)
            params              = optax.apply_updates(params, updates)
            vit.SetParameters(params)
            
            # Metrics
            totalLoss       += loss
            logits          = vit.forward(images_jax)
            preds           = jnp.argmax(logits, axis=-1)
            batchCorrect    = jnp.sum(preds == labels_jax)
            correct         += batchCorrect
            total           += labels_jax.shape[0]

            if __Debug:
                # Print batch stats
                if step % 50 == 0:

                    vit.training = False
                        
                    # Get a single batch for visualization
                    with jax.disable_jit():
                        eval_logits = vit.forward(images_jax)
                                
                    # Convert attention weights to numpy for visualization
                    numpy_attention_weights = [w.block_until_ready().copy() for w in vit.DebugAttentionWeights]
                    VisualizeAttention(numpy_attention_weights, epoch)
                            
                    # Set back to training mode
                    vit.training = True

                    batchAcc = 100 * batchCorrect / labels_jax.shape[0]
                    print(f"Epoch {epoch+1}, Step {step} | Loss: {loss:.4f} | Batch Acc: {batchAcc:.1f}% | Grad norm: {gradNorm:.4f}")
        
        # Epoch summary            
        epochLoss   = totalLoss / (step + 1)
        accuracy    = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {epochLoss:.4f} | Accuracy: {accuracy:.2f}%")
    
    print("Training completed successfully!")
    
    return vit, params, float(accuracy)

if __name__ == "__main__":
    vit, params = Train()