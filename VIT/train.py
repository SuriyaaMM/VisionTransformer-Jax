from VisionTransformer import *

import jax
import jax.numpy as jnp

import optax

from torchvision.datasets.mnist     import MNIST
from torchvision.transforms         import ToTensor, Compose
from torch.utils.data.dataloader    import DataLoader

def crossEntropyLoss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def FNLoss(params, model, images, labels):
    model.setParameters(params)
    logits = model.forward(images)
    return crossEntropyLoss(logits, labels)

# path for dataset
datasetPath = "./dataset"

# load training dataset
trainDataset    = MNIST(datasetPath, download = True, transform = Compose([ToTensor()]))
trainData       = DataLoader(trainDataset, batch_size = 4, shuffle = True)

# extract first batch
images, label = next(iter(trainData))

# debug info
print(f" Image Shape: {images.shape}")

# convert the torch.tensor to numpy.ndarray
imagesNP = images.numpy()

#debug info
print(f" Image Shape (Numpy): {imagesNP.shape}")

vit = VisionTransformer(imagesNP.shape, 7, 512, 1, 4, 1, 2, jax.random.PRNGKey(69))

logit = vit.forward(imagesNP)

lr = 1e-4

optimizer = optax.adam(lr)
optimizerState = optimizer.init()

grad_fn = jax.jit(jax.value_and_grad(FNLoss))

# Training loop
for epoch in range(5):
    total_loss = 0
    for images, labels in trainData:
        images_np = images.numpy()
        labels_np = labels.numpy()

        loss, grads = grad_fn(params, vit, images_np, labels_np)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        vit.setParameters(params)

        total_loss += loss

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")


