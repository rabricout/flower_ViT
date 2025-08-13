import torch
import torch.nn as nn
import os
import random
import torchvision.models
import modular.data_setup
import modular.engine
import modular.utils
import modular.predictions
from pathlib import Path
from torchinfo import summary
from torchvision import transforms

# hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001
NUM_WORKERS = 1


# Data directories
data_dir = 'datasets/flower_photos/'


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create a model to fine-tune
weights = torchvision.models.ViT_B_16_Weights.DEFAULT
model = torchvision.models.vit_b_16(weights=weights)
data_transform = weights.transforms()


# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = modular.data_setup.create_dataloaders(
    data_dir=data_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)


# Freezing the model
for param in model.parameters():
    param.requires_grad = False


# Adding custom head
model.heads = nn.Sequential(
    nn.Linear(model.heads[0].in_features, len(class_names))
)

# Set loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)


# Start training
modular.engine.train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=NUM_EPOCHS,
               device=device)


# Make predictions on and plot some random images
num_images_to_plot = 5
test_image_path_list = list(Path(data_dir).glob("*/*.jpg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

for i, image_path in enumerate(test_image_path_sample):
    modular.predictions.pred_and_plot_image(model=model, 
                        image_path=image_path,
                        class_names=class_names,
                        transform=data_transform,
                        image_size=(224, 224))

    
# Save the model with help from utils.py
modular.utils.save_model(model=model,
                 target_dir="models",
                 model_name="flowers_discriminator_ViT.pth")
