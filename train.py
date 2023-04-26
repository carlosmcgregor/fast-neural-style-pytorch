import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np
import time
import argparse
import logging

import vgg
import transformer
import utils

# GLOBAL SETTINGS
TRAIN_IMAGE_SIZE = 256
# DATASET_PATH = "dataset"
NUM_EPOCHS = 1
# STYLE_IMAGE_PATH = "images/mosaic.jpg"
BATCH_SIZE = 4
CONTENT_WEIGHT = 17 # 17
STYLE_WEIGHT = 50 # 25
ADAM_LR = 0.001
SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/out/"
SAVE_MODEL_EVERY = 500 # 2,000 Images with batch size 4
SEED = 35
PLOT_LOSS = 1

DATASET_PATH="datasets/coco_2014"
STYLE_IMAGE_PATH="style/style.jpg"
DEVICE = None


def train():
    # Seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    device = DEVICE or device

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load networks
    TransformerNetwork = transformer.TransformerNetwork().to(device)
    VGG = vgg.VGG16().to(device)

    # Get Style Features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1,3,1,1).to(device)
    style_image = utils.load_image(STYLE_IMAGE_PATH)
    style_tensor = utils.itot(style_image).to(device)
    style_tensor = style_tensor.add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = utils.gram(value)

    # Optimizer settings
    optimizer = optim.Adam(TransformerNetwork.parameters(), lr=ADAM_LR)

    # Loss trackers
    content_loss_history = []
    style_loss_history = []
    total_loss_history = []
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0

    # Optimization/Training Loop
    batch_count = 1
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch+1, NUM_EPOCHS))
        for content_batch, _ in train_loader:
            # Get current batch size in case of odd batch sizes
            curr_batch_size = content_batch.shape[0]

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Zero-out Gradients
            optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch[:,[2,1,0]].to(device)
            generated_batch = TransformerNetwork(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # Content Loss
            MSELoss = nn.MSELoss().to(device)
            content_loss = CONTENT_WEIGHT * MSELoss(generated_features['relu2_2'], content_features['relu2_2'])            
            batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0
            for key, value in generated_features.items():
                s_loss = MSELoss(utils.gram(value), style_gram[key][:curr_batch_size])
                style_loss += s_loss
            style_loss *= STYLE_WEIGHT
            batch_style_loss_sum += style_loss.item()

            # Total Loss
            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss.item()

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            # Save Model and Print Losses
            if (((batch_count-1)%SAVE_MODEL_EVERY == 0) or (batch_count==NUM_EPOCHS*len(train_loader))):
                # Print Losses
                print("========Iteration {}/{}========".format(batch_count, NUM_EPOCHS*len(train_loader)))
                print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
                print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
                print("Time elapsed:\t{} seconds".format(time.time()-start_time))

                # Save Model
                checkpoint_path = SAVE_MODEL_PATH + "checkpoint_" + str(batch_count-1) + ".pth"
                torch.save(TransformerNetwork.state_dict(), checkpoint_path)
                print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = utils.ttoi(sample_tensor.clone().detach())
                sample_image_path = SAVE_IMAGE_PATH + "sample0_" + str(batch_count-1) + ".png"
                utils.saveimg(sample_image, sample_image_path)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                content_loss_history.append(batch_total_loss_sum/batch_count)
                style_loss_history.append(batch_style_loss_sum/batch_count)
                total_loss_history.append(batch_total_loss_sum/batch_count)

            # Iterate Batch Counter
            batch_count+=1

    stop_time = time.time()
    # Print loss histories
    print("Done Training the Transformer Network!")
    print("Training Time: {} seconds".format(stop_time-start_time))
    print("========Content Loss========")
    print(content_loss_history) 
    print("========Style Loss========")
    print(style_loss_history) 
    print("========Total Loss========")
    print(total_loss_history) 

    # Save TransformerNetwork weights
    TransformerNetwork.eval()
    TransformerNetwork.cpu()
    final_path = SAVE_MODEL_PATH + "transformer_weight.pth"
    print("Saving TransformerNetwork weights at {}".format(final_path))
    torch.save(TransformerNetwork.state_dict(), final_path)
    print("Done saving final model")

    # Plot Loss Histories
    if (PLOT_LOSS):
        utils.plot_loss_hist(content_loss_history, style_loss_history, total_loss_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO
    )
    parser.add_argument(
        "-tis",
        "--train-image-size",
        help="Train image size.",
        dest="TRAIN_IMAGE_SIZE",
        type=int,
        default=256
    )
    parser.add_argument(
        "-ne",
        "--num-epochs",
        help="Number of epochs.",
        dest="NUM_EPOCHS",
        type=int,
        default=1
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        help="Batch size.",
        dest="BATCH_SIZE",
        type=int,
        default=4
    )
    parser.add_argument(
        "-cw",
        "--content-weight",
        help="Content weight.",
        dest="CONTENT_WEIGHT",
        type=int,
        default=8
    )
    parser.add_argument(
        "-sw",
        "--style-weight",
        help="Style weight.",
        dest="STYLE_WEIGHT",
        type=int,
        default=50
    )
    parser.add_argument(
        "-alr",
        "--adam-lr",
        help="Adam LR.",
        dest="ADAM_LR",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "-mo",
        "--save-model-path",
        help="Save model path.",
        dest="SAVE_MODEL_PATH",
        type=str,
        default="models/"
    )
    parser.add_argument(
        "-io",
        "--save-image-path",
        help="Save image path.",
        dest="SAVE_IMAGE_PATH",
        type=str,
        default="images/out/"
    )
    parser.add_argument(
        "-sme",
        "--save-model-every",
        help="Save model every so many images.",
        dest="SAVE_MODEL_EVERY",
        type=int,
        default=500
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed.",
        dest="SEED",
        type=int,
        default=35
    )
    parser.add_argument(
        "-pl",
        "--plot-loss",
        help="Plot loss.",
        dest="PLOT_LOSS",
        type=int,
        default=1
    )
    parser.add_argument(
        "-dp",
        "--dataset-path",
        help="Dataset path.",
        dest="DATASET_PATH",
        type=str,
        default="datasets/coco_2014"
    )
    parser.add_argument(
        "-de",
        "--device",
        help="Force device use.",
        dest="DEVICE",
        type=str,
        default=""
    )
    parser.add_argument(
        "-st",
        "--style-image-path",
        help="Style image path.",
        dest="STYLE_IMAGE_PATH",
        type=str,
        required=True
    )

    args = parser.parse_args()

    TRAIN_IMAGE_SIZE = args.TRAIN_IMAGE_SIZE
    NUM_EPOCHS = args.NUM_EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    CONTENT_WEIGHT = args.CONTENT_WEIGHT
    STYLE_WEIGHT = args.STYLE_WEIGHT
    ADAM_LR = args.ADAM_LR
    SAVE_MODEL_PATH = args.SAVE_MODEL_PATH
    SAVE_IMAGE_PATH = args.SAVE_IMAGE_PATH
    SAVE_MODEL_EVERY = args.SAVE_MODEL_EVERY
    SEED = args.SEED
    PLOT_LOSS = args.PLOT_LOSS
    DATASET_PATH = args.DATASET_PATH
    STYLE_IMAGE_PATH = args.STYLE_IMAGE_PATH
    DEVICE = args.DEVICE

    logging.basicConfig(level=args.loglevel)

    train()
