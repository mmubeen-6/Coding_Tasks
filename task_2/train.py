import io
import os
import random
from pprint import pprint

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils
import utils.transforms as et
import wandb

# import utils.utils as utils
from datasets import Cityscapes, VOCSegmentation
from metrics import StreamSegMetrics
from networks import (
    convert_to_separable_conv,
    deeplabv3plus_resnet32,
    deeplabv3plus_resnet50,
)
from PIL import Image
from torch.utils import data
from tqdm import tqdm


def read_configs():
    from config import configs

    return configs


def get_dataset(
    dataset_name: str, dataset_root: str, crop_size: int, crop_val: bool
):
    """
    Returns Dataset And Augmentation

    Args:
        dataset_name: name of dataset
        dataset_root: path to dataset
        crop_size: crop size
        crop_val: if True, crop validation set

    Returns:
        train_dst: train dataset
        val_dst: validation dataset
    """
    if dataset_name.lower() == "vocsegmentation":
        train_transform = et.ExtCompose(
            [
                # et.ExtResize(size=opts.crop_size),
                et.ExtRandomScale((0.5, 2.0)),
                et.ExtRandomCrop(
                    size=(crop_size, crop_size), pad_if_needed=True
                ),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if crop_val:
            val_transform = et.ExtCompose(
                [
                    et.ExtResize(crop_size),
                    et.ExtCenterCrop(crop_size),
                    et.ExtToTensor(),
                    et.ExtNormalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            val_transform = et.ExtCompose(
                [
                    et.ExtToTensor(),
                    et.ExtNormalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        train_dst = VOCSegmentation(
            root=dataset_root,
            year="2012",
            image_set="train",
            download=False,
            transform=train_transform,
        )
        val_dst = VOCSegmentation(
            root=dataset_root,
            year="2012",
            image_set="val",
            download=False,
            transform=val_transform,
        )

    elif dataset_name.lower() == "cityscapes":
        train_transform = et.ExtCompose(
            [
                # et.ExtResize( 512 ),
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5
                ),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = et.ExtCompose(
            [
                # et.ExtResize( 512 ),
                et.ExtToTensor(),
                et.ExtNormalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dst = Cityscapes(
            root=dataset_root, split="train", transform=train_transform
        )
        val_dst = Cityscapes(
            root=dataset_root, split="val", transform=val_transform
        )

    else:
        raise NotImplementedError

    return train_dst, val_dst


def validate(
    opts: dict,
    model: nn.Module,
    epoch: int,
    loader: data.DataLoader,
    device: torch.device,
    metrics: StreamSegMetrics,
    experiment_path: str,
    ret_samples_ids=None,
):
    """
    Does validation and return specified samples

    Args:
        opts: config dict
        model: model to validate
        epoch: current epoch
        loader: data loader
        device: device to use
        metrics: metrics to use
        experiment_path: path to experiment
        ret_samples_ids: ids of samples to return

    Returns:
        score: score of validation
        samples_numpy: list of samples to return
    """
    metrics.reset()
    samples = []
    samples_numpy = []

    images_dir = os.path.join(experiment_path, "results", f"epoch_{epoch}")
    os.makedirs(images_dir, exist_ok=True)
    denorm = utils.Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if (
                ret_samples_ids is not None and i in ret_samples_ids
            ):  # get vis samples
                samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0])
                )

        score = metrics.get_results()

    if opts["save_val_results"]:
        for idx in range(len(samples)):
            image = samples[idx][0]
            target = samples[idx][1]
            pred = samples[idx][2]

            image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
            target = loader.dataset.decode_target(target).astype(np.uint8)
            pred = loader.dataset.decode_target(pred).astype(np.uint8)

            Image.fromarray(image).save(f"{images_dir}/{idx}_image.png")
            Image.fromarray(target).save(f"{images_dir}/{idx}_target.png")
            Image.fromarray(pred).save(f"{images_dir}/{idx}_pred.png")

            fig = plt.figure()
            plt.imshow(image)
            plt.axis("off")
            plt.imshow(pred, alpha=0.7)
            ax = plt.gca()
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.savefig(
                f"{images_dir}/{idx}_overlay.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            image_overlay = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            plt.close()

            samples_numpy.append((image, target, pred, image_overlay))

    return score, samples_numpy


def init_wandb(
    project: str, experiment_name: str, experiment_path: str, config: dict
) -> wandb.run:
    """
    Initialize wandb

    Args:
        project: project name
        experiment_name: experiment name
        experiment_path: path to experiment
        config: config dict

    Returns:
        run: wandb run
    """
    return wandb.init(
        project=project,
        name=experiment_name,
        config=config,
        dir=experiment_path,
    )


def send_images_wandb(images_list: list, step: int):
    """
    Send images to wandb

    Args:
        images_list: list of images to send, each image is a tuple of (image, target, pred, image_overlay)
        step: current step/epoch
    """
    images_input = []
    images_over = []
    images_target = []
    images_pred = []

    for image, target, pred, image_overlay in images_list:
        images_input.append(wandb.Image(image, caption="input_image"))
        images_over.append(wandb.Image(image_overlay, caption="overlay_image"))
        images_target.append(wandb.Image(target, caption="target_image"))
        images_pred.append(wandb.Image(pred, caption="pred_image"))

    wandb.log({"input_images": images_input}, step=step)
    wandb.log({"overlay_images": images_over}, step=step)
    wandb.log({"target_images": images_target}, step=step)
    wandb.log({"pred_images": images_pred}, step=step)


def setup_project(
    output_path: str,
    project_name: str,
    experiment_name: str,
    project_exist_ok: bool,
):
    """
    Setup project and experiment

    Args:
        output_path: output path
        project_name: project name
        experiment_name: experiment name
        project_exist_ok: if True, allow to overwrite existing project

    Returns:
        experiment_path: path to experiment
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    project_path = os.path.join(output_path, project_name)
    if not os.path.exists(project_path):
        os.mkdir(project_path)

    experiment_path = os.path.join(project_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    else:
        if not project_exist_ok:
            raise Exception("Project and experiment already exists")

    ckpt_path = os.path.join(experiment_path, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    results_path = os.path.join(experiment_path, "results")
    os.makedirs(results_path, exist_ok=True)
    wandb_path = os.path.join(experiment_path, "wandb")
    os.makedirs(wandb_path, exist_ok=True)

    return experiment_path


def save_ckpt(
    path: str,
    cur_itrs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_score: float,
):
    """
    Save checkpoint to path

    Args:
        path: path to save checkpoint
        cur_itrs: current iteration
        model: model to save
        optimizer: optimizer to save
        scheduler: scheduler to save
        best_score: best score to save
    """
    # check if model in nn.DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.save(
        {
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        },
        path,
    )
    print(f"Model saved at: {path}")


def main():
    opts = read_configs()
    pprint("Configs: ")
    pprint(opts)

    experiment_path = setup_project(
        opts["output_path"],
        opts["project_name"],
        opts["experiment_name"],
        opts["project_exist_ok"],
    )
    run = init_wandb(
        opts["project_name"], opts["experiment_name"], experiment_path, opts
    )

    if opts["dataset_name"].lower() == "vocsegmentation":
        opts["num_classes"] = 21
    elif opts["dataset_name"].lower() == "cityscapes":
        opts["num_classes"] = 19

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts["gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts["random_seed"])
    np.random.seed(opts["random_seed"])
    random.seed(opts["random_seed"])

    if opts["enable_vis"]:
        opts["val_batch_size"] = 1  # vis only supports batch_size=1

    train_dst, val_dst = get_dataset(
        dataset_name=opts["dataset_name"],
        dataset_root=opts["dataset_path"],
        crop_size=opts["crop_size"],
        crop_val=opts["crop_val"],
    )
    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts["batch_size"],
        shuffle=True,
        num_workers=opts["num_workers"],
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dst, batch_size=opts["val_batch_size"], shuffle=True, num_workers=1
    )
    print(
        f"Dataset: {opts['dataset_name']}, Train set: {len(train_dst)}, Val set: {len(val_dst)}"
    )

    # Set up model
    if opts["model_name"] == "deeplabv3plus_resnet32":
        model = deeplabv3plus_resnet32(num_classes=opts["num_classes"])
    elif opts["model_name"] == "deeplabv3plus_resnet50":
        model = deeplabv3plus_resnet50(num_classes=opts["num_classes"])
    else:
        raise NotImplementedError(f"Unknown model name: {opts['model_name']}")
    if opts["separable_conv"]:
        convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = model.to(device)

    # Set up metrics
    metrics = StreamSegMetrics(opts["num_classes"])

    # Set up optimizer
    optimizer = torch.optim.SGD(
        params=[
            {"params": model.backbone.parameters(), "lr": 0.1 * opts["lr"]},
            {"params": model.classifier.parameters(), "lr": opts["lr"]},
        ],
        lr=opts["lr"],
        momentum=0.9,
        weight_decay=opts["weight_decay"],
    )

    # set up learning rate scheduler
    total_batches = len(train_loader)
    if opts["lr_policy"] == "poly":
        scheduler = utils.PolyLR(
            optimizer, opts["epochs"] * total_batches, power=0.9
        )
    elif opts["lr_policy"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts["step_size"], gamma=0.1
        )
    else:
        raise NotImplementedError(f"Unknown lr policy: {opts['lr_policy']}")

    # Set up criterion
    if opts["loss_type"] == "focal_loss":
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts["loss_type"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    else:
        raise NotImplementedError(f"Unknown loss type: {opts['loss_type']}")

    # sample idxs for visualization
    vis_sample_id = (
        np.random.randint(
            0, len(val_loader), opts["vis_num_samples"], np.int32
        )
        if opts["vis_num_samples"] > 0
        else None
    )

    best_score = 0.0
    cur_itrs = 0
    interval_loss = 0

    # ==========   Train Loop   ==========#
    for cur_epoch in range(1, opts["epochs"] + 1):
        epoch_loss = 0.0  # Reset epoch loss
        num_iters = 0

        # =====  Train  =====
        model.train()
        for images, labels in train_loader:
            cur_itrs += 1
            num_iters += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            epoch_loss += np_loss

            if cur_itrs % 20 == 0:
                interval_loss = interval_loss / 20
                print(
                    f"Epoch {cur_epoch}, Itrs {cur_itrs}, Loss={interval_loss:.4f}"
                )
                interval_loss = 0.0

        epoch_loss /= num_iters
        wandb.log(data={"train": {"loss": epoch_loss}}, step=cur_epoch)
        if cur_epoch % opts["model_save_interval"] == 0:
            save_ckpt(
                os.path.join(
                    experiment_path,
                    f"checkpoints/epoch_{cur_epoch}.pt",
                ),
                cur_itrs,
                model,
                optimizer,
                scheduler,
                best_score,
            )

        # =====  Validation  =====
        if cur_epoch % opts["val_interval"] == 0:
            save_ckpt(
                os.path.join(experiment_path, f"checkpoints/latest.pt"),
                cur_itrs,
                model,
                optimizer,
                scheduler,
                best_score,
            )

            print("Validation...")
            val_score, ret_samples = validate(
                opts=opts,
                model=model,
                epoch=cur_epoch,
                loader=val_loader,
                device=device,
                metrics=metrics,
                ret_samples_ids=vis_sample_id,
                experiment_path=experiment_path,
            )
            print(metrics.to_str(val_score))
            if val_score["Mean IoU"] > best_score:  # save best model
                best_score = val_score["Mean IoU"]
                save_ckpt(
                    os.path.join(experiment_path, "checkpoints/best.pt"),
                    cur_itrs,
                    model,
                    optimizer,
                    scheduler,
                    best_score,
                )

            wandb.log(data={"val": metrics.to_dict(val_score)}, step=cur_epoch)
            send_images_wandb(images_list=ret_samples, step=cur_epoch)


if __name__ == "__main__":
    main()
