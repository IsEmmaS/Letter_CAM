import torch
import torch.nn as nn
import cv2
import numpy as np

from torch.utils.data import DataLoader
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_batch(model, data, optimizer, criterion):
    """
    Train a model in one batch
    :return: Tuple of loss and accuracy
    """

    model.train()
    images, labels, _ = data
    _preds = model(images)
    optimizer.zero_grad()
    loss, acc = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    """
    Validate a model in one batch
    :return: Tuple of loss and accuracy
    """

    model.eval()
    images, labels, _ = data
    _preds = model(images)
    loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()


if __name__ == "__main__":
    from torch_snippets.torch_loader import Report
    from torch_snippets import resize, subplots

    from dataloader import load_alphabet_data
    from model import letterClassifier

    # fixed this, path should be abs-path otherwise may occur overflow 
    train_dl, val_dl, train_ds, val_ds = load_alphabet_data("/home/emma/ocr/archive")
    model = letterClassifier().to(device)
    criterion = model.compute_metrics  # self Made Indicator Calculations
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 2

    log = Report(num_epochs)
    for epoch in trange(num_epochs):
        N = len(train_dl)
        for batchx, data in enumerate(train_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log.record(epoch + (batchx + 1) / N, trn_loss=loss, trn_acc=acc, end="\r")

        N = len(val_dl)
        for batchx, data in enumerate(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            log.record(epoch + (batchx + 1) / N, val_loss=loss, val_acc=acc, end="\r")

        log.report_avgs(epoch + 1)

    im2fmap = nn.Sequential(
        *(list(model.model[:5].children()) + list(model.model[5][:2].children()))
    )

    def im2gradCAM(x):
        model.eval()
        logits = model(x)
        activations = im2fmap(x)
        print(activations.shape)

        pred = logits.max(-1)[-1]

        model.zero_grad()
        logits[0, pred].backward(retain_graph=True)

        pooled_grads = model.model[-7][1].weight.grad.data.mean((0, 2, 3))

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=1)[0].cpu().detach()

        return heatmap, "Uninfected" if pred.item() else "Parasitized"

    Size = 28

    def upsampleHeatmap(map, img):
        m, M = map.min(), map.max()
        map = 255 * (map - m) / (M - m)
        map = np.uint8(map)
        map = cv2.resize(map, (Size, Size))
        map = cv2.applyColorMap(255 - map, cv2.COLORMAP_JET)
        map = np.uint8(map)
        map = np.uint8(map * 0.7 + img * 0.3)
        return map

    N = 16
    _val_dl = DataLoader(
        val_ds, batch_size=N, shuffle=True, collate_fn=val_ds.collate_fn
    )

    x, y, z = next(iter(_val_dl))

    for i in range(N):
        img = resize(z[i], Size)
        heatmap, pred = im2gradCAM(x[i : i + 1])

        if pred == "Uninfected":
            continue

        heatmap = upsampleHeatmap(heatmap, img)
        subplots([img, heatmap], nc=2, figsize=(5, 3), suptitle=pred)
