from __future__ import annotations

import argparse
import sys
import time

# from contextlib import nullcontext
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Color:
    reset = bold = dim = red = green = yellow = cyan = magenta = ""

    @classmethod
    def enable(cls, on: bool) -> None:
        if not on:
            return
        cls.reset = "\033[0m"
        cls.bold = "\033[1m"
        cls.dim = "\033[2m"
        cls.red = "\033[31m"
        cls.green = "\033[32m"
        cls.yellow = "\033[33m"
        cls.cyan = "\033[36m"
        cls.magenta = "\033[35m"


def c(style: str, text: str) -> str:
    s = getattr(Color, style, "")
    return f"{s}{text}{Color.reset}" if s else text


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    log(c("dim", "Loading FashionMNIST ..."))
    t0 = time.perf_counter()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.FashionMNIST(".", train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(".", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    log(
        c("dim", f"  train={len(train_ds)}  test={len(test_ds)}  "
                 f"batch={batch_size}  ({time.perf_counter()-t0:.2f}s)")
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Train / Test / Infer
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    log()
    log(c("bold", "Train"))
    log(c("dim", f"  device={device}  epochs={epochs}  lr={lr}  params={n_params:,}"))

    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        total, correct, running_loss = 0, 0, 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        acc = correct / total
        avg_loss = running_loss / total
        elapsed = time.perf_counter() - t0
        log(
            f"  epoch {ep}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"acc={c('bold', f'{acc:.4f}')}  "
            f"{c('dim', f'{elapsed:.1f}s')}"
        )

    torch.save(model.state_dict(), "fashion_mnist.pt")
    log(c("green", "  Saved fashion_mnist.pt"))


@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.to(device)
    model.eval()

    log()
    log(c("bold", "Test"))
    t0 = time.perf_counter()
    total, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    acc = correct / total
    elapsed = time.perf_counter() - t0
    log(
        f"  accuracy={c('bold', c('green', f'{acc:.4f}'))}  "
        f"({correct}/{total})  {c('dim', f'{elapsed:.2f}s')}"
    )


@torch.no_grad()
def infer(model: nn.Module, device: torch.device = torch.device("cpu")) -> None:
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST(".", train=False, download=True, transform=tfm)
    x, y = ds[0]
    x = x.unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    logits = model(x)
    pred = logits.argmax(1).item()
    conf = logits.softmax(1)[0, pred].item()

    log()
    log(c("bold", "Inference"))
    log(c("dim", f"  sample index=0  device={device}"))
    log(f"  true  = {classes[y]}")
    log(f"  pred  = {c('bold', classes[pred])}  (conf={conf:.3f})")
    status = c("green", "PASS") if pred == y else c("red", "MISS")
    log(f"  {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Minimal FashionMNIST (train / test / infer)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--infer", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    use_color = (not args.no_color) and sys.stdout.isatty()
    Color.enable(use_color)

    device = torch.device(args.device)

    log(c("bold", c("cyan", "FashionMNIST (minimal)")))
    log(c("dim", f"PyTorch {torch.__version__}  |  device={device}"))
    if device.type == "cuda":
        log(c("dim", f"  {torch.cuda.get_device_name(device)}"))
    log()

    model = Net()
    train_loader, test_loader = get_loaders(args.batch)

    if args.train:
        train(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)

    if args.test or args.infer:
        try:
            state = torch.load("fashion_mnist.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)
            log(c("dim", "Loaded fashion_mnist.pt"))
        except FileNotFoundError:
            log(c("red", "fashion_mnist.pt not found — run with --train first"))
            return 1

    if args.test:
        test(model, test_loader, device=device)

    if args.infer:
        infer(model, device=device)

    if not (args.train or args.test or args.infer):
        p.print_help()
        return 0

    log()
    log(c("green", "Done."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
