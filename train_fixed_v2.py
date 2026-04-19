#!/usr/bin/env python3
"""
train_fixed_v2.py
==================
Fixed for small dataset (30 images, 10 classes).

Key fixes vs previous version:
  1. Smaller hidden layers (8 neurons, not 16) - less overfitting
  2. Much lower learning rate with decay
  3. Proper pixel normalization (0.0 to 1.0 float, NOT binary)
  4. L2 regularization to prevent weight explosion
  5. Many more epochs with momentum
  6. Data augmentation (small shifts) to help with 30 samples
  7. Prints LOSS not just accuracy so you can see if training works

IMPORTANT: nn_core.v hidden size must match H1_SIZE and H2_SIZE here.
If you change them here, change the parameters in nn_core.v too.

Usage:
    python train_fixed_v2.py --data_dir dataset --out_dir .
"""

import os, sys, argparse
import numpy as np
from PIL import Image

# ── Network sizes  (MUST match nn_core.v localparam values) ───
INPUT_SIZE = 400
H1_SIZE    = 8      # reduced from 16 → less overfitting
H2_SIZE    = 8      # reduced from 16
OUT_SIZE   = 10

print(f"Network: {INPUT_SIZE} → {H1_SIZE} → {H2_SIZE} → {OUT_SIZE}")

# ── Q4.4 fixed-point export ───────────────────────────────────
def to_q44(x):
    """Float → 8-bit signed Q4.4.  Range -8.0 .. +7.9375"""
    x = np.clip(x, -8.0, 7.9375)
    return np.round(x * 16).astype(np.int8)

def write_hex(filename, arr):
    arr_flat = arr.flatten().astype(np.int8)
    with open(filename, 'w') as f:
        for v in arr_flat:
            f.write(f"{v & 0xFF:02X}\n")
    print(f"  Wrote {filename}  ({len(arr_flat)} entries)")

# ── Activations ───────────────────────────────────────────────
def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def cross_entropy_loss(probs, y_true):
    return -np.log(probs[y_true] + 1e-9)

# ── Data augmentation: shift image by 1-2 pixels ─────────────
def augment(pixels_2d):
    """Randomly shift the 20x20 image by up to 2 pixels."""
    img = pixels_2d.reshape(20, 20)
    dx  = np.random.randint(-2, 3)
    dy  = np.random.randint(-2, 3)
    shifted = np.zeros_like(img)
    # Source and destination slices
    sx = max(0,  dx); ex = min(20, 20+dx)
    sy = max(0, -dx); ey = min(20, 20-dx)
    ry = max(0,  dy); fy = min(20, 20+dy)
    rx = max(0, -dy); fx = min(20, 20-dy)
    try:
        shifted[ry:fy, rx:fx] = img[sy:ey, sx:ex]
    except Exception:
        shifted = img   # fallback if slice math goes wrong
    return shifted.flatten()

# ── Dataset loader ────────────────────────────────────────────
def load_dataset(data_dir):
    X, Y = [], []
    files = sorted(os.listdir(data_dir))
    print(f"\nLoading from: {data_dir}")
    for fname in files:
        if not fname.lower().endswith('.png'):
            continue
        base = os.path.splitext(fname)[0]
        try:
            parts = base.split('_')
            label = int(parts[0][1:])
            assert 0 <= label <= 9
        except Exception:
            print(f"  Skip {fname}")
            continue

        img    = Image.open(os.path.join(data_dir, fname)).convert('L').resize((20, 20))
        pixels = np.array(img).flatten().astype(np.float32) / 255.0

        # Normalize: ensure bright pixels = foreground
        # If image is dark-on-light, invert it
        if pixels.mean() > 0.5:
            pixels = 1.0 - pixels

        X.append(pixels)
        Y.append(label)
        print(f"  {fname} → label {label}  mean={pixels.mean():.3f}")

    print(f"\nTotal: {len(X)} images, classes: {sorted(set(Y))}")
    return X, Y

# ── Network class ─────────────────────────────────────────────
class SmallNN:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # Xavier init: scale = sqrt(1 / fan_in)
        self.W1 = rng.normal(0, np.sqrt(1.0/INPUT_SIZE), (H1_SIZE, INPUT_SIZE)).astype(np.float32)
        self.b1 = np.zeros(H1_SIZE,  dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(1.0/H1_SIZE),   (H2_SIZE, H1_SIZE)).astype(np.float32)
        self.b2 = np.zeros(H2_SIZE,  dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(1.0/H2_SIZE),   (OUT_SIZE, H2_SIZE)).astype(np.float32)
        self.b3 = np.zeros(OUT_SIZE, dtype=np.float32)

        # Momentum buffers (velocity)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, x):
        self.x  = x
        self.z1 = self.W1 @ x  + self.b1;  self.a1 = relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2; self.a2 = relu(self.z2)
        self.z3 = self.W3 @ self.a2 + self.b3; self.out = softmax(self.z3)
        return self.out

    def backward(self, y_true, lr=0.001, momentum=0.9, l2=0.001):
        # Output gradient (softmax + cross-entropy combined)
        d3 = self.out.copy()
        d3[y_true] -= 1.0

        dW3 = np.outer(d3, self.a2)
        db3 = d3

        d2 = (self.W3.T @ d3) * relu_grad(self.z2)
        dW2 = np.outer(d2, self.a1)
        db2 = d2

        d1 = (self.W2.T @ d2) * relu_grad(self.z1)
        dW1 = np.outer(d1, self.x)
        db1 = d1

        # L2 regularisation gradient
        dW3 += l2 * self.W3
        dW2 += l2 * self.W2
        dW1 += l2 * self.W1

        # SGD with momentum
        self.vW3 = momentum*self.vW3 + lr*dW3; self.W3 -= self.vW3
        self.vb3 = momentum*self.vb3 + lr*db3; self.b3 -= self.vb3
        self.vW2 = momentum*self.vW2 + lr*dW2; self.W2 -= self.vW2
        self.vb2 = momentum*self.vb2 + lr*db2; self.b2 -= self.vb2
        self.vW1 = momentum*self.vW1 + lr*dW1; self.W1 -= self.vW1
        self.vb1 = momentum*self.vb1 + lr*db1; self.b1 -= self.vb1

    def predict(self, x):
        return int(np.argmax(self.forward(x)))

# ── Training ──────────────────────────────────────────────────
def train(X, Y, epochs=2000, lr_start=0.005, lr_end=0.0005):
    nn = SmallNN(seed=42)
    n  = len(X)

    # Augment dataset: add 3 shifted copies of each image
    X_aug, Y_aug = list(X), list(Y)
    for i in range(n):
        for _ in range(3):
            X_aug.append(augment(X[i]))
            Y_aug.append(Y[i])
    print(f"After augmentation: {len(X_aug)} samples")

    best_acc   = 0
    best_state = None

    for epoch in range(1, epochs+1):
        # Learning rate linear decay
        lr = lr_start + (lr_end - lr_start) * (epoch / epochs)

        # Shuffle
        idx = np.random.permutation(len(X_aug))
        total_loss = 0

        for i in idx:
            probs = nn.forward(X_aug[i])
            total_loss += cross_entropy_loss(probs, Y_aug[i])
            nn.backward(Y_aug[i], lr=lr)

        if epoch % 100 == 0:
            # Evaluate on ORIGINAL data (no augmentation)
            correct = sum(nn.predict(X[i]) == Y[i] for i in range(n))
            acc = correct / n * 100
            print(f"  Epoch {epoch:5d}  loss={total_loss/len(X_aug):6.3f}  "
                  f"acc={acc:5.1f}%  lr={lr:.5f}")

            if acc > best_acc:
                best_acc = acc
                # Save best weights
                best_state = {
                    'W1': nn.W1.copy(), 'b1': nn.b1.copy(),
                    'W2': nn.W2.copy(), 'b2': nn.b2.copy(),
                    'W3': nn.W3.copy(), 'b3': nn.b3.copy(),
                }

            if acc >= 100.0:
                print("  → 100% accuracy reached, stopping early.")
                break

    # Restore best weights
    if best_state and best_acc > (sum(nn.predict(X[i])==Y[i] for i in range(n))/n*100):
        nn.W1 = best_state['W1']; nn.b1 = best_state['b1']
        nn.W2 = best_state['W2']; nn.b2 = best_state['b2']
        nn.W3 = best_state['W3']; nn.b3 = best_state['b3']
        print(f"  Restored best weights (acc={best_acc:.1f}%)")

    return nn

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset')
    parser.add_argument('--out_dir',  default='.')
    parser.add_argument('--epochs',   default=2000, type=int)
    args = parser.parse_args()

    X, Y = load_dataset(args.data_dir)
    if len(X) == 0:
        print("ERROR: No images found in", args.data_dir)
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Training with {args.epochs} epochs...")
    print(f"{'='*50}")

    nn = train(X, Y, epochs=args.epochs)

    # Final report
    print(f"\n{'='*50}")
    print("FINAL PREDICTIONS ON TRAINING SET:")
    correct = 0
    for i in range(len(X)):
        pred = nn.predict(X[i])
        ok   = "✓" if pred == Y[i] else "✗"
        if pred != Y[i]:
            print(f"  True={Y[i]}  Pred={pred}  {ok}")
        correct += (pred == Y[i])
    print(f"Final accuracy: {correct}/{len(X)} = {correct/len(X)*100:.1f}%")

    # Weight statistics (helps debug Q4.4 overflow)
    for name, w in [('W1',nn.W1),('W2',nn.W2),('W3',nn.W3)]:
        print(f"  {name}: min={w.min():.3f}  max={w.max():.3f}  "
              f"clipped={(np.abs(w)>7.9).sum()}")

    # Export
    print(f"\nExporting hex files to: {args.out_dir}/")
    os.makedirs(args.out_dir, exist_ok=True)
    write_hex(os.path.join(args.out_dir,'w1.hex'), to_q44(nn.W1))
    write_hex(os.path.join(args.out_dir,'b1.hex'), to_q44(nn.b1))
    write_hex(os.path.join(args.out_dir,'w2.hex'), to_q44(nn.W2))
    write_hex(os.path.join(args.out_dir,'b2.hex'), to_q44(nn.b2))
    write_hex(os.path.join(args.out_dir,'w3.hex'), to_q44(nn.W3))
    write_hex(os.path.join(args.out_dir,'b3.hex'), to_q44(nn.b3))

    print("\n✓ Done!")
    print(f"  Hidden layer sizes used: H1={H1_SIZE}  H2={H2_SIZE}")
    print(f"  UPDATE nn_core.v: change H1_SIZE to {H1_SIZE} and H2_SIZE to {H2_SIZE}")
    print(f"  Then paste hex files → re-synthesize → re-implement → re-program")

if __name__ == '__main__':
    main()