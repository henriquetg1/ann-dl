"""Single-layer perceptron written from scratch (NumPy only).

The same class is used in both exercises. Training always keeps the pocket
(best-so-far weights); on separable data the pocket simply ends up equal to
the final weights, so Exercise 1 can ignore it.
"""
import numpy as np


def step(z):
    """Heaviside activation: 1 if z >= 0, else 0."""
    return (z >= 0).astype(int)


class Perceptron:
    def __init__(self, w0, b0=0.0, eta=0.01):
        # Copies, so the caller's initial weights are never modified in place
        # (the eta = 1.0 re-run starts from exactly the same w0).
        self.w = np.array(w0, dtype=float)
        self.b = float(b0)
        self.eta = eta

    def predict(self, X):
        """y_hat = step(w . x + b) for every row of X."""
        return step(X @ self.w + self.b)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def fit(self, X, y, max_epochs=100):
        """Online perceptron training with the (y - y_hat) error-driven rule.

        Samples are visited in the order they appear in X (the data is
        shuffled once, at generation). Stops after an epoch with no update
        or after max_epochs. Returns a history dict with, per epoch, the
        accuracy of the current weights, the best-so-far (pocket) accuracy,
        the number of updates and the weights at the end of the epoch; and,
        per update, the weights and the accuracy right after it.
        """
        # Pocket starts with the initial weights and their accuracy.
        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_acc, self.pocket_epoch = self.accuracy(X, y), 0

        hist = {"epoch": [], "accuracy": [], "pocket_accuracy": [], "updates": [],
                "w": [], "b": [], "update_epoch": [], "update_accuracy": [],
                "update_w": [], "update_b": []}
        self.converged = False
        for epoch in range(1, max_epochs + 1):
            n_updates = 0
            for i, (xi, yi) in enumerate(zip(X, y)):
                y_hat = int(step(xi @ self.w + self.b))
                error = yi - y_hat          # 0 if correct, +1 / -1 on a mistake
                if error != 0:
                    self.w += self.eta * error * xi
                    self.b += self.eta * error
                    n_updates += 1
                    # Pocket algorithm: keep the best weights ever seen.
                    acc = self.accuracy(X, y)
                    hist["update_epoch"].append(epoch - 1 + (i + 1) / len(y))
                    hist["update_accuracy"].append(acc)
                    hist["update_w"].append(self.w.copy())
                    hist["update_b"].append(self.b)
                    if acc > self.pocket_acc:
                        self.pocket_w, self.pocket_b = self.w.copy(), self.b
                        self.pocket_acc, self.pocket_epoch = acc, epoch

            hist["epoch"].append(epoch)
            hist["accuracy"].append(self.accuracy(X, y))
            hist["pocket_accuracy"].append(self.pocket_acc)
            hist["updates"].append(n_updates)
            hist["w"].append(self.w.copy())
            hist["b"].append(self.b)
            if n_updates == 0:          # a full clean pass: converged
                self.converged = True
                break

        self.epochs = epoch
        return hist
