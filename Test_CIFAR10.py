import unittest
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import transforms
import os

from KAN_w_cumulative_polynomials import FixedKANConfig


# -----------------------------------------------------------
# Assume your KAN code is defined in "KAN_w_cumulative_polynomials.py"
# from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN
# -----------------------------------------------------------

class TestQKANonCIFAR10(unittest.TestCase):
    def setUp(self):
        """Set up:
           1) QKAN single-layer config ([3072 -> 10])
           2) Tiny MLP for comparison
           3) CIFAR-10 (small subset) => x_data, y_data (one-hot).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -----------------------
        # 1) QKAN Config
        # -----------------------
        self.qkan_config = FixedKANConfig(
            network_shape=[3072,32,16,10],  # single-layer, 10 neurons => 10 classes
            max_degree=7,             # small polynomial degree
            complexity_weight=0.0,    # no L2 penalty for demo
            trainable_coefficients=True,
            skip_qubo_for_hidden=False
        )

        # -----------------------
        # 2) Tiny MLP as baseline
        # -----------------------
        # Something like: input=3072 -> hidden=32 -> output=10
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3072, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        ).to(self.device)

        # -----------------------
        # 3) CIFAR-10 (subset)
        # -----------------------
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        subset_indices = list(range(30000))  # 1k samples
        train_subset = Subset(dataset, subset_indices)
        loader = DataLoader(train_subset, batch_size=len(subset_indices), shuffle=False)
        images, labels = next(iter(loader))

        # Flatten => [N, 3072]
        self.x_data = images.view(images.size(0), -1).to(self.device)

        # Convert to one-hot => [N, 10]
        num_classes = 10
        y_data_oh = torch.zeros((labels.size(0), num_classes), device=self.device)
        y_data_oh.scatter_(1, labels.unsqueeze(1).to(self.device), 1.0)
        self.y_data = y_data_oh
        self.labels = labels.to(self.device)  # keep the raw labels for plotting

        # We'll just keep them in memory for demonstration
        # Optionally, create train/test splits in a real scenario.

    def test_1_qkan_training(self):
        """
        Train single-layer QKAN on CIFAR-10 subset:
         1) QUBO-based optimize => polynomial degrees
         2) MSE training on one-hot labels
         3) Save model & load it back
         4) Quick accuracy check
         5) Visualize some predictions
        """
        print("\n==== [QKAN] Single-Layer Training on CIFAR10 Subset ====")

        # 1) Build QKAN
        from KAN_w_cumulative_polynomials import FixedKAN
        qkan = FixedKAN(self.qkan_config).to(self.device)

        # 2) QUBO-based Optimize
        qkan.optimize(self.x_data, self.y_data)

        # 3) Train with MSE on one-hot
        qkan.train_model(
            x_data=self.x_data,
            y_data=self.y_data,
            num_epochs=500,
            lr=2e-3,
            complexity_weight=0.0,
            do_qubo=False  # Already did QUBO
        )
        # 5) Evaluate quick accuracy
        def compute_accuracy(model, x_data, y_data_onehot, raw_labels):
            with torch.no_grad():
                preds = model(x_data)  # [N, 10]
                predicted_labels = preds.argmax(dim=1)
                acc = (predicted_labels == raw_labels).float().mean().item() * 100
            return acc
        acc_original = compute_accuracy(qkan, self.x_data, self.y_data, self.labels)
        # 4) Save & Load the model
        save_path = f"./models/qkan_cifar10_single_layer_{datetime.now()}_{acc_original:.4f}.pth"
        qkan.save_model(save_path)
        print(f"QKAN model saved to: {save_path}")

        # Load it back
        from KAN_w_cumulative_polynomials import FixedKAN
        loaded_qkan = FixedKAN.load_model(save_path).to(self.device)



        # Compare original & loaded

        acc_loaded   = compute_accuracy(loaded_qkan, self.x_data, self.y_data, self.labels)
        print(f"[QKAN Original] Accuracy on 1k-subset: {acc_original:.2f}%")
        print(f"[QKAN Loaded  ] Accuracy on 1k-subset: {acc_loaded:.2f}%")
        self.assertAlmostEqual(acc_original, acc_loaded, delta=1e-5,
             msg="Loaded QKAN's accuracy should match the original model's accuracy")

        # Optional: visualize predictions
        self._visualize_predictions(loaded_qkan, title="[QKAN] CIFAR-10 Predictions")

        # # Cleanup
        # if os.path.exists(save_path):
        #     os.remove(save_path)

    def test_2_tiny_mlp_training(self):
        """
        Compare QKAN with a tiny MLP (3072->32->10).
        We'll do a quick MSE training on the one-hot labels as well, for a rough comparison.
        """
        print("\n==== [Tiny MLP] Training on CIFAR10 Subset ====")

        # We'll do the same number of epochs and MSE on one-hot for direct apples-to-apples
        # (Though usually we use cross-entropy for classification.)
        mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        num_epochs = 100

        for epoch in range(num_epochs):
            mlp_optimizer.zero_grad()
            preds = self.mlp(self.x_data)          # [N,10]
            mse = torch.mean((preds - self.y_data)**2)
            mse.backward()
            mlp_optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, MSE={mse.item():.6f}")

        # Evaluate accuracy
        with torch.no_grad():
            predicted_labels = self.mlp(self.x_data).argmax(dim=1)
            acc_mlp = (predicted_labels == self.labels).float().mean().item() * 100
        print(f"[Tiny MLP] Accuracy on 1k-subset: {acc_mlp:.2f}%")

        # Visualize
        self._visualize_predictions(self.mlp, title="[Tiny MLP] CIFAR-10 Predictions")

    def _visualize_predictions(self, model, title="[Model] Predictions"):
        """
        Optional: Show a small grid of predictions vs. truth.
        Just to get a quick sense, we'll pick e.g. 8 images at random from the subset.
        """
        model.eval()
        import random
        import math

        # Pick random indices
        indices = random.sample(range(self.x_data.size(0)), 8)
        images_to_show = self.x_data[indices]      # shape [8, 3072]
        raw_labels = self.labels[indices]          # shape [8]
        # forward => [8, 10]
        with torch.no_grad():
            preds = model(images_to_show)
            pred_labels = preds.argmax(dim=1)

        # Reshape images to [8, 3, 32, 32] for plotting
        images_to_show_4d = images_to_show.view(-1, 3, 32, 32).cpu().numpy()

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        fig, axes = plt.subplots(2, 4, figsize=(12,6))
        fig.suptitle(title)
        for i, ax in enumerate(axes.flat):
            img = images_to_show_4d[i]
            # Denormalize from [-1,1] to [0,1] if needed (assuming you normalized (0.5,0.5,0.5))
            # This is optional. Just quick approximate:
            img = (img * 0.5) + 0.5
            # Convert (3,32,32) -> (32,32,3) for imshow
            img = np.transpose(img, (1,2,0))
            # Clip to [0,1]
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            true_lbl = class_names[raw_labels[i].item()]
            pred_lbl = class_names[pred_labels[i].item()]
            ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)