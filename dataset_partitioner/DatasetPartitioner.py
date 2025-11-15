import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DatasetPartitioner:
    """
    A class to handle dataset loading, partitioning (IID/non-IID/real-world style),
    transformations, and visualization for Federated Learning.
    """

    def __init__(self, dataset_name="cifar10", n_clients=10, batch_size=32, max_class_per_client=10, seed=42, verbose=True):
        self.dataset_name = dataset_name.lower()
        self.n_clients = n_clients
        self.batch_size = batch_size
        self.max_class_per_client = max_class_per_client
        self.seed = seed
        self.verbose = verbose

        # Hold client datasets and dataloaders
        self.client_datasets = []
        self.client_dataloader_list = []

        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    # --------------------------------------------------------------------------
    # 🔹 Data Transforms
    # --------------------------------------------------------------------------
    def get_default_data_transforms(self):
        """Return default transforms for the dataset."""
        transforms_train = {
            'cifar10': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        }

        transforms_eval = {
            'cifar10': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        }

        if self.verbose:
            print("\nData preprocessing: ")
            for t in transforms_train[self.dataset_name].transforms:
                print(" -", t)
            print()

        return transforms_train[self.dataset_name], transforms_eval[self.dataset_name]

    # --------------------------------------------------------------------------
    # 🔹 Real-World Partition for CIFAR-10
    # --------------------------------------------------------------------------
    def split_cifar10_realworld(self):
        """Splits CIFAR-10 into clients using a 'real world' style distribution."""
        if self.dataset_name != "cifar10":
            raise ValueError("Currently only CIFAR-10 is supported for real-world split.")

        # Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        data = np.array(trainset.data)
        labels = np.array(trainset.targets)

        # Convert to NCHW format
        data = data.transpose((0, 3, 1, 2))

        # Helper to break integer into random parts
        def break_into(n, m):
            parts = [1 for _ in range(m)]
            for _ in range(n - m):
                idx = random.randint(0, m - 1)
                parts[idx] += 1
            return parts

        n_classes = len(set(labels))
        classes = list(range(n_classes))
        np.random.shuffle(classes)
        label_indices = [list(np.where(labels == c)[0]) for c in classes]

        # Decide number of classes per client
        tmp = [np.random.randint(1, self.max_class_per_client + 1) for _ in range(self.n_clients)]
        total_partition = sum(tmp)
        class_partition = break_into(total_partition, len(classes))
        class_partition = sorted(class_partition, reverse=True)

        # Partition data by class
        class_partition_split = {}
        for ind, class_ in enumerate(classes):
            class_partition_split[class_] = [
                list(i) for i in np.array_split(label_indices[ind], class_partition[ind])
            ]

        # Assign data to each client
        clients_split = []
        for i in range(self.n_clients):
            n = tmp[i]
            j = 0
            indices = []
            while n > 0:
                class_ = classes[j]
                if len(class_partition_split[class_]) > 0:
                    indices.extend(class_partition_split[class_][-1])
                    class_partition_split[class_].pop()
                    n -= 1
                j += 1
            classes = sorted(classes, key=lambda x: len(class_partition_split[x]), reverse=True)
            clients_split.append((data[indices], labels[indices]))

        # Print summary
        if self.verbose:
            print("Data split summary:")
            for i, client in enumerate(clients_split):
                split = np.sum(client[1].reshape(1, -1) == np.arange(10).reshape(-1, 1), axis=1)
                print(f" - Client {i}: {split}")

        transform_train, _ = self.get_default_data_transforms()

        # Convert client data to datasets
        client_datasets = []
        for x_client, y_client in clients_split:
            x_tensor = torch.tensor(x_client, dtype=torch.uint8)
            y_tensor = torch.tensor(y_client, dtype=torch.long)
            dataset = [(transform_train(img), label) for img, label in zip(x_tensor, y_tensor)]
            client_datasets.append(dataset)

        self.client_datasets = client_datasets
        self.client_dataloader_list = [
            DataLoader(ds, batch_size=self.batch_size, shuffle=True) for ds in client_datasets
        ]

        return self.client_datasets, self.client_dataloader_list

    # --------------------------------------------------------------------------
    # 🔹 Label Distribution
    # --------------------------------------------------------------------------
    @staticmethod
    def get_label_distribution(trainloader, num_classes=10):
        """Return label distribution for a client’s dataloader."""
        all_labels = []
        for _, labels in trainloader:
            all_labels.extend(labels.tolist())

        label_counts = Counter(all_labels)
        for i in range(num_classes):
            label_counts.setdefault(i, 0)

        return dict(sorted(label_counts.items()))

    def print_label_distribution(self):
        """Print a table showing label distribution across all clients."""
        header = ["Client"] + list(range(10)) + ["Total"]
        col_widths = [8] + [6] * 10 + [8]
        print("".join(f"{h:>{w}}" for h, w in zip(header, col_widths)))

        all_totals = []

        for i, loader in enumerate(self.client_dataloader_list):
            dist = self.get_label_distribution(loader)
            total = sum(dist.values())
            all_totals.append(total)
            row = [i] + [dist[k] for k in range(10)] + [total]
            print("".join(f"{val:>{w}}" for val, w in zip(row, col_widths)))

        print(f"\nClients Total Images: {all_totals}")
        print(f"Total Images: {sum(all_totals)}")

    # --------------------------------------------------------------------------
    # 🔹 Visualization
    # --------------------------------------------------------------------------
    def plot_client_data_distribution(self, save_path=None):
        """Plot per-client label distributions."""
        num_clients = len(self.client_dataloader_list)
        num_classes = 10
        rows, cols = (5, 4) if num_clients > 4 else (1, num_clients)

        fig, axs = plt.subplots(rows, cols, figsize=(14, 11), sharey=True)
        axs = axs.flatten()
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

        for i in range(num_clients):
            trainloader = self.client_dataloader_list[i]
            label_counts = defaultdict(int)

            for _, labels in trainloader:
                for label in labels:
                    label_counts[int(label)] += 1

            labels = np.arange(num_classes)
            counts = [label_counts[lbl] for lbl in labels]

            axs[i].bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
            axs[i].set_title(f"Client {i+1}", fontsize=14, fontweight="bold", pad=3)
            axs[i].set_xticks(labels)
            axs[i].grid(axis="y", linestyle="--", alpha=0.4)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)

        for j in range(num_clients, len(axs)):
            axs[j].axis("off")

        fig.text(0.5, 0.03, "Class Label", ha="center", fontsize=16, fontweight="bold")
        fig.text(0.02, 0.5, "Number of Images", va="center", rotation="vertical",
                 fontsize=16, fontweight="bold")

        plt.suptitle(f"Image Class Distribution Among {num_clients} Clients",
                     fontsize=18, fontweight="bold", y=0.995)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08,
                            wspace=0.28, hspace=0.45)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")

        plt.show()

# Example
if __name__ == "__main__":
    NUM_CLIENTS = 20
    BATCH_SIZE = 32

    dp = DatasetPartitioner(
        n_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        max_class_per_client=6,
        seed=42,
        verbose=True
    )

    client_datasets, client_dataloaders = dp.split_cifar10_realworld()
    dp.print_label_distribution()
    dp.plot_client_data_distribution(save_path="./client_distribution.png")