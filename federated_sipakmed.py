"""
Local PyTorch federated-learning simulation for binary SIPaKMeD classification.

Expected data layout:
    data/sipakmed_binary/
        im_Dyskeratotic/
            im_Dyskeratotic/CROPPED/
        im_Koilocytotic/
            im_Koilocytotic/CROPPED/
        im_Metaplastic/
            im_Metaplastic/CROPPED/
        im_Parabasal/
            im_Parabasal/CROPPED/
        im_Superficial-Intermediate/
            im_Superficial-Intermediate/CROPPED/

When a CROPPED folder is present, only its individual cell images are used. The
binary grouping below treats dyskeratotic and koilocytotic cells as abnormal, and
metaplastic, parabasal, and superficial-intermediate cells as normal. Verify this
mapping against your study protocol before reporting results. The majority class is
undersampled to balance the binary dataset.

The prompt does not specify the paper's numeric hyperparameters, so the defaults
below are configurable starting points rather than a claim of exact paper replication.

This simulates clients and a central server in one process. It does not implement
network transport, differential privacy, or secure aggregation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS


# Keys must match the names of the immediate class folders under --data.
# Positive class (1) is treated as Abnormal when computing metrics.
CLASS_TO_BINARY = {
    "im_dyskeratotic": 1,
    "im_koilocytotic": 1,
    "im_metaplastic": 0,
    "im_parabasal": 0,
    "im_superficial-intermediate": 0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BinaryImageFolder(ImageFolder):
    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        root = Path(directory)
        classes = sorted(
            class_dir.name
            for class_dir in root.iterdir()
            if class_dir.is_dir()
            and any(
                image.is_file() and image.name.casefold().endswith(IMG_EXTENSIONS)
                for image in class_dir.rglob("*")
            )
        )
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folders containing supported images in {directory}."
            )
        return classes, {class_name: index for index, class_name in enumerate(classes)}

    def __init__(
        self,
        root: Path,
        transform: transforms.Compose,
        class_to_binary: Dict[str, int],
    ) -> None:
        super().__init__(str(root), transform=transform)

        cropped_samples = [
            sample
            for sample in self.samples
            if any(part.casefold() == "cropped" for part in Path(sample[0]).parts)
        ]
        if cropped_samples:
            self.samples = cropped_samples
            self.imgs = self.samples
            self.targets = [class_index for _, class_index in self.samples]
            self.image_source = "CROPPED cell images"
        else:
            self.image_source = "all supported images"

        mapping = {name.strip().casefold(): value for name, value in class_to_binary.items()}
        active_class_indices = {class_index for _, class_index in self.samples}
        missing = [
            self.classes[class_index]
            for class_index in active_class_indices
            if self.classes[class_index].casefold() not in mapping
        ]
        if missing:
            raise ValueError(
                f"No binary label configured for folder(s) {missing}. "
                f"Edit CLASS_TO_BINARY; found folders: {self.classes}"
            )

        labels = {
            mapping[self.classes[class_index].casefold()]
            for class_index in active_class_indices
        }
        if labels != {0, 1}:
            raise ValueError("The configured class folders must map to both 0 and 1.")

        self.binary_targets = [
            mapping[self.classes[class_index].casefold()]
            for _, class_index in self.samples
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, _ = super().__getitem__(index)
        return image, self.binary_targets[index]


def stratified_train_validation_split(
    labels: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    indices_by_label: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)

    train_indices: List[int] = []
    validation_indices: List[int] = []

    for label in sorted(indices_by_label):
        indices = indices_by_label[label]
        if len(indices) < 2:
            raise ValueError(
                f"Need at least two images for each binary label to create a "
                f"stratified train/validation split; label {label} has {len(indices)}."
            )
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def balanced_binary_indices(labels: Sequence[int], seed: int) -> List[int]:
    """Undersample the majority binary class to match the minority class."""
    indices_by_label: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)

    if set(indices_by_label) != {0, 1}:
        raise ValueError("Balancing requires examples for both binary labels 0 and 1.")

    sample_count = min(len(indices) for indices in indices_by_label.values())
    if sample_count == 0:
        raise ValueError("Cannot balance classes when either class has no images.")

    rng = random.Random(seed)
    balanced_indices: List[int] = []
    for label in (0, 1):
        class_indices = indices_by_label[label]
        rng.shuffle(class_indices)
        balanced_indices.extend(class_indices[:sample_count])

    rng.shuffle(balanced_indices)
    return balanced_indices


def partition_iid(
    indices: Sequence[int],
    client_count: int,
    seed: int,
) -> List[List[int]]:
    """Randomly distribute shuffled examples; client sizes differ by at most one."""
    if client_count < 1:
        raise ValueError("--clients must be at least 1.")

    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::client_count] for i in range(client_count)]


def cluster_clients_by_label_distribution(
    client_indices: Sequence[Sequence[int]],
    labels: Sequence[int],
    cluster_count: int,
    seed: int,
    iterations: int = 30,
) -> List[List[int]]:
    """Cluster clients using their local binary-label frequency vectors."""
    if cluster_count < 1 or cluster_count > len(client_indices):
        raise ValueError("--clusters must be between 1 and --clients.")

    profiles = torch.zeros((len(client_indices), 2), dtype=torch.float32)
    for client_id, indices in enumerate(client_indices):
        if not indices:
            raise ValueError(f"Client {client_id} has no training examples.")
        for index in indices:
            profiles[client_id, labels[index]] += 1
        profiles[client_id] /= profiles[client_id].sum()

    rng = random.Random(seed)
    initial_ids = rng.sample(range(len(client_indices)), cluster_count)
    centers = profiles[initial_ids].clone()
    assignments = torch.zeros(len(client_indices), dtype=torch.long)

    for _ in range(iterations):
        distances = torch.cdist(profiles, centers, p=2)
        new_assignments = distances.argmin(dim=1)
        new_centers = centers.clone()

        for cluster_id in range(cluster_count):
            members = profiles[new_assignments == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(dim=0)

        if torch.equal(assignments, new_assignments) and torch.equal(centers, new_centers):
            assignments = new_assignments
            break
        assignments = new_assignments
        centers = new_centers

    groups = [
        [client_id for client_id, assignment in enumerate(assignments.tolist())
         if assignment == cluster_id]
        for cluster_id in range(cluster_count)
    ]
    return [group for group in groups if group]


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_dict_nbytes(state: Dict[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in state.values())


class WeightedStateAccumulator:
    """Accumulate a sample-weighted model average without retaining every client model."""

    def __init__(self) -> None:
        self.total_weight = 0
        self.float_sums: Dict[str, torch.Tensor] = {}
        self.float_dtypes: Dict[str, torch.dtype] = {}
        self.non_float_values: Dict[str, torch.Tensor] = {}

    def add(self, state: Dict[str, torch.Tensor], weight: int) -> None:
        if weight <= 0:
            raise ValueError("Aggregation weights must be positive.")

        if self.total_weight == 0:
            for name, value in state.items():
                if torch.is_floating_point(value):
                    accumulator_dtype = (
                        torch.float32
                        if value.dtype in (torch.float16, torch.bfloat16)
                        else value.dtype
                    )
                    self.float_dtypes[name] = value.dtype
                    self.float_sums[name] = value.to(accumulator_dtype).mul(weight)
                else:
                    self.non_float_values[name] = value.clone()
        else:
            for name, value in state.items():
                if torch.is_floating_point(value):
                    self.float_sums[name].add_(
                        value.to(self.float_sums[name].dtype),
                        alpha=weight,
                    )
                else:
                    self.non_float_values[name] = torch.maximum(
                        self.non_float_values[name],
                        value,
                    )

        self.total_weight += weight

    def finish(self, fallback: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.total_weight == 0:
            return fallback

        result: Dict[str, torch.Tensor] = {}
        for name, value in self.float_sums.items():
            result[name] = (value / self.total_weight).to(self.float_dtypes[name])
        result.update(self.non_float_values)
        return result


class FederatedServer:
    """Central aggregation logic; FedProx changes local training, not server averaging."""

    def fedavg(
        self,
        accumulator: WeightedStateAccumulator,
        previous_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return accumulator.finish(previous_state)

    def fedprox(
        self,
        accumulator: WeightedStateAccumulator,
        previous_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.fedavg(accumulator, previous_state)

    def clustered_fedavg(
        self,
        accumulators: Sequence[WeightedStateAccumulator],
        previous_states: Sequence[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            accumulator.finish(previous_state)
            for accumulator, previous_state in zip(accumulators, previous_states)
        ]


class YoloNanoBinaryClassifier(nn.Module):
    """YOLO classification backbone with an adaptive-pooling binary head."""

    def __init__(self, backbone: nn.Module, feature_channels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_channels, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim != 4:
            raise RuntimeError(
                f"Expected a 4D feature map from the YOLO backbone, got "
                f"shape {tuple(features.shape)}."
            )
        features = self.pool(features).flatten(1)
        return self.classifier(features)


def build_model(weights: str, image_size: int) -> YoloNanoBinaryClassifier:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install dependencies with: "
            "pip install torch torchvision ultralytics pillow"
        ) from exc

    detector = YOLO(weights)
    if getattr(detector, "task", None) != "classify":
        raise ValueError(
            f"{weights!r} is not an Ultralytics classification model. "
            "Use a classification checkpoint such as yolov8n-cls.pt."
        )

    layers = list(detector.model.model.children())
    if len(layers) < 2:
        raise RuntimeError("Could not find a YOLO backbone and classification head.")

    # Drop the original classification head; this model has no box-detection branch.
    backbone = nn.Sequential(*layers[:-1])
    with torch.inference_mode():
        features = backbone(torch.zeros(1, 3, image_size, image_size))
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if features.ndim != 4:
        raise RuntimeError(
            f"Expected the YOLO backbone to return a 4D feature map, got "
            f"shape {tuple(features.shape)}."
        )

    model = YoloNanoBinaryClassifier(backbone, int(features.shape[1]))
    del detector
    return model


def local_train(
    model_template: nn.Module,
    initial_state: Dict[str, torch.Tensor],
    dataset: Dataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    regularizer: Optional[str],
    regularizer_coefficient: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    if not indices:
        raise ValueError("A selected client has no local examples.")

    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(initial_state, strict=True)
    model.train()

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )

    reference_params: Dict[str, torch.Tensor] = {}
    if regularizer is not None:
        for name, parameter in model.named_parameters():
            reference_params[name] = initial_state[name].to(
                device=device,
                dtype=parameter.dtype,
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    total_steps = max(1, epochs * len(loader))
    step = 0

    for _ in range(epochs):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float().view(-1, 1)

            progress = step / max(1, total_steps - 1)
            adaptive_lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = adaptive_lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)

            if regularizer is not None:
                distance_squared = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    distance_squared = distance_squared + (
                        parameter - reference_params[name]
                    ).square().sum()

                if regularizer == "fedprox":
                    loss = loss + (regularizer_coefficient / 2.0) * distance_squared
                elif regularizer == "personalization":
                    loss = loss + regularizer_coefficient * distance_squared
                else:
                    raise ValueError(f"Unknown local regularizer: {regularizer}")

            loss.backward()
            optimizer.step()
            step += 1

    result = cpu_state_dict(model)
    del model, loader, optimizer, criterion, reference_params
    return result


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    states: Sequence[Dict[str, torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    if not states:
        raise ValueError("Evaluation requires at least one model state.")

    model.eval()
    true_positive = true_negative = false_positive = false_negative = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long().view(-1)

        logits_sum: Optional[torch.Tensor] = None
        for state in states:
            model.load_state_dict(state, strict=True)
            logits = model(images)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        probabilities = torch.sigmoid(logits_sum / len(states)).view(-1)
        predictions = (probabilities >= 0.5).long()

        true_positive += int(((predictions == 1) & (targets == 1)).sum().item())
        true_negative += int(((predictions == 0) & (targets == 0)).sum().item())
        false_positive += int(((predictions == 1) & (targets == 0)).sum().item())
        false_negative += int(((predictions == 0) & (targets == 1)).sum().item())

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    accuracy = divide(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)

    positive_f1 = divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    negative_f1 = divide(
        2 * true_negative,
        2 * true_negative + false_positive + false_negative,
    )

    return {
        "top1_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local federated simulation for binary SIPaKMeD image classification."
    )
    parser.add_argument("--data", type=Path, default=Path("data/sipakmed_binary"))
    parser.add_argument("--weights", default="yolov8n-cls.pt")
    parser.add_argument("--clients", type=int, default=5, choices=(1, 3, 4, 5, 10, 20))
    parser.add_argument(
        "--algorithm",
        choices=("fedavg", "fedprox", "clustered"),
        default="fedavg",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx coefficient.")
    parser.add_argument(
        "--personalization-lambda",
        type=float,
        default=0.01,
        help="Clustered personalization coefficient.",
    )
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("--rounds and --local-epochs must be positive.")
    if not 0.0 < args.client_fraction <= 1.0:
        raise ValueError("--client-fraction must be in (0, 1].")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1).")
    if args.algorithm == "clustered" and not 1 <= args.clusters <= args.clients:
        raise ValueError("For clustered training, --clusters must be in [1, --clients].")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    dataset = BinaryImageFolder(
        args.data,
        transform=image_transform,
        class_to_binary=CLASS_TO_BINARY,
    )

    balanced_indices = balanced_binary_indices(dataset.binary_targets, args.seed)
    balanced_labels = [dataset.binary_targets[index] for index in balanced_indices]
    train_indices, validation_indices = stratified_train_validation_split(
        balanced_labels,
        args.validation_fraction,
        args.seed,
    )
    train_indices = [balanced_indices[index] for index in train_indices]
    validation_indices = [balanced_indices[index] for index in validation_indices]
    client_indices = partition_iid(train_indices, args.clients, args.seed)

    if any(not indices for indices in client_indices):
        raise ValueError(
            "At least one client has no examples. Reduce --clients or provide more data."
        )

    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model_template = build_model(args.weights, args.image_size)
    initial_state = cpu_state_dict(model_template)
    parameter_count = sum(parameter.numel() for parameter in model_template.parameters())
    model_bytes = state_dict_nbytes(initial_state)

    active_classes = sorted(
        {dataset.classes[class_index] for _, class_index in dataset.samples}
    )
    print(
        f"Image source: {dataset.image_source}; active classes: {active_classes} "
        f"(binary mapping: {CLASS_TO_BINARY})"
    )
    print(
        f"Balanced images: {len(balanced_indices)} total "
        f"({len(balanced_indices) // 2} normal, {len(balanced_indices) // 2} abnormal); "
        "the majority class is undersampled."
    )
    print(
        f"Train/validation images: {len(train_indices)}/{len(validation_indices)}; "
        f"client sizes: {[len(indices) for indices in client_indices]}"
    )
    print(f"Trainable model parameters: {parameter_count:,}")
    print(
        f"Model state payload: {model_bytes / (1024 ** 2):.2f} MiB; "
        "communication estimate excludes protocol overhead."
    )

    cluster_groups: List[List[int]] = []
    client_to_cluster: Dict[int, int] = {}
    cluster_states: List[Dict[str, torch.Tensor]] = []

    if args.algorithm == "clustered":
        cluster_groups = cluster_clients_by_label_distribution(
            client_indices,
            dataset.binary_targets,
            args.clusters,
            args.seed,
        )
        for cluster_id, members in enumerate(cluster_groups):
            for client_id in members:
                client_to_cluster[client_id] = cluster_id
        cluster_states = [copy.deepcopy(initial_state) for _ in cluster_groups]
        print(
            f"Label-distribution clustering produced {len(cluster_groups)} "
            f"non-empty cluster(s): {cluster_groups}"
        )

    server = FederatedServer()
    global_state = initial_state
    evaluation_model = copy.deepcopy(model_template).to(device)

    for round_number in range(1, args.rounds + 1):
        selected_count = max(1, math.ceil(args.clients * args.client_fraction))
        selected_clients = sorted(
            random.sample(range(args.clients), selected_count)
        )

        accumulator_count = (
            len(cluster_groups) if args.algorithm == "clustered" else 1
        )
        accumulators = [
            WeightedStateAccumulator() for _ in range(accumulator_count)
        ]

        for client_id in selected_clients:
            if args.algorithm == "clustered":
                cluster_id = client_to_cluster[client_id]
                client_start_state = cluster_states[cluster_id]
                regularizer = "personalization"
                coefficient = args.personalization_lambda
                target_accumulator = cluster_id
            else:
                client_start_state = global_state
                target_accumulator = 0
                if args.algorithm == "fedprox":
                    regularizer = "fedprox"
                    coefficient = args.mu
                else:
                    regularizer = None
                    coefficient = 0.0

            updated_state = local_train(
                model_template=model_template,
                initial_state=client_start_state,
                dataset=dataset,
                indices=client_indices[client_id],
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
                epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                regularizer=regularizer,
                regularizer_coefficient=coefficient,
                seed=args.seed + round_number * 1009 + client_id,
            )
            accumulators[target_accumulator].add(
                updated_state,
                weight=len(client_indices[client_id]),
            )

            # Client model and activations are released; only the CPU aggregation
            # accumulator remains resident between clients.
            del updated_state
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.algorithm == "clustered":
            cluster_states = server.clustered_fedavg(accumulators, cluster_states)
            evaluation_states = cluster_states
        elif args.algorithm == "fedprox":
            global_state = server.fedprox(accumulators[0], global_state)
            evaluation_states = [global_state]
        else:
            global_state = server.fedavg(accumulators[0], global_state)
            evaluation_states = [global_state]

        metrics = evaluate(
            evaluation_model,
            evaluation_states,
            validation_loader,
            device,
        )

        # One full model download and upload per selected client.
        round_communication_bytes = 2 * selected_count * model_bytes
        print(
            f"Round {round_number:03d}/{args.rounds} | "
            f"clients={selected_clients} | "
            f"Top-1 Acc={metrics['top1_accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Specificity={metrics['specificity']:.4f} | "
            f"Macro F1={metrics['macro_f1']:.4f} | "
            f"Communication={round_communication_bytes / (1024 ** 2):.2f} MiB"
        )

    del evaluation_model, model_template, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
