import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vision3d.utils.registry import MODELS


class MLP(nn.Module):
    """Multi-layer perceptron (Feed-forward network)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        """
        Initialize MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of layers
        """
        super().__init__()
        self.num_layers = num_layers

        # Create layer dimensions
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@MODELS.register()
class SimpleBbox3DHead(nn.Module):
    """Simple 3D bounding box head for predicting center, size, rotation, and class logits."""

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 256):
        super(SimpleBbox3DHead, self).__init__()

        self.num_classes = num_classes

        # Center head
        self.center_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=3,  # x, y, z coordinates
            num_layers=2,
        )

        # Size head: Log-space prediction for better numerical stability
        self.log_size_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=3,  # length, width, height
            num_layers=2,
        )

        # Quaternion head
        self.rotation_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=4,  # quaternion (x, y, z, w)
            num_layers=2,
        )

        # Classification head
        self.class_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=num_classes,  # num_classes for classification
            num_layers=2,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict 3D bounding box parameters.

        Args:
            queries: Input tensor of shape (B, num_q, d_model).

        Returns:
            Output tensor of shape (B, num_q, 10), where 10 = [center (3), size (3), quaternion (4)].
        """
        if queries.dim() != 3:
            raise ValueError(
                f"Expected input of shape (B, num_q, d_model), but got {queries.shape}"
            )

        # Predict center
        center = self.center_head(queries)

        # Predict size in log space for numerical stability
        log_size = self.log_size_head(queries)
        size = torch.exp(log_size)

        # Predict quaternion (x, y, z, w) for rotation
        quaternion = self.rotation_head(queries)
        # Normalize quaternion to ensure valid rotations
        quaternion = F.normalize(quaternion, p=2, dim=-1)

        # Predict class logits and probabilities
        class_logits = self.class_head(queries)

        # Concatenate predictions along the last dimension
        pred_bbox3d = torch.cat([center, size, quaternion], dim=-1)  # (B, num_q, 10)

        output = {
            "pred_bbox3d": pred_bbox3d,  # (B, num_q, 10)
            "pred_logits": class_logits,  # (B, num_q, num_classes)
        }

        return output
