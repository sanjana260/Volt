"""
Volt with Articulation Heads for Articulate3D Dataset

Extended Volt model with two additional classification heads:
1. Movable part segmentation (3 classes: background, rotation, translation)
2. Interactable part segmentation (binary)

Author: Sanjana Mohan
"""

import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.volt.volt_base import Volt


@MODELS.register_module()
class VoltArticulate(nn.Module):
    """Volt backbone with articulation heads for Articulate3D dataset."""

    def __init__(
        self,
        backbone_out_channels=128,
        num_seg_classes=100,
        backbone=None,
        freeze_backbone=False,
    ):
        """
        Args:
            backbone_out_channels: output feature dimension from backbone decoder
            num_seg_classes: number of semantic segmentation classes
            backbone: dict config for Volt backbone
            freeze_backbone: whether to freeze backbone parameters
        """
        super().__init__()
        self.backbone_out_channels = backbone_out_channels
        self.num_seg_classes = num_seg_classes
        self.freeze_backbone = freeze_backbone

        # MODIFIED: Build Volt backbone
        from pointcept.models.builder import build_model

        self.backbone = build_model(backbone)

        # MODIFIED: Main semantic segmentation head (existing)
        self.seg_head = nn.Linear(backbone_out_channels, num_seg_classes)

        # MODIFIED: Articulation head 1: Movable part segmentation
        # 3 classes: background (0), rotation (1), translation (2)
        self.movable_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, 3),
        )

        # MODIFIED: Articulation head 2: Interactable part segmentation (binary)
        self.interactable_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, 1),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        """
        Args:
            data_dict: input batch dict with keys like 'coord', 'grid_coord', 'feat', 'batch'

        Returns:
            features: backbone features (N, backbone_out_channels)
            seg_logits: semantic segmentation logits (N, num_seg_classes)
            movable_logits: movable part logits (N, 3)
            interactable_logits: interactable part logits (N, 1)
        """
        # MODIFIED: Extract backbone features
        features = self.backbone(data_dict)

        # MODIFIED: Compute all head outputs
        seg_logits = self.seg_head(features)
        movable_logits = self.movable_head(features)
        interactable_logits = self.interactable_head(features)

        return features, seg_logits, movable_logits, interactable_logits


@MODELS.register_module()
class ArticulateSegmentor(nn.Module):
    """Wrapper for Volt with articulation heads, matching training/inference interface."""

    def __init__(
        self,
        backbone=None,
        num_classes=100,
        backbone_out_channels=128,
        criteria=None,
        articulation_criteria=None,
        articulation_weight=0.5,
        freeze_backbone=False,
    ):
        """
        Args:
            backbone: Volt backbone config
            num_classes: number of semantic segmentation classes
            backbone_out_channels: output channels of backbone decoder
            criteria: list of loss configs for semantic segmentation
            articulation_criteria: loss config for articulation tasks
            articulation_weight: weight of articulation loss relative to seg loss
            freeze_backbone: freeze backbone parameters
        """
        super().__init__()
        self.articulation_weight = articulation_weight

        # MODIFIED: Build model with articulation heads
        self.model = VoltArticulate(
            backbone_out_channels=backbone_out_channels,
            num_seg_classes=num_classes,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
        )

        # MODIFIED: Build loss criteria
        from pointcept.models.losses import build_criteria

        self.seg_criteria = build_criteria(criteria)
        # Articulation loss is computed directly in forward if available
        self.articulation_criteria_cfg = articulation_criteria

        if articulation_criteria is not None:
            self.articulation_criteria = build_criteria([articulation_criteria])
        else:
            self.articulation_criteria = None

    def forward(self, input_dict):
        """
        Args:
            input_dict: batch dict with:
                - geometry/features: coord, grid_coord, feat, batch, etc.
                - labels: segment (semantic labels)
                - articulation (optional): movable_label, interactable_label, has_articulation

        Returns:
            dict with keys:
                - loss: total loss (train mode)
                - seg_logits: semantic segmentation logits
                - movable_logits: movable part logits (optional)
                - interactable_logits: interactable part logits (optional)
        """
        # MODIFIED: Extract features and all head outputs
        features, seg_logits, movable_logits, interactable_logits = self.model(
            input_dict
        )

        return_dict = {}

        # MODIFIED: Training mode
        if self.training:
            # Semantic segmentation loss
            seg_loss = self.seg_criteria(seg_logits, input_dict["segment"])
            total_loss = seg_loss

            # Articulation loss (if labels available)
            if (
                self.articulation_criteria is not None
                and "movable_label" in input_dict
                and "interactable_label" in input_dict
            ):
                pred_dict = {
                    "movable_logits": movable_logits,
                    "interactable_logits": interactable_logits,
                }
                artic_loss = self.articulation_criteria(pred_dict, input_dict)
                if isinstance(artic_loss, torch.Tensor) and artic_loss.item() > 0:
                    total_loss = seg_loss + self.articulation_weight * artic_loss

            return_dict["loss"] = total_loss
            return return_dict

        # MODIFIED: Evaluation/Test mode
        return_dict["seg_logits"] = seg_logits
        return_dict["movable_logits"] = movable_logits
        return_dict["interactable_logits"] = interactable_logits

        if "segment" in input_dict:
            seg_loss = self.seg_criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = seg_loss

        return return_dict
