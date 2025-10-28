import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TensorPrompter(nn.Module):
    def __init__(self, spatial_dim, feature_dim, prompt_size):
        """
        Args:
            spatial_dim (int): The spatial dimensions of the input tensor (assumes square, e.g., 16 for [16, 16]).
            feature_dim (int): The feature dimension of the input tensor (e.g., 512).
            prompt_size (int): The size of the learnable prompt.
        """
        super(TensorPrompter, self).__init__()
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.prompt_size = prompt_size

        # Initialize learnable parameters for the prompt
        # Shape: [spatial_dim, spatial_dim, feature_dim]
        self.learnable_prompt = nn.Parameter(torch.randn(spatial_dim, spatial_dim, feature_dim) * 1e-5)

    def forward(self, x):
        """
        Add the learnable prompt to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [b, spatial_dim, spatial_dim, feature_dim].

        Returns:
            torch.Tensor: Tensor with the learnable prompt added.
        """
        batch_size = x.size(0)
        assert x.size(1) == self.spatial_dim and x.size(2) == self.spatial_dim and x.size(3) == self.feature_dim, \
            f"Input tensor shape must be [b, {self.spatial_dim}, {self.spatial_dim}, {self.feature_dim}]"

        # Add the learnable prompt to the input
        return x + self.learnable_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1)

class EdgePrompter(nn.Module):
    def __init__(self, spatial_dim, feature_dim, prompt_size=1):
        """
        Args:
            spatial_dim (int): The spatial dimensions of the input tensor (e.g., 16 for [16, 16]).
            feature_dim (int): The feature dimension of the input tensor (e.g., 512).
            prompt_size (int): The size of the prompt applied to the 16th edge region (default is 1).
        """
        super(EdgePrompter, self).__init__()
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.prompt_size = prompt_size

        # Learnable parameters for the edges (top, bottom, left, right)
        self.edge_prompt_top = nn.Parameter(torch.zeros(1, feature_dim))
        self.edge_prompt_bottom = nn.Parameter(torch.zeros(1, feature_dim))
        self.edge_prompt_left = nn.Parameter(torch.zeros(1, feature_dim))
        self.edge_prompt_right = nn.Parameter(torch.zeros(1, feature_dim))

    def forward(self, x):
        """
        Add the learnable prompt only at the edges of the tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [b, spatial_dim, spatial_dim, feature_dim].

        Returns:
            torch.Tensor: Tensor with the edge prompts added.
        """
        batch_size, h, w, c = x.size()
        assert h == self.spatial_dim and w == self.spatial_dim and c == self.feature_dim, \
            f"Input tensor must be of shape [b, {self.spatial_dim}, {self.spatial_dim}, {self.feature_dim}]"

        # Clone the input to avoid in-place modification
        output = x.clone()

        # Apply the learnable parameters to the edges
        # Top edge (row 0)
        output[:, 0, :, :] += self.edge_prompt_top.unsqueeze(0).expand(batch_size, -1, -1)

        # Bottom edge (row 15, if spatial_dim=16)
        output[:, -1, :, :] += self.edge_prompt_bottom.unsqueeze(0).expand(batch_size, -1, -1)

        # Left edge (column 0)
        output[:, :, 0, :] += self.edge_prompt_left.unsqueeze(0).expand(batch_size, -1, -1)

        # Right edge (column 15, if spatial_dim=16)
        output[:, :, -1, :] += self.edge_prompt_right.unsqueeze(0).expand(batch_size, -1, -1)

        return output


class PadPrompter(nn.Module):
    def __init__(self, image_size, prompt_size):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size

        self.base_size = image_size - pad_size*2
        # self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        # self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        # self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        # self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

        # Initialize with small values (scaled by 1e-5)
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]) * 1e-5)
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]) * 1e-5)
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]) * 1e-5)
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]) * 1e-5)

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left.cuda(), base, self.pad_right.cuda()], dim=3)
        prompt = torch.cat([self.pad_up.cuda(), prompt, self.pad_down.cuda()], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt

class BoundingBoxPrompter(nn.Module):
    def __init__(self, image_size, max_boxes=6, base_prompt_size=(32, 32)):
        """
        3D bounding-box prompter over feature maps of shape [B, H, W, C].

        Args:
            image_size (int): The height/width of the original input image (assumed square).
            max_boxes (int): Maximum number of bounding boxes to handle.
            base_prompt_size (tuple): (height, width) of the base spatial prompt.
        """
        super(BoundingBoxPrompter, self).__init__()
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.base_prompt_height, self.base_prompt_width = base_prompt_size

        # Ensure a registered parameter exists for checkpoint compatibility.
        # Will be resized lazily on first forward based on feature channels.
        self.base_prompt = nn.Parameter(torch.empty(0, 0, 0))  # [Hb, Wb, C]

    def _maybe_init_base_prompt(self, channels, device):
        needs_init = (self.base_prompt is None) or (self.base_prompt.numel() == 0) or (self.base_prompt.size(-1) != channels)
        if needs_init:
            base = torch.randn(self.base_prompt_height, self.base_prompt_width, channels, device=device) * 1e-5
            # Replace the parameter with the correctly shaped tensor
            self.base_prompt = nn.Parameter(base)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Feature tensor of shape [B, H, W, C].
            y (torch.Tensor): Bounding boxes tensor of shape [max_boxes, 4] with raw image coords
                              [x1, y1, x2, y2]. Use [-1, -1, -1, -1] for unused boxes.

        Returns:
            torch.Tensor: Tensor with prompts applied to specified bounding boxes on [H, W] and all C.
        """
        assert x.dim() == 4, "Expected x to have shape [B, H, W, C]"
        batch_size, H, W, C = x.size()
        # y can be [max_boxes, 4] (applied to all B) or [B, max_boxes, 4]
        if y.dim() == 2:
            assert y.size(0) == self.max_boxes and y.size(1) == 4, \
                f"y should have shape [{self.max_boxes}, 4]"
            y_batched = y.unsqueeze(0).expand(batch_size, -1, -1)
        elif y.dim() == 3:
            assert y.size(0) == batch_size and y.size(1) == self.max_boxes and y.size(2) == 4, \
                f"y should have shape [B, {self.max_boxes}, 4]"
            y_batched = y
        else:
            raise AssertionError("y must be of shape [max_boxes, 4] or [B, max_boxes, 4]")

        self._maybe_init_base_prompt(C, x.device)

        # Combined prompt accumulator
        combined_prompt = torch.zeros_like(x)

        # Track where prompts have already been applied to avoid overlaps (per spatial location)
        applied_mask = torch.zeros((batch_size, H, W, 1), device=x.device)

        # Scale factors from raw image coords to feature map grid
        scale_x = W / float(self.image_size)
        scale_y = H / float(self.image_size)

        for b in range(batch_size):
            for i in range(self.max_boxes):
                box = y_batched[b, i]
                if torch.all(box >= 0):
                    x1_raw, y1_raw, x2_raw, y2_raw = box

                    # Map raw image coords to feature grid and clamp
                    x1g = torch.clamp(torch.floor(x1_raw * scale_x), 0, W - 1)
                    y1g = torch.clamp(torch.floor(y1_raw * scale_y), 0, H - 1)
                    x2g = torch.clamp(torch.floor(x2_raw * scale_x), 0, W - 1)
                    y2g = torch.clamp(torch.floor(y2_raw * scale_y), 0, H - 1)

                    x_min_t, x_max_t = torch.min(x1g, x2g).long(), torch.max(x1g, x2g).long()
                    y_min_t, y_max_t = torch.min(y1g, y2g).long(), torch.max(y1g, y2g).long()

                    # Python ints for slicing
                    x_min, x_max = int(x_min_t.item()), int(x_max_t.item())
                    y_min, y_max = int(y_min_t.item()), int(y_max_t.item())

                    bbox_height = y_max - y_min + 1
                    bbox_width = x_max - x_min + 1
                    if bbox_height <= 0 or bbox_width <= 0:
                        continue

                    # Resize base prompt spatially to bbox size (keep channels)
                    prompt_hw_c = self.base_prompt.permute(2, 0, 1).unsqueeze(0)  # [1, C, Hb, Wb]
                    resized = F.interpolate(prompt_hw_c, size=(bbox_height, bbox_width), mode='bilinear', align_corners=False)
                    resized = resized.squeeze(0).permute(1, 2, 0)  # [bbox_h, bbox_w, C]

                    # Create mask for this box (shared across channels)
                    box_mask = torch.zeros((batch_size, H, W, 1), device=x.device)
                    box_mask[b, y_min:y_max + 1, x_min:x_max + 1, :] = 1.0

                    # Apply only to regions not yet covered (per-sample)
                    new_mask = box_mask * (1.0 - applied_mask)
                    if torch.sum(new_mask[b]) == 0:
                        continue

                    # Place the resized prompt into spatial location and mask
                    prompt_padded = torch.zeros_like(x)
                    prompt_padded[b, y_min:y_max + 1, x_min:x_max + 1, :] = resized
                    prompt_padded = prompt_padded * new_mask.expand(-1, -1, -1, C)

                    combined_prompt += prompt_padded
                    applied_mask += new_mask

        return x + combined_prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)
