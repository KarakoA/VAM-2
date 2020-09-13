import torch
import torch.nn.functional as F

from config.configs import Config

class Retina:
    """
    Extracts a glimpse `phi` around location `l`
    from an image `x`.

    Encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        From config:

        patch_size: size of the first square patch.
        num_patches: number of patches to extract in the glimpse.
        scale: scaling factor that controls the size of
            successive patches.
    """

    def __init__(self, conf:Config):
        self.patch_size = conf.patch_size
        self.num_patches = conf.num_patches
        self.scale = conf.glimpse_scale

    def foveate(self, x, l):
        """Extract `num_patches` square patches of size `patch_size`, centered
        at location `l`. The initial patch is a square of
        size `patch_size`, and each subsequent patch is a square
        whose side is `scale` times the size of the previous
        patch.

        The `num_patches` patches are finally resized to (patch_size, patch_size) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.patch_size

        # extract k patches of increasing size
        for i in range(self.num_patches):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.scale * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.patch_size
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        # phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.

        Args:
        x: a 4D Tensor of shape (B, C, H, W). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2).
        size: a scalar defining the size of the extracted patch.

        Returns:
            patch: a 4D Tensor of shape (B, num_patches, size, size)
        """
        B, C, H, W = x.shape

        start = self.denormalize(H, l)
        end = start + size

        # pad with zeros
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))

        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):
            patch.append(x[i, :, start[i, 1]: end[i, 1], start[i, 0]: end[i, 0]])
        return torch.stack(patch)

    def denormalize(self, T, coords):
        """Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()