"""
Grad-CAM explainability for the DR severity model.

Highlights which regions of the retina image most influenced the model's
prediction, overlaid as a heatmap. This is the "explainable" part of the
project — validate against IDRiD lesion masks (see docs/decisions/0002)
before trusting it in the demo.
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """Returns a normalized heatmap (H, W) in [0, 1] for the given class.
        If class_idx is None, uses the model's top predicted class."""
        self.model.eval()
        input_tensor.requires_grad = True
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=False)

        with torch.no_grad():
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations[0].detach()

            for i in range(activations.shape[0]):
                activations[i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-8)
            probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()

        # Explicitly release references to keep memory usage minimal
        self.gradients = None
        self.activations = None
        del output, score, activations, pooled_gradients

        return heatmap, class_idx, probs


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay the Grad-CAM heatmap on top of the original RGB image for display."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (heatmap_color * alpha + original_img * (1 - alpha)).astype(np.uint8)
    return overlay
