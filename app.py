"""
RetinaScan AI — Hugging Face Spaces (Gradio SDK) Entrypoint.
Enables 100% free hosting on Hugging Face Spaces with 16 GB CPU RAM.
"""
import base64
import io
from PIL import Image
import gradio as gr

from backend.app.inference import load_model, predict, CLASS_NAMES, CLASS_EXPLANATIONS

# Pre-load trained model
load_model()

def screen_retina(input_img):
    if input_img is None:
        return None, None, "Please upload a retinal fundus image.", {}

    # Convert PIL Image to bytes for inference
    buf = io.BytesIO()
    input_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    result = predict(image_bytes)

    # Decode returned base64 images
    heatmap_bytes = base64.b64decode(result["heatmap_overlay_b64"])
    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

    processed_bytes = base64.b64decode(result["processed_image_b64"])
    processed_img = Image.open(io.BytesIO(processed_bytes))

    # Formatted clinical report
    summary = (
        f"## Diagnosis: {result['predicted_label']} (Class {result['predicted_class']})\n"
        f"**Confidence:** `{result['confidence'] * 100:.1f}%`\n\n"
        f"**Clinical Recommendation:**\n> {result['explanation']}"
    )

    probs = {k: float(v) for k, v in result["class_probabilities"].items()}
    return heatmap_img, processed_img, summary, probs

with gr.Blocks(title="RetinaScan AI — Diabetic Retinopathy Screening") as demo:
    gr.Markdown(
        """
        # 👁️ RetinaScan AI
        ### Explainable AI Screening for Diabetic Retinopathy (SIH26038 · MathWorks Track)
        Upload a digital fundus photograph to perform automated ICDR staging and generate Grad-CAM explainability heatmaps.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Retinal Fundus Image")
            scan_button = gr.Button("Scan Retina", variant="primary", size="lg")

        with gr.Column(scale=1):
            report_markdown = gr.Markdown(label="Clinical Summary")
            probs_label = gr.Label(label="Severity Probability Distribution")

    gr.Markdown("### Explainability & Visual Inspection")
    with gr.Row():
        heatmap_output = gr.Image(label="Grad-CAM Saliency Overlay")
        processed_output = gr.Image(label="Preprocessed Fundus (CLAHE)")

    scan_button.click(
        fn=screen_retina,
        inputs=[input_image],
        outputs=[heatmap_output, processed_output, report_markdown, probs_label],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
