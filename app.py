import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
import torch

# -----------------------------
# Load BLIP Model
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# -----------------------------
# Groq Client (API key via HF Secrets)
# -----------------------------
client = Groq()

# -----------------------------
# Core Function
# -----------------------------
def analyze_image(image, question):
    if image is None:
        return "Please upload an image."

    # Generate image caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Build prompt for Groq
    prompt = f"""
You are an AI image analyst.

Image Caption:
"{caption}"

User Question:
"{question if question else 'No specific question asked'}"

Tasks:
1. Describe the image clearly.
2. Identify main objects and the scene.
3. Answer the user's question if provided.
4. Provide brief reasoning.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.choices[0].message.content

    return result


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="AI Image Analyzer") as demo:
    gr.Markdown("## 🖼️ AI Image Analyzer")
    gr.Markdown(
        "Upload an image and optionally ask a question. "
        "The AI will analyze the image, identify objects, "
        "and explain the scene."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        question_input = gr.Textbox(
            label="Ask a question about the image (optional)"
        )

    analyze_btn = gr.Button("Analyze Image")
    output = gr.Textbox(
        label="AI Analysis",
        lines=10
    )

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, question_input],
        outputs=output
    )

demo.launch()
