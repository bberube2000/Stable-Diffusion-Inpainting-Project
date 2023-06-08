"""
Ben Berube
06.05.2023
Meta SAM Model Expirement 

In this program, I combine the output of Segment Anything Model (SAM) 
by Meta AI with Stable Diffusion Inpainting using Hugging Face Diffusers library 
and Gradio.

Sourced from W3 Schools Tutorial
"""

import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
)
pipe = pipe.to(device)

select_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(label="Input")
        mask_image = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
    
    with gr.Row():
        submit = gr.Button("Submit")

    def generate_mask(image, evt: gr.SelectData):
        select_pixels.append(evt.index)
        predictor.set_image(image)
        input_points = np.array(select_pixels)
        input_label = np.ones(input_points.shape[0])
        mask, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            multimask_output=False
        )
        #(1, sz, sz)
        #Change to background
        #mask = np.logical_not(mask)
        mask = Image.fromarray(mask[0, :, :])
        return mask 

    
    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image = image.resize((512,512))
        mask = mask.resize((512,512))

        output = pipe(
            prompt=prompt, 
            image=image, 
            mask_image=mask,
        ).images[0]

        return output
    
    input_image.select(generate_mask, [input_image], [mask_image])
    submit.click(
        inpaint, 
        inputs=[input_image, mask_image, prompt_text], 
        outputs=[output_img],
    )

if __name__ == "__main__":
   print(torch.cuda.is_available())
   demo.launch()


