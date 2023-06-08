import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_graphics as tfg
import numpy as np
from PIL import Image
import gradio as gr
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# No need to convert to device in TensorFlow

predictor = SamPredictor(sam)

# Use TensorFlow Hub model for StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
)

select_pixels = []

with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
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
            # (1, sz, sz)
            # Change to background
            # mask = np.logical_not(mask)
            mask = Image.fromarray(mask[0, :, :])
            return mask

        def inpaint(image, mask, prompt):
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            image = image.resize((512, 512))
            mask = mask.resize((512, 512))

            # Convert PIL images to TensorFlow tensors
            image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
            mask = tf.convert_to_tensor(np.array(mask), dtype=tf.float32)

            # Expand dimensions for batch size
            image = tf.expand_dims(image, axis=0)
            mask = tf.expand_dims(mask, axis=0)

            output = sess.run(
                pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                ).images[0]
            )

            output_img.value = output.astype(np.uint8)
            return output

        input_image.select(generate_mask, [input_image], [mask_image])
        submit.click(
            inpaint,
            inputs=[input_image, mask_image, prompt_text],
            outputs=None,  # Remove 'output' parameter
        )


    if __name__ == "__main__":
        print("TensorFlow with Metal is not supported on all systems. Make sure your system is compatible.")
        demo.launch()
