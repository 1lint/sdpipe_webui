
# inpaint pipeline with fix to avoid noise added to latents during final iteration of denoising loop
from inpaint_pipeline import SDInpaintPipeline as StableDiffusionInpaintPipelineLegacy

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    #StableDiffusionInpaintPipelineLegacy # uncomment this line to use original inpaint pipeline
)

import diffusers.schedulers
import gradio as gr
import torch
import random
from multiprocessing import cpu_count
import json

import importlib

_xformers_available = importlib.util.find_spec("xformers") is not None
device = "cuda" if torch.cuda.is_available() else "cpu"
low_vram_mode = False

# scheduler dict includes superclass SchedulerMixin (it still generates reasonable images)
scheduler_dict = {
    k: v
    for k, v in diffusers.schedulers.__dict__.items()
    if "Scheduler" in k and "Flax" not in k
}
scheduler_dict.pop(
    "VQDiffusionScheduler"
)  # requires unique parameter, unlike other schedulers
scheduler_names = list(scheduler_dict.keys())
default_scheduler = scheduler_names[3]  # expected to be DPM Multistep

model_ids = [
    "andite/anything-v4.0",
    "hakurei/waifu-diffusion",
    "prompthero/openjourney-v2",
    "runwayml/stable-diffusion-v1-5", 
    "johnslegers/epic-diffusion",
    "stabilityai/stable-diffusion-2-1",
]

loaded_model_id = ""


def load_pipe(
    model_id, scheduler_name, pipe_class=StableDiffusionPipeline, pipe_kwargs="{}"
):
    global pipe, loaded_model_id

    scheduler = scheduler_dict[scheduler_name]

    # load new weights from disk only when changing model_id
    if model_id != loaded_model_id:
        pipe = pipe_class.from_pretrained(
            model_id,
            # torch_dtype=torch.float16,
            # revision='fp16',
            safety_checker=None,
            requires_safety_checker=False,
            scheduler=scheduler.from_pretrained(model_id, subfolder="scheduler"),
            **json.loads(pipe_kwargs),
        )
        loaded_model_id = model_id

    # if same model_id, instantiate new pipeline with same underlying pytorch objects to avoid reloading weights from disk
    elif pipe_class != pipe.__class__ or not isinstance(pipe.scheduler, scheduler):
        pipe.components["scheduler"] = scheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = pipe_class(**pipe.components)

    if device == 'cuda':
        pipe = pipe.to(device)
        if _xformers_available:
            pipe.enable_xformers_memory_efficient_attention()
            print("using xformers")
        if low_vram_mode:
            pipe.enable_attention_slicing()
            print("using attention slicing to lower VRAM")

    return pipe


pipe = None
pipe = load_pipe(model_ids[0], default_scheduler)


def generate(
    model_name,
    scheduler_name,
    prompt,
    guidance,
    steps,
    n_images=1,
    width=512,
    height=512,
    seed=0,
    image=None,
    strength=0.5,
    inpaint_image=None,
    inpaint_strength=0.5,
    inpaint_radio='',
    neg_prompt="",
    pipe_class=StableDiffusionPipeline,
    pipe_kwargs="{}",
):

    if seed == -1:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator("cuda").manual_seed(seed)

    pipe = load_pipe(
        model_id=model_name,
        scheduler_name=scheduler_name,
        pipe_class=pipe_class,
        pipe_kwargs=pipe_kwargs,
    )

    status_message = (
        f"Prompt: '{prompt}' | Seed: {seed} | Guidance: {guidance} | Scheduler: {scheduler_name} | Steps: {steps}"
    )

    if pipe_class == StableDiffusionPipeline:
        status_message = "Text to Image " + status_message

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=n_images,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )

    elif pipe_class == StableDiffusionImg2ImgPipeline:

        status_message = "Image to Image " + status_message
        print(image.size)
        image = image.resize((width, height))
        print(image.size)

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=n_images,
            image=image,
            num_inference_steps=int(steps),
            strength=strength,
            guidance_scale=guidance,
            generator=generator,
        )

    elif pipe_class == StableDiffusionInpaintPipelineLegacy:
        status_message = "Inpainting " + status_message

        init_image = inpaint_image["image"].resize((width, height))
        mask = inpaint_image["mask"].resize((width, height))

        
        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_images_per_prompt=n_images,
            image=init_image,
            mask_image=mask,
            num_inference_steps=int(steps),
            strength=inpaint_strength,
            preserve_unmasked_image=(inpaint_radio == inpaint_options[0]),
            guidance_scale=guidance,
            generator=generator,
        )

    else:
        None, f"Unhandled pipeline class: {pipe_class}"

    return result.images, status_message


default_img_size = 512

with open("header.html") as fp:
    header = fp.read()

with open("footer.html") as fp:
    footer = fp.read()

with gr.Blocks(css="style.css") as demo:

    pipe_state = gr.State(lambda: StableDiffusionPipeline)

    gr.HTML(header)

    with gr.Row():

        with gr.Column(scale=70):

            # with gr.Row():
            prompt = gr.Textbox(
                label="Prompt", placeholder="<Shift+Enter> to generate", lines=2
            )
            neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="", lines=2)

        with gr.Column(scale=30):
            model_name = gr.Dropdown(
                label="Model", choices=model_ids, value=loaded_model_id
            )
            scheduler_name = gr.Dropdown(
                label="Scheduler", choices=scheduler_names, value=default_scheduler
            )
            generate_button = gr.Button(value="Generate", elem_id="generate-button")

    with gr.Row():

        with gr.Column():

            with gr.Tab("Text to Image") as tab:
                tab.select(lambda: StableDiffusionPipeline, [], pipe_state)

            with gr.Tab("Image to image") as tab:
                tab.select(lambda: StableDiffusionImg2ImgPipeline, [], pipe_state)

                image = gr.Image(
                    label="Image to Image",
                    source="upload",
                    tool="editor",
                    type="pil",
                    elem_id="image_upload",
                ).style(height=default_img_size)
                strength = gr.Slider(
                    label="Denoising strength",
                    minimum=0,
                    maximum=1,
                    step=0.02,
                    value=0.8,
                )

            with gr.Tab("Inpainting") as tab:
                tab.select(lambda: StableDiffusionInpaintPipelineLegacy, [], pipe_state)

                inpaint_image = gr.Image(
                    label="Inpainting",
                    source="upload",
                    tool="sketch",
                    type="pil",
                    elem_id="image_upload",
                ).style(height=default_img_size)
                inpaint_strength = gr.Slider(
                    label="Denoising strength",
                    minimum=0,
                    maximum=1,
                    step=0.02,
                    value=0.8,
                )
                inpaint_options = ["preserve non-masked portions of image", "output entire inpainted image"]
                inpaint_radio = gr.Radio(inpaint_options, value=inpaint_options[0], show_label=False, interactive=True)

            with gr.Row():
                batch_size = gr.Slider(
                    label="Batch Size", value=1, minimum=1, maximum=8, step=1
                )
                seed = gr.Slider(-1, 2147483647, label="Seed", value=-1, step=1)

            with gr.Row():
                guidance = gr.Slider(
                    label="Guidance scale", value=7.5, minimum=0, maximum=20
                )
                steps = gr.Slider(
                    label="Steps", value=20, minimum=1, maximum=100, step=1
                )

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    value=default_img_size,
                    minimum=64,
                    maximum=1024,
                    step=32,
                )
                height = gr.Slider(
                    label="Height",
                    value=default_img_size,
                    minimum=64,
                    maximum=1024,
                    step=32,
                )

        with gr.Column():
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            ).style(height=default_img_size, grid=2)

            generation_details = gr.Markdown()

            pipe_kwargs = gr.Textbox(label="Pipe kwargs", value="{\n\t\n}")

            # if torch.cuda.is_available():
            #  giga = 2**30
            #  vram_guage = gr.Slider(0, torch.cuda.memory_reserved(0)/giga, label='VRAM Allocated to Reserved (GB)', value=0, step=1)
            #  demo.load(lambda : torch.cuda.memory_allocated(0)/giga, inputs=[], outputs=vram_guage, every=0.5, show_progress=False)

    gr.HTML(footer)

    inputs = [
        model_name,
        scheduler_name,
        prompt,
        guidance,
        steps,
        batch_size,
        width,
        height,
        seed,
        image,
        strength,
        inpaint_image,
        inpaint_strength,
        inpaint_radio,
        neg_prompt,
        pipe_state,
        pipe_kwargs,
    ]
    outputs = [gallery, generation_details]

    prompt.submit(generate, inputs=inputs, outputs=outputs)
    generate_button.click(generate, inputs=inputs, outputs=outputs)

demo.queue(concurrency_count=cpu_count())
demo.launch()
