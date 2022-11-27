import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# model_id = "stabilityai/stable-diffusion-2"
model_id = "stabilityai/stable-diffusion-2-base"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
# pipe.enable_attention_slicing() # for less VRAM usage (to the cost of speed)

prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt, height=768, width=768)
image = images.images[0]
    
image.save("astronaut_rides_horse.png")
