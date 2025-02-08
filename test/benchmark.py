import time
import torch
from diffusers.utils import make_image_grid

from src.compel.compel import Compel
from diffusers import StableDiffusionPipeline


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


if __name__ == '__main__':

    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo",torch_dtype=torch.float16).to(device)
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, device=device)

    # warmup
    for _ in range(10):
        compel.build_weighted_embedding('dummy')

    # compel
    times = []
    images = []
    for i in range(9):

        # prompt
        prompt = f"jungle (snow)0.{i}"  # (jungle)0.{9 - i}"

        # time compel
        start = time.perf_counter()
        # compel.build_conditioning_tensor(prompt)
        prompt_embeds = compel.build_weighted_embedding(prompt)
        times.append(time.perf_counter() - start)

        # generate image
        image = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            guidance_scale=1.0,
            num_inference_steps=1
        ).images[0]
        images.append(image)

    print(f"Average compel time: {sum(times) / len(times)}")

    grid = make_image_grid(images, rows=3, cols=3)
    grid.show()
