import re
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

def parse_prompt_subprompts(prompt: str):
    """
    Parses a prompt to extract subprompts and their weights.

    Example:
        Input: "sunset on (snowy)1.5 mountain"
        Output: [("sunset on", 1.0), ("snowy", 1.5), ("mountain", 1.0)]
    """
    pattern = r"\(([^)]+)\)([\d.]+)"
    segments = []
    last_end = 0

    for match in re.finditer(pattern, prompt):
        text_before = prompt[last_end:match.start()].strip()  # Strip spaces
        if text_before:  # Avoid adding empty or whitespace-only segments
            segments.append((text_before, 1.0))

        bracketed_text = match.group(1).strip()  # Strip spaces from inside the parentheses
        weight = float(match.group(2))
        if bracketed_text:  # Ensure there's actual text
            segments.append((bracketed_text, weight))

        last_end = match.end()

    text_after = prompt[last_end:].strip()  # Strip spaces at the end
    if text_after:
        segments.append((text_after, 1.0))

    return segments

def faster_compel(prompt, device):

    segments = parse_prompt_subprompts(prompt)
    fragments = [segment[0] for segment in segments]
    weights = [segment[1] for segment in segments]

    this_conditioning = compel.conditioning_provider.get_embeddings_for_weighted_prompt_fragments(
        text_batch=[fragments],
        fragment_weights_batch=[weights],
        device=device
    )
    return this_conditioning


if __name__ == '__main__':

    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo",torch_dtype=torch.float16).to(device)
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, device=device)

    # warmup
    for _ in range(10):
        faster_compel('dummy')

    # compel
    times = []
    images = []
    for i in range(9):

        # prompt
        prompt = f"jungle (snow)0.{i}"  # (jungle)0.{9 - i}"

        # time compel
        start = time.perf_counter()
        # compel.build_conditioning_tensor(prompt)
        prompt_embeds = faster_compel(prompt, device)
        times.append(time.perf_counter() - start)

        # generate image
        # image = pipe(
        #     prompt=None,
        #     prompt_embeds=prompt_embeds,
        #     guidance_scale=1.0,
        #     num_inference_steps=1
        # ).images[0]
        # images.append(image)

    print(f"Average compel time: {sum(times) / len(times)}")

    # grid = make_image_grid(images, rows=3, cols=3)
    # grid.show()
