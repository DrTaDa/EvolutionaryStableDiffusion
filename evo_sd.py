import requests
import io
import base64
from PIL import Image
import numpy
import numpy as np
import pygame
import random
import pathlib
import copy

NEGATIVE_PROMPT = "lowres, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, " \
                  "signature, watermark, username, censorship, amateur drawing, (((duplicate))), (bad legs), (bad face), (bad eyes), ((bad hands, " \
                  "bad anatomy, bad feet, missing fingers, fewer digits, cropped:1.0)), wide face, ((fused fingers)), (((too many fingers))), " \
                  "lowres, out of frame, (cloned face:1.3), (gross proportions:1.3), (mutated hands:1.3), (poorly drawn hands:1.3), " \
                  "(bad hands:1.3), (extra fingers:1.3), (poorly drawn feet:1.3), long neck, extra limbs, broken limb, asymmetrical eyes " \
                  "cell shading, artist name"

def generate_images(prompts, previous_images):

    images = []
    url = "http://127.0.0.1:7860"
    
    for prompt in prompts:
    
        if prompt in previous_images:
            images.append(previous_images[prompt])

        else:
            
            payload = {
              "enable_hr": False,
              "denoising_strength": 0,
              "firstphase_width": 0,
              "firstphase_height": 0,
              "hr_scale": 2,
              "hr_second_pass_steps": 0,
              "hr_resize_x": 0,
              "hr_resize_y": 0,
              "prompt": prompt,
              "seed": -1,
              "subseed": -1,
              "subseed_strength": 0,
              "seed_resize_from_h": -1,
              "seed_resize_from_w": -1,
              "batch_size": 1,
              "n_iter": 1,
              "steps": 50,
              "cfg_scale": 12,
              "width": 512,
              "height": 512,
              "restore_faces": False,
              "tiling": False,
              "negative_prompt": NEGATIVE_PROMPT,
              "eta": 0,
              "s_churn": 0,
              "s_tmax": 0,
              "s_tmin": 0,
              "s_noise": 1,
              "override_settings": {},
              "override_settings_restore_afterwards": True,
              "script_args": [],
              "sampler_index": "Euler"
            }

            response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

            r = response.json()

            for i in r['images']:

                image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
                images.append(image)
        
    return images

    
class EvolutionaryStrategy:

    def __init__(self, pop_size, prompt_length, dictionary_length, mutation_rate=0.1):
        self.pop_size = pop_size
        self.prompt_length = prompt_length
        self.dictionary_length = dictionary_length
        self.mutation_rate = mutation_rate
        self.pop = [[
            np.random.randint(0, self.dictionary_length) for i in range(self.prompt_length)
        ] for j in range(self.pop_size)]

    def evolve(self, selected):

        self.pop = [ind for i, ind in enumerate(self.pop) if i in selected]

        while len(self.pop) < self.pop_size:

            idx1 = numpy.random.randint(0, len(self.pop))
            idx2 = numpy.random.randint(0, len(self.pop))
            new_ind = []
            for j in range(self.prompt_length):
                if numpy.random.random() > 0.5:
                    new_ind.append(self.pop[idx1][j])
                else:
                    new_ind.append(self.pop[idx2][j])
 
            for j in range(self.prompt_length):

                # Mutation swap
                if numpy.random.random() > (1 - self.mutation_rate):
                    idx2 = numpy.random.randint(self.prompt_length)
                    new_ind[j], new_ind[idx2] = copy.copy(new_ind[idx2]), copy.copy(new_ind[j])

                # Mutation change
                if numpy.random.random() > (1 - self.mutation_rate):
                    new_ind[j] = numpy.random.randint(self.dictionary_length)

            self.pop.append(new_ind)

        random.shuffle(self.pop)


def main():
    
    output_directory = pathlib.Path("./outputs/")
    output_directory.mkdir(parents=True, exist_ok=True)

    # ES parameters
    prompt_length = 20
    pop_size = 8
    assert pop_size % 2 == 0

    # Create the list of words possible
    with open("words.txt") as file:
        dictionary = [line.rstrip() for line in file if line.rstrip()]
    print(len(dictionary))
    dictionary = list(set(dictionary))
    print(len(dictionary))

    # Create a word to index dict and index to word dict
    word_to_index = {i: d for i, d in enumerate(dictionary)}

    # Init the evolution strategy
    es = EvolutionaryStrategy(
        pop_size=pop_size,
        prompt_length=prompt_length,
        dictionary_length=len(word_to_index)
    )

    pygame.init()

    width = round(pop_size / 2) * 512
    height = 512 * 2
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('image')

    # Iterate forever
    gen = 1
    previous_images = {}
    while True:
 
        # Transform the population in prompt
        prompts = [", ".join(word_to_index[i] for i in ind) for ind in es.pop]

        # Generate the images
        images = generate_images(prompts, previous_images)
        first_row = [numpy.array(img) for i, img in enumerate(images) if i < pop_size /2.]
        second_row = [numpy.array(img) for i, img in enumerate(images) if i >= pop_size /2.]
        first_row = numpy.concatenate([img for img in first_row], axis=1)
        second_row = numpy.concatenate([img for img in second_row], axis=1)
        np_imgs = numpy.concatenate([first_row, second_row], axis=0)

        # Save output
        im = Image.fromarray(np_imgs)
        im.save(str(output_directory / f"{gen}.jpeg"))
        with open(str(output_directory / f"{gen}.txt"), 'w') as f:
            for prompt in prompts:
                f.write(prompt + "\n")

        # Display the images in a row
        surf = pygame.Surface((np_imgs.shape[1], np_imgs.shape[0]))
        pygame.surfarray.blit_array(surf, np_imgs.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (X, Y))

        selected = []
        while len(selected) < round(pop_size / 2.):

            screen.fill((0, 0, 0))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    col_clicked = int(pos[0] / 512.)
                    row_clicked = int(pos[1] / 512.)
                    index_clicked = int((row_clicked * (pop_size /2.)) + col_clicked)
                    selected.append(index_clicked)

        # Mate and mutate
        es.evolve(selected)
        gen += 1
        
        previous_images = {}
        for p, img in zip(prompts, images):
            previous_images[p] = img

 
if __name__ == "__main__":
    main()
