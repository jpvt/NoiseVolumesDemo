import pyfastnoisesimd as fns
import numpy as np
from helpers.noise_functions import generate_thresholded_tissues, rescale, pad_mask_to_match_shape
import tifffile
import json

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k) if k.isnumeric() else k: tuple(v) if isinstance(v, list) else v for k,v in x.items()}
    return x

with open("data.json", "r") as f:
    generator_parameters = json.load(f, object_hook=jsonKeys2int)

template_mask = tifffile.imread(generator_parameters["template_mask_path"])
volume_size = int(max(template_mask.shape))

noise_tissues = generate_thresholded_tissues(
    n_volumes=generator_parameters["num_volumes"],
    noise_type=fns.NoiseType.Simplex if generator_parameters["noise_type"] == "Simplex" else fns.NoiseType.Perlin,
    shape=[volume_size, volume_size, volume_size],
    octave_thresholds=generator_parameters["octave_thresholds"],
    lacunarity=generator_parameters["lacunarity"],
    persistence=generator_parameters["persistence"],
    threads=generator_parameters["threads"],
    seed=generator_parameters["seed"],
    min_values= generator_parameters["min_values"],
    max_values= generator_parameters["max_values"],
    layers= generator_parameters["layers"],
)

for label in noise_tissues.keys():
    noise_tissues[label] = np.uint8(rescale(noise_tissues[label] , 0, 255))

result_volume = np.zeros_like(template_mask,dtype=np.uint8)
for label in noise_tissues.keys():
    label_mask = pad_mask_to_match_shape(template_mask, noise_tissues[label].shape)
    label_mask = np.where(label_mask == label, 1, 0)
    label_mask *= noise_tissues[label]

    result_volume += np.uint8(label_mask[:template_mask.shape[0], :template_mask.shape[1], :template_mask.shape[2]])

tifffile.imwrite(generator_parameters["output_file"], result_volume)