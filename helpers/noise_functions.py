import pyfastnoisesimd as fns
import numpy as np

def rescale(array, new_min, new_max) -> np.ndarray:
    return ((array-array.min()) * ((new_max-new_min)/(array.max()-array.min()))) + new_min

def pad_mask_to_match_shape(mask, target_shape):
    pad_width = []
    
    for m_dim, t_dim in zip(mask.shape, target_shape):
        padding_needed = max(t_dim - m_dim, 0)
        pad_width.append((0, padding_needed))
    
    padded_mask = np.pad(mask, pad_width=pad_width)
    
    return np.uint8(padded_mask)

def generate_threshold_matrix(layers, volumes, min_values, max_values):
    assert len(min_values) == 2, "min_values must have 2 elements."
    assert len(max_values) == 2, "max_values must have 2 elements."

    min_gradient = np.linspace(min_values[0], min_values[1], layers).reshape(-1, 1)
    max_gradient = np.linspace(max_values[0], max_values[1], layers).reshape(-1, 1)
    
    column_scaling = np.linspace(0, 1, volumes)
    threshold_matrix = min_gradient + (max_gradient - min_gradient) * column_scaling

    return threshold_matrix

def perturb(volume: np.ndarray) -> np.single:
    """
    Function to perturb distribution of a volume
    Args:
        volume: Volume to be perturbed
    Returns: 
        Pertubed volume as np.single
    """
    return np.single((1-np.abs(volume))**10)

def calculate_frequencies(shape: list, lacunarity: float) -> dict:
    """
    Calculates frequencies for each octave from volume size and lacunarity.
    Args:
        shape: List that describes the size of each dimension of the volume
        lacunarity: Frequency factor between two octaves (float)
    Returns:
        Frequencies dict mapped by octave
    """
    max_size = np.max(shape)
    freq = 1
    octaves = 0
    frequencies = {}
    while max_size / freq >= 1:
        freq = round(freq * lacunarity)
        octaves += 1
        frequencies[octaves] = freq

    return frequencies

def generate_noise(
        frequency: int, 
        noise_type: int = fns.NoiseType.Simplex, 
        shape: list = [500, 500, 500], 
        threads: int = 8, 
        seed=None
) -> np.single:
    """
    Function that generates noise distributions using the pyfastnoisesimd library.
    Args:
        frequency: int that represents frequency of the generated noise
        noise_type: fns.NoiseType or int(Check fns documentation to know which ints are mapped to each type) that represents the noise type
        shape: list with the size of each volume dimension
        threads: int with the number of the threads that will be used to generate the noise
        seed: int value to represent randomness
    Returns:
        Noise volume as np.single

    """
    if seed is None:
        seed=np.random.randint(2**31)
    noisegen = fns.Noise(seed=seed, numWorkers=threads)
    noisegen.noiseType = noise_type
    noisegen.frequency = 1/frequency
    noisegen.perturb.perturbType = fns.PerturbType.NoPerturb
    return np.single(noisegen.genAsGrid(shape))

def combine_base(base, thresholds, shape):
    layers, volume_thresholds = thresholds.shape
    finished = np.zeros(shape, dtype=np.uint8)

    for layer in range(layers):
        combined = np.zeros(shape, dtype=np.uint8)
        for volume_threshold in range(volume_thresholds):
            combined += np.where(base[volume_threshold] > thresholds[layer, volume_threshold], 1, 0).astype(np.uint8)

        finished += combined

    return finished

def generate_thresholded_volume(
        volumes: int = 7, 
        shape: list = [100,100,100], 
        lacunarity: float = 1.5, 
        persistence: float = 0.7,
        octave_threshold: tuple = (0,7),  
        noise_type: int = fns.NoiseType.Simplex,
        threads: int = 8,
        seed: int = None,
        min_values: tuple = (0.94, 0.6),
        max_values: tuple = (0.99, 0.98),
        layers: int = 63,
) -> np.ndarray:
    """
    Generate a 3D noise volume with specified parameters.

    Parameters:
    - volumes (int): The number of volumes combined to generate the result.
    - shape (list): The shape of the output volume (e.g. [100, 100, 100] for a 100x100x100 volume).
    - lacunarity (float): The frequency factor between two octaves ("step" from one octave to the other).
    - persistence (float): The scaling factor between two octaves ("weight" of an octave).
    - octave_threshold (tuple): Interval of octaves you want to compose your volume.
    - noise_type (int): The type of noise to generate (e.g., Simplex or Perlin).
    - threads (int): The number of threads used for generating the noise.
    - seed (int, optional): The seed for deterministic results.

    Returns:
    - volume (ndarray): A 3D noise volume as a NumPy array.
    """
    if np.size(persistence) == 1:
        persistence = np.array([persistence, persistence])
    
    frequencies =  calculate_frequencies(shape, lacunarity)
    #volume = np.zeros(shape, dtype=np.single)
    generated_volumes = []
    
    for _ in range(volumes):
        vol = np.zeros(shape, dtype=np.single)
        count = len(frequencies)
        
        for jj in range(1,len(frequencies)+1):

            noise = generate_noise(
                frequency=frequencies[jj],
                noise_type=noise_type,
                shape=shape,
                threads=threads,
                seed=seed
            )

            if octave_threshold[0] <= jj <= octave_threshold[1]:
                vol += (persistence[0] ** count) * noise
                count -= 1

        generated_volumes.append(perturb(vol))

    threshold_matrix = generate_threshold_matrix(
        layers=layers,
        volumes=volumes,
        min_values=min_values,
        max_values=max_values
    )
    volume = combine_base(generated_volumes, threshold_matrix, shape)

    return volume

def generate_volume(
        volumes: int = 7, 
        shape: list = [100,100,100], 
        lacunarity: float = 1.5, 
        persistence: float = 0.7,
        octave_threshold: tuple = (0,7),  
        noise_type: int = fns.NoiseType.Simplex,
        threads: int = 8,
        seed: int = None,
) -> np.ndarray:
    """
    Generate a 3D noise volume with specified parameters.

    Parameters:
    - volumes (int): The number of volumes combined to generate the result.
    - shape (list): The shape of the output volume (e.g. [100, 100, 100] for a 100x100x100 volume).
    - lacunarity (float): The frequency factor between two octaves ("step" from one octave to the other).
    - persistence (float): The scaling factor between two octaves ("weight" of an octave).
    - octave_threshold (tuple): Interval of octaves you want to compose your volume.
    - noise_type (int): The type of noise to generate (e.g., Simplex or Perlin).
    - threads (int): The number of threads used for generating the noise.
    - seed (int, optional): The seed for deterministic results.

    Returns:
    - volume (ndarray): A 3D noise volume as a NumPy array.
    """
    if np.size(persistence) == 1:
        persistence = np.array([persistence, persistence])
    
    frequencies =  calculate_frequencies(shape, lacunarity)
    volume = np.zeros(shape, dtype=np.single)
    
    for _ in range(volumes):
        vol = np.zeros(shape, dtype=np.single)
        count = len(frequencies)
        
        for jj in range(1,len(frequencies)+1):

            noise = generate_noise(
                frequency=frequencies[jj],
                noise_type=noise_type,
                shape=shape,
                threads=threads,
                seed=seed
            )

            if octave_threshold[0] <= jj <= octave_threshold[1]:
                vol += (persistence[0] ** count) * noise
                count -= 1

        volume += perturb(vol)

    return volume

def generate_tissues(
        n_volumes: int = 7, 
        shape: list = [100,100,100], 
        lacunarity: float = 1.5, 
        persistence: float = 0.7,
        octave_thresholds: dict = {0: (0,7)},  
        noise_type: int = fns.NoiseType.Simplex,
        threads: int = 8,
        seed: int = None,
) -> dict:
    """
    Generate a 3D noise volume with specified parameters.

    Parameters:
    - volumes (int): The number of volumes combined to generate the result.
    - shape (list): The shape of the output volume (e.g. [100, 100, 100] for a 100x100x100 volume).
    - lacunarity (float): The frequency factor between two octaves ("step" from one octave to the other).
    - persistence (float): The scaling factor between two octaves ("weight" of an octave).
    - octave_thresholds (dict): Intervals of octaves you want to compose your tissues.
    - noise_type (int): The type of noise to generate (e.g., Simplex or Perlin).
    - threads (int): The number of threads used for generating the noise.
    - seed (int, optional): The seed for deterministic results.

    Returns:
    - tissues (ndarray): A 3D noise volume as a NumPy array.
    """
    if np.size(persistence) == 1:
        persistence = {label: persistence for label in octave_thresholds.keys()}
        
    frequencies =  calculate_frequencies(shape, lacunarity)
    tissues = {label: np.zeros(shape, dtype=np.single) for label in octave_thresholds.keys()}
    for _ in range(n_volumes):
        volumes = {label: np.zeros(shape, dtype=np.single) for label in octave_thresholds.keys()}
        counts = {label: len(frequencies) for label in octave_thresholds.keys()}
        
        for jj in range(1,len(frequencies)+1):

            noise = generate_noise(
                frequency=frequencies[jj],
                noise_type=noise_type,
                shape=shape,
                threads=threads,
                seed=seed
            )

            for label, octave_threshold in octave_thresholds.items():

                if octave_threshold[0] <= jj <= octave_threshold[1]:
                    volumes[label] += (persistence[label] ** counts[label]) * noise
                    counts[label] -= 1
        
        for label in octave_thresholds.keys():
            tissues[label] += perturb(volumes[label])

    return tissues

def generate_thresholded_tissues(
        n_volumes: dict = {0: 7},
        shape: list = [100,100,100], 
        lacunarity: float = 1.5, 
        persistence: dict = {0: 0.7},
        octave_thresholds: dict = {0: (0,7)},  
        noise_type: int = fns.NoiseType.Simplex,
        threads: int = 8,
        seed: int = None,
        min_values: dict = {0: (0.94, 0.6)},
        max_values: dict = {0: (0.99, 0.98)},
        layers: dict = {0: 63},
) -> dict:
    """
    Generate a 3D noise volume with specified parameters.

    Parameters:
    - volumes (int): The number of volumes combined to generate the result.
    - shape (list): The shape of the output volume (e.g. [100, 100, 100] for a 100x100x100 volume).
    - lacunarity (float): The frequency factor between two octaves ("step" from one octave to the other).
    - persistence (float): The scaling factor between two octaves ("weight" of an octave).
    - octave_thresholds (dict): Intervals of octaves you want to compose your tissues.
    - noise_type (int): The type of noise to generate (e.g., Simplex or Perlin).
    - threads (int): The number of threads used for generating the noise.
    - seed (int, optional): The seed for deterministic results.

    Returns:
    - tissues (ndarray): A 3D noise volume as a NumPy array.
    """
        
    frequencies =  calculate_frequencies(shape, lacunarity)
    tissues = {label: [] for label in octave_thresholds.keys()}
    for ii in range(max(n_volumes.values())):
        volumes = {label: np.zeros(shape, dtype=np.single) for label in octave_thresholds.keys()}
        counts = {label: len(frequencies) for label in octave_thresholds.keys()}
        
        for jj in range(1,len(frequencies)+1):

            noise = generate_noise(
                frequency=frequencies[jj],
                noise_type=noise_type,
                shape=shape,
                threads=threads,
                seed=seed
            )

            for label, octave_threshold in octave_thresholds.items():
                if ii < n_volumes[label]:
                    if octave_threshold[0] <= jj <= octave_threshold[1]:
                        volumes[label] += (persistence[label]** counts[label]) * noise #BB 
                        counts[label] -= 1
        
        for label in octave_thresholds.keys():
            if ii < n_volumes[label]: 
                tissues[label].append(perturb(volumes[label]))
    
    for label in octave_thresholds.keys():
        threshold_matrix = generate_threshold_matrix(
            layers=layers[label],
            volumes=n_volumes[label],
            min_values=min_values[label],
            max_values=max_values[label]
        )
        tissues[label] = np.asarray(combine_base(tissues[label], threshold_matrix, shape))

    return tissues