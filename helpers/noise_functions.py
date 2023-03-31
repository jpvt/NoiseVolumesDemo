import pyfastnoisesimd as fns
import numpy as np

def rescale(array, new_min, new_max) -> np.ndarray:
    return ((array-array.min()) * ((new_max-new_min)/(array.max()-array.min()))) + new_min

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

def generate_volume(
        volumes: int = 7, 
        shape: list = [100,100,100], 
        lacunarity: float = 1.5, 
        persistence: float = 0.7,
        octave_threshold: int = 2,  
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
    - octave_threshold (int): The number of octaves you wish to skip.
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

            if jj >= octave_threshold:
                vol += (persistence[0] ** count) * noise
                count -= 1

        volume += perturb(vol)

    return volume
