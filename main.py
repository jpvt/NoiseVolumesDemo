import pyfastnoisesimd as fns
import numpy as np
import os
from helpers.noise_functions import generate_tissues, generate_thresholded_tissues, rescale, pad_mask_to_match_shape
import tifffile
import json

#import imagej
import zipfile
import xml.etree.ElementTree as ET, xml.etree.ElementInclude as EI
import shutil

#Parse dic
def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k) if k.isnumeric() else k: tuple(v) if isinstance(v, list) else v for k,v in x.items()}
    return x

# Iterate directory
dir_path = r'data/'
files = []
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        files.append(path)

count=0
for file in files:

    #Decompress mask
    with zipfile.ZipFile(dir_path+"/"+file,"r") as zip_ref:
        zip_ref.extractall(".")

    #Create json
    in_file = file.replace(".zip", ".tif")
    out_file = in_file.replace("phantom","noise")

    dic = {
        "template_mask_path": in_file,
        "num_volumes": {"2": 2, "3": 2, "4": 3},
        "noise_type": "Simplex",
        "octave_thresholds": {"2": [3, 7], "3": [5, 13], "4": [5, 12]},
        "lacunarity": 1.5,
        "persistence": {"2": 0.55, "3": 0.65, "4": 0.70},
        "threads": 4,
        "seed": None,
        "min_values": {"2": [0.90, 0.95], "3": [0.88, 0.94], "4": [0.70, 0.94]},
        "max_values": {"2": [0.96, 1.00], "3": [0.90, 1.00], "4": [0.96, 0.99]},
        "layers": {"2": 31, "3": 31, "4": 31},
        "hist_threshold": {"2": [160, 180], "3": [145, 150], "4": [140, 255]},
        "output_file": out_file
    }

    #selection
    count+=1
    if count > 2:
        count=0
 
    # Serializing json
    json_object = json.dumps(dic, indent=2)
 
    # Writing to data.json
    with open("data_B.json", "w") as outfile:
        outfile.write(json_object)

    with open("data_B.json", "r") as f:
        generator_parameters = json.load(f, object_hook=jsonKeys2int)

    print(in_file)
    
    template_mask = tifffile.imread(generator_parameters["template_mask_path"])
    
    #Replace ligaments here
    template_mask = np.where(template_mask == 5, 3, template_mask)
    template_mask = np.where(template_mask == 6, 4, template_mask)
    template_mask = np.where(template_mask == 7, 3, template_mask)
    template_mask = np.where(template_mask == 8, 4, template_mask)
    tifffile.imwrite('template_mask.tif', template_mask)

    volume_size = int(max(template_mask.shape))
    print('Volume size: ', volume_size)
    if generator_parameters["min_values"] == None or generator_parameters["max_values"] == None or generator_parameters["layers"] == None:
        noise_tissues = generate_tissues(
            n_volumes=generator_parameters["num_volumes"],
            noise_type=fns.NoiseType.Simplex if generator_parameters["noise_type"] == "Simplex" else fns.NoiseType.Perlin,
            shape=[volume_size, volume_size, volume_size],
            octave_thresholds=generator_parameters["octave_thresholds"],
            lacunarity=generator_parameters["lacunarity"],
            persistence=generator_parameters["persistence"],
            threads=generator_parameters["threads"],
            seed=generator_parameters["seed"]
        )

    else:
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
    
    print('Noise done!')

    #Normalize and threshold here for each tissue type
    thr = generator_parameters["hist_threshold"]
    for label in noise_tissues.keys():
        tifffile.imwrite("noise_"+str(label)+'.tif', np.uint8(rescale(noise_tissues[label] , 0, 255)))
        noise_tissues[label] = np.uint8(rescale(noise_tissues[label] , 0, 255))
        thr0 = thr[label]
        noise_tissues[label] = np.where(noise_tissues[label] <= thr0[0], 1, noise_tissues[label])
        noise_tissues[label] = np.where(noise_tissues[label] >= thr0[1], 1, noise_tissues[label])
        noise_tissues[label] = np.where(noise_tissues[label] > 1, 2, noise_tissues[label])

    #Crete Result Mask
    result_volume = np.zeros_like(template_mask,dtype=np.uint8)

    for label in noise_tissues.keys():
        label_mask = pad_mask_to_match_shape(template_mask, noise_tissues[label].shape)
        label_mask = np.where(label_mask == label, 1, 0)
        #label_mask = np.single(label_mask) #delete this
        label_mask *= noise_tissues[label]
        tifffile.imwrite("tissue_"+str(label)+'.tif', np.uint8(label_mask[:template_mask.shape[0], :template_mask.shape[1], :template_mask.shape[2]]))

        result_volume += np.uint8(label_mask[:template_mask.shape[0], :template_mask.shape[1], :template_mask.shape[2]])
        #result_volume += label_mask[:template_mask.shape[0], :template_mask.shape[1], :template_mask.shape[2]]
        
    tifffile.imwrite('result_volume.tif', result_volume)

    #Save
    path_dir = out_file.replace(".tif", "")
    if os.path.exists(path_dir):
        shutil.rmtree(path_dir)
    os.mkdir(path_dir)

    result_volume = np.flip(result_volume, 2) #flip horizontally
    result_volume.astype('uint8').tofile(path_dir+"/Phantom.dat")
    
    #Parse XML
    tree = ET.parse('../temp/Phantom.xml')
    root = tree.getroot()

    values, counts = np.unique(result_volume, return_counts=True)

    #Write XML
    for child in root:
        if child.tag == 'Phantom_Name':
            child.text = path_dir

        elif child.tag == 'Total_Non_Air_Voxels':
            child.text = str(counts[0]+counts[1])

        elif child.tag == 'Glandular_Count':
            child.text =str(counts[1])

        elif child.tag == 'Thickness_mm':
            child.find('X').text = str(template_mask.shape[2]*0.1) #hardcoded
            child.find('Y').text = str(template_mask.shape[1]*0.1) #hardcoded
            child.find('Z').text = str(template_mask.shape[0]*0.1) #hardcoded

        elif child.tag == 'Voxel_Array':
            child.find('VOXEL_NUM').find('X').text = str(template_mask.shape[2])
            child.find('VOXEL_NUM').find('Y').text = str(template_mask.shape[1])
            child.find('VOXEL_NUM').find('Z').text = str(template_mask.shape[0])
            child.find('VOXEL_SIZE_MM').find('X').text = str(0.1) #hardcoded
            child.find('VOXEL_SIZE_MM').find('Y').text = str(0.1) #hardcoded
            child.find('VOXEL_SIZE_MM').find('Z').text = str(0.1) #hardcoded
            
        elif child.tag == 'Deformation':
             child.find('Deformation_Mode').text = 'DEFORM_CC'
        
    tree.write(path_dir+'/Phantom.xml')

    #Copy additional files
    path_subdir = path_dir+"/Private"
    os.mkdir(path_subdir)
    shutil.copy2('../temp/Private/XPL_AttenuationTable.xml', path_subdir+"/XPL_AttenuationTable.xml")

    #zip
    zf = zipfile.ZipFile(path_dir+".vctx", "w")
    for dirname, subdirs, files in os.walk(path_dir):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename), compress_type=zipfile.ZIP_DEFLATED)
    zf.close()

    #delete temp files and dirs
    shutil.rmtree(path_dir)
    os.remove(in_file)

