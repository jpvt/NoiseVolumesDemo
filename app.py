import streamlit as st
import pyfastnoisesimd as fns
import numpy as np
from helpers.noise_functions import generate_volume, rescale, calculate_frequencies
from helpers.noise_functions import generate_tissues, pad_mask_to_match_shape, generate_thresholded_volume, generate_thresholded_tissues, generate_thresholded_tissues_legacy
import matplotlib.pyplot as plt
import tifffile
import io

st.set_page_config(
     page_title='NBP-3D',
     layout="wide",
     initial_sidebar_state="expanded",
)

def main():
    header()

    select_mode = st.sidebar.selectbox('Choose your mode:', ["Simple Visualizer","Advanced Generator"])

    if select_mode == "Simple Visualizer":
        simple_option_page()
    else:
        advanced_page()


def header():
    st.title('Generating Noise Volumes')
    st.sidebar.title("Modes")
    st.markdown(
        """
        Dashboard to generate noise volumes. 
        You can select different parameters to generate the volume you want and explore options to choose which parameters works best for your texture.
        """
        , unsafe_allow_html=True)
    
def simple_option_page():
    st.title("3D Noise Generator Dashboard")

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Parameters")

        noise_type = st.selectbox(
            "Noise Type", options=["Simplex", "Perlin"], index=0, help="Select between Simplex and Perlin noise. Default = Simplex"
        )
        volume_size = st.slider(
            "Volume Size", min_value=50, max_value=500, value=100, step=1, help="Size of the block, currently supports only cubes. Default = 100"
        )
        num_volumes = st.slider(
            "Number of Volumes", min_value=1, max_value=20, value=6, step=1, help="The number of volumes combined to generate the result. Default = 6"
        )

        lacunarity = st.slider(
            "Lacunarity", min_value=0.01, max_value=5.0, value=1.5, step=0.01, help="The frequency factor between two octaves ('step' from one octave to the other). Default = 1.5"
        )
        persistence = st.slider(
            "Persistence", min_value=0.1, max_value=1.0, value=0.7, step=0.01, help="The scaling factor between two octaves ('weight' of an octave). Default = 0.7"
        )
        
        max_octaves =  len(calculate_frequencies(volume_size, lacunarity))
        octaves_threshold = st.slider(
            "Octaves Thresholds", min_value=1, max_value=max_octaves, value=(1, max_octaves), step=1, help=f"Interval of octaves you want to compose your volume. Default = (0, {max_octaves})"
        )
        
        advanced_options_button = st.checkbox("Show advanced options?")
        if advanced_options_button:
            threads = st.number_input(
                "Threads", min_value=1, value=4, step=1, help="Number of threads used to generate the noise. Default = 4"
            )

            thresholds_option = st.checkbox("Volume Threshold")
            if thresholds_option:
                min_values_threshold = st.slider(
                    "Thresholds Minimum Values ", min_value=0.0, max_value=1.0, value=(0.94, 0.6), step=0.01, 
                    help=f"Interval of minimum values of the gradient that will threshold your volume. Default = (0.94, 0.6)"
                )
                max_values_threshold  = st.slider(
                    "Thresholds Minimum Values ", min_value=0.0, max_value=1.0, value=(0.99, 0.96), step=0.01, 
                    help=f"Interval of maximum values of the gradient that will threshold your volume. Default = (0.94, 0.6)"
                )
                layers  = st.slider(
                    "Gradient Matrix Layers", min_value=1, max_value=63, value=63//2, step=1, help=f"Number of layers between the intervals (Number of rows of gradient matrix)"
                )

            
            seed_button = st.checkbox("Set seed")
            if seed_button:
                seed = st.number_input(
                "Seed", min_value=0, value=0, step=1, help="Seed to guarantee deterministic results. Default = None"
            )
            else:
                seed = None

        else:
            threads = 4
            seed = None

        if st.button("Generate 3D Noise"):
            if thresholds_option:
                noise_volume = generate_thresholded_volume(
                    volumes=num_volumes,
                    noise_type=fns.NoiseType.Simplex if noise_type == "Simplex" else fns.NoiseType.Perlin,
                    shape=[volume_size, volume_size, volume_size],
                    octave_threshold=octaves_threshold,
                    lacunarity=lacunarity,
                    persistence=persistence,
                    threads=threads,
                    seed=seed,
                    min_values = min_values_threshold,
                    max_values = max_values_threshold,
                    layers = layers,
                )
            else:
                noise_volume = generate_volume(
                    volumes=num_volumes,
                    noise_type=fns.NoiseType.Simplex if noise_type == "Simplex" else fns.NoiseType.Perlin,
                    shape=[volume_size, volume_size, volume_size],
                    octave_threshold=octaves_threshold,
                    lacunarity=lacunarity,
                    persistence=persistence,
                    threads=threads,
                    seed=seed,
                )
            st.session_state.noise_volume = np.uint8(rescale(noise_volume, 0, 255))
        elif "noise_volume" not in st.session_state:
            st.session_state.noise_volume = None

        if st.session_state.noise_volume is not None:
            tiff_buffer = io.BytesIO()
            tifffile.imwrite(tiff_buffer, st.session_state.noise_volume)
            tiff_buffer.seek(0)
            st.download_button(
                label="Download Volume as TIFF",
                data=tiff_buffer,
                file_name=f"{noise_type}_nvols-{num_volumes}_octhd-{octaves_threshold}_lac-{lacunarity}_per-{persistence}.tiff",
                mime="image/tiff",
            )

    if st.session_state.noise_volume is not None:
        max_slices = st.session_state.noise_volume.shape[2] - 1
        with right_column:
            z_project = st.checkbox("Show Z-Project")

            if z_project:
                fig, ax = plt.subplots()
                ax.set_title(f"Z-Project")
                image_to_show = np.sum(st.session_state.noise_volume, axis=2)
                ax.imshow(image_to_show, cmap="gray", aspect='equal')
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            else:
                slice_index = st.slider("Slice", min_value=0, max_value=max_slices, value=max_slices//2, step=1)
                fig, ax = plt.subplots()
                ax.imshow(st.session_state.noise_volume[:, :, slice_index], cmap="gray", aspect='equal')
                ax.set_title(f"Slice {slice_index + 1}")
                ax.axis("off")
                #plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

def advanced_page():
    st.title("Advanced 3D Noise Generator Dashboard")
    uploaded_file = st.file_uploader("Upload Mask TIFF File", type=["tiff", "tif"])
    left_column, right_column = st.columns(2)
    
    if "noise_tissues" not in st.session_state:
        st.session_state.noise_tissues = None

    if "result_volume" not in st.session_state:
        st.session_state.result_volume = None

    if uploaded_file is not None:
        template_mask = tifffile.imread(uploaded_file)
        volume_size = int(max(template_mask.shape))
        with left_column:
            st.subheader("Parameters")
            noise_type = st.selectbox(
            "Noise Type", options=["Simplex", "Perlin"], index=0, help="Select between Simplex and Perlin noise. Default = Simplex"
            )

            lacunarity = st.slider(
                "Lacunarity", min_value=0.01, max_value=5.0, value=1.5, step=0.01, help="The frequency factor between two octaves ('step' from one octave to the other). Default = 1.5"
            )

            mask_labels = np.unique(template_mask)
            max_octaves =  len(calculate_frequencies(volume_size, lacunarity))
            num_volumes_per_label = {}
            persistences = {}
            octaves_thresholds = {}

            gradient_matrix_thresholds = st.checkbox("Set Threshold Gradient Matrix")
            if gradient_matrix_thresholds:
                min_values_thresholds = {}
                max_values_thresholds = {}
                layers_per_label = {}
                
            for label in mask_labels:
                st.markdown(f"**Label {label}**")

                num_volumes = st.slider(
                    "Number of Volumes", min_value=1, max_value=20, value=6, step=1,
                    help="The number of volumes combined to generate the result. Default = 6", key=f"nvols_{label}"
                )
                num_volumes_per_label[label] = num_volumes

                persistence = st.slider(
                    "Persistence", min_value=0.1, max_value=1.0, value=0.7, step=0.01,
                    help="The scaling factor between two octaves ('weight' of an octave). Default = 0.7", key=f"per_{label}"
                )
                persistences[label] =  np.float32(persistence)

                octaves_threshold = st.slider(
                    "Octaves Thresholds", min_value=1, max_value=max_octaves, value=(1, max_octaves), step=1,
                    help=f"Interval of octaves you want to compose your volume. Default = (0, {max_octaves})", key=label
                )
                octaves_thresholds[label] = octaves_threshold

                if gradient_matrix_thresholds:
                    min_values_threshold = st.slider(
                    "Thresholds Minimum Values ", min_value=0.0, max_value=1.0, value=(0.94, 0.6), step=0.01, 
                    help=f"Interval of minimum values of the gradient that will threshold your volume. Default = (0.94, 0.6)", key=f"minv_{label}"
                    )
                    min_values_thresholds[label] = min_values_threshold

                    max_values_threshold  = st.slider(
                        "Thresholds Minimum Values ", min_value=0.0, max_value=1.0, value=(0.99, 0.96), step=0.01, 
                        help=f"Interval of maximum values of the gradient that will threshold your volume. Default = (0.94, 0.6)", key=f"maxv_{label}"
                    )
                    max_values_thresholds[label] = max_values_threshold

                    layers  = st.slider(
                        "Gradient Matrix Layers", min_value=1, max_value=63, value=63//2, step=1, 
                        help=f"Number of layers between the intervals (Number of rows of gradient matrix)", key=f"layers_{label}"
                    )
                    layers_per_label[label] = layers


            advanced_options_button = st.checkbox("Show advanced options?")
            if advanced_options_button:
                threads = st.number_input(
                    "Threads", min_value=1, value=4, step=1, help="Number of threads used to generate the noise. Default = 4"
                )
                
                seed_button = st.checkbox("Set seed")
                if seed_button:
                    seed = st.number_input(
                    "Seed", min_value=0, value=0, step=1, help="Seed to guarantee deterministic results. Default = None"
                )
                else:
                    seed = None

            else:
                threads = 4
                seed = None

            if st.button("Generate Noise for Labels"):

                if gradient_matrix_thresholds:
                    noise_tissues = generate_thresholded_tissues(
                        n_volumes=num_volumes_per_label,
                        noise_type=fns.NoiseType.Simplex if noise_type == "Simplex" else fns.NoiseType.Perlin,
                        shape=[volume_size, volume_size, volume_size],
                        octave_thresholds=octaves_thresholds,
                        lacunarity=lacunarity,
                        persistence=persistences,
                        threads=threads,
                        seed=seed,
                        min_values= min_values_thresholds,
                        max_values= max_values_thresholds,
                        layers= layers_per_label,
                    )
                else:
                    noise_tissues = generate_tissues(
                        n_volumes=num_volumes_per_label,
                        noise_type=fns.NoiseType.Simplex if noise_type == "Simplex" else fns.NoiseType.Perlin,
                        shape=[volume_size, volume_size, volume_size],
                        octave_thresholds=octaves_thresholds,
                        lacunarity=lacunarity,
                        persistence=persistences,
                        threads=threads,
                        seed=seed,
                    )

                for label in mask_labels:
                    print(label, type(noise_tissues[label]))
                    print(noise_tissues[label].shape)
                    noise_tissues[label] = np.uint8(rescale(noise_tissues[label].astype(np.ushort) , 0, 255))

                st.session_state.noise_tissues = noise_tissues
    
        if st.session_state.noise_tissues is not None:
            
            with right_column:
                st.subheader("Visualization")
                viz_mode = st.selectbox(
                "Select visualization", options=["Tissues", "Result", "Thresholding"], index=0, help="Select between visualization modes"
                )
                if viz_mode == "Tissues":
                    fig, ax = plt.subplots()
                    label_index = st.slider("Label", min_value=0, max_value=len(mask_labels)-1, value=0, step=1)
                    selected_label = mask_labels[label_index]
                    central_index = (st.session_state.noise_tissues[selected_label].shape[2]-1)//2
                    image_to_show = st.session_state.noise_tissues[selected_label][:, :, central_index]

                    r_col1, r_col2 = right_column.columns(2)
                    if r_col1.button("Invert"):
                        st.session_state.noise_tissues[selected_label] = np.max(st.session_state.noise_tissues[selected_label])-st.session_state.noise_tissues[selected_label]
                        image_to_show = st.session_state.noise_tissues[selected_label][:, :, central_index]

                    if r_col2.checkbox("Z-Project"):
                        ax.set_title(f"Z-Project of label {selected_label}")
                        image_to_show = np.sum(st.session_state.noise_tissues[selected_label], axis=2)
                    else:
                        ax.set_title(f"Label {selected_label}")
                        image_to_show = st.session_state.noise_tissues[selected_label][:, :, central_index]
                        

                    ax.imshow(image_to_show, cmap="gray", aspect='equal')
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

                    if st.session_state.noise_tissues is not None:
                        tiff_buffer = io.BytesIO()
                        tifffile.imwrite(tiff_buffer, st.session_state.noise_tissues[selected_label])
                        tiff_buffer.seek(0)
                        st.download_button(
                            label="Download Volume as TIFF",
                            data=tiff_buffer,
                            file_name=f"tissue_{selected_label}_{noise_type}_nvols-{num_volumes}_lac-{lacunarity}_per-{persistence}.tiff",
                            mime="image/tiff",
                        )
                
                elif viz_mode == "Result":
                    result_volume = np.zeros_like(template_mask,dtype=np.uint8)
                    for label in mask_labels:
                        label_mask = pad_mask_to_match_shape(template_mask, st.session_state.noise_tissues[label].shape)
                        label_mask = np.where(label_mask == label, 1, 0)
                        label_mask *= st.session_state.noise_tissues[label]

                        result_volume += np.uint8(label_mask[:template_mask.shape[0], :template_mask.shape[1], :template_mask.shape[2]])
                    
                    st.session_state.result_volume = result_volume
                    axis = st.selectbox(
                            'Which axis would you like to slice along?',
                            ('X', 'Y', 'Z')
                        )

                    z_project_result = st.checkbox("Z-Project")
                    if z_project_result:
                        fig, ax = plt.subplots()
                        if axis == "X":
                            ax.imshow(np.sum(st.session_state.result_volume, axis=0), cmap="gray", aspect='equal')
                        elif axis == "Y":
                            ax.imshow(np.sum(st.session_state.result_volume, axis=1), cmap="gray", aspect='equal')
                        elif axis == "Z":
                            ax.imshow(np.sum(st.session_state.result_volume, axis=2), cmap="gray", aspect='equal')
                        ax.set_title(f"Z-Project")
                        ax.axis("off")
                        #plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        max_slices = st.session_state.result_volume.shape[2] - 1
                        slice_index = st.slider("Slice", min_value=0, max_value=max_slices, value=max_slices//2, step=1)
                        fig, ax = plt.subplots()

                        if axis == "X":
                            ax.imshow(st.session_state.result_volume[slice_index, :, :], cmap="gray", aspect='equal')
                        elif axis == "Y":
                            ax.imshow(st.session_state.result_volume[:, slice_index, :], cmap="gray", aspect='equal')
                        elif axis == "Z":
                            ax.imshow(st.session_state.result_volume[:, :, slice_index], cmap="gray", aspect='equal')

                        ax.set_title(f"Slice {slice_index + 1}")
                        ax.axis("off")
                        #plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        if st.session_state.result_volume is not None:
                            tiff_buffer = io.BytesIO()
                            tifffile.imwrite(tiff_buffer, st.session_state.result_volume)
                            tiff_buffer.seek(0)
                            st.download_button(
                                label="Download Volume as TIFF",
                                data=tiff_buffer,
                                file_name=f"result_{noise_type}_nvols-{num_volumes}_lac-{lacunarity}_per-{persistence}.tiff",
                                mime="image/tiff",
                            )
                
                elif viz_mode == "Thresholding":
                    fig, ax = plt.subplots()
                    result_volume = np.zeros_like(template_mask,dtype=np.uint8)
                    
                    hist_thresholds = {}
                    for label in mask_labels:
                        st.markdown(f"*Label {label}*")

                        hist_threshold = st.slider(
                        "Thresholds Values ", min_value=0, max_value=255, value=(80, 120), step=1, 
                        help=f"Binary Mask threshold values. Default = (0, 255)", key=f"{label}_hist_threshold"
                        )
                        hist_thresholds[label] = hist_threshold

                    
                    label_index = st.slider("Label", min_value=0, max_value=len(mask_labels)-1, value=0, step=1)
                    selected_label = mask_labels[label_index]
                    
                    r_col1, r_col2, r_col3 = right_column.columns(3)
                    if r_col1.button("Apply Threshold"):
                        st.session_state.noise_tissues_thr = st.session_state.noise_tissues.copy()
                        for label in st.session_state.noise_tissues_thr.keys():
                            st.session_state.noise_tissues_thr[label] = np.uint8(rescale(st.session_state.noise_tissues_thr[label] , 0, 255))
                            thr0 = hist_thresholds[label]
                            st.session_state.noise_tissues_thr[label] = np.where(st.session_state.noise_tissues_thr[label] <= thr0[0], 1,  st.session_state.noise_tissues_thr[label])
                            st.session_state.noise_tissues_thr[label] = np.where(st.session_state.noise_tissues_thr[label] >= thr0[1], 1,  st.session_state.noise_tissues_thr[label])
                            st.session_state.noise_tissues_thr[label] = np.where(st.session_state.noise_tissues_thr[label] > 1, 2,  st.session_state.noise_tissues_thr[label])

                    central_index = (st.session_state.noise_tissues_thr[selected_label].shape[2]-1)//2
                    image_to_show_thr = st.session_state.noise_tissues_thr[selected_label][:, central_index, :]

                    if r_col2.checkbox("Tissues Z-Project"):
                        ax.set_title(f"Z-Project of label {selected_label}")
                        image_to_show_thr = np.sum(st.session_state.noise_tissues_thr[selected_label], axis=2)

                        ax.set_title(f"Z-Project of label {selected_label}")
                        ax.imshow(image_to_show_thr, cmap="gray", aspect='equal')
                        ax.axis("off")
                        st.pyplot(fig)
                        plt.close(fig)
                        

                    else:
                        if r_col3.checkbox("See mask thresholded"): 

                            for label in mask_labels:
                                label_mask = pad_mask_to_match_shape(template_mask, st.session_state.noise_tissues_thr[label].shape)
                                label_mask = np.where(label_mask == label, 1, 0)
                                label_mask *= st.session_state.noise_tissues_thr[label]

                                result_volume += np.uint8(label_mask[:template_mask.shape[0], :template_mask.shape[1], :template_mask.shape[2]])
                            
                            st.session_state.result_volume_thr = result_volume

                            axis = st.selectbox(
                            'Which axis would you like to slice along?',
                            ('X', 'Y', 'Z')
                        )

                            if st.checkbox("Mask Z-Project"):
                    
                                fig, ax = plt.subplots()
                                if axis == "X":
                                    ax.imshow(np.sum(st.session_state.result_volume_thr, axis=0), cmap="gray", aspect='equal')
                                elif axis == "Y":
                                    ax.imshow(np.sum(st.session_state.result_volume_thr, axis=1), cmap="gray", aspect='equal')
                                elif axis == "Z":
                                    ax.imshow(np.sum(st.session_state.result_volume_thr, axis=2), cmap="gray", aspect='equal')
                                ax.set_title(f"Mask Z-Project")
                                ax.axis("off")
                                #plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                            else:
                                max_slices = st.session_state.result_volume_thr.shape[2] - 1
                                slice_index = st.slider("Slice", min_value=0, max_value=max_slices, value=max_slices//2, step=1)
                                fig, ax = plt.subplots()

                                if axis == "X":
                                    ax.imshow(st.session_state.result_volume_thr[slice_index, :, :], cmap="gray", aspect='equal')
                                elif axis == "Y":
                                    ax.imshow(st.session_state.result_volume_thr[:, slice_index, :], cmap="gray", aspect='equal')
                                elif axis == "Z":
                                    ax.imshow(st.session_state.result_volume_thr[:, :, slice_index], cmap="gray", aspect='equal')

                                ax.axis("off")
                                #plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                            
                        else:
                            
                            ax.set_title(f"Label {selected_label}")
                            ax.imshow(image_to_show_thr, cmap="gray", aspect='equal')
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                                
                    

                    

                    if st.session_state.noise_tissues_thr is not None:
                        tiff_buffer = io.BytesIO()
                        tifffile.imwrite(tiff_buffer, st.session_state.noise_tissues_thr[selected_label])
                        tiff_buffer.seek(0)
                        st.download_button(
                            label="Download Volume as TIFF",
                            data=tiff_buffer,
                            file_name=f"tissue_{selected_label}_{noise_type}_nvols-{num_volumes}_lac-{lacunarity}_per-{persistence}.tiff",
                            mime="image/tiff",
                        )

                   

                



if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)

    # df = load_data()

    main()