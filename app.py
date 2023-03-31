import streamlit as st
import pyfastnoisesimd as fns
import numpy as np
from helpers.noise_functions import generate_volume, rescale, calculate_frequencies
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

    select_mode = st.sidebar.selectbox('Choose your mode:', ["Simple Visualizer","Detailed Visualizer"])

    if select_mode == "Simple Visualizer":
        simple_option_page()
    else:
        st.header('Under Construction')


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
            "Volume Size", min_value=50, max_value=1000, value=100, step=1, help="Size of the block, currently supports only cubes. Default = 100"
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
            slice_index = st.slider("Slice", min_value=0, max_value=max_slices, value=0, step=1)
            fig, ax = plt.subplots()
            ax.imshow(st.session_state.noise_volume[:, :, slice_index], cmap="gray", aspect='equal')
            ax.set_title(f"Slice {slice_index + 1}")
            ax.axis("off")
            #plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)

    # df = load_data()

    main()