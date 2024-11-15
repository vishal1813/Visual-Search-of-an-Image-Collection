import os
import cv2
import time
import numpy as np
import scipy.io as sio
from extractRandom_v3 import extractRandomAdv
from HelperFunctions import (
    # gridPlot,
    getDetails,
    getImageVectorParameters,
)
import streamlit as st
import warnings
from cvpr_visualsearch_v3 import visualsearch
from random import randint
from stqdm import stqdm

warnings.filterwarnings("ignore")

DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2"
OUT_FOLDER = "descriptors"
OUT_SUBFOLDER = "globalRGBhisto"

####################     STREAMLIT     ####################

start = time.time()

# Base configuration
st.set_page_config(layout="wide")

# Main panel title
titleContainer = st.container(border=True)
_, tCol1, tColR = titleContainer.columns(
    [
        1,
        3,
        1,
    ]
)
with titleContainer:
    tCol1.title(":blue[Visual Search - Advanced]")
    tCol1.caption(":blue[Assignment - 1]")

# Panel for user options
st.sidebar.title(":blue[Options]")
st.sidebar.divider()

imgSelect = st.sidebar.toggle(":blue[**Random Image**]")
data = getDetails()
if imgSelect:
    index = randint(0, len(data))
    imgSelected = st.sidebar.text_input(
        "Image Input", data.iloc[index]["ImageFile"], disabled=True
    )
else:
    imgSelected = st.sidebar.selectbox("Select Image", data["ImageFile"].to_list())
st.sidebar.divider()

# Create 3 row containers
inputContainer = st.container(border=True)
inCol1, inCol2, inCol3 = inputContainer.columns(
    [3, 3, 3], gap="medium", vertical_alignment="top"
)

with inputContainer:
    cSpace = inCol1.radio(":blue[**Colour Space**]", ["GREY", "LAB", "HSV"])
    fdescriptor = inCol2.radio(":blue[**Feature Descriptor**]", ["SIFT", "ORB"])
    distance = inCol3.radio(
        ":blue[**Distance**]", ["Euclidean", "JensenShannon", "Minkowski", "Cosine"]
    )

# Controlling parameters

IMAGE = imgSelected
COLOUR = cSpace
DESCRIPTOR = fdescriptor
DIST = distance

imgContainer = st.expander(label="Input and Generated Images")
imgCol1, imgCol2, imgCol3 = imgContainer.columns([3, 3, 3])

##### Streamlit Image diaplay panel

if imgSelect:
    run = st.sidebar.button(
        "Execute", type="primary", use_container_width=True, disabled=True
    )
else:
    run = st.sidebar.button(
        "Execute", type="primary", use_container_width=True, disabled=False
    )

logDetails = st.sidebar.expander(":blue[**Execution Log**]", expanded=True)


def displayInputOutputImages(img: np.array, colour: str, descriptor: str) -> None:
    logDetails.empty()
    logDetails.caption(":green[- _Display input and generated images_]")

    # OpenCV reads image in BGR (assuming the image is stored in RGB)
    img = cv2.imread(os.path.join(DATASET_FOLDER, "Images", IMAGE), cv2.IMREAD_COLOR)
    imgCol1.image(image=img[:, :, ::-1])
    imgCol1.caption("Original Image (BGR -> RGB for viewing)")
    if colour == "GREY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif colour == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    imgCol2.image(image=img)
    imgCol2.caption(f"Converted to {COLOUR} colour-space")

    keyPoints, _ = getImageVectorParameters(img, colour=COLOUR, descriptor=descriptor)

    # if COLOUR == "GREY":
    #     black = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    # else:
    #     black = np.zeros([img.shape[0], img.shape[1], img.shape[2]], dtype=np.uint8)

    black = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    image = cv2.drawKeypoints(
        black, keyPoints, black, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    imgCol3.image(image)
    imgCol3.caption(
        f"{fdescriptor} {IMAGE} - {len(keyPoints)} important keypoints choosen"
    )

    return


##########

# Display Search image and generated images in  the panel
displayInputOutputImages(IMAGE, COLOUR, DESCRIPTOR)

resultContainer = st.expander("Results", expanded=True)

if run or imgSelect:
    with resultContainer:
        with st.spinner("Generating descriptors . . ."):
            logDetails.caption(f":green[- _Creating {DESCRIPTOR} descriptors_]")

            # Iterate through all BMP files in the dataset folder
            for filename in stqdm(os.listdir(os.path.join(DATASET_FOLDER, "Images"))):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(DATASET_FOLDER, "Images", filename)
                    img = cv2.imread(img_path)
                    fout = os.path.join(
                        OUT_FOLDER, OUT_SUBFOLDER, filename.replace(".bmp", ".mat")
                    )
                    # st.write(img.shape)
                    # Call extractRandom to get the descriptor
                    F = extractRandomAdv(
                        img,
                        colorSpace=COLOUR,
                        featureDescriptor=DESCRIPTOR,
                    )

                    sio.savemat(fout, {"F": F})
                    # break
            logDetails.caption(":green[- _Saved image vectors as .mat files_]")
            resultContainer.success("Descriptors created!")
            st.toast("Descriptors created", icon=":material/select_check_box:")

    resultsContainerCol1, resultsContainerCol2 = resultContainer.columns([3, 2])
    with resultsContainerCol1:
        with st.spinner("Image search in progress . . ."):
            visualsearch(
                resultsContainerCol1,
                resultsContainerCol2,
                dist=DIST,
                RANDOM_IMG=os.path.join(DATASET_FOLDER, "Images", IMAGE),
                colourSpace=cSpace,
            )
            resultsContainerCol1.success("Search complete!")
            st.toast("Visual search completed", icon=":material/select_check_box:")

    refContainer = st.container(border=True)
    with refContainer:
        refContainer.write("**References**")
        refContainer.write(
            """
            1. OpenCV Computer Vision Library, version 4.10.0 - https://docs.opencv.org/4.10.0/index.html
            2. SciPy, Scientific Computing in Python, version 1.14.1 - https://docs.scipy.org/doc/
            3. Scikit-Learn, Machine Learning in Python, version 1.5.1 - https://scikit-learn.org/1.5/index.html
            4. Numpy, Scientific Computing in Python, version 1.26.4 - https://numpy.org/doc/stable/
            5. Pandas, for Data Analysis in Python, version 2.2.2 - https://pandas.pydata.org/docs/user_guide/index.html
            6. Streamlit, Python framework for Web frontend, version 1.37.1 - https://docs.streamlit.io/
            7. Teaching materials and of course, the infinite resources of Internet.
    """
        )
end = time.time()
with tColR:
    tColR.metric(":blue[Execution Time, sec]", np.round((end - start), 0))
