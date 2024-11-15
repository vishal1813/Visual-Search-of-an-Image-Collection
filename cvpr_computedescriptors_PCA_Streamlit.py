import os
import cv2
import time
import numpy as np
from HelperFunctions import (
    getDetails,
    myPCA,
)
import streamlit as st
import warnings
from cvpr_visualsearch_v3 import visualsearch
from random import randint
from stqdm import stqdm

# from visualsearch import visualsearch

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
    tCol1.title(":blue[Visual Search - PCA]")
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
inCol1, inCol2 = inputContainer.columns([3, 3], gap="medium", vertical_alignment="top")

with inputContainer:
    cSpace = inCol1.radio(":blue[**Colour Space**]", ["RGB", "BGR", "LAB"])
    components = inCol2.slider(
        ":blue[**PCA Components**]", min_value=1, max_value=300, value=5
    )

# Controlling parameters

IMAGE = imgSelected
COLOR = cSpace
n_components = components

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


def displayInputOutputImages(img: np.array, colour: str) -> None:
    logDetails.empty()
    logDetails.caption(":green[- _Display input and generated images_]")

    # OpenCV reads image in BGR (assuming the image is stored in RGB)
    img = cv2.imread(os.path.join(DATASET_FOLDER, "Images", IMAGE), cv2.IMREAD_COLOR)
    imgCol1.image(image=img[:, :, ::-1])
    imgCol1.caption("Original Image (BGR -> RGB for viewing)")
    if cSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif cSpace == "RGB":
        img = img[:, :, ::-1]

    imgCol2.image(image=img)
    imgCol2.caption(f"Converted to {COLOR} colour-space")
    pcaImg = myPCA(img, cSpace, n_components=n_components)
    imgCol3.image(pcaImg)
    imgCol3.caption("PCA compressed image")

    return


# ##########

# Display Search image and generated images in  the panel
displayInputOutputImages(IMAGE, COLOR)

resultContainer = st.expander("Results", expanded=True)

if run or imgSelect:
    with resultContainer:
        with st.spinner("Generating descriptors . . ."):
            logDetails.caption(":green[- _Generating descriptors_]")

            # Iterate through all BMP files in the dataset folder
            for filename in stqdm(os.listdir(os.path.join(DATASET_FOLDER, "Images"))):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(DATASET_FOLDER, "Images", filename)
                    img = cv2.imread(img_path)
                    pcaImg = myPCA(img, cSpace, n_components=n_components)
                    fout = os.path.join(
                        OUT_FOLDER, OUT_SUBFOLDER, filename.replace(".bmp", ".mat")
                    )
                    # st.write(img.shape)

                    # Call extractRandom to get the descriptor
                    # F = extractRandomPCA(
                    #     pcaImg,
                    #     colorSpace=COLOR,
                    # )
                    filename = filename.split(".")[0]
                    filename = DATASET_FOLDER + "/PCA/" + filename + "_PCA" + ".bmp"
                    cv2.imwrite(filename=filename, img=pcaImg)
                    # sio.savemat(fout, {"F": F})
                    # break
            logDetails.caption(":green[- _Saved PCA compressed images_]")
            resultContainer.success("Descriptors created!")
            st.toast("Descriptors created", icon=":material/select_check_box:")

    resultsContainerCol1, resultsContainerCol2 = resultContainer.columns([3, 2])
    with resultsContainerCol1:
        with st.spinner("Image search in progress . . ."):
            visualsearch(
                resultsContainerCol1,
                resultsContainerCol2,
                dist="Mahalanobis",
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
