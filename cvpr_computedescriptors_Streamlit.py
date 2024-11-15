import os
import time
import cv2
import numpy as np
import scipy.io as sio
from extractRandom_v3 import extractRandom, extractRandomGrid
from HelperFunctions import quantizeImage, gridPlot, getDetails
import streamlit as st
from cvpr_visualsearch_v3 import visualsearch
from random import randint
from stqdm import stqdm
import warnings

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
        2,
        2,
        2,
    ]
)
with titleContainer:
    tCol1.title(":blue[Visual Search]")
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

fDesc = st.sidebar.radio(":blue[**Feature Descriptor**]", ["GCH", "HOG"])
st.sidebar.divider()
quant = st.sidebar.toggle(":blue[**Quantization**]")

if quant:
    technique = st.sidebar.radio("**Method**", ["K-Means", "Random"])
    q = st.sidebar.select_slider(
        "***Q Values***", options=[4, 8, 16, 32, 64, 128, 256]
    )  # Q = 256 means "No Quantization"
st.sidebar.divider()

# Create 3 row containers
inputContainer = st.container(border=True)
inCol1, inCol2, inCol3 = inputContainer.columns(
    [3, 3, 3], gap="medium", vertical_alignment="top"
)
with inputContainer:
    cSpace = inCol1.radio(":blue[**Colour Space**]", ["LAB", "RGB", "BGR"])
    grid = inCol3.toggle(":blue[**Grid**]")
    if grid:
        kSize = inCol3.slider("***Grid size***", 2, 4, step=1)
    else:
        kSize = inCol3.slider("***Grid size***", 2, 4, step=1, disabled=True)
    dist = inCol2.radio(
        ":blue[**Distance**]", ["Euclidean", "JensenShannon", "Minkowski", "Cosine"]
    )

###########################################################

# Controlling variables

# Feature extractor
featureDescriptor = fDesc  # GCH or HOG

# Colour Space
ColourSpace = cSpace

# Quantization
QUANTIZE = quant
if QUANTIZE:
    COLOURS = q  # 256 means 'No Quantization

# Gridding
GRID = grid
kSize = kSize  # vals = 2, 3, 4


# RANDOM_INDEX = randint(0, len(getDetails()) - 1)
RANDOM_IMG = imgSelected

# RANDOM_INDEX = 100
imgCount = 1
caption = ""

if imgSelect:
    run = st.sidebar.button(
        "Execute", type="primary", use_container_width=True, disabled=True
    )
else:
    run = st.sidebar.button(
        "Execute", type="primary", use_container_width=True, disabled=False
    )


logDetails = st.sidebar.expander(":blue[**Execution Log**]", expanded=True)
logDetails.empty()

outputExpander = st.expander("**Sample Input and Output images**", expanded=False)
outputExpanderCol1, outputExpanderCol2 = outputExpander.columns([2, 3])
col1, col2 = outputExpanderCol1.columns([3, 3])

if run or imgSelect:
    with outputExpanderCol1:
        # with st.spinner("Creating descriptors . . ."):
        # Ensure the output directory exists
        os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)
        imgCount = 1
        logDetails.caption(":green[- _Creating Image vectors_]")
        # Iterate through all BMP files in the dataset folder
        for filename in stqdm(os.listdir(os.path.join(DATASET_FOLDER, "Images"))):
            if filename.endswith(".bmp"):
                # print(f"Processing file {filename}")
                img_path = os.path.join(DATASET_FOLDER, "Images", filename)
                # img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
                img = cv2.imread(img_path)
                fout = os.path.join(
                    OUT_FOLDER, OUT_SUBFOLDER, filename.replace(".bmp", ".mat")
                )
                if filename == RANDOM_IMG:
                    # Display the original image
                    col1.image(
                        img,
                        caption="Original Image (BGR -> RGB for display)",
                        channels="BGR",
                    )

                if GRID:
                    if QUANTIZE:
                        if technique == "K-Means":
                            imgK = quantizeImage(
                                img,
                                colourSpace=ColourSpace,
                                clusterAlgo="kmeans",
                                divisions=COLOURS,
                            )  # divisons = 256 means, "No Quantization"
                            img = imgK
                            caption = f"Quantization: K-Means - {ColourSpace}"
                        else:
                            imgR = quantizeImage(
                                img,
                                colourSpace=ColourSpace,
                                clusterAlgo="random",
                                divisions=COLOURS,
                            )  # divisons = 256 means, "No Quantization"
                            img = imgR
                            caption = f"Quantization: Random, Pair-wise compressed Image - {ColourSpace}"
                    # Call extractRandom to get the descriptor
                    F, imgList = extractRandomGrid(
                        img,
                        colorSpace=ColourSpace,
                        kSize=kSize,
                        featureDescriptor=featureDescriptor,
                    )

                    # Show the image
                    if filename == RANDOM_IMG:
                        if len(caption) == 0:
                            caption = f"Colour Space - {ColourSpace}"
                        if ColourSpace == "LAB":
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        elif ColourSpace == "RGB":
                            img = img[:, :, ::-1]
                        col2.image(
                            img,
                            caption=caption,
                        )
                        if featureDescriptor == "HOG":
                            for i in range(len(imgList)):
                                _, imgList[i] = cv2.threshold(
                                    imgList[i].astype(np.uint8),
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                            col1.image(
                                imgList[0],
                                caption="HOG Image",
                            )
                            imgList = imgList[1:]
                        gridPlot(
                            imgList,
                            container=outputExpanderCol2,
                            imgFormat=None,
                            # caption="",
                            gridNum=kSize * kSize,
                            columns=4,
                        )
                else:
                    if QUANTIZE:
                        if technique == "K-Means":
                            imgK = quantizeImage(
                                img,
                                colourSpace=ColourSpace,
                                clusterAlgo="kmeans",
                                divisions=COLOURS,
                            )  # divisons = 256 means, "No Quantization"
                            img = imgK
                            caption = f"K-Means compressed Image - {ColourSpace}"
                        else:
                            imgR = quantizeImage(
                                img,
                                colourSpace=ColourSpace,
                                clusterAlgo="random",
                                divisions=COLOURS,
                            )  # divisons = 256 means, "No Quantization"
                            # Call extractRandom to get the descriptor
                            img = imgR
                            caption = (
                                f"Random, Pair-wise compressed Image - {ColourSpace}"
                            )
                    # Call extractRandom to get the descriptor
                    F, imgOut = extractRandom(
                        img,
                        colorSpace=ColourSpace,
                        featureDescriptor=featureDescriptor,
                    )

                    # Show the image
                    if filename == RANDOM_IMG:
                        if len(caption) == 0:
                            caption = f"Colour Space - {ColourSpace}"
                        if ColourSpace == "LAB":
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        elif ColourSpace == "RGB":
                            img = img[:, :, ::-1]
                        col2.image(
                            img,
                            caption=caption,
                        )
                        if featureDescriptor == "HOG":
                            _, imgOut = cv2.threshold(
                                imgOut,
                                0,
                                255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                            )
                            col1.image(
                                imgOut,
                                caption="HOG Image",
                            )
            # break
            imgCount += 1

            # Save the descriptor to a .mat file
            sio.savemat(fout, {"F": F})
        logDetails.caption(":green[- _Saved image vectors as .mat files_]")
        outputExpanderCol1.success("Descriptors created!")
        st.toast("Descriptors generated and saved", icon=":material/select_check_box:")

    resultsContainer = st.expander("**Results**", expanded=True)
    resultsContainerCol1, resultsContainerCol2 = resultsContainer.columns([3, 2])
    with resultsContainerCol1:
        with st.spinner("Image search in progress . . ."):
            visualsearch(
                resultsContainerCol1,
                resultsContainerCol2,
                dist=dist,
                RANDOM_IMG=os.path.join(DATASET_FOLDER, "Images", RANDOM_IMG),
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
