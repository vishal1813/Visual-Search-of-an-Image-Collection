import os
import cv2
import time
import numpy as np
import scipy.io as sio
from HelperFunctions import (
    # gridPlot,
    getDetails,
    getImageVectorParameters,
    TFIDF,
)
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
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
    tCol1.title(":blue[Visual Search - Bag of Visual Words]")
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

###########################################################

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
    if cSpace == "GREY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif cSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    imgCol2.image(image=img)
    imgCol2.caption(f"Converted to {COLOUR} colour-space")

    keyPoints, _ = getImageVectorParameters(img, colour=COLOUR, descriptor=DESCRIPTOR)

    if COLOUR == "GREY":
        black = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    else:
        black = np.zeros([img.shape[0], img.shape[1], img.shape[2]], dtype=np.uint8)

    image = cv2.drawKeypoints(
        black, keyPoints, black, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    imgCol3.image(image)
    imgCol3.caption(
        f"{fdescriptor} {IMAGE} - {len(keyPoints)} important keypoints choosen"
    )

    return


# ##########

# Display Search image and generated images in  the panel
displayInputOutputImages(IMAGE, COLOUR, DESCRIPTOR)

keyDescriptors = []
images = []
imgName = []
n_clusters = 100

resultContainer = st.expander("Results", expanded=True)

if run or imgSelect:
    with resultContainer:
        with st.spinner("Data Generation in Progress . . ."):
            logDetails.caption(f":green[- _Creating {DESCRIPTOR} descriptors_]")
            # Iterate through all BMP files in the dataset folder
            for filename in stqdm(
                os.listdir(os.path.join(DATASET_FOLDER, "Images")),
                desc=f"Generating {DESCRIPTOR} Image Vectors -> ",
            ):
                if filename.endswith(".bmp"):
                    img_path = os.path.join(DATASET_FOLDER, "Images", filename)
                    img = cv2.imread(img_path)
                    # Colour Space conversion
                    if COLOUR == "LAB":
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    elif COLOUR == "GREY":
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    images.append(img)
                    imgName.append(filename)
                    # Get Image descriptors of either SIFT or ORB based on selection
                    if (
                        getImageVectorParameters(
                            img, colour=COLOUR, descriptor=DESCRIPTOR
                        )
                        is not None
                    ):
                        _, kd = getImageVectorParameters(
                            img, colour=COLOUR, descriptor=DESCRIPTOR
                        )
                        keyDescriptors.append(kd)

            logDetails.caption(":green[- _Creating Codebook & Vector frequency_]")

            # Shuffle the descriptors and create the Codebook
            desShuffled = shuffle(keyDescriptors, random_state=100)

            # Flattening the shuffled descriptors into a single array
            desShuffled = np.array([list(k) for kd in desShuffled for k in kd])

            # Group similar features using K-Means algorithm. Clusters taken as 100.
            kmeans = KMeans(
                n_clusters=n_clusters, random_state=1805, n_init="auto"
            ).fit(desShuffled)

            # The code book is the cluster centres
            codebook = kmeans.cluster_centers_

            # Transform visual descriptors into visual words.
            # In effect, you are labeling the descriptors based on cluster labels
            visual_words = []
            for des in keyDescriptors:
                visual_words.append(kmeans.predict(des))

            # Now create the frequency vectors
            freq_vectors = []
            for word in visual_words:
                tmp = np.zeros(n_clusters)
                for val in word:
                    tmp[val] += 1
                freq_vectors.append(tmp)
            freq_vectors = np.stack(freq_vectors)

            logDetails.caption(":green[- _Creating TF-IDF vectors_]")

            # TF-IDF, or term frequency-inverse document frequency, is a statistical measure that determines
            # how important a word is to a document in a collection.
            # We must use tf-idf to adjust the frequency vector to consider relevance.
            tfidf = TFIDF(len(imgName), freqVectors=freq_vectors)

            logDetails.caption(":green[- _Saving TF-IDF vectors to .mat files_]")

            for filename, tf_idf in zip(imgName, tfidf):
                fout = os.path.join(
                    OUT_FOLDER, OUT_SUBFOLDER, filename.replace(".bmp", ".mat")
                )
                F = tf_idf
                sio.savemat(fout, {"F": F})

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
            1. 5 minutes with Cyrill - https://www.youtube.com/watch?v=a4cFONdc6nc
            2. Bag of Visual Words for finding similar images - https://www.youtube.com/watch?v=HWIkcMakz-s
            3. Bag of Words - https://www.pinecone.io/learn/series/image-search/bag-of-visual-words/
            4. OpenCV Computer Vision Library, version 4.10.0 - https://docs.opencv.org/4.10.0/index.html
            5. SciPy, Scientific Computing in Python, version 1.14.1 - https://docs.scipy.org/doc/
            6. Scikit-Learn, Machine Learning in Python, version 1.5.1 - https://scikit-learn.org/1.5/index.html
            7. Numpy, Scientific Computing in Python, version 1.26.4 - https://numpy.org/doc/stable/
            8. Pandas, for Data Analysis in Python, version 2.2.2 - https://pandas.pydata.org/docs/user_guide/index.html
            9. Streamlit, Python framework for Web frontend, version 1.37.1 - https://docs.streamlit.io/
            10. Teaching materials and of course, the infinite resources of Internet.
            """
        )
end = time.time()
with tColR:
    tColR.metric(":blue[Execution Time, sec]", np.round((end - start), 0))
