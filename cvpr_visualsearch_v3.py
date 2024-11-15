import os
import cv2
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from cvpr_compare_v3 import cvpr_compare
from HelperFunctions import gridPlot, getDetails, getMinLengthVectors
# from random import randint

# from random import randint
# from tqdm import tqdm
# import ipdb


def visualsearch(
    resultsContainer1: any,
    resultsContainer2: any,
    dist: str,
    RANDOM_IMG: str,
    colourSpace: str,
) -> None:
    DESCRIPTOR_FOLDER = "descriptors"
    DESCRIPTOR_SUBFOLDER = "globalRGBhisto"
    IMAGE_FOLDER = "MSRC_ObjCategImageDatabase_v2"

    # Load all descriptors
    ALLFEAT = []
    ALLFILES = []
    # for filename in tqdm(
    #     os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)),
    #     colour="green",
    # ):

    for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
        if filename.endswith(".mat"):
            img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            img_actual_path = os.path.join(IMAGE_FOLDER, "Images", filename).replace(
                ".mat", ".bmp"
            )
            # ipdb.set_trace()
            img_data = sio.loadmat(img_path)
            ALLFILES.append(img_actual_path)
            ALLFEAT.append(img_data["F"][0])  # Assuming F is a 1D array
    ALLFEAT = getMinLengthVectors(ALLFEAT)
    # Convert ALLFEAT to a numpy array
    ALLFEAT = np.array(ALLFEAT)

    # Pick a random image as the query
    NIMG = ALLFEAT.shape[0]
    # queryimg = randint(0, NIMG - 1)
    # queryimg = 100  # 100
    # queryimg = RANDOM_INDEX - 1  # To sync indexing

    queryimg = np.where(np.array(RANDOM_IMG) == ALLFILES)[0][0]

    # Compute the distance between the query and all other descriptors
    dst = []
    query = ALLFEAT[queryimg]

    for i in range(NIMG):
        candidate = ALLFEAT[i]
        if dist == "Mahalanobis":
            distance = cvpr_compare(
                ALLFILES[queryimg],
                query,
                ALLFILES[i],
                candidate,
                distAlgo=dist,
                colourSpace=colourSpace,
            )
        else:
            distance = cvpr_compare(
                "", query, "", candidate, distAlgo=dist, colourSpace=colourSpace
            )

        dst.append((distance, i))

    # Sort the distances
    dst.sort(key=lambda x: x[0])

    # Referemce image
    img = cv2.imread(ALLFILES[dst[0][1]])
    fname_ref = ALLFILES[dst[0][1]].split("/")[-1].split(".")[0]
    category = int(fname_ref.split("_")[0])
    # numCatImages = getDetails().groupby(["Category"]).count().iloc[category].values[0]
    df_getDetails = getDetails()
    numCatImages = df_getDetails.loc[df_getDetails.Category == str(category)][
        "ImageFile"
    ].count()
    resultsContainer1.image(img, caption="Search Image", channels="BGR")
    # cv2.imshow(f"Reference: {fname_ref}", img)
    # cv2.waitKey(0)

    # Show the top 15 results
    SHOW = 15
    matches = 0
    imgList = []
    captions = []
    for i in range(SHOW):
        img = cv2.imread(ALLFILES[dst[i][1]])
        imgList.append(img)
        fname = ALLFILES[dst[i][1]].split("/")[-1].split(".")[0]
        if fname_ref.split("_")[0] == fname.split("_")[0]:
            matches += 1
            result = fname + " - Matched"
        else:
            result = fname + " - Mis-Matched"
        captions.append(result)

    ### Grid Plot in Column - 1

    gridPlot(
        imgList,
        container=resultsContainer1,
        imgFormat="BGR",
        caption=captions,
        gridNum=SHOW,
        columns=4,
    )

    ### Stats in Column - 2

    statContainer = resultsContainer2.container(border=True)
    col1, col2 = statContainer.columns([2, 2])
    with statContainer:
        delta = f"{np.round(((matches / SHOW) - 1) * 100, 2)}%"
        col1.metric(":green[***Matches***]", matches, delta)
        col1.metric(":blue[***Sample Size***]", SHOW)

    ### Plot in Column - 2
    resultList = [0 if "Mis" in i else 1 for i in captions]
    Precision = [
        np.round(sum(resultList[: i + 1]) / len(resultList[: i + 1]), 2)
        for i in range(len(resultList))
    ]
    # Calculate Average Precision (AP)
    relevant_docs = sum(resultList)  # Total number of relevant images in the top 15
    AP = np.round(sum([Precision[i] for i in range(len(resultList)) if resultList[i] == 1]) / relevant_docs, 2)
    Recall = [
        np.round(sum(resultList[: i + 1]) / numCatImages, 2)
        for i in range(len(resultList))
    ]
    col2.metric(":blue[***Average Precision***]", AP)
    col2.metric(":blue[***Recall***]", np.round(sum(resultList) / numCatImages, 2))
    col1.metric(f":blue[***Category-{category} images***]", numCatImages)

    ### PR Curve for all Images ###

    # Consider all images
    SHOW = NIMG
    matches = 0
    imgList = []
    captions = []
    for i in range(SHOW):
        fname = ALLFILES[dst[i][1]].split("/")[-1].split(".")[0]
        if fname_ref.split("_")[0] == fname.split("_")[0]:
            matches += 1
            result = fname + " - Matched"
        else:
            result = fname + " - Mis-Matched"
        captions.append(result)

    resultList = [0 if "Mis" in i else 1 for i in captions]
    Precision = [
        np.round(sum(resultList[: i + 1]) / len(resultList[: i + 1]), 2)
        for i in range(len(resultList))
    ]
    Recall = [
        np.round(sum(resultList[: i + 1]) / numCatImages, 2)
        for i in range(len(resultList))
    ]

    fig, ax = plt.subplots()
    # plt.style.use("ggplot")
    plt.style.use("seaborn-v0_8")
    ax.plot(Recall, Precision, "b", marker=".")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR - Curve")

    plotContainer = resultsContainer2.container(border=True)
    with plotContainer:
        plotContainer.pyplot(fig)
