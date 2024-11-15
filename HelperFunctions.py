import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# os.environ["OMP_NUM_THREADS"] = "4"


def quantizeImage(
    img: np.array, colourSpace: str, clusterAlgo: str = "random", divisions: int = 4
) -> np.array:
    if divisions == 256:
        return img

    # Convert the image
    if colourSpace == "RGB":
        img = img[:, :, ::-1]
    elif colourSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Normalize the image
    img = img.astype(np.float32) / 255.0

    assert img.max() <= 1.0

    imgArray = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    # Quantizing using Pairwise distance - Thanks to Scipy Documentation
    # Ref: https://scikit-learn.org/1.5/auto_examples/cluster/plot_color_quantization.html

    if clusterAlgo == "kmeans":
        imgShuffle = shuffle(imgArray, random_state=0, n_samples=1000)
        kmeans = KMeans(n_clusters=divisions, random_state=0).fit(imgShuffle)
        labels = kmeans.predict(imgArray)

        img = kmeans.cluster_centers_[labels].reshape(img.shape[0], img.shape[1], -1)
    else:
        imgShuffle = shuffle(imgArray, random_state=0, n_samples=divisions)
        labels = pairwise_distances_argmin(imgShuffle, imgArray, axis=0)

        img = imgShuffle[labels].reshape(img.shape[0], img.shape[1], -1)

    # Converting the color space back to BGR
    img = img * 255.0
    img = img.astype(np.uint8)

    # Revert back to BGR
    if colourSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    elif colourSpace == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img  # Returns image in BGR format


# def myPCA(img: np.array, colourSpace: str = "LAB", variance: float = 0.95) -> np.array:
def myPCA(img: np.array, colourSpace: str = "LAB", n_components: int = 5) -> np.array:
   
    # Resize the image to 300 x 300
    img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)

    # Convert the image
    if colourSpace == "RGB":
        img = img[:, :, ::-1]
    elif colourSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Normalize the image
    img = img.astype(np.float32) / 255.0

    assert img.max() <= 1.0

    # Split the image into it's individual colour components
    ch1, ch2, ch3 = cv2.split(img)

    # Consider till the given variance in data
    pca = PCA(n_components=n_components)

    # PCA fit and transform each channel
    ch1FT = pca.fit_transform(ch1)
    ch1Inv = pca.inverse_transform(ch1FT)

    ch2FT = pca.fit_transform(ch2)
    ch2Inv = pca.inverse_transform(ch2FT)

    ch3FT = pca.fit_transform(ch3)
    ch3Inv = pca.inverse_transform(ch3FT)

    # Stack the channels to get the compressed image
    img_converted = np.dstack((ch1Inv, ch2Inv, ch3Inv))
    img_converted *= 255
    img_converted = img_converted.astype(np.uint8)

    # Revert back to BGR
    if colourSpace == "LAB":
        img_converted = cv2.cvtColor(img_converted, cv2.COLOR_LAB2BGR)
    elif colourSpace == "RGB":
        img_converted = img_converted[:, :, ::-1]  # to BGR

    return img_converted  # Returns image in BGR format


def gridPlot(
    img: list,
    container: any,
    imgFormat: any,
    caption: any = "",
    gridNum: int = 4,
    columns: int = 4,
) -> None:
    inList = [2 for i in range(columns)]
    colNames = ["Col" + str(i) for i in range(columns)]
    colNames = container.columns(inList)

    for i in range(gridNum):
        if not isinstance(caption, list):
            if imgFormat is None:
                colNames[i % columns].image(
                    img[i], use_column_width="always", caption=f"Grid-{i+1}"
                )
            else:
                colNames[i % columns].image(
                    img[i],
                    use_column_width="always",
                    caption=f"Grid-{i+1}",
                    channels=imgFormat,
                )
        else:
            if imgFormat is None:
                colNames[i % columns].image(
                    img[i], use_column_width="always", caption=caption[i]
                )
            else:
                colNames[i % columns].image(
                    img[i],
                    use_column_width="always",
                    caption=caption[i],
                    channels=imgFormat,
                )


def getDetails():
    PATH = "MSRC_ObjCategImageDatabase_v2/Images/"
    data = pd.DataFrame()
    for fname in os.listdir(PATH):
        cat = fname.split("_")[0]
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    np.array([cat, fname]).reshape(1, -1),
                    columns=["Category", "ImageFile"],
                ),
            ]
        )
    return data.sort_values(["Category"])


def getImageVectorParameters(
    img: np.array, colour: str, descriptor: str
) -> tuple[np.array, np.array]:
    # img is already converted based on Colour Space

    # SIFT or ORB is generally used on GREY images.
    # In a 3 channel, colour image, they should be applied to the Luminance channel
    # In HSV, it is the V - channel and in LAB, it is the L - channel

    if colour == "LAB":
        img, _, _ = cv2.split(img)
    elif colour == "HSV":
        _, _, img = cv2.split(img)

    if descriptor == "SIFT":
        # Get Sift output parameters
        sift = cv2.SIFT_create(
            nfeatures=100
        )  # nfeatures, the number of best features to retain
        # It’s worth noting that if an image doesn’t have any noticeable features
        # (e.g., it is a flat image without any edges, gradients, etc.),
        # extraction with SIFT can return None
        try:
            kp, des = sift.detectAndCompute(img, mask=None)
            return (kp, des)
        except:  # noqa: E722
            return None
    else:
        orb = cv2.ORB_create(
            nfeatures=100
        )  # nfeatures, the number of best features to retain
        try:
            kp, des = orb.detectAndCompute(img, mask=None)
            return (kp, des)
        except:  # noqa: E722
            return None


def getMinLengthVectors(vector: list) -> list:
    minVal = 1e10
    for val in vector:
        if len(val) < minVal:
            minVal = len(val)

    vectorList = [val[:minVal] for val in vector]

    return vectorList


def TFIDF(numImages: int, freqVectors: np.array) -> np.array:
    df = np.sum(freqVectors > 0, axis=0)
    idf = np.log(numImages / df)
    tfidf = freqVectors * idf
    return tfidf
