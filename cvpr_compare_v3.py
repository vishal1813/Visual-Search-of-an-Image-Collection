import cv2
import numpy as np
from scipy.spatial import distance


def mahalanobis_distance(
    image1_path: str,
    imgMeanVec1: np.array,
    image2_path: str,
    imgMeanVec2: np.array,
    colourSpace: str,
) -> float:
    # Get the corresponding PCA file name
    image1_path = image1_path.replace("Images", "PCA")
    image1_path = image1_path.replace(".bmp", "_PCA.bmp")

    image2_path = image2_path.replace("Images", "PCA")
    image2_path = image2_path.replace(".bmp", "_PCA.bmp")

    # Read the images from the given path
    img1 = cv2.imread(image1_path, 1)
    img2 = cv2.imread(image2_path, 1)

    # Convert Colour Space
    if colourSpace == "RGB":
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    elif colourSpace == "LAB":
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    channels = img1.shape[2]

    # Normalize the images
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # Reshape the images into 2D
    flat_image1 = img1.reshape(-1, channels)
    flat_image2 = img2.reshape(-1, channels)

    # Calculate the mean
    imgMeanVec1 = np.mean(flat_image1, axis=0)
    imgMeanVec2 = np.mean(flat_image2, axis=0)

    # Calculate the Covariance Matrix
    v = np.cov(
        np.concatenate((flat_image1 - imgMeanVec1, flat_image2 - imgMeanVec2), axis=0).T
    )

    # Calculate the Inverse of Covariance Matix
    iv = np.linalg.pinv(v)

    # Compute Mahalanobis distance between centroids
    mahal_dist = np.sqrt(
        (imgMeanVec1 - imgMeanVec2) @ iv @ (imgMeanVec1 - imgMeanVec2).T
    )

    return mahal_dist


def cvpr_compare(
    imgPath1: str,
    imgVector1: np.array,
    imgPath2: str,
    imgVector2: np.array,
    distAlgo: str = "Euclidean",
    colourSpace: str = "RGB",
) -> float:
    length = min(len(imgVector1), len(imgVector2))
    imgVector1 = imgVector1[:length]
    imgVector2 = imgVector2[:length]

    assert len(imgVector1) == len(imgVector2), print(
        "Length mis-match: ", len(imgVector1, len(imgVector2))
    )

    # Make sure the vectors are arrays and of dtype = np.float32
    if isinstance(imgVector1, list):
        imgVector1 = np.array(imgVector1).astype(np.float32)
        imgVector2 = np.array(imgVector2).astype(np.float32)
    if imgVector1.dtype != np.float32:
        imgVector1 = imgVector1.astype(np.float32)
        imgVector2 = imgVector2.astype(np.float32)

    if distAlgo == "Cosine":
        return np.round(distance.cosine(imgVector1, imgVector2), 9)
    elif distAlgo == "JensenShannon":
        return np.round(
            distance.jensenshannon(imgVector1, imgVector2), 9
        )  # Default log base is e
    elif distAlgo == "Minkowski":
        return np.round(
            distance.minkowski(imgVector1, imgVector2), 9
        )  # p, the order of the norm, is 2 by default
    elif distAlgo == "Mahalanobis":
        return mahalanobis_distance(
            imgPath1, imgVector1, imgPath2, imgVector2, colourSpace
        )
    else:
        return np.round(np.sqrt(sum((imgVector1 - imgVector2) ** 2)), 9)
