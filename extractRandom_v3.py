import numpy as np
import cv2
from skimage.feature import hog
from HelperFunctions import getImageVectorParameters


def extractRandom(
    img: np.array, colorSpace: str = "LAB", featureDescriptor: str = "GCH"
) -> tuple[np.array, np.array]:
    # Convert the image
    if colorSpace == "RGB":
        img = img[:, :, ::-1]
    elif colorSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Normalize the image
    img = img.astype(np.float32) / 255.0

    assert img.max() <= 1.0

    # Generate Image Vector (which is already normalized)
    if featureDescriptor == "HOG":
        # Calculate HOG features
        imgVector, imgH = hog(
            img,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            feature_vector=True,
            channel_axis=-1,
        )

        img = (imgH * 255).astype(np.uint8)

        # Considering 1500 of the feature values
        imgVector = np.array(imgVector[:1500])

    else:
        # Calculating histogram for each channel
        ch1 = cv2.calcHist([img.astype(np.float32)], [0], None, [16], [0.0, 2.0])
        ch2 = cv2.calcHist([img.astype(np.float32)], [1], None, [16], [0.0, 2.0])
        ch3 = cv2.calcHist([img.astype(np.float32)], [2], None, [16], [0.0, 2.0])

        # Normalize the channel histograms
        ch1 = cv2.normalize(ch1, ch1, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX)
        ch2 = cv2.normalize(ch2, ch2, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX)
        ch3 = cv2.normalize(ch3, ch3, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX)

        img = (img * 255).astype(np.uint8)

        # Create the image vector
        imgVector = np.concatenate([ch1, ch2, ch3], axis=0)

        # Make sure the vector is of length 192
        # assert len(imgVector) == 192

    imgVector = imgVector.reshape(-1)

    # Make sure the vector is of length 30
    # assert len(imgVector) == 30

    return (imgVector, img)


def extractRandomGrid(
    img: np.array,
    colorSpace: str = "LAB",
    kSize: int = 2,
    featureDescriptor: str = "GCH",
) -> tuple[np.array, list]:
    # Convert the image
    if colorSpace == "RGB":
        img = img[:, :, ::-1]
    elif colorSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Normalize the image
    img = img.astype(np.float32) / 255.0

    assert img.max() <= 1.0

    img2 = img

    height, width, _ = img.shape
    basket = np.array([])

    imgList = []

    # HOG features of the input image for display
    if featureDescriptor == "HOG":
        _, imgH = hog(
            img,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            feature_vector=True,
            channel_axis=-1,
        )
        imgList.append((imgH * 255).astype(np.uint8))

    for kh in range(kSize):
        for kw in range(kSize):
            x = width / kSize * kw
            y = height / kSize * kh
            h = height / kSize
            w = width / kSize
            img = img[int(y) : int(y + h), int(x) : int(x + w)]

            if featureDescriptor == "HOG":
                features, imgH = hog(
                    img,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    visualize=True,
                    feature_vector=True,
                    channel_axis=-1,
                )

                imgList.append((imgH * 255).astype(np.uint8))

                basket = np.concatenate([basket, features])
            else:
                ch1 = cv2.calcHist(
                    [img.astype(np.float32)], [0], None, [16], [0.0, 2.0]
                )
                ch2 = cv2.calcHist(
                    [img.astype(np.float32)], [1], None, [16], [0.0, 2.0]
                )
                ch3 = cv2.calcHist(
                    [img.astype(np.float32)], [2], None, [16], [0.0, 2.0]
                )

                # Normalize the channel histograms
                ch1 = cv2.normalize(
                    ch1, ch1, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX
                )
                ch2 = cv2.normalize(
                    ch2, ch2, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX
                )
                ch3 = cv2.normalize(
                    ch3, ch3, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX
                )

                imgList.append((img * 255).astype(np.uint8))

                basket = np.concatenate([basket, ch1.flatten()])
                basket = np.concatenate([basket, ch2.flatten()])
                basket = np.concatenate([basket, ch3.flatten()])

            img = img2

    bSlice = 0

    if featureDescriptor == "HOG":
        bSlice = 1200  # based on lengths generated
        imgVector = basket[:bSlice]
    else:
        bSlice = 16 * kSize * kSize
        imgVector = basket[:bSlice]

    assert len(imgVector) == bSlice, print(len(imgVector))

    imgVector = imgVector.reshape(-1)

    return imgVector, imgList


def extractRandomAdv(
    img: np.array, colorSpace: str = "GREY", featureDescriptor: str = "SIFT"
) -> np.array:
    # Convert the image
    if colorSpace == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif colorSpace == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Generate Image Vector (which is already normalized)
    _, imgVector = getImageVectorParameters(
        img, colour=colorSpace, descriptor=featureDescriptor
    )

    imgVector = imgVector.reshape(-1)

    return imgVector
