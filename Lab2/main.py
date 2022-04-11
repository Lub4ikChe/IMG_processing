import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def match_self(des1, des2):
    match = []
    distances = {}

    for i, d_1 in enumerate(des1):
        if np.sum(d_1) != 0:
            d = []
            for _, d_2 in enumerate(des2):
                norm = cv.norm(d_1, d_2, cv.NORM_HAMMING)
                d.append(norm)
            orders = np.argsort(d).tolist()
            if d[orders[0]]/d[orders[1]] <= 0.98:
                match.append((i, orders[0]))
            distances[f'{i}-{orders[0]}'] = d[orders[0]]

    return [cv.DMatch(pair[0], pair[1], distances[f'{pair[0]}-{pair[1]}']) for pair in match]



images = ['img1.jpeg', 'img2.jpeg']

for image in images:
    # Load the image
    training_image = cv.imread(image, cv.IMREAD_GRAYSCALE)

    # resize
    scale_percent = 30  # percent of original size
    width = int(training_image.shape[1] * scale_percent / 100)
    height = int(training_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    training_image = cv.resize(training_image, dim, interpolation=cv.INTER_AREA)

    # Create test image by adding Scale Invariance and Rotational Invariance
    test_image = cv.pyrDown(training_image)
    test_image = cv.pyrDown(test_image)
    num_rows, num_cols = test_image.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    test_image = cv.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

    # initialization of detector and extractor
    fast = cv.FastFeatureDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # key points
    kp1 = fast.detect(training_image, None)
    kp2 = fast.detect(test_image, None)

    # descriptors
    kp1, des1 = brief.compute(training_image, kp1)
    kp2, des2 = brief.compute(test_image, kp2)

    # self match
    matches = match_self(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    result = cv.drawMatches(training_image, kp1, test_image,
                            kp2, matches[:10], None, flags=2)

    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Best Self Matching Points')
    plt.imshow(result)
    plt.show()

    # library match
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches_bf = bf.match(des1, des2)
    matches_bf = sorted(matches_bf, key=lambda x: x.distance)
    result_bf = cv.drawMatches(training_image, kp1, test_image,
                            kp2, matches_bf[:10], None, flags=2)

    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Best BF Matching Points')
    plt.imshow(result_bf)
    plt.show()

