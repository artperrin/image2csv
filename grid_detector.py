import cv2
import numpy as np


def dedupe(pts, thresh):
    """Eliminates redundant lines: points are grouped by distance and the first point is used as representative

    Parameters
    ----------
    pts : array
        array of points to be grouped
    thresh : float
        distance threshold below which two points are grouped

    Returns
    -------
    array
        array of representatives of the groups
    """
    pts = sorted(pts)

    i = 0
    while i < len(pts) - 1:
        if np.abs(pts[i] - pts[i + 1]) < thresh:
            pts.pop(i + 1)
            continue
        i += 1

    return pts


def intersection(o1, p1, o2, p2):
    """Returns the intersection between two lines (o1, p1) and (o2, p2)

    Parameters
    ----------
    o1 : int
        (x,y) coordinates of the beginning of the first line
    p1 : int
        (x,y) coordinates of the end of the first line
    o2 : int
        (x,y) coordinates of the beginning of the second line
    p2 : int
        (x,y) coordinates of the end of the second line

    Returns
    -------
    array
        (x,y) coordinates of the intersection of the two lines
    """
    o1, p1 = np.array(o1), np.array(p1)
    o2, p2 = np.array(o2), np.array(p2)

    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-8:
        return None

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    inter = o1 + d1 * t1
    return (int(inter[0]), int(inter[1]))


def detect_grid(
    img,
    kernel_size=5,
    dist=5 / 100,
    canny_th_low=50,
    canny_th_high=150,
    rho=1,
    theta=np.pi / 180,
    threshold=15,
    min_length_line=50,
    max_length_gap=20,
):
    """Perform the automatic grid detection and returns all the ROI

    Parameters
    ----------
    img : np.array
        image of the grid to be detected
    kernel_size : int, optional
        kernel size of the Gaussian Blur applied for denoizing, by default 5
    dist : float, optional
        distance threshold for line grouping (relative to the image's size), by default 5/100
    canny_th_low : int, optional
        low threshold for OpenCV's contour detection function (see official documentation, link below), by default 50
    canny_th_high : int, optional
        high threshold for OpenCV's contour detection function (see official documentation, link below), by default 150
    rho : int, optional
        parameter of OpenCV's Hough transform function (see official documentation, link below), by default 1
    theta : [type], optional
        parameter of OpenCV's Hough transform function (see official documentation, link below), by default np.pi/180
    threshold : int, optional
        parameter of OpenCV's Hough transform function (see official documentation, link below), by default 15
    min_length_line : int, optional
        parameter of OpenCV's Hough transform function (see official documentation, link below), by default 50
    max_length_gap : int, optional
        parameter of OpenCV's Hough transform function (see official documentation, link below), by default 20

    OpenCV's documentation for Canny function : https://www.docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    OpenCV's documentation for Hough Transform function : https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
    Returns
    -------
    np.array, array
        array of the detected ROI and an array containing the number of lines and rows
    """
    image = img.copy()
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # denoize
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(image, canny_th_low, canny_th_high)

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_length_line, max_length_gap
    )

    hori, vert = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if np.abs(x1 - x2) < dist * w:
                vert.append((x1 + x2) // 2)
            elif np.abs(y1 - y2) < dist * h:
                hori.append((y1 + y2) // 2)

    hori, vert = dedupe(hori, dist * h), dedupe(vert, dist * w)

    intersections = []
    for y in hori:
        o1, p1 = (0, y), (w, y)
        temp = []
        for x in vert:
            o2, p2 = (x, 0), (x, h)
            temp.append(intersection(o1, p1, o2, p2))
        intersections.append(temp)

    boxes = []
    for i in range(len(intersections) - 1):
        for j in range(1, len(intersections[i])):
            a = intersections[i][j - 1]
            b = intersections[i + 1][j]
            box = (a, b)
            boxes.append(box)

    return boxes, [len(hori) - 1, len(vert) - 1]


# to test the script
if __name__ == "__main__":
    img = cv2.imread("./testimages/test.png")
    h, w = img.shape[:2]

    boxes, trash = detect_grid(img, dist=5 / 100)

    line = img.copy()
    for (pt1, pt2) in boxes:  # pt1 top left, pt2 bottom right
        cv2.rectangle(line, pt1, pt2, (0, 0, 255))

    cv2.imshow("output", line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()