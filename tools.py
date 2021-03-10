import cv2
import numpy as np


def imclearborder(imgBW, radius):
    """Clear all object that touches at least one border of an image

    Parameters
    ----------
    imgBW : numpy.ndarray
        source image
    radius : int
        size of the border

    Returns
    -------
    numpy.ndarray
        image without the objects touching at least one border of the input image
    """
    imgBWcopy = imgBW.copy()
    contours, trash = cv2.findContours(
        imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    (imgRows, imgCols) = imgBW.shape
    contourList = []

    for idx in np.arange(len(contours)):
        cnt = contours[idx]

        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            check1 = (rowCnt >= 0 and rowCnt < radius) or (
                rowCnt >= imgRows - 1 - radius and rowCnt < imgRows
            )
            check2 = (colCnt >= 0 and colCnt < radius) or (
                colCnt >= imgCols - 1 - radius and colCnt < imgCols
            )

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

    return imgBWcopy


def pre_process(images, visu, method):
    """Pre-process the images for Tesseract to do a better job

    Parameters
    ----------
    images : array
        images to be pre-processed
    visu : string
        tell if the user wants visualization of the pre-processed image (comparison before/after)
    method : string
        tell what method the user wants (with or without denoize)

    Returns
    -------
    array
        pre-processed images
    """
    for i in range(len(images)):
        imgA = images[i]

        if method == "denoize":
            imgB = cv2.fastNlMeansDenoising(imgA, None, 20)
            trash, imgB = cv2.threshold(imgB, 190, 255, cv2.THRESH_BINARY_INV)
        else:
            trash, imgB = cv2.threshold(imgA, 190, 255, cv2.THRESH_BINARY_INV)

        imgB = cv2.morphologyEx(imgB, cv2.MORPH_CLOSE, np.ones((2, 2)))
        imgB = cv2.erode(imgB, np.ones((4, 4)))
        imgB = imclearborder(imgB, 1)

        if visu == "y":
            cv2.imshow("before segmentation", imgA)
            cv2.imshow("after segmentation (press 0 to close)", imgB)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        images[i] = imgB

        progress_line = ""

        for p in range(30):
            if p <= int((i + 1) / len(images) * 30):
                progress_line += "="
            else:
                progress_line += "."

        print(f"{i + 1}/{len(images)} [{progress_line}]", end="\r", flush=True)
    print("")
    return images


def onmouse(event, x, y, flags, params):
    """Function for the user to drawn a rectangle"""
    img = params
    rectangle = None
    global rect, ix, iy, rect_over
    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            rect = [(ix, iy), (x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False
        rect_over = True
        sceneCopy = img.copy()
        rect = [(ix, iy), (x, y)]
        cv2.rectangle(sceneCopy, rect[0], rect[1], (0, 0, 255))
        cv2.imshow("Drawn rectangle (press 0 to close)", sceneCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        rect = None


def draw_region(img):
    """Driver function for the user to draw a rectangle

    Parameters
    ----------
    img : numpy.dnarray
        image to draw a rectangle on

    Returns
    -------
    array
        coordinates of the rectangle the user has draw
    """
    while True:
        # displaying the image
        cv2.imshow("image (draw a rectangle)", img)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback("image (draw a rectangle)", onmouse, img)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
        return rect


def region_creator(img, initBoxes):
    """Create a grid on an image given an initial region

    Parameters
    ----------
    img : numpy.ndarray
        image the user wants a grid to be drawn on
    initBoxes : array
        the initial rectangle coordinates to build the grid from

    Returns
    -------
    endBoxes : array
        coordinates of each regions composing the grid

    lineLenghts : array
        number of regions in each grid's line
    """

    xi = initBoxes[0][0][0]
    yi = initBoxes[0][0][1]
    wi = initBoxes[0][1][0]
    hi = initBoxes[0][1][1]
    height = img.shape[0]
    width = img.shape[1]

    Xshift = initBoxes[0][1][0] - initBoxes[0][0][0]
    Yshift = initBoxes[0][1][1] - initBoxes[0][0][1]
    initPos = initBoxes[0][0][0]

    lineLenghts = []
    endBoxes = [[(xi, yi), (wi, hi)]]

    Xoffset = 0
    Yoffset = 0
    search = True
    test = 0
    while search:
        print("----------------------------------")
        print(
            "Automatic region extractor : enter offsets values for the grid to cover all numbers (end with 0 0)"
        )
        Xoffset = float(input(f"X offset ? : (previously = {str(Xoffset)}) "))
        Yoffset = float(input(f"Y offset ? : (previously = {str(Yoffset)}) "))
        if int(Xoffset) == 0 and int(Yoffset) == 0 and test != 0:
            search = False
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()
            test += 1
            pos = [initPos, initBoxes[0][0][1]]
            endBoxes = [[(xi, yi), (wi, hi)]]
            lineLenghts = []
            raw = 0
            while True:
                width_test = pos[0] + 0.5 * (Xshift + Xoffset) >= width
                height_test = pos[1] + 0.5 * (Yshift + Yoffset) >= height
                if (
                    width_test or height_test
                ):  # if there are no more regions on the right
                    if height_test:  # if there are no more regions below
                        break
                    else:  # restart from the line below
                        pos[1] += Yshift + int(Yoffset)
                        pos[0] = initPos
                        lineLenghts.append(raw)
                        raw = 0
                else:  # if there are still regions on the right
                    newBox = [
                        (pos[0], pos[1]),
                        (pos[0] + int(Xshift), pos[1] + int(Yshift)),
                    ]
                    pos[0] += Xshift + int(Xoffset)
                    endBoxes.append(newBox)
                    raw += 1

        print(f"Number of regions extracted : {len(endBoxes)}")

        # Show the regions over the image
        if search:
            imgCopy = img.copy()
            for i in range(len(endBoxes)):
                (x, y) = endBoxes[i][0]
                (w, h) = endBoxes[i][1]
                cv2.rectangle(imgCopy, (x, y), (w, h), (0, 0, 255), 1)
            cv2.imshow("(temporary) extracted regions (press 0 to close)", imgCopy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    endBoxes.pop(0)
    array_shape = [len(lineLenghts), lineLenghts[0]]
    return endBoxes, array_shape


def to_matrix(el, h, w):
    """Convert a list to a matrix given the number of elements per line

    Parameters
    ----------
    el : array
        list to be converted
    lineLenghts : array
        list of number of elements per line to build the matrix

    Returns
    -------
    array
        lines of the matrix formed by the elements of the input list
    """
    res = []
    for i in range(h):
        lineLenght = w
        temp = []
        for j in range(lineLenght):
            temp.append(el[j + i * lineLenght])
        res.append(temp)
    return np.array(res)