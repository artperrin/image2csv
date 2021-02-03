import cv2
import numpy as np

def pre_process(images,visu,method):
    for i in range(len(images)):
        imgA = images[i]
        if method=='quality':
            imgB = cv2.fastNlMeansDenoising(imgA,None,20)
            temp,imgB = cv2.threshold(imgB,190,255,cv2.THRESH_BINARY_INV)
        else:
            temp,imgB = cv2.threshold(imgA,190,255,cv2.THRESH_BINARY_INV)
        imgB = cv2.morphologyEx(imgB, cv2.MORPH_CLOSE, np.ones((2,2)))
        imgB = cv2.erode(imgB, np.ones((4,4)))
        if visu=='y':
            cv2.imshow('before segmentation',imgA)        
            cv2.imshow('after segmentation (press 0 to close)',imgB)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        images[i] = imgB
        print('Pre-processed '+str(i+1)+' images out of '+str(len(images)))
        
    return images

def onmouse(event,x,y,flags,params):
    img = params
    rectangle = None
    global rect,ix,iy,rect_over
    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            rect = [(ix,iy),(x,y)]

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False
        rect_over = True
        sceneCopy = img.copy()
        rect = [(ix,iy),(x,y)]
        cv2.rectangle(sceneCopy,rect[0],rect[1],(0,0,255))
        cv2.imshow('Drawn rectangle (press 0 to close)',sceneCopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        rect = None

def draw_region(img):
    while True:
        # displaying the image
        cv2.imshow('image (draw a rectangle)', img) 
    
        # setting mouse hadler for the image 
        # and calling the click_event() function 
        cv2.setMouseCallback('image (draw a rectangle)', onmouse, img) 
    
        # wait for a key to be pressed to exit 
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
        return rect

def region_creator(img, initBoxes):
    xi = initBoxes[0][0][0]
    yi = initBoxes[0][0][1]
    wi = initBoxes[0][1][0]
    hi = initBoxes[0][1][1]
    height = img.shape[0]
    width = img.shape[1]

    Xshift = initBoxes[0][1][0] - initBoxes[0][0][0]
    Yshift = initBoxes[0][1][1] - initBoxes[0][0][1]
    initPos = initBoxes[0][0][0]

    Xoffset = 0
    Yoffset = 0
    search = True
    test = 0
    while search:
        print('----------------------------------')
        print('Automatic region extractor : enter offsets values for the grid to cover all numbers (end with 0 0)')
        Xoffset = int(input('X offset ? : (previously = '+str(Xoffset)+')  '))
        Yoffset = int(input('Y offset ? : (previously = '+str(Yoffset)+')  '))
        if Xoffset==0 and Yoffset==0 and test!=0:
            search = False
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()
            test+=1
            pos = [initPos,initBoxes[0][0][1]]
            endBoxes = [[(xi,yi),(wi,hi)]]

            while True:
                width_test = pos[0]+0.5*(Xshift+Xoffset)>=width
                height_test = pos[1]+0.5*(Yshift+Yoffset)>=height
                if width_test or height_test: # if their are no more regions on the right
                    if height_test: # if their are no more regions below
                        break
                    else: # restart from the line below
                        pos[1] += (Yshift+Yoffset)
                        pos[0] = initPos
                else: # if their are still regions on the right
                    newBox = [(pos[0],pos[1]),(pos[0]+Xshift,pos[1]+Yshift)]
                    pos[0] += (Xshift+Xoffset)
                    endBoxes.append(newBox)

        print('Number of regions extracted :', len(endBoxes))

        # Show the regions over the image
        if search:
            imgCopy = img.copy()
            for i in range(len(endBoxes)):
                (x,y) = endBoxes[i][0]
                (w,h) = endBoxes[i][1]
                cv2.rectangle(imgCopy, (x,y), (w,h), (0,0,255), 1)
            cv2.imshow('(temporary) extracted regions (press 0 to close)',imgCopy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    endBoxes.pop(0)
    return endBoxes