import argparse
import cv2
from region_drawer import draw_region
from region_drawer import region_creator
import numpy as np
import pytesseract
from segmentation import pre_process
import matplotlib.pyplot as plt
import time

# internal parameters
window_maximizer = 6

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, 
	help="path to input image")
ap.add_argument("-v", "--visualization", default='n', type=str, help="y/[n] to see each extracted regions individualy")
args = vars(ap.parse_args())

visu = args["visualization"]

# load the input image from disk
print('[INFO] Load grayscale image...')
image = cv2.imread(args["image"])

# create regions to be scanned
print('[INFO] Extracting regions...')
boxes = [draw_region(image)]
boxes = region_creator(image,boxes)

boxes.pop(0)
ROI = []
for i in range(len(boxes)):
	X1 = boxes[i][0][0]
	X2 = boxes[i][1][0]
	Y1 = boxes[i][0][1]
	Y2 = boxes[i][1][1]
	roi = image[Y1:Y2,X1:X2]
	roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	(h, w) = roi.shape[:2]
	roi = cv2.resize(roi, (w*window_maximizer, h*window_maximizer),interpolation=cv2.INTER_CUBIC)
	ROI.append(roi)

# Preprocessing the images
print("[INFO] Pre-processing the regions...")
ROI = pre_process(ROI,visu)

if args["image"]=='test.png':
    test = range(1,101)
elif args["image"]=='test2.png':
    test = range(0,10)
else:
    test = [1,2,11,12]

RecError = np.array([0 for i in range(len(test))])
PerError = []
Times = []

for psm in [6,7,9]:
# Give regions to machine learning model in order to be classified
    print('=================================================')
    print("[INFO] Applying OCR to the regions with psm ="+str(psm))
    custom_config = r'--oem 0 --psm '+str(psm)
    numbers = []
    NbError = 0
    startTime = time.time()
    for i in range(len(ROI)):
        region = ROI[i]

        try:
            number = pytesseract.image_to_string(region, config=custom_config)
        except:
            print("Error encountered due to this psm value ("+str(psm)+"), skipping this region ("+str(i+1)+"/"+str(len(ROI))+")")
            continue

        isInt = True
        try:
            number = int(number)
        except ValueError:
            isInt = False

        if not isInt:
            number = 'error'
            NbError += 1
            numbers.append(number)
        else:
            numbers.append(number)
        
        print('OCR applied to '+str(i+1)+' regions out of '+str(len(ROI)))
        if visu=='y':
            print('-----------------------------')
            cv2.imshow('region processing',region)
            print("res = ",number)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    Times.append(time.time()-startTime)
    
    # print("[INFO] End of OCR, found "+str(NbError)+" errors out of "+str(len(ROI))+" regions")    
    if len(numbers)>0:
        temp = [1 if test[i]!=numbers[i] else 0 for i in range(len(test))]
        Nbtest = sum(temp)/len(numbers)*100
        RecError += np.array(temp)
    else:
        Nbtest = 100

    PerError.append(Nbtest)
    print('Erreurs :' ,Nbtest)
    print('---------------------')

# plt.figure()
# plt.grid()
# plt.bar(test,RecError)
# plt.title('Number of errors per item for all psm')
# plt.xlabel('Item')
# plt.ylabel('Total number of errors')
# plt.savefig('output\\recError_'+args["image"]+'bis_.png')

plt.figure()

plt.subplot(211)
plt.grid()
plt.bar([6,7,9],PerError)
plt.title('Percentages of error per psm config')
plt.xlabel('PSM value')
plt.ylabel('Percentage of error')

plt.subplot(212)
plt.grid()
plt.bar([6,7,9],Times)
plt.title('Time spent for each psm config')
plt.xlabel('PSM value')
plt.ylabel('Time')

plt.savefig('output\\PerError+Times_'+args["image"]+'_.png')