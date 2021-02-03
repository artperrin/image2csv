# Convert an image of numbers to a .csv file

This Python program aims to convert images of array numbers to corresponding .csv files. It uses OpenCV for Python to process the given image and Tesseract for number recognition.

![Output Example](https://raw.githubusercontent.com/artperrin/Image2csv/master/readme_figures/example.PNG)

The repository includes:
* the source code of image2csv.py ;
* the tools.py file where useful functions are implemented ;
* a folder with some files used for test.

The code is not well documented nor fully efficient as I'm a beginner in programming, and this project is a way for me to improve with skills, in particular in Python programming.

# How to use the program 

In a python terminal, use the command line:
```sh
$ python image2csv.py --image path\to\image
```

Their are a few optionnal arguments: 
* `--path path\to\output\csv\file`
* `--visualization [y]/n`
* `--method [fast]/denoize`

and one can find their usage using the command line:
```sh
$ python image2csv.py --help
```

When then program is running, the user has to interact with it via its mouse or the terminal :
1. the image is opened in a window for the user to draw a rectangle around the first (top left) number.
As this rectangle is used as a base to create a grid afterward, keep in mind that all the numbers should fit into the box.
2. A new window is opened showing the image with the drawn rectangle. Press any key to close and continue.
3. Based on the drawn rectangle, a grid is create to extract each number one by one. This grid is controlled by the user via two "offset" values. The user has to enter those values in the terminal, then the image is opened in a window with the created grid. Press any key to close and continue.
If the grid does not fit the numbers, the user can change the offset values and repeat the step. When the grid matches the user's expectations, he can set the offset values to 0 to continue.
4. The numbers are extracted from the image and the results are shown in the terminal.
(be carefoul though, the indicated number of errors represent the number of errors encountered by Tesseract, but Tesseract can predict a wrong number which will not be counted !)
5. The .csv file is created with the numbers predicted by Tesseract. If Tesseract found an error, it will show up on the .csv file as an infinite value.

# Hypothesis and limits

For the program to run correctly, the input image must verify some hypothesis (just a few _simple_ ones):
* the line and row width must be constant, as the build grid is just a repetition of the initial rectangle with offsets ;
* if there is a grid already on the input image, the user has to be careful not see it into each grid region, or Tesseract will act badly (there is a need of a better pre-proccessing not yet implemented) ;
* it is recommended to have a good input image size, to control the offsets more easily.

At last, this program is not perfect (I know you thought so, with its smooth workflow and simple hypothesis, sorry to disappoint...) and does not work with decimal numbers... But does a great job on negatives ! Also the user must be careful with the slashed zero which seems to be identified by Tesseract as a six.