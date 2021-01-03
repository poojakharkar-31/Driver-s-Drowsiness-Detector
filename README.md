# Driver-s-Drowsiness-Detector
Installation of  Libraries: 

Scipy- SciPy package is used to compute the Euclidean distance between facial landmarks points in the eye aspect ratio. 

Open CV- It is an computer vision library. 

Numpy- numpy for basic processing and calcutions of arrays and data 

Dlib- Dlib is a general purpose cross-platform software library written in the programming language C++.it has various functions for machine learning,compression,computer vision,etc. 

Playsound-to play the alarm 

Threading-to run two different process simultaneously 

 

Install OpenCV3 on Windows to complete Step 1, 2 and 3. 

Step 1: Install Visual Studio 2015 

Step 2: dowload the required python and dlib libraries 

Step 3: Install CMake v3.8.2 

Step 4: configure the cmake files 

Step 5: build OpenCV libraries 

Step 6: Download Dlib 

Step 7: Build Dlib Library 



Algorithm: 

Step 1- Take image as input from a camera. 

Step 2- resizing the image to maintain the aspect ratio. 

Step 3 - face detection and eye landmark detection using dlib. 

Step 4 - converting co-ordinates to numpy array 

Step 5 - Passing co-ordinates in form of array to EAR function to eye aspect ratio 

Step 6- if EAR is less than 0.2 for more than 100 frames of consecutive image alarm goes on else the counter and alarm status is reset. 

  

 

 
