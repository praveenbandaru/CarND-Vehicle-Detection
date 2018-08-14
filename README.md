## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/Sample_Car.png
[image2]: ./examples/Sample_Non_Car.png
[image3]: ./examples/Histogram_Car.png
[image4]: ./examples/Histogram_Non_Car.png
[image5]: ./examples/Spatial_Car.png
[image6]: ./examples/Spatial_Non_Car.png
[image7]: ./examples/HOG_Car.png
[image8]: ./examples/HOG_Non_Car.png
[image9]: ./examples/params_selection.png
[image10]: ./examples/Combined_Features.png
[image11]: ./examples/Single_Scale.png
[image12]: ./examples/Search_Windows.png
[image13]: ./examples/Multi_Scale.png
[image14]: ./examples/HeatMap.png
[image15]: ./examples/Car_Detection.png
[image16]: ./examples/Draw_Boxes.png
[image17]: ./examples/Test_Images.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the eighth code cell of the IPython notebook under section **HOG Features**.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of few of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces for histograms, spatial binnings of color and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the output of the above methods looks like.

Here is an example using the `RGB` color space and Histograms of Color parameters of `bins=32`:
![alt text][image3]
![alt text][image4]

Here is an example using the `RGB` color space and Spatial Binning of Color parameters of `size=(32, 32)`:
![alt text][image5]
![alt text][image6]

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image7]
![alt text][image8]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and calculated trial runs of the `extract_features()` function and selected the best combination with higher accuracy as below:
![alt text][image9]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` along with Histograms of Color parameters of `bins=32` and  Spatial Binning of Color parameters of `size=(32, 32)`. The code for this step is contained in the 12th and 14th cells of the IPython notebook. I've also normalized the combined feature vector using `sklearn.preprocessing.StandardScaler`. Here is the comparison of the feature vector before and after normalization:
![alt text][image10]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented sliding window search in the 17th code cell of the IPython notebook. I used an 8x8 window to search for cars on bottom half part of the image(`ystart = 400,  ystop = 720`). I also used 0.75 overlap for windows. I've implemented two scales `1 & 1.5` to perform the search on different areas of the image to increase the chances of finding cars.

Here is the image with detections using search windows from single scale (1):
![alt text][image11]

Here is the image with total search windows from both scales (1 & 1.5):
![alt text][image12]

Here is the image with detections using combined search windows from both scales (1 & 1.5):

![alt text][image13]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I tried to optimize the performance of the classifier by extracting features using hog sub-sampling. I've also improved my pipeline to skip processing on some frames but use the boxes from last processed frame to draw on them to speed up processing time. Here are some example images:

![alt text][image17]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)

 [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/IO9c_PQrEbk/0.jpg)](https://www.youtube.com/watch?v=IO9c_PQrEbk)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this is contained in the 23rd, 25th and 26th code cells in the IPython notebook.
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the final image:

### Here is the output of sliding window search and its corresponding heatmap detection:

![alt text][image14]

### Here is the output of `scipy.ndimage.measurements.label()`:
![alt text][image15]

### Here the resulting bounding boxes are drawn onto the final image:
![alt text][image16]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced in my implementation of the project is to select the best possible parameter combination that can yield high accuracy while reducing the processing time. After some trial and error approach, I decided on the parameter selection which yields higher accuracy and compensated the processing time by improving the pipeline to skip some frames and reuse the previously detected boxes for them. 

The pipeline was not able to perform well in recognizing the white cars on a lighter background but reusing the previously detected boxes from last 'n' frames solved the problem to an extent. The pipeline would obviously fail to detect the cars correctly under different lighting conditions and also where the shapes of the car look different from the training images when captured at certain angles.

To make the pipeline more robust, we can augment the training set with a much better labelled dataset and try to reduce the size of the feature vector. Also we can try to parallelize the feature extraction and evaluation steps to increase the processing speed. Also just a thought, instead of processing each frame, we can extract the car image using HOG from the first frame and try to do template matching on the next few frames.

