## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[imagea]: ./output_images/random_images_1208_5594.png
[imageb]: ./output_images/random_images_2641_8582.png
[imagec]: ./output_images/random_images_4217_2820.png
[imaged]: ./output_images/random_images_5757_4035.png
[imagee]: ./output_images/car_histograms_RGB.png
[imagef]: ./output_images/noncar_histograms_RGB.png
[imageg]: ./output_images/car_histograms_HLS.png
[imageh]: ./output_images/noncar_histograms_HLS.png
[imagei]: ./output_images/car_image0922_hog_RGB.png
[imagej]: ./output_images/noncar_extra3849_hog_RGB.png
[imagek]: ./output_images/car_image0922_hog_HLS.png
[imagel]: ./output_images/noncar_extra3849_hog_HLS.png
[imagem]: ./output_images/car_image0922_hog_RGB_pix16.png
[imagen]: ./output_images/noncar_extra3849_hog_RGB_pix16.png
[imageo]: ./output_images/car_image0922_hog_RGB_pix16_blk1.png
[imagep]: ./output_images/test_bboxes_test1.jpg
[imageq]: ./output_images/test_bboxes_test5.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In the "Data Exploration" section of the `Vehicle Detection` jupyter notebook, I loaded and counted all the images available for training. There are 8792 car images and 8968 noncar images provided in the project. Here are a few random samples of the images:

![alt text][imagea]
![alt text][imageb]
![alt text][imagec]
![alt text][imaged]

I then examined different color spaces in the "Color Exploration" section of the notebook, specifically regard to the distinguishability of their color histograms.

The `aggregate_histograms` function takes an array of images along with a color space and builds an aggregated color histogram for all of the images. Because these histograms are not spatially-oriented, but rather sums of all pixels in an image, this aggregation scheme makes sense.

I cycled through six different color spaces (`['RGB','HSV','YCrCb','LUV', 'HLS', 'YUV']`), and compared aggregated histograms from 100 car images to 100 noncar images. The most promising color space seemed to be `RGB`, whose histogram peaks were distinct through all three channels:

![alt text][imagee]
![alt text][imagef]

While the H and L channels of the `HLS` channel also showed promise:
![alt text][imageg]
![alt text][imageh]

I decided to focus on these 2 color spaces for the HOG parameters going forward.

#### 2. Explain how you settled on your final choice of HOG parameters.

 This is carried out in the "HOG Parameters" Section of the notebook. I picked a collection of random images and plotted their HOG images (from the `get_hog_images`) method for all 3 channels of `RGB` and `HLS`. Though this was an anecdotal exploration, it seemed as if the RGB channels were still the clearest distinguisher between classes, as there seemed to be clear differences in the HOG images for all channels.

Here are some example images:
![alt text][imagei]
![alt text][imagej]
![alt text][imagek]
![alt text][imagel]

I also experimented a bit with the HOG parameters in the RGB space. When expanding the pixels per cell to 16 instead of 8, I felt like the results were still very distinctive, but at a lower computational cost because less cells needed to be calculated:

![alt text][imagem]
![alt text][imagen]

I wasn't able to get much more distinct figures from adjusting the cells per block or the orientations parameter of the HOG filter, but I was able to get very similar-looking figures using only 1 cell per block, which reduces the number of features compared to 2 cells per block:

![alt text][imagem]
![alt text][imageo]

The final parameters I chose were:
* `RGB` color space, all channels
* 8 orientations
* 16 pixels per cell
* 1 cell per block

**_NOTE: These parameter decisions ended up not working very well in practice, so I changed them for the final vehicle detection steps (see the "Sliding Window" section). I didn't really have time to dig back into manual feature interpretation, especially because my insights are apparently not that useful._**

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Using the parameter set above, I trained a linear support vector machine in the "Training the Linear Classifer" section of the notebook. I used sklearn's `LinearSVC` tool, with default settings.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is built into my `find_cars` method, which also handles efficient HOG sampling through subsampling of different sized windows based on a `scale` parameter.

I implemented a very simplistic 4-part tiered scaling of the windows, in `tiered_window_search`. Essentially I examine the image at through 4 differenet sized windows, where only the largest windows are applied all the way to the bottom of the image and the smallest windows are only used near the horizon points. This is because vehicles only appear at small scale near the horizon, while they can be rather large in the foreground. At the smallest scale, I also limit the horizontal (x) search range to eliminate noise from the scenery on the sides of the road.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here a funny thing happened. I applied all the parameters and features that I described above, and the classifiers did a horrible job! This tells me that I'm not very good at eyeballing the usefulness of various features. Rather than exhaustively go back to square 1 and manually choose a color scale, etc, I decided to just randomly try different parameters and see which one worked the best. In the end, I used the following parameters:

* `YCrCb` color space, all channels
* 9 orientations
* 16 pixels per cell
* 4 cell per block

Here are some examples of the predictions I got from the test images. Note that small boxes only exist near the horizon, and there are many false positives, but hopefully their low relative density will enable me to identify and discard them
![alt text][imagep]
![alt text][imageq]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I first applied the raw box-detection algorithm on the entire video, to get a view of what all the raw detections looked like. This can be viewed [here](./output_video_images/vehicles_rough2.mp4)

I tried to implement a very basic scheme for discarding false positives and combining overlapping and subsequent bounding boxes but it didn't quite work before I ran out of time. I preserved a "heat map" of the previous frame's pixels that were within positive bounding boxes using a global variable `recent_heat_map`. Then I built the current frame's heat map by summing once on each pixel for every positive bounding box within which it existed, and added `recent_heat_map` to it. Then I zeroed out any pixels that had a value less than a threshold of 1 using the  `apply_threshold` function provided in the lesson. Finally, I used `scipy.ndimage.measurements.label` to identify the remaining clusters of points and used `draw_labeled_bboxes` to highlight them in the image. It didn't improve things and in fact seemed to add false positives. In any case the resulting video is called [`vehicles_filtered`.](./output_video_images/vehicles_filtered.mp4)


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I was a bit surprised at the fact that all the features I hand-picked to use ended up not being very effective (e.g. the histogram of the RGB color scale). I also noticed that the training images i saw all seemed to be a bit darker than the test images. I wonder if that had anything to do with the difficulties I had training. All these failures around figuring out classification rules makes me wonder if this project would have been more easily accomplished using a neural net.

In general, I didn't have great success with this classification algorithm. One big problem was that I ran out of time to properly implement a detector for multiples and false positives. To do that correctly would probably involve a much more sophisticated way of tracking the state of predictions, over more than just one previous frame.
