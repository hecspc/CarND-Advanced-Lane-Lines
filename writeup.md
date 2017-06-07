## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration]: ./output_images/calibration.png "calibration"
[chessboard]: ./output_images/chessboard.png "chessboard"
[color_spaces]: ./output_images/color_spaces.png "color_spaces"
[example]: ./test_images/straight_lines1.jpg "example"
[hls_l_channel]: ./output_images/hls_l_channel.png "hls_l_channel"
[hls_s_channel]: ./output_images/hls_s_channel.png "hls_s_channel"
[lab_b_channel]: ./output_images/lab_b_channel.png "lab_b_channel"
[lanes_radius]: ./output_images/lanes_radius.png "lanes_radius"
[previous_polyfit]: ./output_images/previous_polyfit.png "previous_polyfit"
[sliding_window_search]: ./output_images/sliding_window_search.png "sliding_window_search"
[sobel_abs]: ./output_images/sobel_abs.png "sobel_abs"
[sobel_dir]: ./output_images/sobel_dir.png "sobel_dir"
[sobel_mag_dir]: ./output_images/sobel_mag_dir.png "sobel_mag_dir"
[sobel_mag]: ./output_images/sobel_mag.png "sobel_mag"
[test_pipeline]: ./output_images/test_pipeline.png "test_pipeline"
[undistorted]: ./output_images/undistorted.png "undistorted"
[unwarped]: ./output_images/unwarped.png "unwarped"
[output]: ./output.mp4 "Video"
[challenge]: ./challenge_output.mp4 "Video Challenge"
[harder_challenge]: ./harder_challenge_output.mp4 "Video Harder Challenge"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in `project.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Chessboard calibration][calibration]

The holes in the grid indicates that the funciton `findChessboardCorners` could not detect the corners for that image.

Once we have found the corners we can apply the calibration and distortion coefficients to an a image to test them

![Chessboard][chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][example]

I have chosen one with the straight lines so it is more evident the pipeline works.

We applied the undistorstion from the camera calibration to the image

![undistort][undistorted]

The effect is very subtle, but you can see some difference on the shape of the hood of the car.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried several methods. First one is using sobel gradient thresholds and color channel thresholds.

![undistort][sobel_mag_dir]

And on the different color spaces

![undistort][color_spaces]

In the end, I chose to use the L channel of the HLS color space to get the white lines and the B channel of the LAB space to get the yellow ones. Using only color space with the correct thresholds was enough to isolates both lines from the lane. I did not use any gradient threshold. In the notebook, I made an interactive cell for every threshold so I could easily find the best values for every color channel or gradient.
 
![undistort][hls_l_channel]
![lab b][lab_b_channel]

Applying the pipeline to the test images

![test_pipeline][test_pipeline]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is titled "Perspective Transform" in the project jupyter notebook in the 8th cell.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][unwarped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the Jupyter notebook, in the section titled "Sliding Window Search", I define the funcion `sliding_window_search`, which indentifies the lane lines and fit their positions to a second order polynomial. It uses the methodology explained in the class. It calculates a histogram in the lower half of the image. The most prominent peaks will be the x-position of the base lines for the lane lines. These peaks are the maxima between a quarter of the width of the image to left and right of the middle point, so we avoid other noise of lines from other lanes. It generates 10 windows to identify the lane pixels. Each windows is centered on the midpoint from the window below. So this follows the lines up to the top of the image. Once we have the line area thanks to the windows, it identifies the pixels of the binary image inside the windows and we fit them to a second order polynomial. The image below show the whole process.

![alt text][sliding_window_search]


Because in a video the fit of the current frame depends on the fit of the previous frame, I also have a method to calculate the fit polynomial based on a previous fit. Giving the fit polynomial, I calculate the fluctuation area posible to search for the next time so we don't need to use the sliding windows. With this method, we assume there is no abrupt change in the curvature of the road. In the next figure, the yellow area shows search area to find the pixels be fit.

![previous_polyfit][previous_polyfit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature, I followed the steps from [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

Given the polynomial of second order, the radius will be

```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

where `left_fit_cr` and `right_fit_cr` are the coefficients for the polynomial fit for the left and the right lines.

The position of the vehicle respect to the center of the lane is calculated with this code

```python
car_position = bin_img.shape[1]/2
l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * xm_per_pix
```
Assuming the camera is in the middle of the car, the car position is the midpoint of the image. We calculate the interception of both lines with the bottom edge (`y=719`) these will be `l_fit_x_int` and `r_fit_x_int`. The midpoint of this points will be the center of the lane. We calculate the difference between this value and the car position.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented in the code cell titled  "Draw Curvature Radius and Distance from Center Data onto the Original Image" in the Jupyter notebook. A polygon is generated based on plots of the left and right fits, warped back to the perspective of the original image using the inverse perspective matrix `Minv` and overlaid onto the original image. The image below is an example of the results of the draw_lane function:

![lanes_radius][lanes_radius]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I had to face were due to lighting conditions and shadows and road status. I found that using sobel gradient some times I did not have enough data to have a good fit. So I tried to use color spaces. Isolating the yellow line under sunny conditions was extremely difficult, that's why I decided to go with the [LAB Space](https://en.wikipedia.org/wiki/Lab_color_space) which helped me to isolate easily the yellow lines, even in situations where the contrast were very low.
The problem using only color spaces will come with low contrast or same color scenarios like snow, white or yellow cars in front, road paints.
Using a polyfit from a previous frame helped to smooth and avoid errors when data is not enough but also it fails when the road curvature is bigger. In the video for the [harder challenge](./harder_challenge_output.mp4) we can see that it fails all the time.

The accuracy of the algorithm is strongly based on the threshold values for the different color channels. We could dinamically assign these values given diferent conditions of the whole image or slicing the image on different parts. Now both lines are calculated indepently, maybe we can avoid more erors, assigning a confidence level for both fits and rejecting those who are out of this confidence level.

