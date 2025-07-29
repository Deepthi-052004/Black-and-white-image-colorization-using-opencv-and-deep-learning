# Black-and-white-image-colorization-using-opencv-and-deep-learning
This project demonstrates how to colorize black and white (grayscale) images using a pre-trained deep learning model with OpenCV. It uses a convolutional neural network (CNN) that has been trained on large datasets to predict the colors of grayscale images.
##  Features
* Converts grayscale images to color automatically
* Utilizes a pre-trained Caffe model from OpenCV's `dnn` module
* Fast and easy to use on images of various sizes
* No need to train your own model
##  Requirements

Install the following dependencies:
pip install opencv-python numpy

##  Project Structure

├── colorize.py            # Main script for colorization
├── model
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
├── input
│   └── grayscale.jpg      # Your grayscale image(s)
├── output
│   └── colorized.jpg      # Output image(s)
└── README.md              # Project documentation
 Download Pre-trained Models


## Usage

python colorize.py --input input/grayscale.jpg --output output/colorized.jpg

##  How It Works

The model is based on the research paper:
> Richard Zhang, Phillip Isola, Alexei A. Efros. "Colorful Image Colorization", ECCV 2016.
It works by:

1. Converting the input image to the LAB color space
2. Using the pre-trained CNN to predict the `a` and `b` color channels
3. Combining the predicted channels with the original `L` channel to form the final colorized image

---

## 🖼 Example
**Input:**
![Grayscale](input/grayscale.jpg)

**Output:**
![Colorized](output/colorized.jpg)

## 📚 References
* [Colorful Image Colorization Paper (ECCV 2016)](https://richzhang.github.io/colorization/)
* [OpenCV dnn Module Documentation](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)



##  License

This project is for educational purposes only. Refer to the original authors for model licenses.




