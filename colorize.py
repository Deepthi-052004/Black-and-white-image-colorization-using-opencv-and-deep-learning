import cv2 as cv
import numpy as np
import requests
import random

# Load model and color centers
pts = np.load("pts_in_hull.npy")
net = cv.dnn.readNetFromCaffe("colorization_deploy_v2.prototxt", "colorization_release_v2.caffemodel")

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype(np.float32)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# ✅ Updated: Valid, direct image URLs (safe and tested)
image_urls = [
    "https://www.zilliondesigns.com/blog/wp-content/uploads/greyscale.jpg"
]

# Function to download and decode image
def fetch_image(url):
    try:
        print(f"Downloading image from: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print("Failed to fetch image (HTTP", response.status_code, ")")
            return None
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error fetching image:", e)
        return None

# Choose a random image URL
url = random.choice(image_urls)
img = fetch_image(url)

if img is None:
    print("Image could not be loaded. Exiting.")
    exit()

# Convert to grayscale then back to BGR
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# Preprocess
img_rgb = (gray_bgr[:, :, [2, 1, 0]] / 255.0).astype(np.float32)
img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
l_channel = img_lab[:, :, 0]
l_rs = cv.resize(l_channel, (224, 224)) - 50

# Predict ab channels
net.setInput(cv.dnn.blobFromImage(l_rs))
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_up = cv.resize(ab_dec, (gray_bgr.shape[1], gray_bgr.shape[0]))

# Combine with original L
lab_out = np.concatenate((l_channel[:, :, np.newaxis], ab_up), axis=2)
bgr_out = cv.cvtColor(lab_out.astype(np.float32), cv.COLOR_Lab2BGR)
bgr_out = np.clip(bgr_out, 0, 1)
bgr_out = (bgr_out * 255).astype(np.uint8)

# Display side-by-side
combined = np.hstack((gray_bgr, bgr_out))

cv.namedWindow("Random Online Image Colorization", cv.WINDOW_NORMAL)
cv.setWindowProperty("Random Online Image Colorization", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.imshow("Random Online Image Colorization", combined)
cv.waitKey(0)
cv.destroyAllWindows()
