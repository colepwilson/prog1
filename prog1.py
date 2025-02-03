# Image stitching program

import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt



# Function that loads the directory containing the image slices
def load_images(input_dir):
    image_files = sorted(os.listdir(input_dir))  # Sort alphabetically or numerically
    return [cv2.imread(os.path.join(input_dir, file)) for file in image_files]



# Function that takes directory of image slices and transforms them into one panoramic image
def main(input_dir, output_file):
    images = load_images(input_dir)
    panorama = stitch_images(images)
    save_image(panorama, output_file)



# Function determining how to match image slices together
def detect_and_match_features(image1, image2):
    sift = cv2.SIFT_create()  # Initialize SIFT detector
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Matches images based on a ratio
    good_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]
    return keypoints1, keypoints2, good_matches



# Function that computes the homography matrix
def compute_homography(keypoints1, keypoints2, matches):
    if len(matches) < 4:
        print(f"Not enough good matches ({len(matches)}) to compute homography.")
        return None

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    return H


# Find best canvas size to update to during stitching procedure
def get_output_canvas_size(image1, image2, H):
    height, width = image1.shape[:2]

    corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
    
    transformed_corners = cv2.perspectiveTransform(corners, H)

    all_corners = np.concatenate((corners, transformed_corners), axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    return canvas_width, canvas_height, -x_min, -y_min



# Function that stitches two images into a panorama
def stitch_two_images(image1, image2, H):
    height, width = image1.shape[:2]

    panorama = cv2.warpPerspective(image2, H, (int(width * 1.1), height))

    mask = (image1 > 0).astype(np.uint8) * 255
    panorama = cv2.seamlessClone(image1, panorama, mask, (width // 2, height // 2), cv2.NORMAL_CLONE)

    return panorama


# Uncommenting the function call in stitch_images allows the user to visualize the matches
def visualize_matches(image1, image2, keypoints1, keypoints2, matches):
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Good Matches", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Function that stitches all images into one panorama
def stitch_images(images):
    result = images[0]
    for i in range(1, len(images)):
        print(f"Stitching image {i} of {len(images)}")

        keypoints1, keypoints2, matches = detect_and_match_features(result, images[i])
        print(f"Found {len(matches)} matches between image {i} and image {i + 1}.")

        # Uncomment the function call in the line below to visualize matches
        # visualize_matches(result, images[i], keypoints1, keypoints2, matches)

        H = compute_homography(keypoints1, keypoints2, matches)

        if H is None:
            print(f"Skipping image {i}: not enough matches.")
            continue

        result = stitch_two_images(result, images[i], H)

    return result



# Function that saves final panorama to an output file
def save_image(image, output_file):
    cv2.imwrite(output_file, image)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prog1.py input_dir output_file")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(input_dir, output_file)
