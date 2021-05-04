import cv2
import numpy as np

def main():
    img1 = cv2.imread('images/bugdroid.png')

    scale = 1.5
    angle = 60.0 # deg

    center = tuple(np.array(img1.shape[:2][::-1]) // 2)

    T = cv2.getRotationMatrix2D(center, angle, scale)

    img2 = cv2.warpAffine(img1, T, tuple(np.array(center) * 2))

    feature_descriptor = cv2.SIFT_create()

    kp1, des1 = feature_descriptor.detectAndCompute(img1, None)
    kp2, des2 = feature_descriptor.detectAndCompute(img2, None)

    kp_img1 = cv2.drawKeypoints(img1, kp1, None)

    cv2.imwrite('output/keypoints1.png', kp_img1)

    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])

    good_match_distances = map(lambda x: x[0].distance, good_matches)

    # Sort by distances
    good_matches = np.array(sorted(zip(good_match_distances, good_matches)), dtype=object)[:15, 1]

    good_match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('output/good_matches.png', good_match_img) 



if __name__ == '__main__':
    main()