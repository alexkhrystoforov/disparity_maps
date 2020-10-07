import cv2


def find_best_matches(matches):
    """
    Filter matches by distance
    Args:
         matches: list
    Returns:
        best_matches: list
    """
    best_matches = []
    for m in matches:
        if m.distance < 50:
            best_matches.append(m)

    return best_matches


class Matcher:

    def __init__(self, sift, img0, img1):
        self.kp1, self.des1 = sift.detectAndCompute(img0, None)
        self.kp2, self.des2 = sift.detectAndCompute(img1, None)
        self.norm_hamming = cv2.NORM_HAMMING

    def BF_matcher(self):
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(self.des2, self.des1)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = find_best_matches(matches)

        return best_matches