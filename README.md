disparity maps pipeline:

	1. Camera calibration
	2. Matching ( orb + bf / sift + bf). Experimented with matching via chessboard corners, but not enough points and they only in specific area => bad result
	3. Find Essential matrix/ Fundamental matrix => E gave better results
	4. Image Rectification
	5. Post filtering
	6. Deep tuning

