import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
import sys

def defineLane(start, end):
	'''
	Auxilliary function used to correctly determine which pixels to look at in the lane, regardless of whether the pixel 
	numbers go from high to low or low to high
	'''
	points = []
	temp = end - start
	if temp < 0:
		for i in range(end, start):
			points.append(i)
	else:
		for i in range(start, end):
			points.append(i)
	return points

def flowAnalysis(videoName, location, start, end, axes='H', threshold=13):
	'''
	Given a video file, return a list of the flow profile of the moving front. Each index will represent a frame of the 
	original video file, and its value will represent how many pixels the wetting front has flowed through

	Parameters:
	"videoName" = the file name of the video file
	"Location" = if the axis is horizontal, then location is the pixel row containing the desired lane, while if the
				 axis vertical, then location is the pixel column contained the desired lane
	"start" = the starting pixel in the row/column (dictated by "Location" parameter) of the desired lane
	"end" = the ending pixel in the row/column (dictated by the "Location" parameter) of the desired lane
	"axes" = determines whether the desired flow lane is horizontal or vertical in the video file (MUST BE EITHER "H" OR "V")
	'''
	
	# NOTE: if you're not seeing any movement in your data, change this threshold value! It represents the brightness change
	# in a pixel within the flow lane that is considered to be high enough to conclude that the moving front has moved to
	# that location. 
	threshold = threshold

	possibleAxes = ['H', 'V']
	if axes not in possibleAxes:
		print('Axes parameter not allowed, must be either H or V')
		sys.exit()
	laneLength = defineLane(start, end)

	flowProfile = []
	video = cv2.VideoCapture(videoName)
	print('Video name: {}'.format(videoName))
	print('Video length (frames): {}'.format(video.get(cv2.CAP_PROP_FRAME_COUNT)))
	success = True
	imageNumber = 1
	while success:
		# print('Image Number: {}'.format(imageNumber))
		success, image = video.read()
		if success:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			if imageNumber == 1:
				initialLane = []
				for i in laneLength:
					triplet = []
					for j in range(-1, 2):
						if axes == 'H':
							triplet.append(gray[location + j, i])
						elif axes == 'V':
							triplet.append(gray[i, location + j])
					initialLane.append(np.average(triplet))

			currentLane = []
			differences = 0
			for i in laneLength:
				triplet = []
				for j in range(-1, 2):
					if axes == 'H':
							triplet.append(gray[location + j, i])
					elif axes == 'V':
						triplet.append(gray[i, location + j])
				currentLane.append(np.average(triplet))

			for i in range(0, len(laneLength)):
				difference = initialLane[i] - currentLane[i]
				if difference >= threshold:
					differences += 1

			flowProfile.append(differences)
			imageNumber += 1
	print('Maximum flow (pixels): {}'.format(np.amax(flowProfile)))
	videoFPS = video.get(cv2.CAP_PROP_FPS)
	videoDuration = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / videoFPS
	print('Total Velocity: {:2f}'.format(float(np.amax(flowProfile) / videoDuration)))
	print('------------------')

	plt.figure()
	plt.suptitle('Flow Profile (Pixels vs. Frame Number)')
	plt.plot(flowProfile)
	plt.show()
	return flowProfile

def exampleBounds(videoName, location, start, end, axes='H'):
	'''
	Given a video file, the function will save a grayscale image depicting where the analysis algorithm would consider
	the lane to be (by using a solid black line over the first frame of the video)

	Parameters:
	"videoName" = the file name of the video file
	"Location" = if the axis is horizontal, then location is the pixel row containing the desired lane, while if the
				 axis vertical, then location is the pixel column contained the desired lane
	"start" = the starting pixel in the row/column (dictated by "Location" parameter) of the desired lane
	"end" = the ending pixel in the row/column (dictated by the "Location" parameter) of the desired lane
	"axes" = determines whether the desired flow lane is horizontal or vertical in the video file (MUST BE EITHER "H" OR "V")
	'''
	possibleAxes = ['H', 'V']
	if axes not in possibleAxes:
		print('Axes parameter not allowed, must be either H or V')
		sys.exit()
	laneLength = defineLane(start, end)
	video = cv2.VideoCapture(videoName)
	success = True
	success, image = video.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for i in laneLength:
		if axes == 'H':
			gray[location, i] = 0
		elif axes == 'V':
			gray[i, location] = 0
	cv2.imwrite('test bounds.png', gray)


if __name__ == '__main__':
	os.chdir(os.path.join(r'C:\Users\alexd\Desktop\Sangsik Flow\8.24.20 Update'))
##	testVideo = cv2.VideoCapture('E5_1.mp4')
##	fps = testVideo.get(cv2.CAP_PROP_FPS)
	

	# exampleBounds('D4 10^5_3.mp4', 600, 260, 500, axes='H')


	flowAnalysis('D4 10^1_1.mp4', 800, 260, 500, axes='H', threshold=9)
	flowAnalysis('D4 10^1_2.mp4', 720, 260, 500, axes='H', threshold=9)
	flowAnalysis('D4 10^1_3.mp4', 620, 260, 500, axes='H', threshold=9)

	flowAnalysis('D4 10^3_1.mp4', 800, 260, 500, axes='H', threshold=9)
	flowAnalysis('D4 10^3_2.mp4', 720, 260, 500, axes='H', threshold=9)
	flowAnalysis('D4 10^3_3.mp4', 620, 260, 500, axes='H', threshold=9)

	flowAnalysis('D4 10^5_1.mp4', 800, 260, 500, axes='H', threshold=9)
	flowAnalysis('D4 10^5_2.mp4', 700, 260, 500, axes='H', threshold=9)
	flowAnalysis('D4 10^5_3.mp4', 600, 260, 500, axes='H', threshold=9)
