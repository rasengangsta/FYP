#!/usr/bin/env python


from copy import deepcopy

import math
import numpy as np
import cv2
import cv
from scipy.ndimage import measurements
import time
import sys

help_message = '''
USAGE: OpticalFlow1.py [<video_source>]
Initialising...
'''

def nothing(x):
    pass

def drawProcessedImage(img, lines, outlierMatrix, step):
    outlierMatrix = list(outlierMatrix) 
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    
    for i in range(len(lines)):
        point1 = (lines[i][0][0], lines[i][0][1])
        point2 = (lines[i][1][0], lines[i][1][1])
        cv2.line(vis, point1, point2, (0, 255, 0), 1) 
        if i in outlyingFlowPoints:
            point1 = (lines[i][0][0], lines[i][0][1])
            point2 = (point1[0] + 5, point1[1] + 5)
    cv2.rectangle(vis, (len(outlierMatrix)*step, len(outlierMatrix[1])*step), (10, 10), (255, 0, 0), thickness=2, lineType=8, shift=0)

    for i in range(len(outlierMatrix)):
        for j in range(len(outlierMatrix[0])):
            if outlierMatrix[i][j] == 1:
                cv2.rectangle(vis, (j*step, i*step), (j*step+10, i*step+10), (255, 0, 0), thickness=2, lineType=8, shift=0)
            if outlierMatrix[i][j] == 2:
                cv2.rectangle(vis, (j*step, i*step), (j*step+10, i*step+10), (0, 255, 0), thickness=2, lineType=8, shift=0)
            if outlierMatrix[i][j] == 3:
                cv2.rectangle(vis, (j*step, i*step), (j*step+10, i*step+10), (0, 0, 255), thickness=2, lineType=8, shift=0)
            if outlierMatrix[i][j] == 4:
                cv2.rectangle(vis, (j*step, i*step), (j*step+10, i*step+10), (0, 255, 255), thickness=2, lineType=8, shift=0)
            
    return vis

def differentiateFlow(lines, smoothingConstant):
    pointsX = []
    pointsY = []
    diffPointsX = []
    diffPointsY = []
    counter = 0
    for (x1, y1), (x2, y2) in lines:
        pointsX.append((x1,x2-x1))
        pointsY.append((y1,y2-y1))
        counter = counter + 1
    pointsSortedX = sorted(pointsX, key=lambda t: t[1])
    pointsSortedY = sorted(pointsY, key=lambda t: t[1])
    numPoints = len(pointsX)
    for i in range(0, numPoints-smoothingConstant):      
        originalPositionX = pointsSortedX[i][0]
        originalPositionY = pointsSortedY[i][0]
        diffPointsX.append((originalPositionX, pointsSortedX[i+smoothingConstant][1] - pointsSortedX[i][1]))
        diffPointsY.append((originalPositionY, pointsSortedY[i+smoothingConstant][1] - pointsSortedY[i][1]))
    return diffPointsX, diffPointsY

def findOutliers(diffX, diffY, threshold):
    tempX = []
    tempY = []
    for i in range(len(diffX)):
        if diffX[i][1] != 0:
            tempX.append(diffX[i][1])
    for i in range(len(diffY)):
        if diffY[i][1] != 0:
            tempY.append(diffY[i][1])
    if len(tempX) == 0:
        tempX.append(0)
    if len(tempY) == 0:
        tempY.append(0)
    minX = min(tempX)
    minY = min(tempY)
    numPoints = len(diffX)
    outlyingPoints = []
    for i in range(numPoints):
        if (minX >= 0 and int(diffX[i][1]) > minX * threshold):   
            outlyingPoints.append((diffX[i][0], diffY[i][0]))
        if (minY >= 0 and int(diffY[i][1]) > minY * threshold):
            outlyingPoints.append((diffX[i][0], diffY[i][0]))
    return outlyingPoints

def quadrantLines(lines):
    topLeft = []
    topRight = []
    botLeft = []
    botRight = []
    for line in lines:     
        if (line[0][0] < 358 and line[0][1] < 235):
            topLeft.append(line)
        elif (line[0][0] > 358 and line[0][1] < 235):
            topRight.append(line)
        elif (line[0][0] < 358 and line[0][1] > 235):
            botLeft.append(line)
        elif (line[0][0] > 358 and line[0][1] > 235):    
            botRight.append(line)
    return (topLeft, topRight, botLeft, botRight)

def gaussianOfXY(x, y, xCenter, yCenter, sigma):
    f = ((x-xCenter)**2) / (2*sigma)
    g = ((y-yCenter)**2) / (2*sigma)
    exponent = 2*math.exp(-(f+g))
    return exponent

def adjustVectorsForCentrality(lines, sigma):
    for line in lines:
        gause = gaussianOfXY(line[0][0], line[0][1], 100, 400, sigma)
        line[1][0] = line[0][0] + ((line[1][0] - line[0][0]) * gause)
        line[1][1] = line[0][1] + ((line[1][1] - line[0][1]) * gause)       

def matrifyOutliers(lines, outliers, xSize, ySize):
    outlierMatrix = [[0 for y in range(ySize)] for x in range(xSize)]      
    x = 0
    y = 0
    outliersNp = np.asarray([list(elem) for elem in outliers])
    for line in lines:
        for outlier in outliersNp:
            if (line[0][0] == outlier[0] and line[0][1] == outlier[1]):   
                outlierMatrix[line[0][1]/ySize][line[0][0]/xSize] = 1    
    return outlierMatrix

if __name__ == '__main__':   
    print help_message
    try: fn = sys.argv[1]
    except: fn = 0
    cam = cv2.VideoCapture( 'road2.avi' )
    fourcc = cv2.cv.CV_FOURCC(*'IYUV')
    frameDimensions = (int(cam.get(cv.CV_CAP_PROP_FRAME_WIDTH/2)), int(cam.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('boxe45.avi',fourcc, 10.0, frameDimensions)
    print("Output file opened: " + str(out.isOpened()))
    ret, prev = cam.read()
    
    height = len(prev)
    width = len(prev[0])
    prev = prev[0:height, width/2:width]
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image')
    cv2.createTrackbar('Threshold','image',5,20,nothing)
    cv2.createTrackbar('Coefficient','image',18,20,nothing)
    cv2.createTrackbar('Vectors','image',18,40,nothing)
    cv2.createTrackbar('Centrality','image',8000,100000,nothing)

    while True:       
        threshold = cv2.getTrackbarPos('Threshold','image')
        coefficient = cv2.getTrackbarPos('Coefficient','image')
        step = cv2.getTrackbarPos('Vectors','image')
        centralityConstant = float(cv2.getTrackbarPos('Centrality','image'))
        ret, img = cam.read()
        if (img is None):
            print("Finnished processing...")
            out.release()
            break
        img = img[0:height, width/2:width]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevGray, gray, 0.5, 3, 15, 3, 2, 1.2, 0)
        prevGray = gray
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.1)
        adjustVectorsForCentrality(lines, centralityConstant)    
        diffX, diffY = differentiateFlow(lines, coefficient)
        outlyingFlowPoints = (findOutliers(diffX, diffY, threshold))
        outlierMatrix = matrifyOutliers(lines, outlyingFlowPoints, w/step, h/step)   
        lw, num = measurements.label(outlierMatrix)
        flowImage = drawProcessedImage(gray, lines, outlierMatrix, step)
        cv2.imshow('flow', flowImage)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            out.release()
            break
        
    cv2.destroyAllWindows()
