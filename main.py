import math as m
import numpy as np
import scipy as sp
from scipy.ndimage import convolve
import os
import sys
import time
from copy import deepcopy
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt 

from funcs import *
from defs import *

if __name__ == '__main__':
    folderName = sys.argv[1]
	
    filePath = 'proc/' + folderName
    
    # Check if input exists
    if not os.path.exists(filePath + '/input.jpg'):
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        # Take picture

    # Load image
    inImg = Image.open(filePath + '/input.jpg').convert('RGB')

    # Resize image
    origImSize = inImg.size
    if MAX_DIMS[0] / origImSize[0] > MAX_DIMS[1] / origImSize[1]:
        MAX_DIMS[1] = int(origImSize[1] * MAX_DIMS[0] / origImSize[0])
        # inImg.resize((MAX_DIMS[0], ), Image.Resampling.HAMMING)
    else:
        MAX_DIMS[0] = int(origImSize[0] * MAX_DIMS[1] / origImSize[1])
    inImg = inImg.resize(MAX_DIMS, Image.Resampling.HAMMING)

    # Load array of raw pixels
    rawImg = np.array(inImg.convert('L'))

    # Plot hist of input image values if requested
    plt.hist(rawImg.flatten(), bins=256)
    # plt.axvline(MAX_VAL, color='orange')
    plt.title("Distribution of shade in raw image")
    plt.savefig(filePath + '/plt_pixDistribution.jpg')

    # Linearize image
    if DO_IMAGE_LINEARIZATION:
        rawImg = linearizeImage(rawImg)

    # Identify white background
    pixelWhiteness = np.array(rawImg, dtype=np.double)
    colPix = np.array(inImg)
    stdev = np.std(colPix, axis=2)
    
    # Plot image STDev distribution
    plt.cla()
    plt.hist(np.array(stdev).flatten(), bins=50)
    plt.title("Standard deviation of RGB of each pixel")
    plt.savefig(filePath + '/plt_pixStd.jpg')

    # Set all stdevs < 5 to 5
    STD_TRUNC_VAL = 10
    stdev[stdev < STD_TRUNC_VAL] = STD_TRUNC_VAL
    stdev /= STD_TRUNC_VAL

    saveArbitraryImage(stdev, filePath+'/img_stdev.jpg')

    # Actually adjust for whiteness
    pixelWhiteness /= stdev
    backgroundVal = np.min(pixelWhiteness[:3, :])
    rawImg[pixelWhiteness > backgroundVal] = 255

    # Plot pixel whiteness
    pixelWhiteness[pixelWhiteness > backgroundVal] *= 3
    saveArbitraryImage(pixelWhiteness, filePath+'/img_whiteness.jpg')

    adjustImg = np.array(rawImg, dtype=np.double)
    adjustImg /= np.max(adjustImg)

    if DARKEN_NOT_WHITE:
        print(np.max((stdev + STD_TRUNC_VAL)/150))
        adjustImg += (stdev + STD_TRUNC_VAL)/300
        
    if DARKEN_EDGES:
        print("Handling edges")
        smoothKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(SMOOTH_KERNEL_RAD*2+1,SMOOTH_KERNEL_RAD*2+1))
        diffKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(EDGE_KERNEL_RAD*2+1,EDGE_KERNEL_RAD*2+1))
        modColPix = np.array(colPix, dtype=np.int32)
        colDiffs = np.zeros_like(modColPix, dtype=np.double)
        
        print(colPix[:, :, 0].shape)
        for ii in range(3):
            modColPix[:, :, ii] = convolve(modColPix[:, :, ii], smoothKernel) / np.sum(smoothKernel)
            colDiffs[:, :, ii] = convolve(modColPix[:, :, ii], diffKernel)#, mode='same')
        colDiffs = modColPix - colDiffs/np.sum(diffKernel)
        # colDiffs -= np.min(colDiffs)
        # print(colDiffs)

        # Plot image edge selection distribution
        plt.cla()
        plt.hist(np.sum(colDiffs, axis=2).flatten(), bins=50)
        plt.title("Absolute sum of RGB Difference in each axis")
        plt.savefig(filePath + '/plt_edgeDist.jpg')

        colDiffs[colDiffs > 0] = 0

        colDiffs = np.max(np.abs(colDiffs), axis=2)
        colDiffs[colDiffs > 100] = 100
        saveArbitraryImage(colDiffs, filePath+'/img_edges.jpg', mode='L')
        
        # adjustImg -= colDiffs/500
        # adjustImg = np.min([adjustImg, 1.0-colDiffs/150], axis=0)
        adjustImg -= colDiffs/200
        adjustImg[adjustImg < -0.0] = -0.0

    # Save greyscale image
    # saveArbitraryImage(adjustImg, filePath+'/img_tmp.jpg', mode='L')
    whitePts = np.where(rawImg == 255)
    adjustImg -= np.min(adjustImg)
    adjustImg *= 255/np.max(adjustImg)
    rawImg = np.array(adjustImg, dtype=np.uint8)
    rawImg[whitePts] = 255
    Image.fromarray(rawImg, mode='L').save(filePath + '/img_grey.jpg')

    # Plot image edge selection distribution
    plt.cla()
    shadePts = rawImg.flatten()
    plt.hist(shadePts[shadePts < 255], bins=256)
    plt.title("Distribution of shade in final greyscale image")
    plt.savefig(filePath + '/plt_shadeDist.jpg')

    # Convert to pts
    pointMap = convertToPoints(rawImg, skipToEveryNth=SKIP_TO_NTH, subdivSize=SUBDIV_SIZE, subDivRad=SUBDIV_RAD, maxVal=MAX_VAL, minDist=MIN_DIST, scaleFact=SCALE_FACT)

    # Connect lines

    # Display line output
    points = Image.new("RGB", inImg.size, "white")
    draw = ImageDraw.Draw(points)
    for pt in iterateOverFullMap(pointMap):
        # draw.point(pt, fill="black")
        drawSize = 3
        draw.ellipse([(pt[1]-drawSize, pt[0]-drawSize), (pt[1]+drawSize, pt[0]+drawSize)], fill="black", )
    points.save(filePath + '/out_points.jpg')
    # points.show()

    # Generate lines
    lineData = pointsToLines(pointMap, subdivSize=SUBDIV_SIZE, subDivRad=SUBDIV_RAD*2, maxLineLen=MAX_LINE_LEN)
    print(len(lineData))

    # Plot lines
    lines = Image.new("RGB", inImg.size, "white")
    draw = ImageDraw.Draw(lines)
    for fooLine in lineData:
        for idx in range(len(fooLine)-1):
            draw.line([(fooLine[idx][1], fooLine[idx][0]), (fooLine[idx+1][1], fooLine[idx+1][0])], fill="black", width=3)
        # # draw.point(pt, fill="black")
        # drawSize = 3
        # draw.ellipse([(pt[1]-drawSize, pt[0]-drawSize), (pt[1]+drawSize, pt[0]+drawSize)], fill="black")
    lines.save(filePath + '/out_lines.jpg')

    