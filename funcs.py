import math as m
import numpy as np
import os
import sys
import time
from copy import deepcopy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 
import ezdxf
import pickle as pkl
import cv2
from scipy.ndimage import convolve


# Shorthand for magnitude of vector
def magnitude(vect):
    return(np.sqrt(np.sum(np.power(vect,2))))
    # if len(vect.shape) == 2:
    # 	return(np.sqrt(np.sum(pow(vect,2), axis=0)))
    # else:
    # 	return(np.sqrt(np.sum(pow(vect,2))))

def saveArbitraryImage(arr, path, mode='L'):
    washOutImage = Image.fromarray(np.array(((arr-np.min(arr))/(np.max(arr)-np.min(arr)))*255, dtype=np.uint8), mode = mode)
    washOutImage.save(path)

def linearizeImage(arr):
    vals, count = np.unique(arr, return_counts=True)
    
    totalCount = np.sum(count)
    valFractions = np.zeros((256), np.double)
    valFractions[vals] = count/totalCount
    
    sumValFracs = np.zeros((256), np.uint8)
    for jj in range(256):
        sumValFracs[jj] = np.sum(valFractions[0:jj+1])*256

    # sumValFracs[1:] = (sumValFracs[:-1] + sumValFracs[1:]) / 2

    return sumValFracs[arr]

# Make empty subdivision map
def initRawSubdivMap(shape, subdivSize):
    return [[[] for jj in range(int(shape[1]/subdivSize + 1.0))] for ii in range(int(shape[0]/subdivSize + 1.0))]

# Iterate over every object in range on map
def subDivMapIterator(map, pos, rad, includeIdx=False):
    xInd = pos[0]
    yInd = pos[1]
    xSize = len(map)
    ySize = len(map[1])
    for xx in range(xInd-rad, xInd+rad+1):
        if xx < 0: continue
        if xx >= xSize: continue
        
        for yy in range(yInd-rad, yInd+rad+1):
            if yy < 0: continue
            if yy >= ySize: continue


            if includeIdx:
                idx = 0
                while idx < len(map[xx][yy]):
                    yield map[xx][yy][idx], (xx, yy, idx)
                    idx += 1
            else:
                for foo in map[xx][yy]:
                    yield foo

def iterateOverFullMap(map, destroy=False):
    for xSet in map:
        for ySet in xSet:
            for fooObj in ySet:
                yield fooObj
                if destroy:
                    ySet.remove(fooObj)
    
# Convert RGB image to points
def convertToPoints(imgArr, skipToEveryNth = 10, subdivSize = 50, subDivRad = 1, maxVal=200, minDist=10, scaleFact=4.0):
    # Load input data
    iW, iH = imgArr.shape

    pointMap = initRawSubdivMap(imgArr.shape, subdivSize)
    
    pixTestList = np.meshgrid(np.arange(0, iW, skipToEveryNth), np.arange(0, iH, skipToEveryNth))
    pixTestList = np.array([foo.flatten() for foo in pixTestList])
    valList = imgArr[*pixTestList]

    sortList = np.argsort(valList)
    valList = valList[sortList]
    pixTestList = pixTestList[:, sortList]

    for xx, yy in np.swapaxes(pixTestList, 0, 1):
        val = imgArr[xx, yy]

        if val > maxVal:
            continue

        doPlace = True
        xMapInd = int(xx/subdivSize)
        yMapInd = int(yy/subdivSize)
        for fooPt in subDivMapIterator(pointMap, (xMapInd, yMapInd), subDivRad):
            dist = magnitude([fooPt[0] - xx, fooPt[1] - yy])

            if dist < scaleFact * float(val)/maxVal + minDist:
                doPlace = False
                break

        if doPlace:
            pointMap[xMapInd][yMapInd].append([xx, yy])
            print(val, ' ', xx, ' ', yy)
    return(pointMap)

def pointsToLines(inputPointMap, subDivRad = 1, maxLineLen = 50):
    pointMap = deepcopy(inputPointMap)

    outLines = []
    for xx in range(len(pointMap[0])):
        for yy in range(len(pointMap[1])):
            idx = 0
            while idx < len(pointMap[xx][yy]):
                # Starting new line
                currIdx = (xx, yy, idx)
                currPt = pointMap[xx][yy][idx]
                outLines.append([])
                print(len(outLines))


                # Iterate until we fail to find a match
                while True:
                    minDist = maxLineLen
                    bestPt = None
                    bestIdx = None

                    # print(currIdx)
                    # print(len(pointMap[currIdx[0]][currIdx[1]]))

                    pointMap[currIdx[0]][currIdx[1]].pop(currIdx[2])
                    outLines[-1].append(currPt)

                    for cmpPt, cmpIdx in subDivMapIterator(pointMap, (currIdx[0], currIdx[1]), subDivRad, includeIdx=True):
                        dist = magnitude([currPt[0] - cmpPt[0], currPt[1] - cmpPt[1]])
                        if dist < minDist:
                            minDist = dist
                            bestPt = cmpPt
                            bestIdx = cmpIdx
                    

                    if bestPt != None:
                        print(minDist)
                        currIdx = bestIdx
                        currPt = bestPt
                    else:
                        if len(outLines[-1]) <= 1:
                            outLines.pop(-1)
                        break
    return outLines

def connectPoints(inputPointMap, subdivSize=50, subDivRad = 1, maxLineLen = 50):
    pointMap = deepcopy(inputPointMap)

    outLines = []
    for currPt in iterateOverFullMap(pointMap, destroy=True):
        xMapInd = int(currPt[0]/subdivSize)
        yMapInd = int(currPt[1]/subdivSize)
    
        minDist = maxLineLen
        bestPt = None

        for cmpPt, cmpIdx in subDivMapIterator(pointMap, (xMapInd, yMapInd), subDivRad, includeIdx=True):
            if currPt == cmpPt: 
                continue
            dist = magnitude([currPt[0] - cmpPt[0], currPt[1] - cmpPt[1]])
            if dist < minDist:
                print(dist)
                minDist = dist
                bestPt = cmpPt

        if bestPt != None:
            # if [currPt, bestPt] not in outLines:
            outLines.append([currPt, bestPt])

    return outLines



def connectPointsWithTangents(inputPointMap, sobelDir, sobelMag, sobelFactor = 10.0, subdivSize=50, subDivRad = 1, maxLineLen = 50):
    pointMap = deepcopy(inputPointMap)

    outLines = []
    for currPt in iterateOverFullMap(pointMap, destroy=True):
        currPtNp = np.array(currPt)
        sobelDirPt = sobelDir[*currPtNp]
        sobelMagPt = sobelMag[*currPtNp]

        xMapInd = int(currPt[0]/subdivSize)
        yMapInd = int(currPt[1]/subdivSize)
    
        minDist = np.inf
        bestPt = None

        print('\n---------')
        for cmpPt, cmpIdx in subDivMapIterator(pointMap, (xMapInd, yMapInd), subDivRad, includeIdx=True):
            if currPt == cmpPt: 
                continue
            diff = [cmpPt[0] - currPt[0], cmpPt[1] - currPt[1]]
            ang = np.arctan2(*diff)
            
            angDiff =  sobelDirPt - ang
            while angDiff > np.pi: angDiff -= np.pi*2
            angDiff = np.abs(angDiff)

            print('   ', sobelDirPt, ' ', diff, ' -> ', ang, '   ', angDiff)

            dist = magnitude(diff)

            score = dist + angDiff*sobelFactor
            if score < minDist:
                # print(dist)
                minDist = dist
                bestPt = cmpPt

        if bestPt != None:
            # if [currPt, bestPt] not in outLines:
            outLines.append([currPt, bestPt])
            
    return outLines

def convertToPairs(inLines):
    return [[(x[0], x[1]), (x[2], x[3])] for x in inLines]

def squareRecreation(imgArr, sobelHorz, sobelVert, offset = 2.0, divMode="MIDPOINT"):
    # Load input data
    iW, iH = imgArr.shape

    cellList = [[0, 0, iW, iH, 0.0]]
    outputLines = []

    while len(cellList) > 0:
        fooCell = cellList.pop(0)
        x0 = fooCell[0]
        y0 = fooCell[1]
        x1 = fooCell[2]
        y1 = fooCell[3]
        prevFrac = fooCell[4]
        
        # Get subset of image that is current shape
        cellSubset = imgArr[int(np.floor(x0)):int(np.ceil(x1)), int(np.floor(y0)):int(np.ceil(y1))]
        cellSums = np.sum(cellSubset)
        cellCount = cellSubset.shape[0]*cellSubset.shape[1]
        netShade = (cellCount*255) - cellSums
        avgShade = netShade / float(cellCount*255)

        avgShade = (255.0-np.median(cellSubset)) / 255.0
        
        if cellCount == 0: continue
        exitLoop = False
        for foo in cellSubset.shape:
            if foo < 3:
                exitLoop = True
        if exitLoop: continue
        
        # Check if dividing
        reqShadeThresh = 0.9    
        # minShadeThresh = 1.0
        

        borderCount = (cellSubset.shape[0]-offset*2)*(cellSubset.shape[1]-offset*2)
        currFrac = (cellCount - borderCount) / cellCount
        if borderCount < 0 or currFrac < 0.0: continue
        reqShadeThresh *= currFrac
        # minShadeThresh *= np.sqrt(currFrac)

        print(f"{len(cellList): 8d} : {avgShade: 3.2f} {currFrac: 3.2f} {netShade: 12.2f} {prevFrac: 3.2f} {fooCell[:4]}")

        subsetHorz = sobelHorz[int(np.floor(x0)):int(np.ceil(x1)), int(np.floor(y0)):int(np.ceil(y1))]
        subsetVert = sobelVert[int(np.floor(x0)):int(np.ceil(x1)), int(np.floor(y0)):int(np.ceil(y1))]
        # divThreshInterp = np.swapaxes([
        #     [1e3, 0.0, 0.0],
        #     [1e2, 0.1, 0.1],
        #     [100, 0.4, 0.8],
        #     [1, 0.4, 0.8],
        # ], 0, 1)

        shadeOffset = 0.2 * (1.0 - currFrac)

        # reqShadeThresh = np.interp(
        #     cellCount,
        # )

        drawOutline = ((avgShade - shadeOffset) > (prevFrac + currFrac))
        # drawOutline = True

        # if netShade > 255*divFactor or avgShade > reqShadeThresh:
        if cellSums > 2000:
            if divMode == "CENTER":
                xSplit = (x0+x1)/2
                ySplit = (y0+y1)/2
            elif divMode == "MIDPOINT":
                xCum = np.cumsum(np.sum(cellSubset, axis=1))
                xSplit = np.interp(float(xCum[-1])/2, xCum, np.arange(cellSubset.shape[0])) + x0
                yCum = np.cumsum(np.sum(cellSubset, axis=0))
                ySplit = np.interp(float(yCum[-1])/2, yCum, np.arange(cellSubset.shape[1])) + y0
            elif divMode == "SOBEL":
                xSplit = np.argmax(np.sum(np.abs(subsetVert), axis=1)* np.interp(np.arange(cellSubset.shape[0]), [x0, x0+x1/2, x1], [0.0, 1.0, 0.0])) + x0
                ySplit = np.argmax(np.sum(np.abs(subsetHorz), axis=0)* np.interp(np.arange(cellSubset.shape[1]), [y0, y0+y1/2, y1], [0.0, 1.0, 0.0])) + y0
            else:
                print(f"divMode {divMode} not real")
                exit()

            nextFrac = prevFrac
            if False:
                nextFrac += currFrac
                cellList.append(np.array([x0+offset, y0+offset, xSplit-offset/2.0, ySplit-offset/2.0, nextFrac]))
                cellList.append(np.array([x0+offset, ySplit+offset/2.0, xSplit-offset/2.0, y1-offset, nextFrac]))
                cellList.append(np.array([xSplit+offset/2.0, y0+offset, x1-offset, ySplit-offset/2.0, nextFrac]))
                cellList.append(np.array([xSplit+offset/2.0, ySplit+offset/2.0, x1-offset, y1-offset, nextFrac]))
            else:
                cellList.append(np.array([x0, y0, xSplit-offset/2.0, ySplit-offset/2.0, nextFrac]))
                cellList.append(np.array([x0, ySplit+offset/2.0, xSplit-offset/2.0, y1, nextFrac]))
                cellList.append(np.array([xSplit+offset/2.0, y0, x1, ySplit-offset/2.0, nextFrac]))
                cellList.append(np.array([xSplit+offset/2.0, ySplit+offset/2.0, x1, y1, nextFrac]))

        # elif drawOutline:
        if drawOutline:
            outputLines.append([x0, y0, x0, y1])
            outputLines.append([x1, y0, x1, y1])
            outputLines.append([x0, y0, x1, y0])
            outputLines.append([x0, y1, x1, y1])


    return convertToPairs(outputLines)


def exportLines(inputLines, outputLabel, inImg, MM_PER_PIX, imgScale=6):
    print(inImg.size)
    lines = Image.new("RGB", (inImg.size[0]*imgScale, inImg.size[1]*imgScale), "white")
    draw = ImageDraw.Draw(lines)
    
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    for fooLine in inputLines:
        draw.line([(imgScale*fooLine[0][1], imgScale*fooLine[0][0]), (imgScale*fooLine[1][1], imgScale*fooLine[1][0])], fill="black", width=2)
        msp.add_line((fooLine[0][1]*MM_PER_PIX, - fooLine[0][0]*MM_PER_PIX), (fooLine[1][1]*MM_PER_PIX, - fooLine[1][0]*MM_PER_PIX), dxfattribs={"color": 2})
    
    lines.save(outputLabel+'.png')
    pkl.dump(inputLines, open(outputLabel+'.pkl', 'wb'))
    # lines.show()
    doc.saveas(outputLabel+'.dxf')

# Yoinked from Pretuft Project
def reorderPathsToMinimizeTravel(pathList, joinPaths=True):
    # Remove paths with no points
    idx = 0
    while idx < len(pathList):
        if len(pathList[idx]) == 0:
            pathList.pop(idx)
        else:
            idx += 1
            
    # If there is only one path just return that
    if len(pathList) <= 1: return(pathList)

    # Start output path with only first path, which should be the outline
    outputPathList = [pathList[0]]
    startPoints = np.array([foo[0] for foo in pathList[1:]])
    endPoints = np.array([foo[-1] for foo in pathList[1:]])

    # Starting point is end of outline
    currPt = pathList[0][-1]
    remainingPaths = deepcopy(pathList[1:])
    while len(remainingPaths) > 0:
        startPointDists = np.array([np.linalg.norm(startPoints - currPt, axis=1), np.linalg.norm(endPoints - currPt, axis=1)])

        # Get closest point
        minDist = np.min(startPointDists)
        minIdx = np.argmin(startPointDists) # Returns min of flattened array
        endIsCloser = (minIdx>=len(startPoints))
        if endIsCloser:
            minIdx -= len(startPoints)

        # Remove closest points
        startPoints = np.delete(startPoints, minIdx, axis=0)
        endPoints = np.delete(endPoints, minIdx, axis=0)
        nextPath = remainingPaths.pop(minIdx)

        if endIsCloser:
            nextPath = np.flip(nextPath, axis=0)

        currPt = nextPath[-1]
        if joinPaths and minDist < 0.0001:
            print(f"   Joining paths")
            outputPathList[-1] = np.concatenate([outputPathList[-1], nextPath])
        else:
            outputPathList.append(nextPath)

    return(outputPathList)

def postProcImage(inputImg, lines):
    if type(inputImg) != np.ndarray:
        img = cv2.cvtColor(np.array(inputImg), cv2.COLOR_RGB2BGR)
    else:
        img = inputImg

    # img = img[1310:3320, 770:2900]
    img = img[770:2850, 1310:3320]

    img_gray = np.array(np.average(img, axis=2), dtype=np.uint8)

    cv2.imwrite("tmp/crop.png", img_gray)

    # kernelRad = 20
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelRad*2+1, kernelRad*2+1))
    # smooth_img = convolve(np.array(img_gray, dtype=np.int32), kernel) / np.sum(kernel)
    # cv2.imwrite("tmp/smooth_img.png", smooth_img)


    img_sel = np.zeros_like(img_gray)
    img_sel[img_gray > 250] = 255
    cv2.imwrite("tmp/img_sel.png", img_gray)

    # Dead simple CV
    xSum = np.where(np.sum(img, axis=0) > 50000)
    ySum = np.where(np.sum(img, axis=1) > 50000)
    bounds = ((xSum[0], ySum[0]), (xSum[-1], ySum[-1]))

    return img_sel, bounds

    # # Get picture from laser
    # img = f1.getPhoto(filePath+"/pic_before")
    # selImg, bounds = postProcImage(img) 
    # # 50 px per mm
    # # dead cent is 2300 x 2800 px, 110x100 mm
    # bounds = np.array(bounds, dtype=np.double)
    # bounds[:, 0] -= 2300
    # bounds[:, 1] -= 2300
    # bounds /= 50.0

