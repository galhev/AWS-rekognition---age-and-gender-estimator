import cv2
import tkinter as tk
import boto3
import time
import json
import numpy as np
from botocore.exceptions import ClientError
from sort import *

def crop_Image(face, width, height, customers_pic):
    padding = 70
    padding2 = 120
    x = int(face['BoundingBox']['Left'] * width)
    y = int(face['BoundingBox']['Top'] * height)
    w = int((face['BoundingBox']['Left'] + face['BoundingBox']['Width']) * width)
    h = int((face['BoundingBox']['Top'] + face['BoundingBox']['Height']) * height)
    point = (x, y)
    x_face = max(0, x - padding)
    y_face = max(0, y - padding)
    w_face = min(width, w + padding)
    h_face = min(height, h + padding2)
    print("x_face "+str(x_face))
    print("y_face "+str(y_face))
    print("w_face "+str(w_face))
    print("h_face "+str(h_face))
    crop_img = customers_pic[y_face:h_face, x_face:w_face]
    cv2.imwrite("frame.png", crop_img)
    faceImg = r"C:\Users\user\PycharmProjects\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FaceDetection\frame.png"
    ret, facePhoto = cv2.imencode('.jpg', crop_img)
    return point, faceImg, facePhoto, x_face, y_face, w_face, h_face


def index_face(rekognition, facePhoto, collection_name, keyInd, keys, facesDic, x, y, w, h):
    resInd = rekognition.index_faces(Image={'Bytes': facePhoto.tobytes()}, CollectionId=collection_name)
    if (len(resInd['FaceRecords'])==0):
        key = 0
    else:
        key = resInd['FaceRecords'][0]['Face']['FaceId']
        keyInd += 1
        keys[key] = keyInd
        if not facesDic.get(key, False):
            # initializes sub-dictionary
            facesDic[key] = {}
        if not facesDic[key].get("toAlert", False):
            facesDic[key]["toAlert"] = {}
        facesDic[key]["toAlert"] = True
        if not facesDic[key].get("Location", False):
            facesDic[key]["Location"] = {}
        facesDic[key]["Location"]["X"] = x
        facesDic[key]["Location"]["Y"] = y
        facesDic[key]["Location"]["W"] = w
        facesDic[key]["Location"]["H"] = h
    return keyInd, keys, key, facesDic


def checkSimilarFace(facesDic, frameNum, similarKeys, x, y, w, h):
    print("similar keys: " + str(similarKeys))
    minVal = 10000000
    suspectedKey = 0
    # Check existing keys:
    for ind, similarKey in enumerate(similarKeys, 1):
        x_diff = 1 if abs(x - facesDic[similarKey]["Location"]["X"]) < 35 else 0
        y_diff = 1 if abs(y - facesDic[similarKey]["Location"]["Y"]) < 35 else 0
        w_diff = 1 if abs(w - facesDic[similarKey]["Location"]["W"]) < 35 else 0
        h_diff = 1 if abs(h - facesDic[similarKey]["Location"]["H"]) < 35 else 0
        abs(x - facesDic[similarKey]["Location"]["X"])
        diffTotal = x_diff + y_diff + w_diff + h_diff
        x_diff_val = abs(x - facesDic[similarKey]["Location"]["X"])
        y_diff_val = abs(y - facesDic[similarKey]["Location"]["Y"])
        w_diff_val = abs(w - facesDic[similarKey]["Location"]["W"])
        h_diff_val = abs(h - facesDic[similarKey]["Location"]["H"])
        print("key {0} location is ".format(similarKey))
        print(facesDic[similarKey]["Location"]["X"])
        print(facesDic[similarKey]["Location"]["Y"])
        print(facesDic[similarKey]["Location"]["W"])
        print(facesDic[similarKey]["Location"]["H"])
        val = 0
        if (diffTotal >= 3):
            val = x_diff_val + y_diff_val + w_diff_val + h_diff_val
            print("total_diff between key and {0} is: {1}".format(similarKey,str(val)))
            frameDiff = frameNum - max(facesDic[similarKey]["FrameNum"])
            if ((frameDiff < 20) and (frameDiff > 0)):
                if minVal > val:
                    minVal = val
                    suspectedKey = similarKey
                    print("suspected Key:" + str(suspectedKey))
    return suspectedKey


def save_customer_details(face, facesDic, key):
    lowage = face['AgeRange']['Low']
    highage = face['AgeRange']['High']
    gender = face['Gender']['Value']
    if not facesDic[key].get("FrameNum", False):
        facesDic[key]["FrameNum"] = []
    facesDic[key]["FrameNum"].append(frameNum)
    if not facesDic[key].get("Gender", False):
        facesDic[key]["Gender"] = []
    facesDic[key]["Gender"].append(gender)
    if not facesDic[key].get("HighAge", False):
        facesDic[key]["HighAge"] = []
    facesDic[key]["HighAge"].append(highage)
    if not facesDic[key].get("LowAge", False):
        facesDic[key]["LowAge"] = []
    facesDic[key]["LowAge"].append(lowage)
    highfreqGender = most_common(facesDic[key]["Gender"])
    highfreqLowAge = most_common(facesDic[key]["LowAge"])
    highfreqHighAge = most_common(facesDic[key]["HighAge"])
    ageRange = str(highfreqLowAge) + "-" + str(highfreqHighAge)
    return facesDic, highfreqGender, highfreqLowAge, highfreqHighAge, ageRange


def updateFaceLocation(facesDic, key, x, y, w, h):
    facesDic[key]["Location"]["X"] = x
    facesDic[key]["Location"]["Y"] = y
    facesDic[key]["Location"]["W"] = w
    facesDic[key]["Location"]["H"] = h
    return facesDic


def most_common(set):
    most_common = None
    qty_most_common = 0
    for item in set:
        qty = set.count(item)
        if qty > qty_most_common:
            qty_most_common = qty
            most_common = item
    return most_common


def add_details_to_frame(face, keyInd, interest_area, width, height, red, point, font, font_scale, thickness, highfreqGender, ageRange):
    labelPic = str(keyInd) + " , " + ageRange
    cv2.rectangle(interest_area,
                  (int(face['BoundingBox']['Left'] * width),
                   int(face['BoundingBox']['Top'] * height)),
                  (int((face['BoundingBox']['Left'] + face['BoundingBox']['Width']) * width),
                   int((face['BoundingBox']['Top'] + face['BoundingBox']['Height']) * height)),
                  green if highfreqGender == 'Female' else red, thickness)
    cv2.putText(interest_area, labelPic, point, font, font_scale, (255, 255, 255), thickness)


def alert(ageRange, highfreqGender, facesDic, key):
    label = 'Targeted Customer: ' + ageRange + ' , Gender: ' + highfreqGender
    facesDic[key]["toAlert"] = False
    root = tk.Tk()
    logo = tk.PhotoImage(file=faceImg)
    w1 = tk.Label(root, image=logo).pack(side="right")
    w2 = tk.Label(root,
                  justify=tk.LEFT,
                  padx=10,
                  text=label).pack(side="left")
    root.mainloop()

####################################################################################################################

if __name__ == "__main__":
    cap = cv2.VideoCapture(r"C:\Users\user\Desktop\גל\גל\Cyou\videos\cashier2.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    rekognition = boto3.client('rekognition')

    mot_tracker = Sort()  # create instance of the SORT tracker

    #Setup:
    # Modiin interactions
    #x_area, y_area, w_area, h_area = 0, 0, 1400, 1400
    # Superfarm
    #x_area, y_area, w_area, h_area = 800, 0, 1400, 800
    # Azrieli entrance
    x_area, y_area, w_area, h_area = 400, 250, 1300, 600
    blue = (255,0,0)
    green = (0,255,0)
    red = (0,0,255)
    thickness, font_scale, font = 2, 1, cv2.FONT_HERSHEY_SIMPLEX
    sec = 0
    frameRate = 1 #it will capture image in each 1 second
    success = True
    frameNum = 0
    keyInd = 0

    # Create a new collection
    collection_name = 'cyouCollection'
    rekognition.delete_collection(CollectionId=collection_name)
    rekognition.create_collection(CollectionId=collection_name)

    # Create faces dictionary
    facesDic = {}
    keys = {}
    sortKeys = {}
    similarKeys = []
    frameKeys = {}

    while success:
        # Capture frame-by-frame
        start = time.time()
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # change to 1 fps
        hasFrames, frame = cap.read()
        success = False
        if hasFrames:
            frameNum+=1
            NewFrame = True
            # change fps
            sec = sec + frameRate
            sec = round(sec, 2)
            success = True
            interest_area = frame[y_area:h_area, x_area:w_area]
            height, width, channels = interest_area.shape
            ret, pic = cv2.imencode('.jpg', interest_area)
            customers_pic = interest_area.copy()
            faces = rekognition.detect_faces(Image={'Bytes': pic.tobytes()}, Attributes=['ALL'])
            # Check faces by AWS
            for face in faces['FaceDetails']:
                key = 0
                print("aws detcted {0} confidence for face".format(face['Confidence']))
                # Check face confidence
                if face['Confidence'] >= 90:
                    print("face passed threshold > 90")
                    hasFace = True
                    point, faceImg, facePhoto, x, y, w, h = crop_Image(face, width, height, customers_pic)
                    if(NewFrame):
                        dets = np.array([[x, y, w, h]])
                        NewFrame = False
                    else:
                        dets = np.append(dets, np.array([[x, y, w, h]]), axis=0)
                        frameKeys = np.append(dets, np.array([[x, y, w, h]]), axis=0)
                    print(dets)
                    print("origin key Location is {0}, {1}, {2}, {3}".format(x, y, w, h))
                    try:
                        # Search for similar faces
                        similarFaces = rekognition.search_faces_by_image(Image={'Bytes': facePhoto.tobytes()}, CollectionId=collection_name, FaceMatchThreshold=30, MaxFaces=4)
                    except ClientError as e:
                        hasFace = False
                        print("search didnt find face")
                    # If didn't find similar faces -> index face
                    similarKeys = []
                    suspectedKey = 0
                    if hasFace:
                        print("search find face!!!")
                        if (len(similarFaces['FaceMatches']) == 0):
                            print("search find face - but no similar faces")
                            keyInd, keys, key, facesDic = index_face(rekognition, facePhoto, collection_name, keyInd, keys, facesDic, x, y, w, h)
                        else:
                            for similarFace in similarFaces['FaceMatches']:
                                key = similarFace['Face']['FaceId']
                                similarKeys.append(key)
                            key = 0
                            suspectedKey = checkSimilarFace(facesDic, frameNum, similarKeys, x, y, w, h)
                    if (not hasFace):
                        for key in facesDic.keys():
                            similarKeys.append(key)
                        key = 0
                        suspectedKey = checkSimilarFace(facesDic, frameNum, similarKeys, x, y, w, h)
                    if suspectedKey != 0:
                        print("update location")
                        key = suspectedKey
                        print("chosen Key is:" + str(key))
                        facesDic = updateFaceLocation(facesDic, key, x, y, w, h)
                    # If didnt find similarFaces
                    if (key != 0):
                        keyInd = keys.get(key)
                        print(key)
                        print(facesDic[key]["Location"]["X"])
                        print(facesDic[key]["Location"]["Y"])
                        print(facesDic[key]["Location"]["W"])
                        print(facesDic[key]["Location"]["H"])
                        # For each face save frame number, age range and gender
                        facesDic, highfreqGender, highfreqLowAge, highfreqHighAge, ageRange= save_customer_details(face, facesDic, key)
                        # Add details to the frame
                        add_details_to_frame(face, keyInd, interest_area, width, height, red, point, font, font_scale, thickness, highfreqGender, ageRange)
                        # Add an alert when customer stayed in the interest area more than 30 seconds
                        if (len(np.unique(np.array(facesDic[key]["FrameNum"]))) > 3) and (facesDic[key]["toAlert"]):
                            alert(ageRange, highfreqGender, facesDic, key)
            # Display the resulting frame
            trackers = mot_tracker.update(dets)
            print("#################################keys trackers##############################")
            print(trackers)
            cv2.imshow('frame', interest_area)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    with open('faceDictionary.json', 'w') as fp:
        json.dump(facesDic, fp)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


























