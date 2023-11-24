## Stop every x Seconds
## Read the number in given Area

import json
import argparse
import os
import pytesseract
import csv
import ast
#from imutils import contours
#import imutils
import time
import cv2
import numpy as np
#from pynput.mouse import Listener
from PIL import Image
#from number_recognition import NumberRecognizer

from multiprocessing import Pool
## Shadystuff for global mouseAction event
x,y,h,w = 0,0,100,100
frame,clone = None,None
rectangles = [(893, 397, 54, 54), (1213, 388, 78, 78)]

DEST = "Screenshots\\"
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Niklas\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
##! Not used

def get_screenshots_rect(interval):
    """Saves a 28x28 screenshot every interval seconds"""
    global clone
    assert type(interval) == int
    cap = cv2.VideoCapture("test.mp4")

    if (not cap.isOpened()):
        print("File did not load")
        exit()

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame",mouseAction)

    while (cap.isOpened()):
        ret, frame, = cap.read()
        if ret:
            clone = frame.copy()
            pressedKey = cv2.waitKey(0) & 0xFF
            if pressedKey == ord("n"):
                frame = cv2.rectangle(frame,(x,y,w,h),(255,0,0),2)
                cv2.imshow('Frame', frame)
                # next
                continue
            elif pressedKey == ord('q'):
                break

        else:
            break
    cap.release()

    cv2.destroyAllWindows()


##* Ab hier gehts los


def  mouseAction(event,x_loc,y_loc,flags,param):
    global x,y,h,w, frame,i, clone
    if event == 1:  # Left Click
        x = x_loc - w//2
        y = y_loc - h //2
    elif event == 10: #Mousewheel
        if flags > 0:
            h = h-2
            w = w-2
        elif flags < 0:
            h = h+2
            w = w+2
    elif event == 2: # Right Click
        rectangles.append((x,y,w,h))
        print(rectangles)
        frame = cv2.rectangle(frame,(x,y,w,h),(255,0,0),2)
        cv2.imshow('Frame', frame)
        clone = frame.copy()
        return
        
        
    frame = clone.copy()
    frame = cv2.rectangle(frame,(x,y,w,h),(255,0,0),2)
    cv2.imshow('Frame', frame)


def save_screenshots(path:str,interval:int=10,fps:int=30,start:int=26,dil=(1,1),threshold=130,end=-1):
    """Saves a screenshot from pathvideo every interval seconds
       Frame000i.png
       -> returns array of array with parsed numbers
    """
    global rectangles
    numbers = []
    if not interval: interval = 10
    if not start: start = 26
    if not fps: fps = 30 
    assert type(interval) == int
    assert type(fps) == int

    cap = cv2.VideoCapture(path)
    if (not cap.isOpened()):
        print("File did not load")
        raise SystemExit
    i = 0

    while (cap.isOpened()):
        pos = i * fps * interval + (fps * start)
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos-1)
        ret, frame = cap.read()
        if ret and i != end:
            ##print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)),pos)
            #cv2.imwrite(f'{DEST}Frame{i:04d}.png',frame)
            numbers.append(split_frame_read_number(rectangles,frame,i,dil,threshold))
            #print(f'Frame{i:04d}.png')
            if i % 50 == 0: print(".",end="")
            i = i + 1
        else:
            break
    cap.release()

    cv2.destroyAllWindows()

    return numbers

def transform_frame(img,dil,threshold):
    """Takes cv2.img and makes it teseract readable
        return img converted for teseract
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, img) = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones(dil, np.uint8) 
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)
    #cv2.imwrite("test.png",img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

def make_screen_right_again_noisy(img,threshhold,dil):
    """NOISY: Takes cv2.img and makes it teseract readable"""

    
    cv2.imshow("Frame",img)
    cv2.waitKey(0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame",img)
    cv2.waitKey(0)

    (_, img) = cv2.threshold(img, threshhold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Frame",img)
    cv2.waitKey(0)
    kernel = np.ones(dil, np.uint8) 
    img = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow("Frame",img)
    cv2.waitKey(0)

    img = cv2.bitwise_not(img)
    cv2.imshow("Frame",img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    cv2.imwrite("test.png",img)
    return img
def choose_area(path):
    """Loads Video and select n areas"""
    global clone,rectangles

    cap = cv2.VideoCapture(path)
    if (not cap.isOpened()):
        print("File did not load")
        raise SystemExit
    cv2.namedWindow('Frame')
    cv2.setMouseCallback("Frame",mouseAction)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            clone = frame.copy()

            pressedKey = cv2.waitKey(0) & 0xFF
            if pressedKey == ord("n"):
                frame = cv2.rectangle(frame,(x,y,w,h),(255,0,0),2)
                cv2.imshow('Frame', frame)
                # next
                continue
                
            elif pressedKey == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return rectangles

def split_frame_read_number(rectangles,frame,i,dil,threshold):
    """Takes array of rectangles (x,y,w,h) and cuts them out frame i
        -> return number from all frames as array of strings
    """
    global DEST
    out = []
    j = 0
   
    for (x,y,w,h) in rectangles:
        j+=1
      
        #cv2.imwrite(f'{DEST}Frame{i:04d}_{j}.png',transform_frame(frame[y:y+h,x:x+w],dil,threshold))
        num = get_numbers(transform_frame(frame[y:y+h,x:x+w],dil,threshold)).replace("\n","")
        out.append(num)
        # if not num:
        #     print(f'Frame{i:04d}_{j}.png',num,"<--------------------")
            
        # else:
        #     print(f'Frame{i:04d}_{j}.png',num)
    return out

def init_argparse():
    """Parse args; make Folder for Screenshots"""
    global DEST
    parser = argparse.ArgumentParser()
    parser.add_argument('-video',type=str,help="Path to the file")
    parser.add_argument('-fps',type=int,help="set fps value [default 30]")
    parser.add_argument('-interval',type=int,help="set interval in seconds [default 10]")
    parser.add_argument('-start',type=int,help="timestamp for the first screenshot in seconds [default 26]")
    parser.add_argument('-dest',type=str,help ="Path to destination folder")
    
    args = parser.parse_args()
    if args.dest: 
        os.makedirs(args.dest,exist_ok=True)
        DEST = args.dest+"\\"
    else: 
        #os.rmdir("Screenshots")
        os.makedirs("Screenshots",exist_ok=True)
    return args.video, args.fps,args.interval,args.start

def get_numbers(img):
    """reads digits from a picture via teseract"""
    return pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

def read_values_from_csv(path:str):
    """reads values from csv-file evaluated by hand
        return tuple array -> (gefahrene km, Restreichweite)
    """
    out = [] #x,y -> Gefahrene km, Restreichweite

    with open(path,newline="") as csvfile:
        spamreader = csv.reader(csvfile,delimiter=";")
        for row in spamreader:
            out.append((row[1].replace(",","."),row[2]))
    out = out[1:] # erste Zeile skippen
    return out
            
            
def worker(param):
    """
    returns dict(threshold, (#mistakes0, #mistakes1), [wrong values (index frame, 0 or 1,wrong value, right value)])
    """
    t = time.time()
    right_values,dil,threshold = param
    print(threshold)
    path, fps, interval,start = init_argparse()
    ocr_values = save_screenshots(path,interval,fps,start,dil,threshold)
    mistake_count = [0,0]
    wrong_values = []
    for i in range(min(len(ocr_values),len(right_values))):
        if not ocr_values[i][0].replace(".","").isdigit() or not ocr_values[i][0] or right_values[i][0].replace(".","") != ocr_values[i][0]:
            mistake_count[0] +=1
            wrong_values.append((i,0,ocr_values[i][0],right_values[i][0]))
        if not ocr_values[i][1].isdigit() or not ocr_values[i][1] or right_values[i][1] != ocr_values[i][1]:
            mistake_count[1] +=1
            wrong_values.append((i,1,ocr_values[i][1],right_values[i][1]))
    print("workertime:",time.time()-t)
    return {"threshold":threshold,"mistakes":mistake_count,"wrong_values":wrong_values,"right_values":right_values,"ocr_values":ocr_values}
def sort_0(elem):
    return elem["mistakes"][0]
def sort_1(elem):
    return elem["mistakes"][1]
def test_bench(worker_threads:int,threshold_start:int, threshold_end:int,threshold_stepsize:int):
    
    path, fps, interval,start = init_argparse()
    right_values = read_values_from_csv("Daten_IONIQ5_WLTC_Eco_handgeschrieben.csv")
    dil = (1,1)
    t = time.time()
    arg_list = [[right_values,dil,i] for i in range(threshold_start,threshold_end,threshold_stepsize)]
    all = None
    ## Multitprocessing start
    with Pool(worker_threads) as p:
         all = p.map(worker,arg_list)
    print()
    print(time.time()-t)
    #print("len difference: ", len(right_values) - len(ocr_values))

    
    print_candidates(all,10)

    #temp = transform_frame(img)
    #get_numbers(temp)


    ## Output to file
    f = open(f"output_{''.join(time.asctime().split(' ')[1:4]).replace(':','_')}SortParam1.txt","w")
    all.sort(key=sort_0,reverse=False)
    for s in all:
        f.write("-"*25+"\n")

        for key in s.keys():
            f.write(key+" : "+str(s[key])+"\n")
    f.close()
    all.sort(key=sort_1,reverse=False)
    f = open(f"output_{''.join(time.asctime().split(' ')[1:4]).replace(':','_')}SortParam2.txt","w")
    for s in all:
        f.write("-"*25+"\n")

        for key in s.keys():
            f.write(key+" : "+str(s[key])+"\n")
    f.close()

    return 
def print_candidates(arr,limit=10,fromFile=False):
    """Takes array of dicts and print the best n
        use txt with fromFile=True 
    """

    temp = []
    if fromFile:
        f = open(arr,"r").read().split("\n")
        
        for line in f:
            temp.append(ast.literal_eval(line))
        arr = temp
    
    print(f'{"Best Values" :=^35}')
    print(f'{"Parameter 1" :-^35}')
    
    arr.sort(key=sort_0,reverse=False)
    print(f'{"Threshold": <10}{"Mistakes": ^10}')
    for t in arr[:limit]:
        print(f'{t["threshold"]: <10}{t["mistakes"][0]: ^10}')
    print("\n")

    print(f'{"Parameter 2" :-^35}')
    arr.sort(key=sort_1,reverse=False)
    print(f'{"Threshold": <10}{"Mistakes": ^10}')
    for t in arr[:limit]:
        print(f'{t["threshold"]: <10}{t["mistakes"][1]: ^10}')
def main():

    path, fps, interval,start = init_argparse()
    #get_screenshots(10)
    #get_numbers()
    #transform_frame(1,1)
    t = time.time()
    #choose_area(path)
    
    save_screenshots(path,interval,fps,start)
    print(time.time()-t)



if __name__ == "__main__":
    #main()
    # worker_threads, start_value, end_value, stepsize
    test_bench(12,5,250,1)
    #read_values_from_csv("Daten_IONIQ5_WLTC_Eco_handgeschrieben.csv")
    #print_candidates("outputNov2117_13_00.txt",10,True)
    pass
