## Stop every x Seconds
## Read the number in given Area

import json
import argparse
import os
#import pytesseract
import csv
import ast
import time
import cv2
import easyocr
import numpy as np

from PIL import Image

import itertools
from multiprocessing import Pool,Value,Process,Lock
reader = None
## Shadystuff for global mouseAction event
x,y,h,w = 0,0,100,100
frame,clone = None,None
worker_progress, RANGE = 0,0
rectangles = [(963, 494, 84, 84)]
threshold_arg = None
DEST = "Screenshots\\"
#pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Niklas\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
write_screenshot_arg = False
csv_out_path_arg = "output.csv" 
##! Not used

def get_screenshots_rect(interval: int) -> None:
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


def  mouseAction(event: int,x_loc: float,y_loc: float,flags: int,param) -> None:
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


def save_screenshots(path:str,interval:int=10,fps=29.97,start:int=0,dil=(1,1),threshold=130,end=-1):
    """Saves a screenshot from pathvideo every interval seconds
       Frame000i.png 
       -> returns array of array with parsed numbers
    """
    global rectangles,threshold_arg, write_screenshot_arg
    numbers = []
    if not interval: interval = 10
    if not start: start = 0
    if not fps: fps = 29.97
    assert type(interval) == int

    cap = cv2.VideoCapture(path)
    if (not cap.isOpened()):
        print("File did not load")
        raise SystemExit
    i = 0
    if threshold_arg: threshold = threshold_arg
    print(f'[*] Reading... ')
    print(f" {'Interval':16}: {interval}")
    print(f" {'Fps':16}: {fps}")
    print(f" {'threshold':16}: {threshold}")
    print(f" {'dilatation':16}: {dil}")
    print(f" {'Make Screenshots':16}: {write_screenshot_arg}")

    while (cap.isOpened()):
        pos = int(round(i * fps * interval + (fps * start)))
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos-1)
        ret, frame = cap.read()
        if ret and i != end:
            ##print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)),pos)
            #cv2.imwrite(f'{DEST}Frame{i:04d}.png',frame)
            numbers.append(split_frame_read_number(rectangles,frame,i,dil,threshold))
            #print(f'Frame{i:04d}.png')
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

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


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
    rectangles = []
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
        -> return ocr read number from all frames as array of strings
    """
    global DEST,write_screenshot_arg
    out = []
    j = 0
    ##print("FIXED RECTANGLES")
    ##rectangles = [(963, 494, 84, 84)]
    for (x,y,w,h) in rectangles:
        j+=1
        # if j == 2: ##! kinda weird
        #     frame = rotate_image(frame,3)
        if write_screenshot_arg: cv2.imwrite(f'{DEST}Frame{i:04d}_{j}.png',transform_frame(frame[y:y+h,x:x+w],dil,threshold))
        # cv2.imshow(transform_frame(frame[y:y+h,x:x+w],dil,threshold))
        
        tmp = transform_frame(frame[y:y+h,x:x+w],dil,threshold)
        if not np.any(tmp):
            print("Frame consists only of zeros (Black)")
            num = ["Blackscreen"]
        else:
            num = get_numbers_ocr(transform_frame(frame[y:y+h,x:x+w],dil,threshold))
        if num == []: num = ["N/A"]  #get_numbers_ocr can return None if only 0 are given
        print("num gets appended:",num[0], type(num))
        out.append(num[0])
        # if not num:
        #     print(f'Frame{i:04d}_{j}.png',num,"<--------------------")
            
        # else:
        #     print(f'Frame{i:04d}_{j}.png',num)
    return out

def init_argparse():
    """Parse args; make Folder for Screenshots"""
    global DEST, threshold_arg, write_screenshot_arg,csv_out_path_arg
    parser = argparse.ArgumentParser(description="Select areas from the video which get OCRed every *interval* seconds", 
                                     epilog="""1. Select the areas via mouse. Use mousewheel to change size of the rectangle.\n\t 
                                     Use leftclick to set a rectangle. \n\t 
                                     Use q to quit once you finished.
                                     2. 
                                     """)
    parser.add_argument('-video',required=True,type=str,help="Path to the file")
    parser.add_argument('-fps',type=int,help="set fps value [default 30]")
    parser.add_argument('-interval',type=int,help="set interval in seconds [default 10]")
    parser.add_argument('-start',type=int,help="timestamp for the first screenshot in seconds [default 0]")
    parser.add_argument('-dest',type=str,help ="Path to destination folder (for Screenshots)")
    parser.add_argument('-tesseract_path',type=str,help="Path to pytesseract.exe")
    parser.add_argument('-debug','-d',action="store_true")
    parser.add_argument('-threshold', help="Value for the img conversion (0-255) [default 130]")
    parser.add_argument('-screenshot',help="Write Screenshots to dest or Screenshots/ [default False]", action="store_true",default=False)
    parser.add_argument('-out',type=str,default="output.csv", help="Path/name to the output csv [default output.csv]")
    args = parser.parse_args()
    threshold_arg = args.threshold
    csv_out_path_arg = args.out
    write_screenshot_arg = args.screenshot
    # pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    if args.dest: 
        os.makedirs(args.dest,exist_ok=True)
        DEST = args.dest+"\\"
    else: 
        #os.rmdir("Screenshots")
        os.makedirs("Screenshots",exist_ok=True)
    return args.video, args.fps,args.interval,args.start,args.debug

def get_numbers_ocr(img):
    """reads digits from a picture via teseract"""
    global reader
    return reader.readtext(img, detail=0)
    # return pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    

def read_values_from_csv(path:str):
    """reads values from csv-file evaluated by hand
        return tuple array -> (gefahrene km, Restreichweite)
    """
    out = [] #x,y -> Gefahrene km, Restreichweite

    with open(path,newline="") as csvfile:
        spamreader = csv.reader(csvfile,delimiter=";")
        for row in spamreader:
            out.append([row[1].replace(",","."),row[2]])
    out = out[1:] # erste Zeile skippen
    return out

def numbers_to_csv(arr:list[list[str]]) -> None:
    """Takes list with lists of numbers and writes them to dest_path (-out param)"""
    global csv_out_path_arg
    dest_path = csv_out_path_arg
    try: 
        csvfile = open(dest_path, 'w', newline='')
    except:
        print("[ ! ] Can't create csv-file!")
        return
    print("RECTANGLES",rectangles)
    writer = csv.writer(csvfile,dialect='excel')
    column_names = None
    while True:
        column_names = input(f"Write {len(rectangles)} column names as you got rectangles [seperated with ,]:")
        if len(column_names.split(',')) == len(rectangles):
            break
        print("invalid number of names")
        print("Should be", len(rectangles))
    if column_names:
        writer.writerow(column_names.split(','))
    else:
        writer.writerow(["Column"+str(j) for j in range(arr[0])])

    for row in arr:
        writer.writerow(row)
    
    
    csvfile.close()
    print("[+] results in", dest_path)
    return

def worker(param):
    """
    returns dict(threshold, (#mistakes0, #mistakes1), [wrong values (index frame, 0 or 1,wrong value, right value)])
    """
    
    t = time.time()
    right_values,dil,threshold,ran = param
    print("Starting with", (dil,dil), threshold)

    path, fps, interval,start,_ = init_argparse()
    ocr_values = save_screenshots(path,interval,fps,start,(dil,dil),threshold)
    mistake_count = [0,0]
    wrong_values = []
    for i in range(min(len(ocr_values),len(right_values))):
        if not ocr_values[i][0].replace(".","").isdigit() or not ocr_values[i][0] or right_values[i][0].replace(".","") != ocr_values[i][0]:
            mistake_count[0] +=1
            wrong_values.append((i,0,ocr_values[i][0][:2]+"."+ocr_values[i][0][2:],right_values[i][0]))
        if not ocr_values[i][1].isdigit() or not ocr_values[i][1] or right_values[i][1] != ocr_values[i][1]:
            mistake_count[1] +=1
            wrong_values.append((i,1,ocr_values[i][1],right_values[i][1]))

    for i in range(len(ocr_values)): ##! Better readability in logs
        ocr_values[i][0] = ocr_values[i][0][:2] + "." + ocr_values[i][0][2:]


    print("workertime:",time.time()-t)
    return {"threshold":threshold,"dilate":dil,"mistakes":mistake_count,"wrong_values":wrong_values,"right_values":right_values,"ocr_values":ocr_values}
def sort_0(elem):
    return elem["mistakes"][0]
def sort_1(elem):
    return elem["mistakes"][1]
def test_bench(worker_threads:int,threshold_start:int, threshold_end:int,threshold_stepsize:int,dilate_start:int,dilate_end:int,path_to_csv:str):

    ran = ((threshold_end - threshold_start)//threshold_stepsize)*(dilate_end-dilate_start+1)
    path, fps, interval,start,_ = init_argparse()
    right_values = read_values_from_csv("Daten_IONIQ5_WLTC_Eco_handgeschrieben.csv")
    dil = [i for i in range(dilate_start,dilate_end+1)]
    thresholds = [i for i in range(threshold_start,threshold_end,threshold_stepsize)]
    print(f'Searching for the best Threshold [{threshold_start} - {threshold_end}] and Dilate [{dilate_start} - {dilate_end}] value')
    print(f'There are {ran} possibilities')
    print(f'Creating folder "TestBenchOutput"')
    print(f'Starting {worker_threads} worker processes')
    os.makedirs("TestBenchOutput",exist_ok=True)
    t = time.time()
    arg_list = [[right_values,x,y,ran] for x,y in itertools.product(dil,thresholds)]
    all = []
    ## Multitprocessing start
    with Pool(worker_threads) as p:
         all = p.map(worker,(arg_list))
    print()
    print(time.time()-t)
    #print("len difference: ", len(right_values) - len(ocr_values))

    
    print_candidates(all,10)

    #temp = transform_frame(img)
    #get_numbers(temp)


    ## Output to file
    f = open(f"TestBenchOutput\\output_{''.join(time.asctime().split(' ')[1:4]).replace(':','_')}SortParam1.txt","w")
    all.sort(key=sort_0,reverse=False)
    for s in all:
        f.write("-"*25+"\n")

        for key in s.keys():
            f.write(key+" : "+str(s[key])+"\n")
    f.close()
    all.sort(key=sort_1,reverse=False)
    f = open(f"TestBenchOutput\\output_{''.join(time.asctime().split(' ')[1:4]).replace(':','_')}--Threshold{str((threshold_start,threshold_end,threshold_stepsize))}--Dilate{str((dilate_start,dilate_end))}SortParam2.txt","w")
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
    print(f'{"Threshold": <10}{"Dilate": ^10}{"Mistakes": ^10}')
    for t in arr[:limit]:
        print(f'{t["threshold"]: <10}{t["dilate"]: ^10}{t["mistakes"][0]: ^10}')
    print("\n")

    print(f'{"Parameter 2" :-^35}')
    arr.sort(key=sort_1,reverse=False)
    print(f'{"Threshold": <10}{"Dilate": ^10}{"Mistakes": ^10}')
    for t in arr[:limit]:
        print(f'{t["threshold"]: <10}{t["dilate"]: ^10}{t["mistakes"][1]: ^10}')
def main():
    global reader
    path, fps, interval,start, _ = init_argparse()
    #get_screenshots(10)
    #get_numbers()
    #transform_frame(1,1)
    t = time.time()
    # print("FIXED RECTANGLE ACTIVE, SKIPPING CHOOSE_AREA")
    choose_area(path)
    ## TODO implement character space: reader.readtext(input, allowlist ='0123456789')
    reader = easyocr.Reader(['en'])
    results = save_screenshots(path,interval,fps,start)
    print(results)
    numbers_to_csv(results)
    
    print(time.time()-t)
    

def debug_main() -> None:
    """Executes Test/Debug stuff
        Put Changes here
    """
    
    
    # test_bench(4,35,230,3,0,2,"Daten_IONIQ5_WLTC_Eco_handgeschrieben.csv")

def test():
    print("Calling test")
    img = cv2.imread('Screenshots\\Frame0000_1.png')
    print(get_numbers_ocr(img))
if __name__ == "__main__":



    if init_argparse()[4]:
        debug_main()
    else:
        main()
    
    #main()
    # worker_threads, threshold start_value, thresholdend_value,threshold stepsize, dilate start, dilate end, path to csv
    
    #read_values_from_csv("Daten_IONIQ5_WLTC_Eco_handgeschrieben.csv")
    #print_candidates("outputNov2117_13_00.txt",10,True)
    
