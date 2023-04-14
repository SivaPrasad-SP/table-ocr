from flask import Flask
import pytesseract
import cv2 as cv
import numpy as np
from flask import request
import pandas as pd
 
app = Flask(__name__)

# pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR/tesseract.exe" 

def words_on_same_line(d1, d2):
    d1_y_mid = (d1['top']+d1['bottom'])/2
    d2_y_mid = (d2['top']+d2['bottom'])/2
    if d1['top'] <= d2_y_mid <= d1['bottom']:
        return True
    elif d2['top'] <= d1_y_mid <= d2['bottom']:
        return True
    else:
        return False

def cluster(data, maxgap):
    '''Arrange data into groups where successive elements differ by no more than *maxgap*
       example : cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
                 [[1, 6, 9], [100, 102, 105, 109], [134, 139]]
    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups
 
@app.route("/", methods=['GET', 'POST'])
def home_view():
        if request.method == 'POST':
                fs = request.files['file'] #.get('file')
                frame = cv.imdecode(np.frombuffer(fs.read(), np.uint8), cv.IMREAD_UNCHANGED)
        else:
                file_name = 'images/tbl_roi.jpg'
                frame = cv.imread(file_name)
        tbl_img = frame.copy()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, frame_thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        tbl_roi = frame_thresh
        image = frame_thresh
        # tbl_img = frame_thresh

        h, w = frame_thresh.shape #[0], frame.shape[1]
        # print(h, w, ' : ', frame_thresh.shape)

        results = pytesseract.image_to_data(frame_thresh)

        data = list(map(lambda x: x.split('\t'), results.split('\n')))
        df = pd.DataFrame(data[1:], columns=data[0])

        df.dropna(inplace=True) # drop the missing in rows
        df['conf'] = [int(float(i)) for i in df['conf'].values]
        col_int = ['level','page_num','block_num','par_num','line_num','word_num','left','top','width','height','conf']
        df[col_int] = df[col_int].astype(int)

        word_data = []

        for i in range(len(df)): # last row gives none
                temp = {}
                txt = df["text"][i]
                # preprocess the text
                if txt != '' and txt != ' ' and txt != '\n':
                        temp['word_no'] = i
                        temp['level'] = df["level"][i]
                        temp['left'] = df["left"][i]
                        temp['top'] = df["top"][i]
                        temp['top_left'] = (df["top"][i] + df["left"][i])
                        temp['width'] = df["width"][i]
                        temp['height'] = df["height"][i]
                        temp['right'] = (df["left"][i] + df["width"][i])
                        temp['bottom'] = (df["top"][i] + df["height"][i])
                        temp['conf'] = round(float(df["conf"][i]), 2)
                        temp['text'] = txt
                        word_data.append(temp)
        word_data_df = pd.DataFrame(word_data)

        word_data = sorted(word_data, key = lambda i: i['top'])

        new_data = []
        temp = []
        all_words = []
        curr_word = ''

        for i in range(len(word_data)):
                if i+1 < len(word_data)-1:
                        temp.append(word_data[i])
                        curr_word += (' '+word_data[i]['text'])
                        if words_on_same_line(word_data[i], word_data[i+1]):
                                pass
                        else:
                                new_data.append(temp)
                                temp = []
                                all_words.append(curr_word)
                                curr_word = ''
        
        sorted_new_data = []
        sorted_line_lists = []

        for lst in new_data:
                new_lst = sorted(lst, key = lambda i: i['left'])
                for i in new_lst:
                        sorted_new_data.append(i)
                sorted_line_lists.append(new_lst)

        tbl_h, tbl_w = tbl_roi.shape
        c = 0
        for lst in sorted_line_lists:
                y1 = lst[0]['top']
                if c == 0:
                        cv.line(image, (0, y1), (tbl_w, y1), (0, 0, 255), 2)
                elif (y1 - y2) > tbl_w//8:
                        cv.line(image, (0, y2), (tbl_w, y2), (0, 0, 255), 2)
                        cv.line(image, (0, y1), (tbl_w, y1), (0, 0, 255), 2)
                else:
                        mid_y = (y2+y1)//2
                        cv.line(image, (0, mid_y), (tbl_w, mid_y), (0, 0, 255), 2)
                y2 = lst[-1]['bottom']
                if c == len(sorted_line_lists)-1:
                        cv.line(image, (0, y2), (tbl_w, y2), (0, 0, 255), 2)
                # ---
                # cv.line(image, (0, y1), (tbl_w, y1), (0, 0, 255), 2)
                # cv.line(image, (0, y2), (tbl_w, y2), (0, 0, 255), 2)
                #--
                c += 1
                txt = ''
                for i in lst:
                        txt += ' '+i['text']
        
        gray = cv.cvtColor(tbl_img, cv.COLOR_BGR2GRAY)

        # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
        gray = cv.bitwise_not(gray)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

        ah, aw = bw.shape
        h, w = bw.shape

        vertical = np.copy(bw)

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 30
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)

        existed_vert_lines = []

        vert_lines = [0]
        vert_lines += list(np.where(np.count_nonzero(vertical, axis=0) > h/4)[0]) # == 0
        if len(vert_lines) > 3:
                if len(cluster(vert_lines, 5)) > 2:
                        existed_vert_lines = cluster(vert_lines, 5)
        
        final_vert_lines = []

        if len(existed_vert_lines) == 0:
                # optimize..
                print('no_lines')
                blank_vert_lines = [0]
                blank_vert_lines += list(np.where(np.count_nonzero(bw, axis=1) < (w*5/100))[0]) # == 0
                clstr_vert = cluster(blank_vert_lines, 50)
                for i in clstr_vert:
                        print(len(i), ' : ', i[0])
                        line_num = i[0]
                        final_vert_lines.append(line_num)
                        cv.rectangle(tbl_img, (i[0], 0), (i[0], h), (0, 0, 255), 2)
                # ----- use tess words sep as cols cols ---
        else:
                for i in existed_vert_lines:
                        print(len(i), ' : ', i[0])
                        line_num = round(np.median(i)) # i[0]
                        final_vert_lines.append(line_num)
                        cv.rectangle(tbl_img, (i[0], 0), (i[0], h), (0, 0, 255), 2)
        final_vert_lines.append(w-1)

        rows = []

        for i in range(len(final_vert_lines)-1):
                if i == 0:
                        st = 0
                else:
                        st = final_vert_lines[i]
                cols = []
                for lst in sorted_line_lists:
                        #print(len(lst))
                        #val_flag = 0
                        txt = ''
                        for val in lst:
                                if st <= val['right'] <= final_vert_lines[i+1]:
                                        #val_flag += 1
                                        txt += ' '+val['text']
                        #if val_flag:
                        cols.append(txt)
                rows.append(cols)
        
        new_rows = np.array(rows).transpose()
        print(new_rows)

        response_data = {'data': {'no_of_tables ': 2, 'img2tbl': new_rows.tolist(), 'tbl_shapes': [frame_thresh.shape]}}
        return response_data