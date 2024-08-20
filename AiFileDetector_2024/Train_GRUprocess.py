import csv
import os
import pickle
import re
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QProgressBar, QDialog, QLabel, \
    QVBoxLayout
from PyQt5 import uic, QtWidgets
from tensorflow.python.client import device_lib
from simhash import Simhash
import pyautogui
from ttkthemes import ThemedTk

device_lib.list_local_devices()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
form_class = uic.loadUiType("UI_Design\\AiDetector2.ui")[0]

with tf.device('/GPU:0'):
    class ProgressWindow(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("파일 처리 진행 상황")

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setGeometry(30, 50, 250, 20)

            self.label = QLabel("파일 처리 중...", self)

            layout = QVBoxLayout()
            layout.addWidget(self.label)
            layout.addWidget(self.progress_bar)
            self.setLayout(layout)

        def set_progress(self, value):
            self.progress_bar.setValue(value)

        def set_label_text(self, text):
            self.label.setText(text)



    class createtrainclass(QMainWindow, form_class):
        def __init__(self):
            super(createtrainclass, self).__init__()
            self.choice = 0
            self.file_paths = []

            self.setupUi(self)

            # 확장자 필터
            self.extension_list = [" ", ".mp4", ".mov"]
            self.comboBox.addItems(self.extension_list)
            self.comboBox.currentIndexChanged.connect(self.filter_files_by_extension)

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setGeometry(200, 32, 250, 20)

            # self.progress_bar2 = QProgressBar(self)
            # self.progress_bar2.setGeometry(50, 50, 250, 20)
            # 파일시스템 트리
            self.dirModel = QFileSystemModel()
            self.dirModel.setRootPath("\\")
            self.treeView.setModel(self.dirModel)

            self.treeView.setRootIndex(self.dirModel.index('Y:\\'))
            self.treeView.clicked.connect(self.file_selected)

            self.feature_create1.clicked.connect(lambda: setattr(self, 'choice', 1))
            self.create_value2.clicked.connect(lambda: setattr(self, 'choice', 2))
            self.create_sequence3.clicked.connect(lambda: setattr(self, 'choice', 3))
            self.LoadButton.clicked.connect(self.main)

            self.listWidget.itemDoubleClicked.connect(self.remove_selected_file)

        def load_directory(self):
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
            if directory:
                self.dirModel.setRootPath(directory)
                self.treeView.setRootIndex(self.dirModel.index(directory))
                self.treeView.clicked.connect(self.file_selected)

        def filter_files_by_extension(self, extension):
            if extension>=0:
                self.extension = self.comboBox.currentText()
                self.extension = self.extension[1:]
                self.dirModel.setNameFilters([f"*{self.extension}"])
                self.dirModel.setNameFilterDisables(False)
            else:
                self.dirModel.setNameFilters([])
                def on_closing(self, root):
                    root.destroy()

        def file_selected(self, index):
            file_info = self.dirModel.fileInfo(index)
            if file_info.isDir():  # If a directory is selected
                self.select_all_files_in_directory(file_info.absoluteFilePath())
            else:
                try:
                    file_path = file_info.absoluteFilePath()
                    extension = os.path.splitext(file_path)[1][1:].lower()
                    if extension == self.extension.lower():
                        if file_path not in self.file_paths:
                            self.listWidget.addItem(file_path)
                            self.file_paths.append(file_path)
                except Exception as e:
                    pyautogui.alert(e)

        def select_all_files_in_directory(self, directory_path):
            try:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        extension = os.path.splitext(file_path)[1][1:]

                        if extension.lower() == self.extension:
                            if file_path not in self.file_paths:
                                self.listWidget.addItem(file_path)
                                self.file_paths.append(file_path)
            except Exception as e:
                pyautogui.alert(e)

        def remove_selected_file(self, item):
            file_path = item.text()
            self.listWidget.takeItem(self.listWidget.row(item))
            self.file_paths.remove(file_path)

        def load_common(self):
            self.data_list2 = []
            self.newlist2 = []

            with open('text2.txt', 'r') as file:
                for line in file:
                    cleaned_line = line.strip('')[1:-1]
                    sub_list = cleaned_line.split(',')
                    self.data_list2.append(sub_list)

            for a in sub_list:
                b = a.strip('" \'')
                self.newlist2.append(b)

            print(self.newlist2)

        def extract_ngram(self, n, file_paths):
            ngram_sets = []
            self.ngrams_list = []
            self.hex_lists = []

            for file_path in file_paths:
                with open(file_path, 'rb') as file:
                    content = file.read()
                    self.hex_values = content.hex()
                    ngrams = []

                    for i in range(len(self.hex_values) - n + 1):
                        ngram = self.hex_values[i:i + n]
                        ngrams.append(ngram)

                self.hex_lists.append((file_path, self.hex_values))
                self.ngrams_list.append((file_path, ngrams))

        def lcs(self, X, Y):
            m = len(X)
            n = len(Y)

            # memoization table 초기화
            L = [[0] * (n + 1) for _ in range(m + 1)]

            # X[0..m-1]와 Y[0..n-1] LCS 계산
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])

            # LCS
            index = L[m][n]
            lcs_list = [None] * index

            i = m
            j = n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    lcs_list[index - 1] = X[i - 1]
                    i -= 1
                    j -= 1
                    index -= 1
                elif L[i - 1][j] > L[i][j - 1]:
                    i -= 1
                else:
                    j -= 1

            return lcs_list

        def lcs_multiple_lists(self, lists):
            if len(lists) < 2:
                raise ValueError("파일처리 에러(길이)")

            current_lcs = self.lcs(lists[0], lists[1])


            for lst in lists[2:]:
                current_lcs = self.lcs(current_lcs, lst)
                if not current_lcs:
                    return []

            return current_lcs

        def process_files(self):
            progress_window = ProgressWindow()
            progress_window.setModal(True)
            progress_window.show()

            total_files = len(self.file_paths)

            for i, fname in enumerate(self.file_paths):
                full_path = fname
                if os.path.isfile(full_path):
                    self.reres = self.extract_value(full_path, self.extension)
                    self.all_result.append(self.reres)

                # 진행 상황 업데이트
                progress_percentage = (i + 1) / total_files * 100
                self.progress_bar.setValue(progress_percentage)
                QApplication.processEvents()  # UI 업데이트를 처리하기 위해 이벤트 루프를 실행

            progress_window.set_label_text("작업 완료")
            progress_window.exec_()

        def get_files_value(self):
            results = {}
            self.all_result = []
            self.count = -1
            # 폴더 내 모든 파일에 대해 수행
            for i, fname in enumerate(self.file_paths):
                self.count += 1
                full_path = fname

                if os.path.isfile(full_path):
                    self.reres = self.extract_value(full_path, self.extension)
                    self.all_result.append(self.reres)
                #self.all_result.append(self.reres)


                # 진행 상황 업데이트
                progress_percentage = (i + 1) / len(self.file_paths) * 100
                self.progress_bar.setValue(progress_percentage)
                QApplication.processEvents()  # UI 업데이트를 처리하기 위해 이벤트 루프를 실행

            # 파일 처리가 완료되면 최장 리스트를 찾음
            try:
                longest_list = max(self.all_result, key=len)

                # 모든 리스트를 가장 긴 리스트와 비교하며 첫 번째 요소가 없으면 추가하고 두 번째 요소는 0으로 설정
                for lst in self.all_result:
                    for i in range(len(longest_list)):
                        if len(lst) <= i or lst[i][0] != longest_list[i][0]:
                            # 첫 번째 요소가 없으면 추가하고 두 번째 요소를 0으로 설정
                            if len(lst) <= i:
                                lst.insert(i, [longest_list[i][0], '0'])
                            else:
                                lst[i] = [longest_list[i][0], '0']
            except:
                pass

            return self.all_result

        def merge_lists2(self, ngram):
            count, count2, onecount = 0,0,0
            new_list = []
            merged_list=[]
            one_merged_list = []
            previous_gram = ''
            for onegram1 in ngram :

                if previous_gram == onegram1 and onegram1 != '00000000':
                    one_merged_list.append(onegram1)
                    onecount += 1
                    pass

                else :
                    if count == 0:
                        previous_gram = onegram1
                        count += 1
                    else :
                        lengh = (len(onegram1) - 1)
                        previous_gram_2=previous_gram[-lengh:]
                        onegram_2 = onegram1[:-1]

                        if previous_gram_2 == onegram_2 and onegram1 != '00000000':
                            previous_gram = previous_gram+onegram1[-1]
                            #count2 += 1
                        else :
                            previous_gram_2 = ''
                            if count2 != 0 and onegram1 != '00000000':
                                one_merged_list.append(previous_gram)
                                count = 0
                            else :
                                if previous_gram != '00000000':
                                    one_merged_list.append(previous_gram)

                            previous_gram = onegram1
                    if onecount == len(ngram)-1:
                        if previous_gram != '00000000':
                            one_merged_list.append(previous_gram)
                        break

                    onecount += 1

            merged_list.append(one_merged_list)

            return merged_list

        def add_numbers_to_duplicates(self, input_list, opcounts):

            for i in range(len(input_list[0])):
                item = input_list[0][i]

                # 이미 등장한 요소인 경우
                if item in opcounts:
                    opcounts[item] += 1
                    input_list[0][i] = f"{item}_{opcounts[item]}"
                else:
                    opcounts[item] = 0

            return input_list

        def extract_value(self, fpath, extension):
            os.chdir("E:\\AiFileDetectorE")
            file_type = ""
            res = []
            self.reres = []
            check_opcode = None
            with open(fpath, 'rb') as fp:
                try:
                    check_opcode = self.mergelist
                except:
                    mergepkl = str(extension + '\\' + "mergelist.pkl")
                    with open(mergepkl, 'rb') as f:
                        self.mergelist = pickle.load(f)
                    check_opcode = self.mergelist
                opcodecounts = {}  # 요소별로 카운트를 저장할 딕셔너리
                check_opcode = self.add_numbers_to_duplicates(check_opcode, opcodecounts)

            # 파일 경로에서 파일 이름 추출
            file_name = os.path.basename(fpath)
            exit_loop = False

            for file_info in self.ngrams_list:

                _, ngrams = file_info
                _ = os.path.basename(_)
                if file_name == _:  # 파일 이름과 fpath의 파일 이름을 비교
                    res.append(('name', os.path.basename(fpath)))

                    # 데이터 추출
                    count = 0
                    tempvalue = ''
                    mvalue = 0
                    for i in range(len(check_opcode[0]) + 1):
                        for j in range(mvalue, len(ngrams)):
                            try:
                                nowvalue = ngrams[j]
                                headerfeat = check_opcode[0][i]

                                if '_' in headerfeat :
                                    headerfeatemp = headerfeat[:-2]
                                else :
                                    headerfeatemp = headerfeat
                                k = 8
                                m = j
                                if nowvalue in headerfeatemp:

                                    while len(nowvalue) < len(headerfeatemp) and (j + k) < len(ngrams):
                                        temp1 = ngrams[m + k]
                                        nowvalue += temp1[0]
                                        k += 1

                                if nowvalue == headerfeatemp:
                                    count += 1
                                elif count != 0:
                                    lennowvalue = 8
                                    testvalue = j + lennowvalue
                                    for kn in range(testvalue, testvalue + (lennowvalue * 2), 8):
                                        temp2 = ngrams[kn]  # 수정된 부분
                                        tempvalue += temp2

                                    res.append((headerfeat, tempvalue))
                                    tempvalue = ''
                                    count = 0
                                    mvalue = j + 1
                                    break

                                if j == len(ngrams) - 1 and tempvalue == '':
                                    res.append((headerfeat, '0'))
                                    break
                                if len(res) == len(check_opcode[0]):
                                    exit_loop = True

                            except Exception as e:
                                pass
                            if exit_loop:
                                return res

                    if exit_loop:
                        return res
                if exit_loop :
                    return res

            return res
        def extract_rengram(self, result):
            self.intersection_lists = []

            result_set = set(result)
            for name, ngram in self.ngrams_list:
                intersection_list = [onegram for onegram in ngram if onegram in result_set]
                self.intersection_lists.append(intersection_list)

            return self.intersection_lists

        def duplication_count(self):

            self.data_list = []
            self.newlist = []
            duplicates = []

            element_count = {}

            # 모든 리스트에서 요소의 출현 횟수를 카운트
            for k in range(len(self.ngrams_list)):
                for lst in self.ngrams_list[k][1]:
                        if lst in element_count:
                            element_count[lst] += 1
                        else:
                            element_count[lst] = 1

            # 출현 횟수가 2번 이상인 요소만 선택
            basenum = int(len(self.ngrams_list)*0.6)

            self.newlist = [key for key, value in element_count.items() if value >= basenum]

            #중복이 없는 교집합 리스트
            commonlistpkl = str(self.extension + '\\' + "commonlist.pkl")
            with open(commonlistpkl, "wb") as fw: #
                pickle.dump(self.newlist, fw)

        def add_string_if_not_exists(self, filename, target_string):
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    content = file.read()

                # 없으면 추가
                if target_string not in content:
                    with open(filename, 'a', encoding='utf-8') as file:
                        file.write(target_string)

            except Exception as e :
                with open(filename, 'a', encoding='utf-8') as file:
                    file.write(target_string)

        def feature_dictionary(self, hexa, extension):

            array10 = []

            dictpkl = str(extension + '\\' +  "_dict.pkl")
            with open(dictpkl, 'rb') as f:
                newdict = pickle.load(f)

            result = hexa

            for key, value in newdict.items():
                if value in result[1]:
                    array10.append(str(key))

            sequencedem = ", ".join(array10)
            sequencedem = self.simhash(sequencedem)

            return sequencedem


        def save_lists_of_10_to_csv(self, data_list, file_name):
            with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)

                row = [j for j in range(1, len(data_list[0])+1)]
                csv_writer.writerow(row)

                for row in data_list:
                    csv_writer.writerow(row)

        # value로 key찾기
        def find_key_by_value(self, dictionary, value):
            for key, val in dictionary.items():
                if val == value:
                    return key
            return None  # 해당 값과 일치하는 키가 없을 경우 None을 반환

        # Feature 딕셔너리 업데이트 or 딕셔너리 추가
        # 기존 딕셔너리 없으면 생성, 있으면 업데이트
        def save_lists_of_10_to_csv_featuredict(self, data_list):
            data_list = data_list.split(",")
            data_list.remove('name')
            data_set = list(dict.fromkeys(data_list).keys())
            newdict= {}

            try:
                #기존 딕셔너리 사전 열기
                dictpkl = str(self.extension + '\\' + "_dict.pkl")
                with open(dictpkl, "wb") as fw:
                    newdict = pickle.load(fw)

                #값이 기존 딕셔너리 value에 존재하지 않으면 추가
                if newdict:
                    last_key = max(newdict.keys())
                else:
                    last_key = 0

                for idata in data_set:
                    if idata not in newdict.values():
                        new_key = last_key + 1
                        newdict[new_key] = idata

                dictpkl = str(self.extension + '\\' + "_dict.pkl")
                with open(dictpkl, "wb") as fw:
                    pickle.dump(newdict, fw)

            except Exception as e:

                for i, item in enumerate(data_set, start=1):
                    newdict[i] = item

                dictpkl = str(self.extension + '\\' + "_dict.pkl")
                with open(dictpkl, "wb") as fw:
                    pickle.dump(newdict, fw)

            self.makearray(data_list, newdict)

        def make_features(self, input_str):
            length = 3
            input_str = input_str.lower()
            out_str = re.sub(r'[^\w]+', '', input_str)
            return [out_str[i:i + length] for i in range(max(len(out_str) - length + 1, 1))]
        def simhash(self, input_str):
            features = self.make_features(input_str)
            return Simhash(features).value
        def headersimhash(self, input_list):
            string_result = ''.join(map(str, input_list))
            self.simhash(string_result)
        def makearray(self, featurelist, newdict):
            newlist2 = []
            newlist = []
            # 피쳐를 딕셔너리 사전의 10진수값에 매핑
            for item in featurelist:
                a = newdict.values()
                if item in newdict.values():
                    newlist.append(self.find_key_by_value(newdict, item))

            newlist2 =[self.file_paths]
            newlist2.append(newlist)

            pklname = str(self.extension + '\\' + "vectordb.pkl")
            with open(pklname, "wb") as fw:
                pickle.dump(newlist2, fw)

        # 헤더를 csv에 저장
        def save_list_of_indivi_to_csv(self, data_list, file_name):

            with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                sub_strings = data_list.split(',')
                csv_writer.writerow(sub_strings)
        def file_exists(self, folder_path, filename):
            file_path = os.path.join(folder_path, filename)
            return os.path.isfile(file_path)

        def open_csv(self, file_path):
            answer = messagebox.askyesno("CSV 생성 완료", f"{csv_file}를 열겠습니까?")
            if answer:
                os.startfile(self, file_path)

        def input_label(self):
            self.label_data = simpledialog.askstring("입력", "라벨 데이터를 입력하세요.")


            if self.label_data is None:
                return
            try:
                number = float(self.label_data)
            except ValueError:
                messagebox.showerror("에러", "유효한 숫자를 입력해주세요.")
                return

            with open(self.label_datacsv, mode = 'a', newline = '') as file:
                aaa = self.fetch_name_from_csv(self.label_data)
                if aaa == None :
                    self.name = simpledialog.askstring("입력", "매핑되는 속성을 입력하세요")
                    writer = csv.writer(file)
                    writer.writerow([self.label_data, self.name])

            return

        def fetch_name_from_csv(self, max_key):
            filename = self.label_datacsv
            with open(filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == max_key:
                        return row[1]
            return None


        def extract_value_tocsv(self, choice):
            x = self.get_files_value()
            y = x[0]
            second_elements = [tpl[0] for tpl in y]
            header = ','.join(second_elements)

            if choice == 1:
                extractvalue = str(self.extension + '\\' +  "extractvalues_header.csv")  # 헤더추출용
                commonheader2csv = str(self.extension + '\\' +  "common2_header.csv")
                self.save_list_of_indivi_to_csv(header, commonheader2csv)
                with open(extractvalue, 'wt', encoding='utf-8') as fp:
                    fp.write(header + '\n')

            elif choice >= 2:  # 인풋파일들에 대한 value 추출
                if choice == 4:
                    extractvalue = str(self.extension + '\\' + "detectextractvalues.csv")
                else :
                    if self.binButton.isChecked():
                        extractvalue = os.path.join(self.extension, "extractvalues_bin.csv")
                        self.label_datacsv = str(self.extension + '\\' + 'labeldata_bin.csv')
                    elif self.mulButton.isChecked():
                        extractvalue = os.path.join(self.extension, "extractvalues_mul.csv")
                        self.label_datacsv = str(self.extension + '\\' + 'labeldata_mul.csv')

                written_names = set()
                self.isfilecsv = 0
                if 'label' not in second_elements:
                    header += ',label'
                for i in range(len(x)):
                    x[i].append(('label', self.label_data))


                existing_names = set()
                try:
                    #파일이 있다면
                    with open(extractvalue, 'r', newline='', encoding='utf-8') as csvfile:
                        csv_reader = csv.reader(csvfile)

                        # header 처리
                        header = next(csv_reader)
                        name_index = header.index('name') if 'name' in header else None

                        if name_index is not None:
                            # 기존 데이터에서 name 추출
                            for row in csv_reader:
                                if len(row) > name_index:
                                    existing_names.add(row[name_index])

                        header = ','.join(header)
                        csvfile.close()


                except Exception as e :
                    self.isfilecsv += 1

                if choice != 4:
                    with open(extractvalue, 'a', encoding='utf-8') as fp:
                        if self.isfilecsv != 0:
                            fp.write(header + '\n')

                        # 새로운 데이터 중에서 name이 없는 경우에만 추가
                        for data_row in x:
                            name_value = data_row[0][1]

                            if name_value not in existing_names:
                                data = ','.join([tpl[1] for tpl in data_row])
                                fp.writelines(data + '\n')
                                existing_names.add(name_value)
                                print(f"새로운 데이터 : {data_row}")
                            else:

                                print(f"이미 존재함 : {data_row}")
                    # fp.write(header + '\n')
                    # preserve_dict = {}  # 딕셔너리를 사용하여 name_value가 이미 있는 행을 보존
                    # for k in range(len(x)):
                    #     name_value = x[k][0][1]  # Assuming the name is the first tuple's second value
                    #     if name_value in preserve_dict:
                    #         # 이미 있는 경우 해당 행을 가져와서 쓰기
                    #         fp.writelines(preserve_dict[name_value] + '\n')
                    #     else:
                    #         data = ','.join([tpl[1] for tpl in x[k]])
                    #         fp.writelines(data + '\n')
                    #         preserve_dict[name_value] = data  # name_value 행을 딕셔너리에 저장
            if choice != 4:
                self.save_lists_of_10_to_csv_featuredict(header)
            return header

        def center_window(self, root, width=300, height=200):
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            x = (screen_width / 2) - (width / 2)
            y = (screen_height / 2) - (height / 2)

            root.geometry('%dx%d+%d+%d' % (width, height, x, y))

        def extract_middle_text(self, filepath):
            # Regular expression pattern to match the filename and extract the middle text
            filename = os.path.basename(filepath)
            pattern = re.compile(r'^\d+([a-zA-Z]+)[_\d]*\.pkl$')
            match = pattern.match(filename)

            if match:
                return match.group(1)
            else:
                return None
        def merge_and_save_pkl(self, data, pkl_path, directory, max_items_per_file=1000,
                               max_size_in_bytes=1 * 1024 * 1024* 256):  # 1GB
            latest_files = None

            # 파일이 존재하는 경우 기존 데이터 로드


            middle_text = self.extract_middle_text(pkl_path)
            if not middle_text:
                print(f"Unable to extract middle text from {pkl_path}")
                return []

            pattern = re.compile(rf'^{self.label_data}+{middle_text}_(\d+)\.pkl$')

            files_with_numbers_and_values = []
            for filename in os.listdir(directory):
                match = pattern.match(filename)
                if match:
                    number_part = match.group(1)
                    if number_part:
                        number = int(number_part)  # Extract the number part and convert to int
                        files_with_numbers_and_values.append((filename, number))

            # 숫자 기준으로 파일명 정렬
            files_with_numbers_and_values.sort(key=lambda x: x[1])

            # 마지막 숫자를 가진 파일명 리스트 반환
            if files_with_numbers_and_values:
                latest_number = files_with_numbers_and_values[-1][1]
                latest_files = [f[0] for f in files_with_numbers_and_values if f[1] == latest_number]

            else:
                return []

            is_exist = os.path.exists(directory + latest_files[0])


            if is_exist:
                with open(directory + latest_files[0], 'rb') as f:

                    existing_data = pickle.load(f)

                f.close()


            else:
                existing_data = []
                existing = 0

            # 기존 데이터와 새로운 데이터 병합
            existing_file_names = {item[0] for item in existing_data}

            # 새로운 데이터를 필터링합니다.
            new_items = [item for item in data if item[0] not in existing_file_names]
            combined_data = existing_data + new_items

            # 데이터를 파일당 max_items_per_file 개수로 나누어 저장
            base, ext = os.path.splitext(pkl_path)

            #file_index = 1


            #1inputngram_1 에서 _1떼네기


            # 매칭된 부분 추출
            base_root = base.rsplit('_', 1)[0]
            file_index = int(base.rsplit('_', 1)[1])

            # 기존 파일의 인덱스를 확인하여 시작 인덱스를 설정
            # while os.path.exists(f"{base_root}{ext}"):
            #     file_index += 1


            # 분할 저장
            new_part=[]
            accessible_items = existing_data
            accessible = 1
            count = 0
            size_accessible_items_size = sys.getsizeof(pickle.dumps(existing_file_names))
            part_file_path = f"{self.label_data}{middle_text}_{file_index}{ext}"
            for item in new_items:
                count += 1
                if os.path.exists(item[0]):
                    with open(item[0], 'rb') as file:
                        file_contents = file.read()
                    f.close()
                    accessible_items.append(item)
                    size_accessible_items_size = sys.getsizeof(pickle.dumps(accessible_items))

                    if size_accessible_items_size < max_size_in_bytes:
                        with open(self.extension + '\\' +  part_file_path, "wb") as f:
                            pickle.dump(accessible_items, f)
                            f.close()
                    else :
                        accessible_items = []
                        accessible_items.append(item)
                        file_index += 1

                        with open(part_file_path, "wb") as f:
                            pickle.dump(accessible_items, f)
                            f.close()

                        new_part = []


                #new_part.append(item)
                with open(part_file_path, "wb") as f:

                     pickle.dump(accessible_items, f)
                     f.close()

            return combined_data


        def main(self):
            self.ngrams = []
            while True:

                choice = self.choice

                folder_path = os.getcwd()  # 폴더 경로
                filename = 'lcsdata.pkl'  # 확인하고 싶은 파일 이름

                a = self.file_exists(folder_path, filename)

                if choice == 1: #기준 피처를 만들기 위함, 10개 이내의 파일로 파일형식의 피처 생성
                    print("1클릭")
                    os.chdir("E:\\AiFileDetectorE")
                    self.extension = (self.file_paths[0].split('.'))[1]
                    if not os.path.exists(self.extension):
                        os.mkdir(self.extension)
                    if a==False: #lcs pkl 파일이 없으면 생성
                        if self.extension == 'jpg' :
                            n = 6
                        else :
                            n = 8
                        self.extract_ngram(n, self.file_paths)
                        pklname = os.path.join(self.extension, "ngramlist.pkl")
                        with open(pklname, "wb") as fw:
                            pickle.dump(self.ngrams_list, fw)

                        hexlistpkl = str(self.extension + '\\' +  "hexlist.pkl")
                        with open(hexlistpkl, "wb") as fw:
                            pickle.dump(self.hex_lists, fw)


                        self.duplication_count()
                        self.extract_rengram(self.newlist)
                        result = self.lcs_multiple_lists(self.intersection_lists)

                        lcsdatapkl = str(self.extension + '\\' +  "lcsdata(2).pkl")
                        with open(lcsdatapkl, "wb") as fw:
                            pickle.dump(result, fw)

                    else: #lcs pkl 파일이 있으면 열기
                        lcsdatapkl = str(self.extension + '\\' +  "lcsdata(2).pkl")
                        with open(lcsdatapkl, 'rb') as f:
                            result = pickle.load(f)

                    self.mergelist = self.merge_lists2(result)

                    mergepkl = (self.extension + '\\' +  "mergelist.pkl")
                    with open(mergepkl, "wb") as fw:
                        pickle.dump(self.mergelist, fw)
                        fw.close()
                    path = '/base'

                    commonheadercsv = (self.extension + '\\' +  "common_header.csv")
                    self.save_lists_of_10_to_csv(self.mergelist, commonheadercsv)
                    print(f"{filename} exists in {folder_path}")

                    #헤더딕셔너리(기존딕셔너리에 없으면 추가하기 위함)
                    header = self.extract_value_tocsv(choice)
                    headersave = header.replace('name,', '')
                    filename = str(self.extension+ 'header.txt')
                    self.add_string_if_not_exists(filename, headersave)
                    messagebox.showinfo("Notification", "Learning data extraction has been completed")

                    break

                elif choice == 2: #Create Value
                    print("2클릭")

                    while True:
                        os.chdir("E:\\AiFileDetectorE")

                        self.extension = self.file_paths[0].split('.')[-1]

                        if self.binButton.isChecked():
                            extractvalue = os.path.join(self.extension, "extractvalues_bin.csv")
                            self.label_datacsv = str(self.extension + '\\' + 'labeldata_bin.csv')
                        elif self.mulButton.isChecked():
                            extractvalue = os.path.join(self.extension, "extractvalues_mul.csv")
                            self.label_datacsv = str(self.extension + '\\' + 'labeldata_mul.csv')

                        root = tk.Tk()
                        root.title("데이터 입력")
                        self.center_window(root)

                        button = tk.Button(root, text="버튼을 클릭하세요", command=self.input_label)
                        button.pack(pady=20)
                        root.mainloop()

                        if not os.path.exists(self.extension):
                            os.mkdir(self.extension)

                        if self.extension == 'jpg':
                            n = 6
                        else:
                            n = 8
                        mergepkl = str(self.extension  + '\\' +  "mergelist.pkl")
                        with open(mergepkl, 'rb') as f:
                            self.mergelist = pickle.load(f)

                        self.extract_ngram(n, self.file_paths)
                        base_directory = os.path.join(self.extension, "")

                        matchvalue = 0

                        inputngramspkl = str(
                            "E:\\AiFileDetectorE\\" + self.extension + '\\' + str(self.label_data) + "inputngrams.pkl")

                        pattern =  re.compile(rf'^{self.label_data}+inputngrams_(\d+)\.pkl$')  # 숫자로 시작하고 'hexlist_'로 이어지며 '.pkl'로 끝나는 파일 이름 패턴
                        processed_files = set()

                        for filename in os.listdir(base_directory):
                            if pattern.match(filename) and filename not in processed_files:
                                file_path = os.path.join(base_directory, filename)
                                print(f"Processing file: {file_path}")  # 파일 처리 정보 출력

                                self.ngrams_list = self.merge_and_save_pkl(self.ngrams_list, file_path, base_directory)
                                matchvalue = 1
                            else:
                                if filename in processed_files:
                                    print(f"Skipping already processed file: {filename}")
                                    processed_files.add(filename)

                                    break  # 폴더 검사 종료
                                else:
                                    print(f"File does not match pattern or is not found: {filename}")
                                    processed_files.add(filename)

                        #if inputngramspkl not in processed_files:
                            #self.ngrams_list = self.merge_and_save_pkl(self.ngrams_list, inputngramspkl, base_directory)

                        if matchvalue == 0:
                            inputthepkl = str("E:\\AiFileDetectorE\\" + self.extension + '\\' + str(self.label_data) + "inputngrams_1.pkl")

                            with open(inputthepkl, 'wb') as f:
                                pickle.dump(self.ngrams_list, f)
                                f.close()

                        matchvalue = 0
                        pattern =  re.compile(rf'^{self.label_data}+hexlist_(\d+)\.pkl$')# 숫자로 시작하고 'hexlist_'로 이어지며 '.pkl'로 끝나는 파일 이름 패턴
                        processed_files = set()

                        for filename in os.listdir(base_directory):
                            if pattern.match(filename) and filename not in processed_files:
                                file_path = os.path.join(base_directory, filename)
                                print(f"Processing file: {file_path}")  # 파일 처리 정보 출력

                                self.hex_lists = self.merge_and_save_pkl(self.hex_lists, file_path, base_directory)
                                matchvalue = 1
                            else:
                                if filename in processed_files:
                                    print(f"Skipping already processed file: {filename}")
                                    processed_files.add(filename)

                                    break  # 폴더 검사 종료
                                else:
                                    print(f"File does not match pattern or is not found: {filename}")
                                    processed_files.add(filename)

                        if matchvalue == 0:
                            inputthepkl = str("E:\\AiFileDetectorE\\" + self.extension + '\\' + str(self.label_data) + "hexlist_1.pkl")

                            with open(inputthepkl, 'wb') as f:
                                pickle.dump(self.hex_lists, f)
                                f.close()

                        self.extract_value_tocsv(choice)
                        messagebox.showinfo("Notification", "Learning data extraction has been completed")

                        break

                elif choice == 3: #Sequence
                    os.chdir("E:\\AiFileDetectorE")
                    self.file_paths = self.listWidget  # 파일 경로 리스트 가져오기
                    self.extension = self.comboBox.currentText()
                    self.extension = self.extension[1:]

                    self.sequencedem = []

                    selected_option = None
                    if self.binButton.isChecked():
                        extractvalue = os.path.join(self.extension, "extractvalues_bin.csv")
                        self.label_datacsv = str(self.extension + '\\' + 'labeldata_bin.csv')
                    elif self.mulButton.isChecked():
                        extractvalue = os.path.join(self.extension, "extractvalues_mul.csv")
                        self.label_datacsv = str(self.extension + '\\' + 'labeldata_mul.csv')

                    df = pd.read_csv(extractvalue)

                    # 저장된 모든 .pkl 파일 로드
                    hex_values_combined = []
                    base_directory = os.path.join(self.extension, "")
                    pattern = re.compile(r'\d+hexlist_.*\.pkl$')  # 숫자로 시작하고 'hexlist_'로 이어지며 '.pkl'로 끝나는 파일 이름 패턴
                    processed_files = set()

                    for filename in os.listdir(base_directory):
                        if pattern.match(filename) and filename not in processed_files:
                            file_path = os.path.join(base_directory, filename)
                            print(f"Processing file: {file_path}")  # 파일 처리 정보 출력
                            with open(file_path, 'rb') as f:
                                hex_values = pickle.load(f)
                                hex_values_combined.extend(hex_values)  # 데이터 병합
                             # 파일 처리 기록
                        else:
                            if filename in processed_files:
                                print(f"Skipping already processed file: {filename}")
                                processed_files.add(filename)
                                break  # 폴더 검사 종료
                            else:
                                print(f"File does not match pattern or is not found: {filename}")
                                processed_files.add(filename)


                    # 병합된 데이터로 작업 수행
                    try:
                        for hex_value in hex_values_combined:
                            sequence = self.feature_dictionary(hex_value, self.extension)
                            self.sequencedem.append((hex_value[0], sequence))
                    except Exception as e:
                        print("Error processing hex values:", e)
                        pass

                    filename_to_sequence = {}
                    for path, value in self.sequencedem:
                        filename = os.path.basename(path)
                        filename_to_sequence[filename] = value

                    # 데이터프레임 업데이트
                    for index, row in df.iterrows():
                        if row['name'] in filename_to_sequence:
                            df.at[index, 'sequence'] = filename_to_sequence[row['name']]

                    # 열 위치 조정
                    columns = list(df.columns)
                    seq_index = columns.index('sequence')
                    label_index = columns.index('label')
                    columns[seq_index], columns[label_index] = columns[label_index], columns[seq_index]
                    df = df[columns]
                    df.to_csv(extractvalue, index=False)
                    root = ThemedTk(theme="equilux")

                    root.withdraw()

                    # 메시지 박스 표시 후, 메인 창을 닫음
                    messagebox.showinfo("Notification", "Learning data extraction has been completed")

                    # 메시지 박스 닫힐 때 root 창도 닫히도록 설정
                    root.protocol("WM_DELETE_WINDOW", self.on_closing(root))

                    # 이벤트 루프 시작 (이건 메시지 박스가 닫힐 때까지 기다리는 데 사용됨)
                    root.mainloop()
                    break
                elif choice ==5:

                    print("종료")
                    break

    if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = createtrainclass()
        ex.show()
        app.exec_()
