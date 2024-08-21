import csv
import os
import pickle
import re
import struct
import sys
from tkinter import messagebox
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import tensorflow as tf
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QProgressBar, QDialog, QLabel, \
    QVBoxLayout
from PyQt5 import uic, QtWidgets
from tensorflow.python.client import device_lib
from simhash import Simhash

device_lib.list_local_devices()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
form_class = uic.loadUiType("UI_Design\\AiDetector2.ui")[0]

with tf.device('/GPU:0'):
    class ProgressWindow(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("파일 처리 진행 상황")

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setGeometry(80, 50, 250, 20)

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

            self.setupUi(self) # UI 요소 초기화

            # 확장자 필터
            self.extension_list = [".mp4", ".png", ".jpg", ".pdf", ".m4a"]
            self.comboBox.addItems(self.extension_list)
            self.comboBox.currentIndexChanged.connect(self.filter_files_by_extension)

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setGeometry(170, 30, 250, 20)

            # self.progress_bar2 = QProgressBar(self)
            # self.progress_bar2.setGeometry(50, 50, 250, 20)
            # 파일시스템 트리
            self.dirModel = QFileSystemModel()
            self.dirModel.setRootPath(QDir.rootPath())
            self.treeView.setModel(self.dirModel)

            self.treeView.setRootIndex(self.dirModel.index(os.getcwd()))
            self.treeView.clicked.connect(self.file_selected)

            self.feature_create1.clicked.connect(lambda: setattr(self, 'choice', 1))
            self.create_value2.clicked.connect(lambda: setattr(self, 'choice', 2))
            self.create_sequence3.clicked.connect(lambda: setattr(self, 'choice', 3))

            self.LoadButton.clicked.connect(self.main) # Load 버튼 클릭 시 self.main() 호출

            # 파일 목록에서 아이템을 더블 클릭할 때 호출되는 슬롯을 연결합니다.
            self.listWidget.itemDoubleClicked.connect(self.remove_selected_file)

        def load_directory(self): # 디렉토리 선택
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
            if directory:
                self.dirModel.setRootPath(directory)
                self.treeView.setRootIndex(self.dirModel.index(directory))
                self.treeView.clicked.connect(self.file_selected)

        def filter_files_by_extension(self, extension): # 선택된 확장자에 따라 필터링
            if extension:
                self.extension = extension[1:]
                self.dirModel.setNameFilters([f"*{extension}"])
                self.dirModel.setNameFilterDisables(False)
            else:
                self.dirModel.setNameFilters([])

        def file_selected(self, index): # 파일 또는 디렉토리 선택 시 호출
            file_info = self.dirModel.fileInfo(index)
            if file_info.isDir():  # If a directory is selected
                self.select_all_files_in_directory(file_info.absoluteFilePath())
            else:
                file_path = file_info.absoluteFilePath()
                extension = os.path.splitext(file_path)[1]
                self.filter_files_by_extension(extension)
                if file_path not in self.file_paths:
                    self.listWidget.addItem(file_path)
                    self.file_paths.append(file_path)

        def select_all_files_in_directory(self, directory_path): # 선택한 디렉토리의 모든 파일을 선택
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path not in self.file_paths:
                        self.listWidget.addItem(file_path)
                        self.file_paths.append(file_path)

        def remove_selected_file(self, item): # 선택한 파일을 목록에서 제거
            # 더블 클릭한 파일 아이템을 목록에서 제거합니다.
            file_path = item.text()
            self.listWidget.takeItem(self.listWidget.row(item))
            self.file_paths.remove(file_path)

        def load_common(self): # 특정 파일에서 데이터 불러오기
            self.data_list2 = []
            self.newlist2 = []

            # 파일 열기
            with open('text2.txt', 'r') as file:
                for line in file:
                    cleaned_line = line.strip('')[1:-1]
                    sub_list = cleaned_line.split(',')
                    self.data_list2.append(sub_list)

            for a in sub_list:
                b = a.strip('" \'')
                self.newlist2.append(b)

            print(self.newlist2)

        def extract_ngram(self, n, file_paths): # 바이너리 데이터를 n-gram으로 변환하여 n크기 피처 추출
            ngram_sets = []
            self.ngrams_list = []
            self.hex_lists = []

            for file_path in file_paths:
                with open(file_path, 'rb') as file:
                    content = file.read()
                    self.hex_values = content.hex()
                    ngrams = []

                    for i in range(len(self.hex_values) - n + 1): # hex값 돌며 n-gram 추출
                        ngram = self.hex_values[i:i + n]
                        ngrams.append(ngram)

                # 파일 이름과 헥사 값 저장
                self.hex_lists.append((file_path, self.hex_values))

                # 파일 이름과 ngram 저장
                self.ngrams_list.append((file_path, ngrams))
                # [(file_path_1, [ngram_1, ngram_2, ngram_3, ...]),
                # (file_path_2, [ngram_1, ngram_2, ngram_3, ...]), ...]

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

        def lcs_multiple_lists(self, lists): # 가장 긴 공통 서브시퀀스 찾기
            if len(lists) < 2: # 리스트가 2개 이상 있어야 LCS 찾기 가능
                raise ValueError("At least two lists are required")

            current_lcs = self.lcs(lists[0], lists[1]) # 처음 두 리스트의 LCS 계산

            for lst in lists[2:]: # 나머지 리스트와의 LCS 반복 계산
                current_lcs = self.lcs(current_lcs, lst)
                if not current_lcs:
                    return []

            return current_lcs
            # lists = [
            #     ['a', 'b', 'c', 'd', 'e'],
            #     ['b', 'c', 'e', 'f', 'g'],
            #     ['c', 'e', 'g', 'h', 'i']
            # ]

        def process_files(self):
            progress_window = ProgressWindow()
            progress_window.setModal(True)
            progress_window.show()

            total_files = len(self.file_paths)

            for i, fname in enumerate(self.file_paths):
                full_path = fname
                if os.path.isfile(full_path):
                    self.extract_value(full_path)
                    self.all_result.append(self.reres)

                # 진행 상황 업데이트
                progress_percentage = (i + 1) / total_files * 100
                self.progress_bar.setValue(progress_percentage)
                QApplication.processEvents()

            progress_window.set_label_text("작업 완료")
            progress_window.exec_()

        def get_files_value(self): # self.file_paths의 모든 파일 처리하고 self.all_result 저장 및 반환
            results = {}
            self.all_result = []
            self.count = -1
            # 폴더 내 모든 파일에 대해 수행
            for i, fname in enumerate(self.file_paths):
                self.count += 1
                full_path = fname

                if os.path.isfile(full_path):
                    self.extract_value(full_path)
                    self.all_result.append(self.reres)

                # 진행 상황 업데이트
                progress_percentage = (i + 1) / len(self.file_paths) * 100
                self.progress_bar.setValue(progress_percentage)
                QApplication.processEvents()

            # 파일 처리가 완료되면 최장 리스트를 찾음
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

            return self.all_result

        def merge_lists2(self, ngram): # LCS 받아서 연속되거나 유사한 n-gram 병합하여 하나의 긴 패턴으로 생성
            count, count2, onecount = 0,0,0
            new_list = []
            merged_list=[]
            one_merged_list = []
            previous_gram = ''
            for onegram1 in ngram :

                if previous_gram == onegram1 and onegram1 != '00000000': # 00000000 이면 제외
                    one_merged_list.append(onegram1)
                    onecount += 1
                    pass

                else : # 현재 n-gram과 이전 n-gram 마지막 부분이 일치하는지 확인
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
                    if onecount == len(ngram)-1: # 리스트의 마지막 n-gram 처리
                        if previous_gram != '00000000':
                            one_merged_list.append(previous_gram)
                        break

                    onecount += 1

            merged_list.append(one_merged_list)
            return merged_list

        def add_numbers_to_duplicates(self, input_list):
            counts = {}  # 요소별로 카운트를 저장할 딕셔너리

            for i in range(len(input_list[0])):
                item = input_list[0][i]

                # 이미 등장한 요소인 경우
                if item in counts:
                    counts[item] += 1
                    input_list[0][i] = f"{item}_{counts[item]}"
                else:
                    counts[item] = 0

            return input_list

        def extract_value(self, fpath): # 파일에서 n-gram 추출하고 병합된 n-gram 리스트와 비교하여 일치하는 패턴 찾아 res 리스트에 저장
            file_type = ""
            res = []
            self.reres = []

            # 파일 경로에서 파일 이름 추출
            file_name = os.path.basename(fpath)

            for file_info in self.ngrams_list:
                _, ngrams = file_info
                _ = os.path.basename(_)
                if file_name == _:  # 파일 이름과 fpath의 파일 이름을 비교
                    res.append(('name', os.path.basename(fpath)))

                    with open(fpath, 'rb') as fp:
                        check_opcode = self.mergelist
                        check_opcode = self.add_numbers_to_duplicates(check_opcode)

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
                                            temppp = ngrams[m + k]
                                            nowvalue += temppp[0]
                                            k += 1

                                    if nowvalue == headerfeatemp:
                                        count += 1
                                    elif count != 0:
                                        lennowvalue = len(nowvalue)
                                        testvalue = j + lennowvalue
                                        for kn in range(testvalue, testvalue + (lennowvalue * 2), 8):
                                            temppppp = ngrams[kn]  # 수정된 부분
                                            tempvalue += temppppp

                                        res.append((headerfeat, tempvalue))
                                        tempvalue = ''
                                        count = 0
                                        mvalue = j + 1
                                        break

                                    if j == len(ngrams) - 1 and tempvalue == '':
                                        res.append((headerfeat, '0'))

                                        break

                                except Exception as e:
                                    pass

            self.reres = res
            return res

        def extract_rengram(self, result): # self.ngrams_list와 result 간 교집합을 계산해 각 파일별로 공통된 n-gram들을 찾아냄
            self.intersection_lists = []

            result_set = set(result)
            for name, ngram in self.ngrams_list:
                intersection_list = [onegram for onegram in ngram if onegram in result_set]
                self.intersection_lists.append(intersection_list)

            return self.intersection_lists

        def find_duplicates_count(self): # ngrams_list에서 공통적으로 출현하는 요소 찾기

            self.data_list = []
            self.newlist = []
            duplicates = []

            element_count = {} # n-gram 요소 출현 횟수 저장 딕셔너리, 키: 요소, 값: 출현 횟수

            # 모든 리스트에서 요소의 출현 횟수를 카운트
            for k in range(len(self.ngrams_list)):
                for lst in self.ngrams_list[k][1]:
                        if lst in element_count:
                            element_count[lst] += 1
                        else:
                            element_count[lst] = 1

            # 출현 횟수가 2번 이상인 요소만 self.newlist에 저장
            basenum = int(len(self.ngrams_list)*0.7)
            self.newlist = [key for key, value in element_count.items() if value >= basenum]

            #중복이 없는 교집합 리스트를 commonlist.pkl에 저장
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

        def feature_dictionary(self, hexa):

            array10 = []

            dictpkl = str(self.extension + '\\' +  "_dict.pkl")
            with open(dictpkl, 'rb') as f:
                newdict = pickle.load(f)

            result = hexa

            for key, value in newdict.items():
                if value in result[1]:
                    array10.append(str(key))

            sequencedem = ", ".join(array10)
            sequencedem = self.simhash(sequencedem)

            self.sequencedem.append((hexa[0], sequencedem))

        def save_lists_of_10_to_csv(self, data_list, file_name): # 더 긴 패턴(mergelist 리스트) 주어진 데이터를 CSV로 저장
            with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)

                row = [j for j in range(1, len(data_list[0])+1)] # 헤더 작성
                csv_writer.writerow(row)

                for row in data_list:
                    csv_writer.writerow(row)
                    # 1,2,3
                    # abcdef,ghijkl,mnopqr
                    # stuvwx,yzabcd,efghij
                    # klmnop,qrstuv,wxyzab

        # value로 key찾기
        def find_key_by_value(self, dictionary, value):
            for key, val in dictionary.items():
                if val == value:
                    return key
            return None  # 해당 값과 일치하는 키가 없을 경우 None을 반환

        # Feature 딕셔너리 업데이트 or 딕셔너리 추가
        # 기존 딕셔너리 없으면 생성, 있으면 업데이트
        def save_lists_of_10_to_csv_featuredict(self, data_list): # header가 새로운 피처로 저장
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
            # 운영체제별 기본 CSV 뷰어를 사용하여 파일 열기
            answer = messagebox.askyesno("CSV 생성 완료", f"{csv_file}를 열겠습니까?")
            if answer:
                os.startfile(self, file_path)

        def on_button_click(self):
            self.label_data = simpledialog.askstring("입력", "라벨 데이터를 입력하세요.")
            if self.label_data is None:
                return
            try:
                number = float(self.label_data)
            except ValueError:
                messagebox.showerror("에러", "유효한 숫자를 입력해주세요.")
                return

            return

        def extract_value_tocsv(self, choice):
            x = self.get_files_value() # 정리된 데이터
            y = x[0]
            second_elements = [tpl[0] for tpl in y] # y에서 각 튜플의 첫 번째 요소 추출한 리스트
            header = ','.join(second_elements) # second_elemetns 리스트를 콤마로 연결한 문자열로, csv 파일 헤더로 사용됨

            if choice == 1:
                extractvalue = str(self.extension + '\\' +  "extractvalues_header.csv") # 헤더추출용
                commonheader2csv = str(self.extension + '\\' +  "common2_header.csv")
                self.save_list_of_indivi_to_csv(header, commonheader2csv) # common2_header.csv 저장
                with open(extractvalue, 'wt', encoding='utf-8') as fp: # extractvalues_header.csv 저장
                    fp.write(header + '\n')

            elif choice == 2:  # 인풋파일들에 대한 value 추출
                extractvalue = str(self.extension + '\\' +  "extractvalues.csv")
                #extractvalue = str("test.csv")
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

            self.save_lists_of_10_to_csv_featuredict(header)
            return header

        def center_window(self, root, width=300, height=200):
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            x = (screen_width / 2) - (width / 2)
            y = (screen_height / 2) - (height / 2)

            root.geometry('%dx%d+%d+%d' % (width, height, x, y))

        def merge_and_save_pkl(self, data, pkl_path):
            if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0:
                with open(pkl_path, 'rb') as f:
                    existing_data = pickle.load(f)
            else:
                existing_data = []

            # 기존 데이터와 새로운 데이터 병합
            existing_file_names = {item[0] for item in existing_data}
            new_items = [item for item in data if item[0] not in existing_file_names]
            combined_data = existing_data + new_items

            with open(pkl_path, 'wb') as f:
                pickle.dump(combined_data, f)

            return combined_data

        def extract_box_feature(self, file_paths):  # 기연 추가

            all_results = [] # 모든 파일 데이터 저장할 리스트

            for file_path in file_paths:
                results = [] # 각 파일 데이터 저장 리스트
                results.append(('name', os.path.basename(file_path))) # 파일명 열 추가
                print(results)

                # 파일 내 Box 파싱
                def parse_box(f, end_position, depth=0, max_depth=100):
                    if depth > max_depth:
                        print("최대 재귀 깊이 도달 에러")
                        return

                    while f.tell() < end_position:
                        box_header = f.read(8) # 첫 8Bytes Box 헤더
                        if len(box_header) < 8:
                            break

                        box_size, box_type = struct.unpack(">I4s", box_header) # size 4Bytes, type 4Bytes 추출
                        box_type = box_type.decode("utf-8")

                        if box_size == 0: # 파일의 끝까지 Box가 확장됨을 의미
                            break
                        elif box_size == 1: # 실제 크기는 다음 8Bytes에 저장됨
                            large_size = f.read(8)
                            box_size = struct.unpack(">Q", large_size)[0]

                        # 현재 Box의 끝 위치 계산 = 현재 포인터 위치 + (사이즈 - 헤더 8Bytes)
                        box_end_position = f.tell() + (box_size - 8)

                        # 컨테이너 Box 처리
                        if box_type in ('moov', 'trak', 'mdia', 'minf', 'stbl'):
                            parse_box(f, box_end_position, depth + 1, max_depth)  # 재귀 처리로 하위 Box 파싱
                        else: # 컨테이너가 아닌 Box 처리
                            box_data = f.read(box_size - 8)
                            box_data_hex = box_data.hex() # hex

                            print(f"Box Type: {box_type}, Box Size: {box_size}")
                            results.append((box_type, box_data_hex))

                        # 다음 Box로 이동
                        f.seek(box_end_position)

                with open(file_path, 'rb') as f:
                    file_size = f.seek(0, 2)  # 파일 끝으로 커서 옮겨서 파일 크기 계산
                    f.seek(0)  # 커서를 파일 시작 위치로 이동
                    parse_box(f, file_size) # 재귀

                all_results.append(results)  # 각 파일의 결과를 전체 리스트에 추가

            print(all_results) # 확인
            self.save_to_csv(all_results)

        # 기연 추가 - 결과를 CSV로 저장
        def save_to_csv(self, all_data):
            csv_file = 'box_features.csv'

            # 첫 번째 데이터에서 필드명을 추출하여 순서 유지
            if not all_data:
                return

            # 첫 번째 데이터셋에서 필드명(헤더)를 가져오고 순서 유지 .. test1에서만
            first_data = all_data[0]
            print(first_data) # [('name', 'testVideo1.mp4'), ('ftyp', '6d703432000002006d70343269736f6d'), ('mv
            fieldnames = [row[0] for row in first_data]
            print(fieldnames) # ['name', 'ftyp', 'mvhd', 'tkhd', 'edts', 'mdhd',

            # 다른 데이터셋에 있는 추가적인 필드명도 추출
            for data in all_data:
                for row in data:
                    if row[0] not in fieldnames:
                        fieldnames.append(row[0])

            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for data in all_data:
                    row_data = {row[0]: row[1] for row in data}
                    writer.writerow(row_data)

            print(f"Results saved to {csv_file}")

        def main(self):
            self.ngrams = []

            while True:
                choice = self.choice

                folder_path = os.getcwd()  # 폴더 경로
                filename = 'lcsdata.pkl'  # 확인하고 싶은 파일 이름

                a = self.file_exists(folder_path, filename) # lcsdata.pkl 확인

                if choice == 1: #기준 피처를 만들기 위함, 10개 이내의 파일로 파일형식의 피처 생성
                    print("1클릭")
                    self.extension = (self.file_paths[0].split('.'))[1]

                    #헤더딕셔너리(기존딕셔너리에 없으면 추가하기 위함)
                    """header = self.extract_value_tocsv(choice) # 헤더 추출해서 문자열로 반환
                    headersave = header.replace('name,', '') # header에서 name 문자열 제거한 결과 저장
                    filename = str(self.extension+ 'header.txt') # 헤더 정보 저장할 파일 경로, 이름
                    self.add_string_if_not_exists(filename, headersave)
                    messagebox.showinfo("Notification", "Learning data extraction has been completed")"""

                    break

                elif choice == 2:
                    print("2클릭", )
                    print("선택한 파일", self.file_paths)

                    self.extension = (self.file_paths[0].split('.'))[1]

                    # 파일 경로 전달해줘서, 추출하는 메소드 위에 작성 (기연 추가)
                    self.extract_box_feature(self.file_paths)

                    break

                elif choice == 3:
                    self.file_paths = self.listWidget
                    #self.extension = (self.file_paths[0].split('.'))[1]
                    #self.folderpath("inputfile2")
                    # 파일을 순서딕셔너리와 비교하여 1-2-3-4, 2-3-1-4 등 순서 리스트를 만들고, 이를 심해시화
                    self.extension = 'm4a'

                    self.sequencedem = []
                    hexlist = str(self.extension + '\\' +  "hexlist.pkl")
                    with open(hexlist, 'rb') as f:
                        hexvalues = pickle.load(f)
                    for h in range(len(hexvalues)):
                        self.feature_dictionary(hexvalues[h])

                    filename_to_sequence = {}
                    for path, value in self.sequencedem:
                        filename = os.path.basename(path)
                        filename_to_sequence[filename] = value

                    extractvalue = str(self.extension + '\\' +  "extractvalues.csv")
                    df = pd.read_csv(extractvalue)

                    for index, row in df.iterrows():
                        if row['name'] in filename_to_sequence:
                            df.at[index, 'sequence'] = filename_to_sequence[row['name']]

                    df.to_csv(extractvalue, index=False)
                    messagebox.showinfo("Notification", "Learning data extraction has been completed")
                    break

                elif choice ==5:
                    print("종료")
                    break

    if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = createtrainclass()
        ex.show()
        app.exec_()
