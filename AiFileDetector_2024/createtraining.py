import csv
import json
import os
import pickle
import queue
import re
import struct
import sys
import threading
from datetime import datetime
from tkinter import messagebox
import tkinter as tk
from tkinter import simpledialog, messagebox
from frame_compression import process_videos_in_folder
import joblib
import numpy as np
import pandas as pd
import pyautogui
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QProgressBar, QDialog, QLabel, \
    QVBoxLayout, QTableWidgetItem, QMessageBox, QLineEdit, QPushButton, QTableWidget, QInputDialog
from PyQt5 import uic, QtWidgets
from openpyxl.reader.excel import load_workbook
from openpyxl.workbook import Workbook
from simhash import Simhash

from Train_GRUprocess import twoTrainClass
from clustering1 import trainClustering
from Train_GRUprocess_multi import TrainClass
from extractframe_single import extractGOP
from extract_sps import parse_sps
from pps import analyzesps

os.environ["CUDA_VISIBLE_DEVICES"]="0"
form_class = uic.loadUiType("UI_Design\\new.ui")[0]

class ProgressWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("파일 처리 진행 상황")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(100, 50, 300, 20)

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
        self.clustering = trainClustering()
        self.trainclass = twoTrainClass()

        # 확장자 필터
        self.extension_list = ["확장자", ".mp4",  ".mov"]
        self.comboBox.addItems(self.extension_list)
        self.csv_file = ''
        self.comboBox.currentIndexChanged.connect(
            lambda index: self.filter_files_by_extension(self.comboBox.itemText(index)))

        # self.progress_bar2 = QProgressBar(self)
        # self.progress_bar2.setGeometry(50, 50, 250, 20)
        # 파일시스템 트리
        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())

        self.treeView.setModel(self.dirModel)

        input_thread = threading.Thread(target=self.ask_input)
        input_thread.start()
        input_thread.join(timeout=10)  # 10초 대기
        initialcode = 0
        self.detectmode = 0
        if initialcode == 0:
            try:
                print("출력: [", self.direc, "]")
                if self.direc == None or self.direc ==" ":
                    raise ValueError("")

            except Exception as e:

                self.direc = 'Y:\\'
                print("[nonepath] 설정값을 입력하지 않아 Y:\\로 설정되었습니다. "
                      "코드에서 수정을 원할 시에 createtraining에서 nonepath를 검색하여 이동해서 수정하세요")
            initialcode=1



        #self.treeView.setRootIndex(self.dirModel.index(os.getcwd()))
        self.treeView.setRootIndex(self.dirModel.index(self.direc))

        self.treeView.clicked.connect(self.file_selected)
        header = self.treeView.header()
        header.setSectionResizeMode(0, header.Interactive)
        header.resizeSection(0, 400)
        self.feature_create1.clicked.connect(lambda: setattr(self, 'choice', 1))
        self.create_value2.clicked.connect(lambda: setattr(self, 'choice', 2))
        self.create_sequence3.clicked.connect(lambda: setattr(self, 'choice', 3))

        self.structure_val_state = False
        self.structure_seq_state = False
        self.frame_sps_state = False
        self.frame_gop_state = False
        self.frame_ratio_state = False

        self.tabWidget.setCurrentIndex(0)

        self.structure_val_but.stateChanged.connect(self.on_structure_val_changed)
        self.structure_seq_but.stateChanged.connect(self.on_structure_seq_changed)
        self.frame_sps_but.stateChanged.connect(self.on_frame_sps_changed)
        self.frame_gop_but.stateChanged.connect(self.on_frame_gop_changed)
        self.frame_ratio_but.stateChanged.connect(self.on_frame_ratio_changed)


        self.LoadButton.clicked.connect(self.main) # Load 버튼 클릭 시 self.main() 호출
        self.cluster_train.clicked.connect(self.clustermain)
        self.class_train.clicked.connect(self.classmain)

        self.class_detect.clicked.connect(self.load_file_for_prediction)
        # 파일 목록에서 아이템을 더블 클릭할 때 호출되는 슬롯을 연결합니다.
        self.listWidget.itemDoubleClicked.connect(self.remove_selected_file)
        self.list_del.clicked.connect(self.remove_all_file)

        self.label_info.clicked.connect(self.open_data_entry_window)
        self.labelinfofile = "labeldata_bin.csv"
        self.load_excel_data()

        self.label_input_but.clicked.connect(self.input_label)
        self.aimodel = self.model_combo_2.currentText()
        self.trainindex = self.comboBox.currentIndex()



    def on_combobox_select(self, index):
        self.trainclass.index = index


    def clustermain(self):

        self.clustering.gotrain(self.csv_path)

    def classmain(self):
        binstat = self.binButton_3.isChecked()
        mulstat = self.mulButton_3.isChecked()
        if binstat:
            self.trainclass = twoTrainClass()
            self.classmode = 'bin_'
        elif mulstat:
            self.trainclass = TrainClass()
            self.classmode = 'mul_'
        else :
            messagebox.showerror("에러", "바이너리/멀티 모드를 선")

        self.model_combo.activated.connect(self.on_combobox_select)
        self.trainclass.csv_path = self.csv_path
        self.trainclass.comboBox = self.model_combo_2
        try:
            self.trainclass.gotrain(self.classmode, self.aimodel, self.trainindex, self.csv_path)
        except : pass
    def classdetect(self):
        self.detectclass.predict(file_path=self.file_paths[0])

    def load_directory(self): # 디렉토리 선택
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dirModel.setRootPath(directory)
            self.treeView.setRootIndex(self.dirModel.index(directory))
            self.treeView.clicked.connect(self.file_selected)

    def filter_files_by_extension(self, extension):  # 선택된 확장자에 따라 필터링
        if extension and extension != "확장자":  # Ensure it's not the placeholder text
            self.extension = extension  # Directly assign the selected extension
            self.dirModel.setNameFilters([f"*{extension}"])
            self.dirModel.setNameFilterDisables(False)
        else:
            self.dirModel.setNameFilters([])

    def file_selected(self, index): # 파일 또는 디렉토리 선택 시 호출
        file_info = self.dirModel.fileInfo(index)
        try:
            if file_info.isDir():  # If a directory is selected
                if self.extension :
                    self.select_all_files_in_directory(file_info.absoluteFilePath())
            else:
                file_path = file_info.absoluteFilePath()
                extension = os.path.splitext(file_path)[1]
                self.filter_files_by_extension(extension)
                if file_path not in self.file_paths:
                    if extension.lower() == self.extension.lower():
                        self.listWidget.addItem(file_path)
                        self.file_paths.append(file_path)
                if extension == '.csv':
                    self.csv_path = file_path
                    self.open_csv2(file_path, self.tableWidget)
        except Exception as e:
            self.show_alert(str(e))

    def ask_input(self):
        self.direc = input("데이터셋 폴더경로를 입력하세요: ")


    def display_dataframe(self, df, widgettype):
        widgettype.setRowCount(df.shape[0])
        widgettype.setColumnCount(df.shape[1])
        widgettype.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                widgettype.setItem(i, j, item)

    def open_csv2(self, csvfile, widgett):
        file_name = csvfile
        if file_name:
            try:
                df = pd.read_csv(file_name, encoding='UTF-8')
                self.display_dataframe(df, widgettype=widgett)
            except Exception as e:
                self.show_error_message("CSV 파일을 읽는 중 오류가 발생했습니다: " + str(e))

        return


    def select_all_files_in_directory(self, directory_path):
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    extension = os.path.splitext(file_path)[1][0:]

                    #같은 확장자만 담아야한다면 조건추가
                    if extension.lower() == self.extension:
                        if file_path not in self.file_paths:
                            self.listWidget.addItem(file_path)
                            self.file_paths.append(file_path)
        except Exception as e:
            self.show_alert(str(e))


    def remove_selected_file(self, item): # 선택한 파일을 목록에서 제거
        # 더블 클릭한 파일 아이템을 목록에서 제거합니다.
        file_path = item.text()
        self.listWidget.takeItem(self.listWidget.row(item))
        self.file_paths.remove(file_path)

    def remove_all_file(self):
        self.listWidget.clear()
        self.file_paths = []

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

        # 엑셀 파일에서 데이터 읽어오기 (엑셀 파일 경로를 설정해 주세요)
        excel_file = str(self.extension + '\\' + '_dict.xlsx')  # 엑셀 파일 경로
        df = pd.read_excel(excel_file)  # 엑셀 파일 읽기

        # 엑셀 데이터를 딕셔너리로 변환 (엑셀 파일의 첫 번째 열을 key로, 두 번째 열을 value로)
        newdict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        result = hexa

        # newdict의 value가 result[1]에 있으면 key를 array10에 추가
        for key, value in newdict.items():
            if value in result[1]:
                array10.append(str(key))

        # array10을 ","로 구분된 문자열로 만들어서 simhash 계산
        sequencedem = ", ".join(array10)
        sequencedem = self.simhash(sequencedem)

        # sequencedem과 hexa[0]을 함께 저장
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
        answer = messagebox.askyesno("CSV 생성 완료", f"파일을 열겠습니까?")
        if answer:
            os.startfile(self, file_path)


    def extract_value_tocsv(self, choice):
        x = self.get_files_value() # 정리된 데이터
        y = x[0]
        second_elements = [tpl[0] for tpl in y] # y에서 각 튜플의 첫 번째 요소 추출한 리스트
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

    def extract_box_feature(self, file_paths):
        #excel_file = str((self.extension[1:]).lower() + '\\' + '_dict.xlsx')  # 엑셀 파일 경로
        all_results = []  # 모든 파일 데이터 저장할 리스트

        excel_file = str(('mp4').lower() + '\\' + '_dict.xlsx')  # 엑셀 파일 경로
        df = pd.read_excel(excel_file)  # 엑셀 파일 읽기


        # 엑셀 데이터를 딕셔너리로 변환 (엑셀 파일의 첫 번째 열을 key로, 두 번째 열을 value로)
        self.seqdict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        filecount = 0
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            filecount +=1
            results = []
            results.append(('name', os.path.basename(file_path)))  # 파일명 열 추가
            if self.structure_val_state == True or self.structure_seq_state == True:
                 # 각 파일 데이터 저장 리스트

                nameprintformat = f"{filecount}/{len(file_paths)}_{results}"
                print(nameprintformat)
                onesequence = []
                # 파일 내 Box 파싱
                def parse_box(f, end_position, depth=0, max_depth=100):
                    if depth > max_depth:
                        print("최대 재귀 깊이 도달 에러")
                        return

                    while f.tell() < end_position:
                        box_header = f.read(8)  # 첫 8Bytes Box 헤더
                        if len(box_header) < 8:
                            break

                        box_size, box_type = struct.unpack(">I4s", box_header)  # size 4Bytes, type 4Bytes 추출
                        try:
                            lenbox_type = len(box_type)

                            box_type = box_type.decode("utf-8")
                            if len(box_type) != 4 or "\\x" in repr(box_type):
                                raise ValueError(f"잘못된 박스 타입 감지: {box_type}")

                        except Exception as e:
                            f.seek(0,1)
                            continue

                        if box_size == 0:  # 파일의 끝까지 Box가 확장됨을 의미
                            break
                        elif box_size == 1:  # 실제 크기는 다음 8Bytes에 저장됨
                            large_size = f.read(8)
                            actual_box_size = struct.unpack(">Q", large_size)[0]
                        else:
                            actual_box_size = box_size

                        box_end_position = f.tell() + (actual_box_size - 8 if box_size == 1 else box_size - 8)

                        if box_type in ('moov', 'trak', 'mdia', 'minf', 'stbl', 'udta', 'edts', 'moof', 'traf'):
                            # 컨테이너 Box 처리
                            parse_box(f, box_end_position, depth + 1, max_depth)
                        else:  # 컨테이너가 아닌 Box 처리

                            if box_type == 'mdat':
                                f.seek(box_end_position)
                                continue
                            box_data = f.read(actual_box_size - 8)

                            box_data_hex = box_data.hex()

                            if self.structure_seq_state == True :
                                if box_type in self.seqdict :
                                    onesequence.append(str(self.seqdict[box_type]))

                            # 각 Box의 속성을 구체적으로 추출
                            if box_type == 'ftyp':
                                major_brand = box_data[0:4].decode("utf-8")
                                minor_version = struct.unpack(">I", box_data[4:8])[0]
                                compatible_brands = [box_data[i:i + 4].decode("utf-8") for i in range(8, len(box_data), 4)]
                                results.append((box_type,
                                                f"Major Brand: {major_brand}, Minor Version: {minor_version}, Compatible Brands: {', '.join(compatible_brands)}"))

                            elif box_type == 'mvhd':
                                version = box_data[0]
                                if version == 0:
                                    create_time, modify_time, timescale, duration = struct.unpack(">IIII", box_data[4:20])
                                else:
                                    create_time, modify_time = struct.unpack(">QQ", box_data[4:20])
                                    timescale, duration = struct.unpack(">II", box_data[20:28])
                                preferred_rate = struct.unpack(">I", box_data[28:32])[0]
                                preferred_volume = struct.unpack(">H", box_data[32:34])[0]
                                box104_108 = box_data[96:100]
                                next_track_id = struct.unpack(">I", box_data[96:100])[0]
                                results.append((box_type,
                                                f"Create Time: {create_time}, Modify Time: {modify_time}, Timescale: {timescale}, Duration: {duration}, Preferred Rate: {preferred_rate}, Preferred Volume: {preferred_volume}, Next Track ID: {next_track_id}"))

                            elif box_type == 'tkhd':
                                version = box_data[0]
                                if version == 0:
                                    create_time, modify_time, track_id, duration = struct.unpack(">IIII", box_data[4:20])
                                else:
                                    create_time, modify_time = struct.unpack(">QQ", box_data[4:20])
                                    track_id, duration = struct.unpack(">II", box_data[20:28])
                                width, height = struct.unpack(">II", box_data[76:84])
                                results.append((box_type,
                                                f"Track ID: {track_id}, Create Time: {create_time}, Modify Time: {modify_time}, Duration: {duration}, Width: {width}, Height: {height}"))

                            elif box_type == 'mdhd':
                                version = box_data[0]
                                if version == 0:
                                    create_time, modify_time, timescale, duration = struct.unpack(">IIII", box_data[4:20])
                                else:
                                    create_time, modify_time = struct.unpack(">QQ", box_data[4:20])
                                    timescale, duration = struct.unpack(">II", box_data[20:28])
                                language_code = struct.unpack(">H", box_data[20:22])[0]
                                results.append((box_type,
                                                f"Create Time: {create_time}, Modify Time: {modify_time}, Timescale: {timescale}, Duration: {duration}, Language Code: {language_code}"))

                            elif box_type == 'elst':
                                version = box_data[0]
                                entry_count = struct.unpack(">I", box_data[4:8])[0]
                                entries = []
                                offset = 8
                                for _ in range(entry_count):
                                    if version == 1:
                                        segment_duration, media_time, media_rate = struct.unpack(">QqI", box_data[
                                                                                                         offset:offset + 16])
                                        entries.append(
                                            f"Duration: {segment_duration}, Media Time: {media_time}, Rate: {media_rate}")
                                        offset += 16
                                    else:
                                        segment_duration, media_time, media_rate = struct.unpack(">Iii", box_data[
                                                                                                         offset:offset + 12])
                                        entries.append(
                                            f"Duration: {segment_duration}, Media Time: {media_time}, Rate: {media_rate}")
                                        offset += 12
                                results.append((box_type, f"Entry Count: {entry_count}, Entries: {entries}"))

                            elif box_type == 'stsd':
                                version = box_data[0]
                                entry_count = struct.unpack(">I", box_data[4:8])[0]
                                results.append((box_type, f"Entry Count: {entry_count}"))

                            elif box_type == 'stts':
                                version = box_data[0]
                                entry_count = struct.unpack(">I", box_data[4:8])[0]
                                results.append((box_type, f"Entry Count: {entry_count}"))

                            elif box_type == 'stsc':
                                entry_count = struct.unpack(">I", box_data[4:8])[0]
                                results.append((box_type, f"Entry Count: {entry_count}"))

                            elif box_type == 'stsz':
                                sample_size = struct.unpack(">I", box_data[4:8])[0]
                                sample_count = struct.unpack(">I", box_data[8:12])[0]
                                results.append((box_type, f"Sample Size: {sample_size}, Sample Count: {sample_count}"))

                            elif box_type == 'co64':
                                entry_count = struct.unpack(">I", box_data[4:8])[0]
                                results.append((box_type, f"Entry Count: {entry_count}"))

                            else:
                                results.append((box_type, box_data_hex[:5000]))  # Default for other box types

                        # 다음 Box로 이동
                            f.seek(box_end_position)

                with open(file_path, 'rb') as f:
                    file_size = f.seek(0, 2)  # 파일 끝으로 커서 옮겨서 파일 크기 계산
                    f.seek(0)  # 커서를 파일 시작 위치로 이동
                    parse_box(f, file_size)  # 재귀

            if self.frame_gop_state == True:
                onesequence = extractGOP(file_path)
                results.append(('GOP', onesequence))

            if self.frame_ratio_state == True:
                ratio = process_videos_in_folder(file_path)
                results.append(('GOP compression', ratio))


            if self.frame_sps_state == True:
                try :
                    parse_sps(file_path)
                    file_name = os.path.basename(file_path)
                    file_name +=".264"
                    sps_filepath = file_name

                    spsresult = analyzesps(sps_filepath)
                    results.append(('SPS', spsresult))
                finally:
                    if os.path.exists(sps_filepath):
                        os.remove(sps_filepath)
                        print(f"{sps_filepath} 파일이 삭제되었습니다.")
                    else:
                        print(f"{sps_filepath} 파일이 존재하지 않습니다.")


            if self.structure_seq_state == True:

                onesequence = Simhash(onesequence).value
                results.append(('sequence', onesequence))


            all_results.append(results)
            # 각 파일의 결과를 전체 리스트에 추가


        self.save_to_csv(all_results)
        return results

    def on_structure_val_changed(self, state):
        if state == Qt.Checked:
            print('structure_val Box is checked')
            self.structure_val_state = True
            if '_val' not in self.csv_file:
                self.csv_file += '_val'
        else:
            print('structure_val Box is unchecked')
            self.structure_val_state = False
            self.csv_file = self.csv_file.replace('_val', '')

    def on_structure_seq_changed(self, state):
        if state == Qt.Checked:
            print('structure_seq Box is checked')
            self.structure_seq_state = True
            if '_seq' not in self.csv_file:
                self.csv_file += '_seq'
        else:
            print('structure_seq Box is unchecked')
            self.structure_seq_state = False
            self.csv_file = self.csv_file.replace('_seq', '')

    def on_frame_sps_changed(self, state):
        if state == Qt.Checked:
            print('frame_sps Box is checked')
            self.frame_sps_state = True
            if '_sps' not in self.csv_file:
                self.csv_file += '_sps'
        else:
            print('frame_sps Box is unchecked')
            self.frame_sps_state = False
            self.csv_file = self.csv_file.replace('_sps', '')

    def on_frame_gop_changed(self, state):
        if state == Qt.Checked:
            print('frame_gop Box is checked')
            self.frame_gop_state = True
            if '_gop' not in self.csv_file:
                self.csv_file += '_gop'
        else:
            print('frame_gop Box is unchecked')
            self.frame_gop_state = False
            self.csv_file = self.csv_file.replace('_gop', '')

    def on_frame_ratio_changed(self, state):
        if state == Qt.Checked:
            print('frame_ratio Box is checked')
            self.frame_ratio_state = True
            if '_ratio' not in self.csv_file:
                self.csv_file += '_ratio'
        else:
            print('frame_ratio Box is unchecked')
            self.frame_ratio_state = False
            self.csv_file = self.csv_file.replace('_ratio', '')

    # 결과를 CSV로 저장
    def save_to_csv(self, all_data):
        csv_file = self.csv_file
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")

        # 파일 이름에 타임스탬프 추가
        csv_file = f"{csv_file}_{timestamp}.csv"
        csv_file = os.path.join(self.direc, csv_file)

        # 필드명 추출 - 모든 파일의 필드를 확인하여 중복 필드 처리
        fieldnames = []
        key_count_global = {}  # 전체 파일에서의 중복 key 카운트 딕셔너리

        # 모든 파일의 데이터를 순회하여 필드 추출
        for file_data in all_data:
            key_count_local = {}  # 각 파일 내에서의 중복 key 카운트
            for key, value in file_data:
                # 중복 필드 처리 (필드 이름 중복 시 숫자를 붙임)
                if key in key_count_local:
                    key_count_local[key] += 1
                    key_with_count = f"{key}({key_count_local[key]})"
                else:
                    key_count_local[key] = 1
                    key_with_count = key

                # 콜론이 있는 경우 자식 속성 분리
                if isinstance(value, str) and ":" in value:
                    attributes = [attr.strip() for attr in value.split(",")]
                    for attr in attributes:
                        if ":" in attr:
                            attr_name = f"{key_with_count}_{attr.split(':')[0].strip()}"
                            attr_value = attr.split(":")[1].strip()

                            # 필드가 fieldnames에 없고 값이 있는 경우만 필드를 추가
                            if attr_name not in fieldnames and attr_value:
                                fieldnames.append(attr_name)
                else:
                    if key_with_count not in fieldnames:
                        fieldnames.append(key_with_count)

            #print('keycount (local):', key_count_local)
            #print('new_fieldnames:', fieldnames)

        ##1025 레이블 추가
        if 'label' not in fieldnames:
            fieldnames.append('label')


        # GOP 처리 (중복 처리 방식과 동일)
        for onedata in all_data:
            for key, value in onedata:
                if key == 'GOP':
                    if isinstance(value, str) and ":" in value:
                        # 자식 속성 있는 경우 속성 분할(예: “생성 시간: 1234, 수정 시간: 5678”).
                        attributes = [attr.strip() for attr in value.split(",")]
                        for attr in attributes:
                            if ":" in attr:
                                attr_name = f"{key}_{attr.split(':')[0].strip()}"
                                if attr_name not in fieldnames:
                                    fieldnames.append(attr_name)
                    else:
                        if key not in fieldnames:
                            fieldnames.append(key)

        print('최종 필드명 확인: ', fieldnames)

        # CSV에 쓰기
        if self.detectmode == 0:
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                # 헤더 작성
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # 새로운 데이터 쓰기
                for file_data in all_data:
                    row_data = {}
                    key_count_local = {}  # 각 파일 내에서의 중복 key 카운트 리셋

                    for key, value in file_data:
                        if key in key_count_local:
                            key_count_local[key] += 1
                            key_with_count = f"{key}({key_count_local[key]})"
                        else:
                            key_count_local[key] = 1
                            key_with_count = key

                        # 필드 값을 CSV에 저장
                        if isinstance(value, str):
                            # : 있는거 세부 속성 나누기
                            attributes = [attr.strip() for attr in value.split(",")]
                            for attr in attributes:
                                if ":" in attr:
                                    attr_name, attr_value = attr.split(":", 1)
                                    if f"{key_with_count}_{attr_name.strip()}" in fieldnames:
                                        row_data[f"{key_with_count}_{attr_name.strip()}"] = attr_value.strip()
                                else:
                                    if key_with_count in fieldnames:
                                        row_data[key_with_count] = value
                        else:
                            if key_with_count in fieldnames:
                                row_data[key_with_count] = value
                    ##1025 레이블 추가

                    try:
                        if hasattr(self, 'label_data') and self.label_data:
                            row_data['label'] = self.label_data
                        else:
                            print("Warning: 'label_data' is not set or is empty.")
                    except Exception as e:
                        pass

                    # CSV에 기록
                    writer.writerow({key: row_data.get(key, "") for key in fieldnames})

            print(f"Results saved to {csv_file}")
            savemassage = f"학습데이터셋이 파일 {csv_file} 에 저장되었습니다."

            self.show_file_alert(csv_file, savemassage, self.tableWidget_Create)


    @staticmethod
    def calculate_simhash_lib(value):
        simval = Simhash(str(value)).value
        return simval

    def apply_simhash(self, df):
        """Simhash 적용"""
        df.columns = df.columns.astype(str)
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            df[column] = df[column].apply(self.calculate_simhash_lib)
        return df
    def load_file_for_prediction(self):
        """Open a dialog to select a file for prediction."""
        self.detectmode = 1
        file_path= self.file_paths[0]
        if file_path:
            self.predict_on_file(file_path)

    def predict_on_file(self, file_path):
        """Predict the label for a new file while handling missing and extra features."""
        try:
            # Extract and flatten features from the file
            feature_data = self.extract_features_from_file(file_path)
            structured_data = self.flatten_features_for_prediction(feature_data)

            # Load model and scaler
            self.load_model_and_scaler()

            # Predict and show results
            predicted_df = self.predict_data(structured_data)
            print(predicted_df)
            self.detectmode = 0

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed for {file_path}: {str(e)}")


    def flatten_features_for_prediction(self, data):
        """Flatten nested feature structures to match training logic."""
        flattened = {}
        key_count = {}  # Handle duplicate keys

        for key, value in data:
            if key in key_count:
                key_count[key] += 1
                key = f"{key}({key_count[key]})"
            else:
                key_count[key] = 1

            if isinstance(value, str) and ":" in value:
                attributes = [attr.strip() for attr in value.split(",")]
                for attr in attributes:
                    if ":" in attr:
                        attr_name, attr_value = attr.split(":", 1)
                        flattened[f"{key}_{attr_name.strip()}"] = attr_value.strip()
                    else:
                        flattened[key] = value
            elif isinstance(value, list):
                for item in value:
                    if ":" in item:
                        attr_name, attr_value = item.split(":", 1)
                        flattened[f"{key}_{attr_name.strip()}"] = attr_value.strip()
                    else:
                        flattened[key] = item
            else:
                flattened[key] = value

        return flattened


    def extract_features_from_file(self, file_path):
        """Extract features from the selected file, using the same state-based logic."""
        results = []

        if self.structure_val_state:
            box_features = self.extract_box_feature(file_path)
            results.extend(box_features)

        if self.frame_gop_state:
            gop_features = extractGOP(file_path)
            results.append(('GOP', gop_features))

        if self.frame_sps_state:
            sps_result = self.extract_sps_features(file_path)
            results.append(('SPS', sps_result))

        if self.structure_seq_state:
            sequence_feature = Simhash([f[1] for f in results if f[0] != 'name']).value
            results.append(('sequence', sequence_feature))

        return results

    def load_model_and_scaler(self):
        """Load the trained model and scaler from disk."""

        binstat = self.binButton_2.isChecked()
        mulstat = self.mulButton_2.isChecked()
        if binstat:
            self.classmode = 'bin_'
        elif mulstat:
            self.classmode = 'mul_'
        else :
            messagebox.showerror("에러", "바이너리/멀티 모드를 선")
        #self.csv_path = "__1_val_241022214050_binary.csv"
        self.csv_path = '__2_galaxy_original_edit_binary.csv'
        #self.csv_path = '__3_Edit_Application.csv'
        pklname = os.path.join('Y:\\', str(self.csv_path+"_" + self.aimodel + "model.pkl"))
        self.scalername = os.path.join('Y:\\', str(self.csv_path+"_" + self.aimodel + "scaler.pkl"))
        if os.path.exists(pklname) and os.path.exists(self.scalername):
            self.model = joblib.load(pklname)
            self.scaler = joblib.load(self.scalername)
        else:
            raise FileNotFoundError("Model or scaler file not found.")

    def predict_data(self, structured_data):
        """Scale the features and predict the label."""


        df = pd.DataFrame([structured_data])

        # Align DataFrame with model's features
        with open('feature.json', 'r') as f:
            model_features = json.load(f)

        # Add missing features with default value (0) and filter out extra features
        for feature in model_features:
            if feature not in df.columns:
                df[feature] = 0  # Fill missing features with default value

        df = df[model_features]  # Keep only relevant features
        df = df.drop(columns=[col for col in df.columns if col == 'name'], errors='ignore')

        # Apply Simhash to the relevant features
        df = self.apply_simhash(df)

        # Scale features and predict label
        X_new_scaled = self.scaler.transform(df)
        y_pred = self.model.predict(X_new_scaled)
        y_pred_probs = self.model.predict_proba(X_new_scaled)
        predicted_class_probs = y_pred_probs[np.arange(len(y_pred)), y_pred]

        print(predicted_class_probs,"% 확률")
        # Add predictions to DataFrame
        df['predicted_label'] = y_pred

        labeltransferdf = pd.read_excel("labelinfo.xlsx")

        try:
            # y_pred[0] 값을 정수로 변환
            temppred = int(y_pred[0])  # 예측 값이 정수라고 가정
            # labeltransferdf의 컬럼명도 정수로 사용될 수 있도록 확인
            labeltransferdf.columns = [int(col) for col in labeltransferdf.columns]

            # 해당 예측 값에 해당하는 컬럼 필터링
            filtered_df = labeltransferdf[temppred]
        except KeyError as e:
            message = f"해당 라벨({temppred})이 존재하지 않습니다. 라벨을 업데이트하세요."
            self.show_alert(message)
        except Exception as e:
            self.show_alert(str(e))

        try :
            fileaccuracy = "{:.3f}".format(predicted_class_probs[0][1] * 100)
            message = f"{fileaccuracy}% 확률로 {filtered_df[0]}({y_pred}) 입니다"
            self.show_alert(message)
        except Exception as e:
            fileaccuracy = "{:.3f}".format(predicted_class_probs[0] * 100)
            message = f"{fileaccuracy}% 확률로  {filtered_df[0]}({y_pred}) 입니다"
            self.show_alert(message)
        return df

    def show_file_alert(self, file_path, messagea, widgett):
        """파일 경로를 받아 사용자에게 알림을 표시하고 파일을 여는 함수."""
        app = QApplication.instance()  # 이미 실행 중인 QApplication 인스턴스 확인
        if not app:
            app = QApplication(sys.argv)

        # QDialog를 사용해 타이틀 없는 커스텀 알림창 생성
        dialog = QDialog()
        dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 타이틀 바 제거 및 최상단 설정

        # 다크 모드 스타일 적용
        dialog.setStyleSheet("""
                    QDialog {
                        background-color: #2e2e2e;
                        border: 2px solid #444;
                        border-radius: 15px;
                        padding: 20px;
                        font: bold 10pt "Playfair Display";
                    }
                    QLabel {
                        color: #f5f5f5;
                        font-size: 20px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        font: bold 10pt "Playfair Display";
                    }
                    QPushButton {
                        background-color: #444;
                        color: white;
                        border: 1px solid #777;
                        border-radius: 5px;
                        padding: 8px 15px;
                        margin-top: 10px;
                        font: bold 10pt "Playfair Display";
                    }
                    QPushButton:hover {
                        background-color: #555;
                    }
                """)

        # 레이아웃 생성 및 위젯 추가
        layout = QVBoxLayout()
        messages = messagea + "바로 확인하시겠습니까?"
        message_label = QLabel(messages)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        # '확인' 버튼 추가
        open_button = QPushButton("확인")
        open_button.clicked.connect(lambda: self.open_csv2(file_path, widgett))  # 파일 열기 함수 호출
        open_button.clicked.connect(dialog.accept)
        layout.addWidget(open_button)

        # '취소' 버튼 추가
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(dialog.reject)  # 창 닫기
        layout.addWidget(cancel_button)
        dialog.setFixedSize(400, 200)
        dialog.setLayout(layout)

        # 창 크기 조정 및 화면 중앙 배치
        dialog.adjustSize()


        # 알림 창 표시
        dialog.exec_()
        return


    def show_alert(self, message):
        title = "알림"
        app = QApplication.instance()  # 이미 실행 중인 QApplication 인스턴스 확인
        if not app:
            app = QApplication(sys.argv)

        # QDialog를 사용해 타이틀 없는 커스텀 알림창 생성
        dialog = QDialog()
        dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 타이틀 바 제거 및 최상단 설정

        # 다크 모드 스타일 적용
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                border: 2px solid #444;
                border-radius: 15px;
                padding: 20px;
                font: bold 10pt "Playfair Display";
            }
            QLabel {
                color: #f5f5f5;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                font: bold 10pt "Playfair Display";
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #777;
                border-radius: 5px;
                padding: 8px 15px;
                margin-top: 10px;
                font: bold 10pt "Playfair Display";
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)

        # 레이아웃 생성 및 위젯 추가
        layout = QVBoxLayout()
        title_label = QLabel(title)
        message_label = QLabel(message)
        layout.addWidget(title_label)
        layout.addWidget(message_label)

        # 확인 버튼 추가
        button = QPushButton("확인")
        button.clicked.connect(dialog.accept)  # 버튼 클릭 시 창 닫기
        layout.addWidget(button)

        dialog.setLayout(layout)

        # 창 크기 조정 및 화면 중앙 배치
        dialog.adjustSize()
        # screen_center = QApplication.primaryScreen().geometry().center()
        # dialog.move(screen_center - dialog.rect().center())

        # 알림 창 표시
        dialog.exec_()

    def show_prediction_results(self, df):
        """Display prediction results."""
        results = df.to_string(index=False)
        QMessageBox.information(self, "Prediction Results", f"Prediction:\n{results}")

    def on_train_button_click(self):
        """Trigger model training."""
        self.gotrain(self.classmode, self.aimodel, self.trainindex, self.csv_path)

    def align_features_with_model(self, df):
        """Align DataFrame columns with the model's feature set."""
        # Load the feature set used during training (from model or scaler)
        model_features = self.feature_list

        # Add missing features with a default value (e.g., 0)
        for feature in model_features:
            if feature not in df.columns:
                df[feature] = 0

        # Ensure the DataFrame contains only the required features
        aligned_df = df[model_features]

        return aligned_df
    def extract_sps_features(self, file_path):
        """Extract SPS features."""
        try:
            parse_sps(file_path)
            file_name = os.path.basename(file_path) + ".264"
            sps_result = analyzesps(file_name)
            return sps_result
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)




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

                if self.structure_val_state == True or self.frame_sps_state== True:
                    self.extract_box_feature(self.file_paths)



                break

            elif choice == 3:
                self.file_paths = self.listWidget
                #self.extension = (self.file_paths[0].split('.'))[1]
                #self.folderpath("inputfile2")
                # 파일을 순서딕셔너리와 비교하여 1-2-3-4, 2-3-1-4 등 순서 리스트를 만들고, 이를 심해시화
                self.extension = 'mp4'

                self.sequencedem = []
                '''                hexlist = str(self.extension + '\\' +  "hexlist.pkl")
                                with open(hexlist, 'rb') as f:
                                    hexvalues = pickle.load(f)
                                for h in range(len(hexvalues)):
                                    self.feature_dictionary(hexvalues[h])'''


                self.feature_dictionary()

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


    def ask_overwrite_labels(self):
        """덮어쓰기 여부를 묻는 팝업."""
        reply = QMessageBox.question(
            self, "Overwrite Labels", "Do you want to overwrite existing labels?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def open_data_entry_window(self):
        """데이터 입력 창 열기."""
        overwrite_labels = self.ask_overwrite_labels()
        self.data_entry_window = DataEntryWindow(overwrite_labels)
        self.data_entry_window.show()
        self.load_excel_data()

    def load_excel_data(self):
        """엑셀 데이터를 DataFrame으로 불러와 테이블에 표시."""
        if not os.path.exists("labelinfo.xlsx"):
            QMessageBox.warning(self, "Warning", "No Excel file found!")
            return

        df = pd.read_excel("labelinfo.xlsx")  # 엑셀 파일을 DataFrame으로 로드
        df.columns = [str(col) for col in df.columns]

        self.display_dataframe(df, widgettype=self.tableWidget_train)
        self.display_dataframe(df, widgettype=self.tableWidget_detect)

    def show_input_dialog(self, title, label):
        """커스텀 입력창을 표시하고 입력된 텍스트를 반환하는 함수."""
        app = QApplication.instance()  # 이미 실행 중인 QApplication 인스턴스 확인
        if not app:
            app = QApplication([])

        # QDialog 생성 및 스타일 설정
        dialog = QDialog()
        dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 타이틀 바 제거 및 최상단 설정

        dialog.setStyleSheet("""
            QDialog {
                background-color: #2e2e2e;
                border: 2px solid #444;
                border-radius: 10px;
                padding: 20px;
            }
            QLabel {
                color: #f5f5f5;
                font-size: 16px;
            }
            QLineEdit {
                background-color: #444;
                color: #f5f5f5;
                border: 1px solid #777;
                border-radius: 5px;
                padding: 8px;
                margin-top: 10px;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)

        # 레이아웃 생성
        layout = QVBoxLayout()

        # 라벨 추가
        message_label = QLabel(label)
        layout.addWidget(message_label)

        # 입력 필드 추가
        line_edit = QLineEdit()
        layout.addWidget(line_edit)

        # 버튼 생성 및 추가
        button_layout = QVBoxLayout()

        ok_button = QPushButton("확인")
        ok_button.clicked.connect(dialog.accept)  # 확인 클릭 시 다이얼로그 닫기
        layout.addWidget(ok_button)

        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(dialog.reject)  # 취소 클릭 시 다이얼로그 닫기
        layout.addWidget(cancel_button)

        # 다이얼로그에 레이아웃 설정
        dialog.setLayout(layout)

        # 다이얼로그 실행 및 결과 처리
        if dialog.exec_() == QDialog.Accepted:
            return line_edit.text(), True
        return "", False

    ##############라벨입력
    def input_label(self):
        if self.binButton.isChecked():
            self.label_datacsv = 'labeldata_bin.csv'
        elif self.mulButton.isChecked():
            self.label_datacsv = 'labeldata_mul.csv'

        # 라벨 데이터 입력 받기
        self.label_data, ok = self.show_input_dialog("입력", "라벨 데이터를 입력하세요.")
        if not ok or not self.label_data:
            return

        try:
            number = float(self.label_data)  # 숫자 입력 여부 확인
        except ValueError:
            self.show_alert("에러", "유효한 숫자를 입력해주세요.")
            return

        # CSV에서 해당 라벨 데이터를 찾기
        try:
            aaa = self.fetch_name_from_csv(self.label_data)
        except Exception as e:
            self.show_alert("binary/multi class 모드를 선택하세요")
            return

        if aaa is None:  # 해당 데이터가 없을 경우 새로 입력
            name, ok = self.show_input_dialog("입력", "매핑되는 속성을 입력하세요.")
            if not ok or not name:
                return

            # 파일이 없으면 생성
            if not os.path.exists(self.label_datacsv):
                with open(self.label_datacsv, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Label", "Name"])  # 헤더 작성

            # CSV 파일에 새 데이터 추가
            with open(self.label_datacsv, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([self.label_data, name])
        else :
            self.show_alert("해당 라벨이 이미 존재합니다!")

    def fetch_name_from_csv(self, max_key):
        filename = self.label_datacsv

        # 파일이 없을 경우 None 반환
        if not os.path.exists(filename):
            return None

        # 파일에서 해당 키에 해당하는 값을 찾기
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == max_key:
                    return row[1]
        return None

##################라벨입력
class DataEntryWindow(QWidget):
    def __init__(self, overwrite_labels):
        super().__init__()
        self.setWindowTitle("Data Entry")
        self.label_counter = 0
        self.headers = []
        self.values = []
        self.filename = "labelinfo.xlsx"
        if not overwrite_labels:
            self.load_existing_labels()

        # UI 구성
        layout = QVBoxLayout()

        self.header_label = QLabel(f"Current Label: {self.label_counter}")
        layout.addWidget(self.header_label)

        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("Enter value...")
        layout.addWidget(self.value_input)

        add_value_button = QPushButton("Add Value")
        add_value_button.clicked.connect(self.add_value)
        layout.addWidget(add_value_button)

        stop_button = QPushButton("Stop and Save")
        stop_button.clicked.connect(self.stop_and_save)
        layout.addWidget(stop_button)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.update_display()

    def load_existing_labels(self):
        """기존 엑셀 파일에서 라벨 로드."""
        if os.path.exists(self.filename):
            workbook = load_workbook(self.filename)
            sheet = workbook.active
            if sheet.max_row > 0:
                self.headers = [cell.value for cell in sheet[1]]
                self.label_counter = len(self.headers)
            workbook.close()

    def add_value(self):
        """값을 추가하고 다음 라벨로 이동."""
        value = self.value_input.text()
        if not value:
            QMessageBox.critical(self, "Error", "Value cannot be empty!")
            return

        self.values.append(value)
        self.headers.append(str(self.label_counter))
        self.label_counter += 1
        self.value_input.clear()
        self.update_display()
        self.header_label.setText(f"Current Label: {self.label_counter}")

    def update_display(self):
        """현재까지 입력된 데이터를 테이블에 표시."""
        self.table.setRowCount(1)
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        for col, value in enumerate(self.values):
            self.table.setItem(0, col, QTableWidgetItem(value))

    def stop_and_save(self):
        """엑셀에 데이터를 저장하고 창 닫기."""
        if not self.values:
            QMessageBox.warning(self, "Warning", "No data to save!")
            return

        if os.path.exists(self.filename):
            workbook = load_workbook(self.filename)
            sheet = workbook.active
            if sheet.max_row == 0:
                sheet.append(self.headers)
        else:
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(self.headers)

        sheet.append(self.values)
        workbook.save(self.filename)
        workbook.close()

        QMessageBox.information(self, "Success", "Data saved successfully!")
        self.close()









if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = createtrainclass()
    ex.show()
    app.exec_()
