import csv

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from simhash import Simhash
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sys
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QMessageBox, QFileDialog, \
    QTableWidgetItem, QLabel, QDialog, QVBoxLayout, QPushButton
from PyQt5 import uic, QtWidgets
import pickle
import joblib

from createtraining import createtrainclass
from tensorflow.keras.models import load_model

form_class = uic.loadUiType("UI_Design\\Detector.ui")[0]

class DetectorGRU(QMainWindow, form_class):  # QMainWindow, form_class
p
    def __init__(self):
        super(DetectorGRU, self).__init__()
        self.choice = 0
        self.file_paths = []
        os.chdir("Y:\\AiFileDetectorE")
        self.dpath = 'Y:\\AiFileDetectorE\\'
        self.setupUi(self)

        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(self.dpath)
        self.detectTreeView.setModel(self.dirModel)

        self.detectTreeView.setRootIndex(self.dirModel.index(os.getcwd()))
        self.detectTreeView.clicked.connect(self.file_selected)

        #self.extension_list = [".mp4", ".png", ".jpg",".m4a"]
        #self.comboBox.addItems(self.extension_list)
        #self.comboBox.activated.connect(self.on_combobox_select)


        #self.comboBox_1.currentIndexChanged.connect(self.filter_files_by_extension)

        self.comboBox_1.activated.connect(self.on_combobox_select)

        self.LoadButton.clicked.connect(self.detect)
    def load_model2(self):
        """학습 모델 로드"""
        if self.index == 0:
            self.aimodel = self.comboBox_1.currentText()
            with open(file=(str(self.extension + '\\' +self.aimodel + "model.pkl")), mode='rb') as f:
                self.model = pickle.load(f)
        elif self.index == 2 or self.index == 3 or self.index == 3:
            self.aimodel = self.comboBox_1.currentText()
            with open(file=(str(self.extension + '\\' + self.aimodel + "model.pkl")), mode='rb') as f:
                self.model = joblib.load(f)
        elif self.index == 1:
            self.model = load_model(str(self.extension + '\\' + 'model.h5'))
    def apply_simhash(self, df):
        """Simhash 적용"""
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            df[column] = df[column].apply(self.calculate_simhash_lib).astype('int64')
        return df

    def on_combobox_select(self, index):
        self.index = index


    def filter_files_by_extension(self, extension):
        if extension:
            self.extension = extension[1:]
            self.dirModel.setNameFilters([f"*{extension}"])
            self.dirModel.setNameFilterDisables(False)
        else:
            self.dirModel.setNameFilters([])

    def predict_data(self, df):
        """새 데이터 예측"""

        X_new = df.iloc[:, 1:]

        # 모델을 사용하여 각 클래스에 속할 확률을 예측
        y_pred_proba = self.model.predict_proba(X_new)

        # 확률이 가장 높은 클래스를 예측
        y_pred = y_pred_proba.argmax(axis=1)

        # 예측된 클래스를 데이터프레임에 추가
        df['label'] = y_pred

        # 첫 번째 샘플의 확률 분리
        first_sample_proba = y_pred_proba[0]
        class_probas = {f'{i}': prob for i, prob in enumerate(first_sample_proba)}



        return df, y_pred[0], class_probas

    def file_selected(self, index):
        file_path = self.dirModel.fileInfo(index).absoluteFilePath()
        if os.path.isfile(file_path):
            self.listWidget.clear()
            file_name = os.path.basename(file_path)
            self.file_path = file_path


    def display_dataframe(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                self.tableWidget.setItem(i, j, item)
    @staticmethod
    def calculate_simhash_lib(value):
        return Simhash(str(value)).value

    def show_custom_message(self, message):
        dialog = QDialog()
        dialog.setWindowTitle("결과")
        dialog.setStyleSheet('background-color: rgb(69, 71, 83);')
        # 레이아웃 설정
        layout = QVBoxLayout()

        # 메시지 라벨 생성 및 스타일 설정
        label = QLabel(message)
        label.setStyleSheet('color: rgb(255, 255, 255); font: 10pt "Helvetica";')

        # 확인 버튼 생성
        button = QPushButton("확인")
        button.clicked.connect(dialog.accept)

        # 레이아웃에 위젯 추가
        layout.addWidget(label)
        layout.addWidget(button)

        dialog.setLayout(layout)

        dialog.exec_()

    def fetch_name_from_csv(self, max_key):
        if self.binButton.isChecked():
            filename = str(self.extension + '\\' + 'labeldata_bin.csv')
        elif self.mulButton.isChecked():
            filename = str(self.extension + '\\' + 'labeldata_mul.csv')

        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == max_key:
                    return row[1]
        return None

    def detect(self):
        self.savefilepath = []
        self.extension = self.comboBox.currentText()
        self.extension = self.extension[1:]
        extractor = createtrainclass()
        self.savefilepath.append(self.file_path)
        extractor.extract_ngram(8, self.savefilepath)
        extractor.reres = extractor.extract_value(self.file_path, self.extension)
        df_test_melted=extractor.reres
        self.original_labels = [t[0] for t in df_test_melted]
        self.original_labels.append('label')
        sequences=extractor.feature_dictionary(extractor.hex_values, self.extension)
        #df_test_melted.pop(0)

        #if 'sequence' in df_test_melted:

        df_test_melted.append(('sequence', sequences))

        columns = [item[0] for item in df_test_melted]
        values = [item[1] for item in df_test_melted]



        df_test_melted = pd.DataFrame([values], columns=columns)


        #sequence = self.feature_dictionary(values, self.extension)



        df_test_melted = self.apply_simhash(df_test_melted)
        extractvalue = str(self.extension + '\\' + "detectextractvalues.csv")

        #df_test_melted = [t[1] for t in df_test_melted]

        self.loaded_model = self.load_model2()
        #df_test_melted = numpy.array(df_test_melted).reshape((1, -1))

        predicted_data, predicted_label, predicted_proba = self.predict_data(df_test_melted)

        max_key = max(predicted_proba, key=predicted_proba.get)
        max_value = predicted_proba[max_key]
        mapped_name = self.fetch_name_from_csv(max_key)


        predicted_data.to_csv("result2.csv", index=False)


        resultaa = (max_value , '확률로' , max_key , mapped_name , "입니다")
        resultaa = ' '.join(map(str, resultaa))
        self.show_custom_message(str(resultaa))

        print(predicted_label)


    def analyze_prediction(self, df, original_labels):
        """위변조 판단"""
        group_averages = df.groupby('name')['label'].mean()
        print('sdfs', group_averages)
        results = {}
        success_failure = {}

        for name, avg in group_averages.items():
            original_label = original_labels[original_labels['name'] == name]['label'].values[0]
            closest_label = round(avg)
            results[name] = f'기존 label : {original_label}, 예측 label : {closest_label}'

            if int(original_label) == closest_label:
                success_failure[name] = "예측 성공"
            else:
                success_failure[name] = "예측 실패"

        results_df = pd.DataFrame(list(results.items()), columns=['name', 'result'])

        return results, success_failure, results_df



if __name__ == "__main__":
    app = QApplication(sys.argv)
    data_preprocessor = DetectorGRU()
    data_preprocessor.show()
    app.exec_()
