import json
import pickle

import pyautogui
import seaborn as sns
from keras.optimizers import Adam
from keras.regularizers import l2
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, \
    precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow import keras
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding, LeakyReLU, Bidirectional, BatchNormalization
from tensorflow.python.keras.models import load_model
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from simhash import Simhash
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QMessageBox, QFileDialog, \
    QTableWidgetItem
from PyQt5 import uic, QtWidgets
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

form_class = uic.loadUiType("UI_Design\\Training.ui")[0]

'''
device_lib.list_local_devices()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
form_class = uic.loadUiType("Training.ui")[0]

with tf.device('/GPU:0'):
'''


class TrainClass(QMainWindow, form_class):  # QMainWindow, form_class

    def __init__(self):
        super(TrainClass, self).__init__()
        self.choice = 0
        self.file_paths = []
        self.dpath = 'E:\\'
        self.model = None

        self.setupUi(self)

        self.dirModel = QFileSystemModel()
        #self.dirModel.setRootPath("E:\\AiFileDetectorE")
        #os.chdir("E:\\AiFileDetectorE")
        self.treeView.setModel(self.dirModel)

        self.treeView.setRootIndex(self.dirModel.index(os.getcwd()))
        self.treeView.clicked.connect(self.file_selected)

        self.xlsfileext = '.csv'
        self.filter_files_by_extension(self.xlsfileext)
        self.comboBox.activated.connect(self.on_combobox_select)

        self.LoadButton.clicked.connect(self.on_train_button_click)

    def filter_files_by_extension(self, xlsext):
        if xlsext:
            self.xlsext = xlsext[1:]
            self.dirModel.setNameFilters([f"*{xlsext}"])
            self.dirModel.setNameFilterDisables(False)
        else:
            self.dirModel.setNameFilters([])

    def file_selected(self, index):
        file_path = self.dirModel.fileInfo(index).absoluteFilePath()
        if os.path.isfile(file_path):
            self.listWidget.clear()
            file_name = os.path.basename(file_path)
            self.listWidget.addItem(file_name)
            self.csv_path = file_path
            self.open_csv(file_path)


    def open_csv(self, csvfile):
        file_name = csvfile

        if file_name:
            try:
                df = pd.read_csv(file_name,encoding='UTF-8')
                self.display_dataframe(df)
            except Exception as e:
                self.tableWidget.setRowCount(0)
                self.tableWidget.setColumnCount(0)
                #self.show_error_message("CSV 파일을 읽는 중 오류가 발생했습니다: " + str(e))

    def display_dataframe(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                self.tableWidget.setItem(i, j, item)

    def on_combobox_select(self, index):
        self.index = index

    def on_train_button_click(self):
        self.gotrain()

    def load_model2(self):
        """학습 모델 로드"""
        if self.index == 0 or 2 or 3:
            self.model = joblib.load(str(self.extension + '\\' + "model.pkl"))
        elif self.index == 1:
            self.model = load_model(str(self.extension + '\\' + 'model.h5'))

    def analyze_prediction(self, df, original_labels):
        """위변조 판단"""
        group_averages = df.groupby('name')['label'].mean()
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

    def gotrain(self, classmode):
        print("***다중분류 시작***")
        self.classmode = classmode
        df, _ = self.preprocess_data(self.csv_path, is_train=True)


        self.extension = os.path.basename(os.path.dirname(self.csv_path))
        # 훈련 데이터와 테스트 데이터로 분할
        df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

        # 훈련 데이터 전처리
        #df_train = df_test.drop(columns='label')
        df_train_processed = self.apply_simhash(df_train)
        print("전처리한 훈련 데이터:")
        print(df_train_processed)

        # 모델 훈련
        self.train_model(df_train_processed)
        baseline_model, baseline_accuracy =self.train_baseline_model(df_train_processed)

        print("베이스라인 정확도", baseline_accuracy)

        self.save_model2()
        self.original_df_test = df_test
        df_test = df_test.drop(columns='label')
        # 테스트 데이터 전처리
        df_test_processed = self.apply_simhash(df_test)
        self.feature_list = df_train.drop(columns=['label']).columns.tolist()
        #추후 변경 필요 --> 파일이름을 피처 반영되게
        with open('feature.json', 'w') as f:
            json.dump(self.feature_list, f)


        # 모델 로드 및 테스트 데이터 예측
        #self.load_model2()
        predicted_data = self.predict_data(df_test_processed)
        predicted_datalabel = predicted_data['label']
        results, success_failure, results_df = self.analyze_prediction(predicted_data, self.original_df_test[['name', 'label']])
        actual_labels = self.original_df_test['label']
        actual_labels = actual_labels.astype(int)
        predicted_labels = predicted_datalabel

        conf_matrix = self.confusion_matrix2(actual_labels, predicted_labels)
        print(conf_matrix)


        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        print(success_failure)
        print(results_df)

        # 예측 성공률 계산

        total = len(results_df)
        success = sum([1 for row in success_failure.values() if "예측 성공" in row])
        success_rate = (success / total) * 100
        print(f"예측 성공률: {success_rate:.2f}%")

        precision = precision_score(actual_labels, predicted_labels, average = 'weighted')
        recall = recall_score(actual_labels, predicted_labels, average = 'weighted')
        f1 = f1_score(actual_labels, predicted_labels, average = 'weighted')

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")




    def confirmfile(self, makefile):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self)
        file_dialog.setOptions(options)

        if self.csv_path:
            try:
                msg_box = QMessageBox.question(self, '확인', f'See {makefile}?', QMessageBox.Yes | QMessageBox.No,
                                               QMessageBox.No)
                current_directory = os.getcwd()
                absolute_path = os.path.join(current_directory, makefile)

                if msg_box == QMessageBox.Yes:
                    self.open_csv(absolute_path)

            except Exception as e:
                print(f"파일 열기 오류: {str(e)}")

    def preprocess_data(self, filepath, is_train=True):
        """데이터 전처리"""

        df = pd.read_csv(filepath, header=None ,encoding='CP949')
        column_count = df.shape[1]
        original_labels = None

        if is_train:
            features = df.iloc[0, 1:-1].values
            df.columns = ['name'] + list(features) + ['label']
            df = df[1:]


        else:
            features = df.iloc[0, 1:-1].values
            df.columns = ['name'] + list(features) + ['label']
            original_labels = df[['name', 'label']]
            df = df[1:]

        return df, original_labels

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

    def train_model(self, df):
        try :
            if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
                model, accuracy = self.ensemble(df)
                pyautogui.alert(f"정확도 {accuracy}%로 학습되었습니다.")

            elif self.index == 1:
                self.lstm(df)
        except :
            self.index = 0

    def confusion_matrix2(self, y_train, y_pred_classes):
        # Confusion matrix 생성
        cm = confusion_matrix(y_train, y_pred_classes)

        # Confusion matrix 출력
        print("Confusion Matrix:")
        print(cm)

        # Confusion matrix 시각화
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # 추가적으로 classification report도 출력
        print("Classification Report:")
        print(classification_report(y_train, y_pred_classes))


    def ensemble(self, df):

        X = df.iloc[:, 1:-1]
        y = df['label']

        y = y.astype("int")
        if self.index < 4:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # MinMaxScaler 적용
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        params_xgb = {
            'max_depth': [2, 3, 4, 5, 6, 7,8],
            'n_estimators': [150, 200, 250,300],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'eval_metric': ['logloss', 'error']
        }

        params_rf = {
            'n_estimators': [10, 20, 30, 40, 50],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        params_lgbm = {
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [100, 150, 200, 250],
            'learning_rate': [0.01, 0.05, 0.1]
        }

        if self.index == 0:
            self.model = xgb.XGBClassifier(random_state=42)
            grid_search  = GridSearchCV(self.model, params_xgb, cv=3, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
        elif self.index == 2:
            self.model = RandomForestClassifier()
            grid_search = RandomizedSearchCV(self.model, params_rf, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
        elif self.index == 3:
            self.model = LGBMClassifier(objective='multiclass')
            grid_search = RandomizedSearchCV(self.model, params_lgbm, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
        elif self.index == 4:
            self.model = LogisticRegression(solver='lbfgs', max_iter=100)
            self.model.fit(X_train_scaled, y_train)  # LinearRegression에는 RandomizedSearchCV 적용 불필요



        # # 모델 구성
        # if self.index ==0:
        #     self.model = xgb.XGBClassifier(metric='binary:logistic', enable_categorical=True)
        #     #다중분류는 multi:softmax
        # elif self.index == 2:
        #     self.model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        # elif self.index == 3:
        #     self.model = LGBMClassifier(objective='binary',max_depth=5,n_estimators=250)
        # elif self.index == 4:
        #     self.model = LinearRegression()
        #
        # self.model.fit(X_train_scaled, y_train)

        # 성능 평가
        y_pred = self.model.predict(X_test_scaled)
        if not self.index == 5:
            accuracy = accuracy_score(y_test, y_pred)
            self.y_pred = y_pred

            print("Model accuracy:", accuracy)
            message = "Model accuracy: " + str(accuracy)

            self.show_message_box(message)
        else :
            mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
            mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            r2 = r2_score(y_test, y_pred)
            print("Model accuracy:", mae)
            message = "Model accuracy: " + str(mae)+","+ str(mse)+ ","+str(rmse)+","+str(r2)
            self.show_message_box(message)
            accuracy = r2
        return self.model, accuracy


    def show_message_box(self, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Message Box")
        msg_box.setText(message)
        msg_box.exec_()


    def build_model(self, hp):

        model = Sequential()

        # LSTM의 unit 수를 조정
        units = hp.Int('units', min_value=32, max_value=256, step=16)

        # 첫 번째 LSTM 레이어
        model.add(LSTM(units, input_shape=(None, 1), return_sequences=True, activation='tanh'))
        # 두 번째 LSTM 레이어
        model.add(LSTM(units))

        # 출력 레이어: 이진 분류를 위해 뉴런 수를 1로 조정하고 활성화 함수를 sigmoid로 변경
        model.add(Dense(1, activation='sigmoid'))

        # Optimizer의 learning rate를 조정
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # 이진 분류에 적합한 손실 함수
                      metrics=['accuracy'])

        return model

    # def lstm(self, df):
    #     """훈련"""
    #     features = df[df.columns[1:-1]]
    #     labels = df['label']
    #     X = df.iloc[:, 1:-1]
    #
    #     y = df['label']
    #     y = y.astype("int")
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #
    #     self.scaler = MinMaxScaler()
    #     X_train_scaled = self.scaler.fit_transform(X_train)
    #     X_test_scaled = self.scaler.transform(X_test)
    #
    #     X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    #     X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    #     # 패딩
    #
    #     model = Sequential()
    #     # LSTM 모델 구성
    #     model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[2])))
    #
    #     model.add(Dropout(0.2))
    #     model.add(Dense(4, activation='softmax'))
    #     model.compile(optimizer='adam',
    #                   loss='sparse_categorical_crossentropy',
    #                   metrics=['mae'])
    #
    #     # 모델 학습
    #     model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))
    #
    #
    #     self.model = model

    def lstm(self, df):
        """훈련"""
        features = df[df.columns[1:-1]]
        labels = df['label']
        X = df.iloc[:, 1:-1]

        y = df['label']
        y = y.astype("int")

        # Reshape y to be 1D array
        y = y.values.reshape(-1)
        num_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)),
                                input_shape=(1, X_train_scaled.shape[2])))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.01))))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.3))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))

        # Adjust the output layer for multi-class classification
        model.add(Dense(num_classes, activation='softmax'))

        # specify your learning rate
        learning_rate = 0.0005
        # create an Adam optimizer with the specified learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Compute class weights for balanced class training
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=1)

        # Train the model with class weights
        model.fit(X_train_scaled, y_train, epochs=300, batch_size=16, validation_data=(X_test_scaled, y_test),
                  class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr])

        self.model = model

    def save_model2(self):
        """모델 저장"""
        if self.index == 0 or self.index == 2 or self.index == 3:


            self.aimodel = self.comboBox.currentText()

            folder_path = os.getcwd()
            pklname = os.path.join(folder_path, str(self.classmode + self.aimodel + "model.pkl"))
            joblib.dump(self.model, pklname)

            self.scalername = os.path.join(folder_path, str(self.classmode + self.aimodel + "scaler.pkl"))
            with open(self.scalername, 'wb') as f:
                joblib.dump(self.scaler, f)
                f.close()

        elif self.index == 1:
            self.model.save(str(self.extension + '\\' + 'model.h5'))

    def train_baseline_model(self, df):

        X = df.iloc[:, 1:-1]
        y = df['label']
        y = y.astype("int")
        atemp = len(y.unique())
        # 교차 검증 준비
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        baseline_model = RandomForestClassifier(random_state=42)
        baseline_model.fit(X_train_scaled, y_train)

        y_pred = baseline_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print("Baseline Model accuracy:", accuracy)

        # cm = confusion_matrix(y_test, y_pred)d

        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Baseline Model Confusion Matrix')
        #plt.show()

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return baseline_model, accuracy

    def predict_data(self, df):
        """새 데이터 예측"""

        X_new = df.iloc[:, 1:]


        if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
            # y_pred_new = self.model.predict(X_new)
            # df['label'] = y_pred_new
            X_new_scaled = self.scaler.transform(X_new)

            y_pred = self.model.predict(X_new_scaled)
            df['label'] = y_pred

        elif self.index == 1:

            X = df.iloc[:, 1:]
            X_scaled = self.scaler.transform(X)
            X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

            y_pred_prob = self.model.predict(X_scaled)
            y_pred = np.argmax(y_pred_prob, axis=1)

        df['label'] = y_pred

        return df


if __name__ == "__main__":
    app = QApplication(sys.argv)
    data_preprocessor = TrainClass()




    data_preprocessor.show()
    app.exec_()
