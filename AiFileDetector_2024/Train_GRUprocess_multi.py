import json
import math
import pickle
import plotly.express as px
import pyautogui
import seaborn as sns
from PyQt5.QtCore import Qt
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
from sklearn.preprocessing import LabelEncoder
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QMessageBox, QFileDialog, \
    QTableWidgetItem, QPushButton, QVBoxLayout, QLabel, QDialog
from PyQt5 import uic, QtWidgets
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau


'''
device_lib.list_local_devices()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
form_class = uic.loadUiType("Training.ui")[0]

with tf.device('/GPU:0'):
'''


class TrainClass(QMainWindow):  # QMainWindow, form_class

    def __init__(self):
        super(TrainClass, self).__init__()
        self.choice = 0
        self.file_paths = []
        self.dpath = 'E:\\'
        self.model = None

        #self.dirModel.setRootPath("E:\\AiFileDetectorE")
        #os.chdir("E:\\AiFileDetectorE")


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

    def plot_feature_importance(self, importance_df):

        fig = px.bar(
            importance_df,
            y='Feature',
            x='Importance',
            orientation='h',  # 가로 막대 그래프
            title='Feature Importance',
            height=400 + len(importance_df) * 20  # 피처 수에 따른 그래프 높이 조정
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},  # 중요도 순서로 정렬
            showlegend=False,  # 범례

            # 비활성화
            xaxis_title='Importance',
            yaxis_title='Feature',
        )

        fig.show()




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
                success_failure[name] = "!!!예측 실패!!!"

        # Combine results and success_failure into a single DataFrame
        results_df = pd.DataFrame({
            'name': list(results.keys()),
            'result': list(results.values()),
            'success_failure': list(success_failure.values())
        })

        return results, success_failure, results_df


    def gotrain(self, classmode, model, trainindex, csv_path):
        self.model = model
        print("***ver2. 다중분류 시작***")
        print("선택한 모델 : ", model)

        self.csv_path = csv_path
        self.classmode = classmode
        self.index = trainindex
        self.aimodel = model
        df, _ = self.preprocess_data(self.csv_path, is_train=True)
        try:
            df= df.drop(columns='md5')
        except:
            pass
        self.extension = os.path.basename(os.path.dirname(self.csv_path))
        # 훈련 데이터와 테스트 데이터로 분할
        df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

        # 훈련 데이터 전처리
        #df_train = df_train.drop(columns='label')
        df_train_processed = self.apply_simhash(df_train)
        # print("전처리한 훈련 데이터:")
        # print(df_train_processed)

        # 모델 훈련
        self.train_model(df_train_processed)
        #baseline_model, baseline_accuracy =self.train_baseline_model(df_train_processed)

        #print("베이스라인 정확도", baseline_accuracy)
        print(f"----------validation--------------")
        self.save_model2()
        self.original_df_test = df_test
        df_test = df_test.drop(columns='label')
        df_test_processed = self.apply_simhash(df_test)

        predicted_data = self.predict_data(df_test_processed)
        predicted_datalabel = predicted_data['label']
        results, success_failure, results_df = self.analyze_prediction(predicted_data, self.original_df_test[['name', 'label']])
        actual_labels = self.original_df_test['label']
        actual_labels = actual_labels.astype(int)
        predicted_labels = predicted_datalabel

        conf_matrix = self.confusion_matrix2(actual_labels, predicted_labels)
        print(conf_matrix)

        pd.set_option('display.width', 1000)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        #print(success_failure)
        print(results_df)
        #
        # # 예측 성공률 계산
        #
        total = len(results_df)
        success = sum([1 for row in success_failure.values() if "예측 성공" in row])
        success_rate = (success / total) * 100

        print(f"예측 성공률: {success_rate:.2f}%")
        accuracy = accuracy_score(actual_labels, predicted_labels)
        precision = precision_score(actual_labels, predicted_labels, average = 'weighted')
        recall = recall_score(actual_labels, predicted_labels, average = 'weighted')
        f1 = f1_score(actual_labels, predicted_labels, average = 'weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")




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

        sample_df = pd.read_csv(filepath, nrows=1, header=None)
        tempvalue = sample_df.iloc[0, 0]
        # 첫 번째 행의 첫 번째 값이 'name'이 아닌 경우 두 번째 행을 헤더로 설정
        if tempvalue != 'name':
            # 첫 번째 행에 컬럼 이름이 없으면 두 번째 행을 헤더로 설정하여 다시 읽어옵니다
            df = pd.read_csv(filepath, header=1)
        else:
            # 첫 번째 행이 컬럼 이름이면 기본적으로 읽어옵니다
            df = pd.read_csv(filepath)

        column_count = df.shape[1]
        original_labels = None

        if is_train:
            features = [col for col in df.columns if col not in ['name', 'label']]
            df = df[1:]

        else:
            features = df.columns[1:-1]
            df.columns = ['name'] + list(features) + ['label']
            original_labels = df[['name', 'label']]
            df = df[1:]
        return df, original_labels

    @staticmethod
    def calculate_simhash_lib(value):
        try:
            if value in [0, None, ""] or (isinstance(value, float) and math.isnan(value)):
                return -99999999
        except Exception as e:
            pass
        try:
            try:
                simval = Simhash(str(value)).value
            except:
                simval = Simhash(str(value[:200])).value
        except Exception as e:
            print(e)
            simval = -99999999
        return simval

    def apply_simhash(self, df):
        """Simhash 적용"""
        df.columns = df.columns.astype(str)
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            df[column] = df[column].apply(self.calculate_simhash_lib)
        return df

        # df.columns = df.columns.astype(str)
        # columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        #
        # def safe_hex_to_int(value):
        #     try:
        #         # 1. 문자열 값 확인
        #
        #         try:
        #             value = int(value)
        #         except:
        #             pass
        #
        #         if isinstance(value, str):
        #             # 과학적 표기법 확인 및 처리
        #             if "E" in value.upper():
        #                 # 과학적 표기법 값을 정수로 변환
        #                 try:
        #                     changeint =  int(float(value))
        #                 except :
        #                     changeint =  int(float(value[:100]))
        #                 return changeint
        #             # 일반 문자열을 16진수로 변환
        #             else:
        #                 return value
        #
        #         # 2. 이미 숫자인 경우
        #         elif isinstance(value, (int, float)):
        #             return int(value)
        #     except Exception as e :
        #         print(e)
        #         return float('nan')
        #
        # for column in columns_to_process:
        #     df[column] = df[column].apply(safe_hex_to_int)
        #
        # return df



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

    def train_model(self, df):
        try :
            if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
                model, accuracy = self.ensemble(df)
                message = f"정확도 {accuracy}%로 학습되었습니다."
                self.show_alert(message)
            elif self.index == 1:
                self.lstm(df)
        except Exception as e:
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
        plt.savefig(self.resource_path('confusion_matrix.png'), format='png')
        # 추가적으로 classification report도 출력
        print("Classification Report:")
        print(classification_report(y_train, y_pred_classes))

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)


    def remove_highly_correlated_features(self, X, threshold=0.9):
        """
        상관관계가 높은 피처 제거 (중복 방지)
        :param X: 피처 데이터프레임
        :param threshold: 상관계수 임계값 (기본값 0.9)
        :return: 상관관계가 높은 피처가 제거된 데이터프레임
        """
        # 상관관계 행렬 계산
        corr_matrix = X.corr().abs()

        # 상삼각행렬에서 임계값을 초과하는 상관관계를 추출
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 상관계수가 threshold를 초과하는 피처 찾기
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Removing highly correlated features: {to_drop}")

        # 상관관계가 높은 피처 제거
        X = X.drop(columns=to_drop)

        return X

    def ensemble(self, df):

        names = df['name']
        labels = df['label']
        X = df.drop(columns=['label', 'name'])
        y = labels.astype("int")

        # Train-test split
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            X, y, names, test_size=0.25, random_state=42
        )

        # MinMaxScaler 적용
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_scaled = self.scaler.transform(X)
        # Define parameter grids
        params_xgb = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'n_estimators': [150, 200, 250, 300],
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

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Select and train the model based on self.index
        # Select and train the model based on self.index
        sample_weights = y_train.map(lambda x: class_weight_dict[x])
        if self.index == 0:  # XGBoost

            self.model = xgb.XGBClassifier(random_state=42)
            grid_search = GridSearchCV(self.model, params_xgb, cv=3, scoring='accuracy')
            y_train_encoded = LabelEncoder().fit_transform(y_train)
            grid_search.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights)
            self.model = grid_search.best_estimator_

        elif self.index == 2:  # RandomForest
            self.model = RandomForestClassifier(class_weight=class_weight_dict)
            grid_search = RandomizedSearchCV(self.model, params_rf, n_iter=10, cv=3, scoring='accuracy',
                                             random_state=42)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_

        elif self.index == 3:  # LightGBM

            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            scale_pos_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

            # Initialize the LightGBM model with scale_pos_weight
            self.model = LGBMClassifier(objective='multiclass', class_weight='balanced')
            grid_search = RandomizedSearchCV(self.model, params_lgbm, n_iter=10, cv=3, scoring='accuracy',
                                             random_state=42)
            grid_search.fit(X_train_scaled, y_train)  # No scale_pos_weight here
            self.model = grid_search.best_estimator_

        elif self.index == 4:  # Logistic Regression
            self.model = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial')
            self.model.fit(X_train_scaled, y_train, class_weight=sample_weights)

        # Model evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_test_labels = y_test.astype(int)
        # Combine test data with predicted labels
        df_test = pd.DataFrame(X_test, columns=X.columns)
        df_test['name'] = names_test.values
        df_test['label'] = y_pred

        # Original labels for comparison
        original_labels = pd.DataFrame({
            'name': names_test.values,
            'label': y_test.values
        })

        # Analyze prediction using the provided function
        results, success_failure, results_df = self.analyze_prediction(df_test, original_labels)

        # Print results
        print("Prediction Results:")
        print(results_df)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test_labels, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Print evaluation metrics
        if self.index != 5:
            accuracy = accuracy_score(y_test_labels, y_pred)
            precision = precision_score(y_test_labels, y_pred, average='weighted')
            recall = recall_score(y_test_labels, y_pred, average='weighted')
            f1 = f1_score(y_test_labels, y_pred, average='weighted')

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            self.accuracy = accuracy
            self.y_pred = y_pred

            message = f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            self.show_alert(message)
        else:
            # 회귀 모델 평가지표
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

            message = f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}"
            self.show_message_box(message)
            accuracy = r2

        pd.set_option('display.width', 1000)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)


        if self.index <4 :
            if hasattr(self.model, 'feature_importances_'):
                feature_importances = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)

                print("Feature Importance:")
                print(importance_df)
                self.importance_df = importance_df
                # 피처 중요도 시각화
                self.plot_feature_importance(importance_df)

                importance_path = os.path.join(str(self.aimodel + "feature_importance.csv"))
                file_path = os.path.join(os.path.dirname(self.csv_path), importance_path)
                importance_df.to_csv(file_path, index=False)
        else :
            feature_importances = np.abs(self.model.coef_[0])  # 계수의 절대값
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            print("Feature Importance:")
            print(importance_df)
            self.importance_df = importance_df
            # 피처 중요도 시각화
            self.plot_feature_importance(importance_df)


            # CSV 파일로 저장
            importance_path = os.path.join(str(self.aimodel + "feature_importance.csv"))
            file_path = os.path.join(os.path.dirname(self.csv_path), importance_path)
            importance_df.to_csv(file_path, index=False)

        # 추후 변경 필요 --> 파일이름을 피처 반영되게 / self.csv_path랑 동일 경로에 feature.json저장
        self.feature_list = X.columns.tolist()
        jsonpath = os.path.join(os.path.dirname(self.csv_path), "feature.json")
        with open(jsonpath, 'w') as f:
            json.dump(self.feature_list, f)




        return self.model, accuracy

    def show_file_alert(self, file_path, messagea):
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
        open_button.clicked.connect(lambda: self.open_csv2(file_path))  # 파일 열기 함수 호출
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

    def open_csv2(self, file_path, widgett):
        """CSV 파일을 기본 프로그램으로 엽니다."""
        try:

            os.startfile(file_path)  # 윈도우에서는 기본 프로그램으로 파일 열기

        except Exception as e:
            print(f"Error opening file: {str(e)}")
            self.show_alert(f"파일을 열 수 없습니다: {str(e)}")



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


    def lstm(self, df):
        """훈련"""
        temp_feat = df.loc[:, ['name'] + [col for col in df.columns if col not in ['name', 'label']]]

        features = temp_feat.values
        labels = df['label']
        X = df.loc[:, ['name'] + [col for col in df.columns if col not in ['name', 'label']]]

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
        if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:



            folder_path = os.getcwd()
            pklname = os.path.join(folder_path, str(self.csv_path+"_" + self.aimodel + "model.pkl"))
            joblib.dump(self.model, pklname)

            self.scalername = os.path.join(folder_path, str(self.csv_path+"_" + self.aimodel + "scaler.pkl"))
            with open(self.scalername, 'wb') as f:
                joblib.dump(self.scaler, f)
                f.close()



        elif self.index == 1:
            self.model.save(str(self.extension + '\\' + 'model.h5'))

    def train_baseline_model(self, df):

        X = df.loc[:, ['name'] + [col for col in df.columns if col not in ['name', 'label']]]
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
