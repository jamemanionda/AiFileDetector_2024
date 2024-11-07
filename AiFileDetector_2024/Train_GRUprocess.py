import json
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
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QMessageBox, QFileDialog, \
    QTableWidgetItem, QVBoxLayout, QLabel, QPushButton, QDialog
from PyQt5 import uic, QtWidgets
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class twoTrainClass():

    def gotrain(self, classmode, model, index, csv_path):
        self.index = index
        self.aimodel = model
        self.csv_path = csv_path
        self.classmode = classmode
        print("***이진분류 시작***")
        df, _ = self.preprocess_data(self.csv_path, is_train=True)
        try:
            df= df.drop(columns='md5')
        except Exception as e:
            pass
        self.extension = os.path.basename(os.path.dirname(self.csv_path))
        # 훈련 데이터와 테스트 데이터로 분할
        df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

        # 훈련 데이터 전처리
        # df_train = df_test.drop(columns='label')
        df_train_processed = self.apply_simhash(df_train)
        print("전처리한 훈련 데이터:")
        print(df_train_processed)

        self.feature_list = df_train.drop(columns=['label']).columns.tolist()

        # 모델 훈련
        self.train_model(df_train_processed)
        baseline_model, baseline_accuracy = self.train_baseline_model(df_train_processed)

        print("베이스라인 정확도", baseline_accuracy)

        self.save_model2()
        self.original_df_test = df_test
        df_test = df_test.drop(columns='label')

        # 테스트 데이터 전처리
        df_test_processed = self.apply_simhash(df_test)

        # 추후 변경 필요 --> 파일이름을 피처 반영되게 / csv_path랑 동일 경로에 feature.json저장
        jsonpath = os.path.join(os.path.dirname(csv_path), "feature.json")
        with open(jsonpath, 'w') as f:
            json.dump(self.feature_list, f)

        # 모델 로드 및 테스트 데이터 예측
        # self.load_model2()
        predicted_data = self.predict_data(df_test_processed)
        predicted_datalabel = predicted_data['label']
        results, success_failure, results_df = self.analyze_prediction(predicted_data,

                                                                                    self.original_df_test[['name', 'label']])
        actual_labels = self.original_df_test['label']
        actual_labels = actual_labels.astype(int)
        predicted_labels = predicted_datalabel


        #conf_matrix = self.confusion_matrix2(actual_labels, predicted_labels)
        #print(conf_matrix)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(results_df)
        print(success_failure)


        # 예측 성공률 계산

        total = len(results_df)
        success = sum([1 for row in success_failure.values() if "예측 성공" in row])
        success_rate = (success / total) * 100
        print(f"예측 성공률: {success_rate:.2f}%")

        if self.index == 4:
            threshold = 0.5
            predicted_labels = [1 if y >= threshold else 0 for y in predicted_labels]
        precision = precision_score(actual_labels, predicted_labels, average='weighted')
        recall = recall_score(actual_labels, predicted_labels, average='weighted')
        f1 = f1_score(actual_labels, predicted_labels, average='weighted')
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def train_model(self, df):
        try :
            if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
                model, accuracy = self.ensemble(df)
                message = f"정확도 {accuracy}%로 학습되었습니다."
                self.show_alert(message)
            elif self.index == 1:
                self.lstm(df)
        except Exception as e:
            #self.index = 0
            pass

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

    def train_baseline_model(self, df):

        X = df.drop(columns=['label', 'name'])

        # 'label' 컬럼을 출력 변수로 설정
        y = df['label'].astype("int")
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






        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return baseline_model, accuracy

    def preprocess_data(self, filepath, is_train=True):
        """데이터 전처리"""

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


    def save_model2(self):
        """모델 저장"""
        if self.index == 0 or self.index == 2 or self.index == 3:



            folder_path = os.getcwd()
            pklname = os.path.join(folder_path, str(self.csv_path+"_" + self.aimodel + "model.pkl"))
            joblib.dump(self.model, pklname)

            self.scalername = os.path.join(folder_path, str(self.csv_path+"_" + self.aimodel + "scaler.pkl"))
            with open(self.scalername, 'wb') as f:
                joblib.dump(self.scaler, f)
                f.close()

        elif self.index == 1:
            self.model.save(str(self.extension + '\\' + 'model_bin.h5'))

    def apply_simhash(self, df):
        """Simhash 적용"""
        df.columns = df.columns.astype(str)
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            df[column] = df[column].apply(self.calculate_simhash_lib)
        return df

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

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': self.feature_list,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            print(importance_df)
            return importance_df
        else:
            print("Model does not support feature importance.")

    def lstm(self, df):
        """LSTM 이진분류 훈련"""
        features = df.iloc[0, 1:-1].values
        labels = df['label']
        X = df.loc[:, ['name'] + [col for col in df.columns if col not in ['name', 'label']]]

        y = df['label'].astype("int")  # 레이블을 정수형으로 변환

        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # 데이터 정규화 (MinMaxScaler 적용)
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # LSTM에 맞게 입력 데이터 차원 변경 (samples, timesteps, features)
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        # LSTM 모델 정의
        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)),
                                input_shape=(1, X_train_scaled.shape[2])))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.01))))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.3))

        # 출력 레이어: 이진 분류를 위해 뉴런 수를 1로 설정하고 활성화 함수를 sigmoid로 변경
        model.add(Dense(1, activation='sigmoid'))

        # Adam Optimizer 생성 (학습률 조정)
        learning_rate = 0.0005
        optimizer = Adam(learning_rate=learning_rate)

        # 모델 컴파일 (이진 분류에 맞는 손실 함수 사용)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # 조기 종료 및 학습률 감소 콜백 정의
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=1)

        # 모델 훈련
        model.fit(X_train_scaled, y_train, epochs=300, batch_size=16, validation_data=(X_test_scaled, y_test),
                  callbacks=[early_stopping, reduce_lr])

        # 훈련 완료된 모델 저장
        self.model = model

    def ensemble(self, df):
        """이진분류를 위한 앙상블 모델 구성"""
        X = df.drop(columns=['label', 'name'])

        # 'label' 컬럼을 출력 변수로 설정
        y = df['label'].astype("int")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # MinMaxScaler로 정규화
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost, RandomForest, LGBM 등 모델 선택
        if self.index == 0:
            self.model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        elif self.index == 2:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        elif self.index == 3:
            self.model = LGBMClassifier(objective='binary', max_depth=5, n_estimators=250)
        elif self.index == 4:
            #self.model = LinearRegression()
            self.model = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='ovr')


        # 모델 훈련
        self.model.fit(X_train_scaled, y_train)

        # 성능 평가
        y_pred = self.model.predict(X_test_scaled)
        if self.index == 4:
            threshold = 0.5
            y_pred = [1 if y >= threshold else 0 for y in y_pred]
        accuracy = accuracy_score(y_test, y_pred)
        message = f"정확도 {accuracy}%로 학습되었습니다."
        self.show_alert(message)
        print(f"Model Accuracy: {accuracy:.2f}")
        self.accuracy = accuracy

        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            print("Feature Importance:")
            print(importance_df)

            # 피처 중요도 시각화
            self.plot_feature_importance(importance_df)
            importance_path = os.path.join(str("bin_" + self.aimodel + "feature_importance.pkl"))
            file_path = os.path.join(os.path.dirname(self.csv_path), importance_path)
            importance_df.to_csv(file_path, index=False)

        # 추후 변경 필요 --> 파일이름을 피처 반영되게 / self.csv_path랑 동일 경로에 feature.json저장
        self.feature_list = X.columns.tolist()
        jsonpath = os.path.join(os.path.dirname(self.csv_path), "feature.json")
        with open(jsonpath, 'w') as f:
            json.dump(self.feature_list, f)

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

    @staticmethod
    def calculate_simhash_lib(value):
        simval = Simhash(str(value)).value
        return simval

    def confusion_matrix2(self, y_true, y_pred):
        """Confusion Matrix 시각화"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        print("Classification Report:")
        print(classification_report(y_true, y_pred))
