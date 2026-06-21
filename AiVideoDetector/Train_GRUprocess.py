import json
import pickle
import plotly.express as px
import seaborn as sns
from PyQt5.QtCore import Qt
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, \
    precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler, LabelBinarizer
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
import traceback
from datetime import datetime

from Train_GRUprocess_multi import build_label_display_map_from_mapping_json


class twoTrainClass():

    def gotrain(self, classmode, model, index, csv_path):
        self.index = index
        self.aimodel = model
        self.csv_path = csv_path
        self.classmode = classmode
        print("***이진분류 시작***")
        print("선택한 모델 : ", model)
        try:
            if hasattr(self, "progress_callback") and callable(self.progress_callback):
                self.progress_callback("데이터 전처리 중...")
        except Exception:
            pass
        df, _ = self.preprocess_data(self.csv_path, is_train=True)
        try:
            df= df.drop(columns='md5')
        except Exception as e:
            pass
        df = self._sanitize_labels_for_training(df)
        self.extension = os.path.basename(os.path.dirname(self.csv_path))
        # 훈련 데이터와 테스트 데이터로 분할
        #df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

        # 훈련 데이터 전처리
        # df_train = df_test.drop(columns='label')
        try:
            if hasattr(self, "progress_callback") and callable(self.progress_callback):
                self.progress_callback("Simhash 적용 중...")
        except Exception:
            pass
        df_train_processed = self.apply_simhash(df)

        #self.feature_list = df_train_processed.drop(columns=['label']).columns.tolist()

        # 모델 훈련
        try:
            if hasattr(self, "progress_callback") and callable(self.progress_callback):
                self.progress_callback("모델 학습 중...")
        except Exception:
            pass
        self.train_model(df_train_processed)
        #baseline_model, baseline_accuracy = self.train_baseline_model(df_train_processed)

        #print("베이스라인 정확도", baseline_accuracy)

        if getattr(self, "save_training_outputs", True):
            self.save_model2()
        else:
            print("[INFO] 결과 저장 OFF: 모델/스케일러 저장 생략")
        #self.original_df_test = df_test
        #df_test = df_test.drop(columns='label')

        # 테스트 데이터 전처리
        #df_test_processed = self.apply_simhash(df_test)

        # 추후 변경 필요 --> 파일이름을 피처 반영되게 / csv_path랑 동일 경로에 feature.json저장
        # jsonpath = os.path.join(os.path.dirname(csv_path), "feature.json")
        # with open(jsonpath, 'w') as f:
        #     json.dump(self.feature_list, f)

        # 모델 로드 및 테스트 데이터 예측
        # self.load_model2()
        # predicted_data = self.predict_data(df_test_processed)
        # predicted_datalabel = predicted_data['label']
        # results, success_failure, results_df = self.analyze_prediction(predicted_data,
        #
        #                                                                             self.original_df_test[['name', 'label']])
        # actual_labels = self.original_df_test['label']
        # actual_labels = actual_labels.astype(int)
        # predicted_labels = predicted_datalabel


        #conf_matrix = self.confusion_matrix2(actual_labels, predicted_labels)
        #print(conf_matrix)
        # pd.set_option('display.width', 1000)
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # print(results_df)
        # print(success_failure)
        # try:
        #     print(self.importance_df)
        # except:
        #     pass
        #
        #
        # # 예측 성공률 계산
        #
        # total = len(results_df)
        # success = sum([1 for row in success_failure.values() if "성공" in row])
        # success_rate = (success / total) * 100
        # print(f"예측 성공률: {success_rate:.2f}%")
        #
        # if self.index == 4:
        #     threshold = 0.5
        #     predicted_labels = [1 if y >= threshold else 0 for y in predicted_labels]
        # precision = precision_score(actual_labels, predicted_labels, average='weighted')
        # recall = recall_score(actual_labels, predicted_labels, average='weighted')
        # f1 = f1_score(actual_labels, predicted_labels, average='weighted')
        # print(f"Accuracy: {self.accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1:.4f}")

    def train_model(self, df):
        try :
            if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
                model, accuracy = self.ensemble(df)
                message = f"정확도 {accuracy}%로 학습되었습니다."
                self.show_alert(message)
            elif self.index == 1:
                self.lstm(df)
        except Exception as e:
            tb = traceback.format_exc()
            print("[ERROR] train_model 실패:", e)
            print(tb)
            try:
                self.show_alert(f"학습 중 오류가 발생했습니다.\n\n{e}\n\n상세:\n{tb}")
            except Exception:
                pass
            raise

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

    @staticmethod
    def _feature_column_category(col_name):
        """Train_GRUprocess_multi.TrainClass와 동일 규칙 (이진 학습 피처 선택과 일치)."""
        if col_name in ('name', 'label', 'md5'):
            return None
        if col_name == 'sequence':
            return 'seq'
        if isinstance(col_name, str) and (
            col_name.startswith('SPS_') or col_name.startswith('PPS_') or col_name in ('SPS', 'PPS')
        ):
            return 'sps'
        if col_name == 'GOP':
            return 'gop'
        if col_name == 'GOP compression':
            return 'ratio'
        return 'val'

    def _sanitize_labels_for_training(self, df):
        """학습 전 label 컬럼 정제 (NaN/inf/정수아님 제거 후 int 변환)."""
        if 'label' not in df.columns:
            raise ValueError("학습 CSV에 'label' 컬럼이 없습니다.")

        out = df.copy()
        raw = pd.to_numeric(out['label'], errors='coerce')
        finite_mask = np.isfinite(raw.to_numpy(dtype=float, copy=False))
        rounded = np.round(raw)
        int_like_mask = (np.abs(raw - rounded) < 1e-9).fillna(False)
        valid_mask = finite_mask & int_like_mask.to_numpy(dtype=bool, copy=False)

        removed = int((~valid_mask).sum())
        if removed > 0:
            print(f"[WARN] label 정제: 비정상 라벨 {removed}행 제거 (NaN/inf/정수아님)")
            out = out.loc[valid_mask].copy()
            raw = raw.loc[valid_mask]

        if out.empty:
            raise ValueError("유효한 label 행이 없어 학습할 수 없습니다.")

        out['label'] = np.round(raw).astype(int).to_numpy()
        uniq = out['label'].nunique(dropna=True)
        if uniq < 2:
            raise ValueError(f"유효한 클래스 수가 부족합니다. (현재 {uniq}개)")
        return out

    def _filter_columns_by_selected_features(self, df):
        """selected_feature_set(val_seq_sps_gop_ratio)에 맞춰 학습 컬럼만 유지. 비어 있으면 전체."""
        selected = getattr(self, 'selected_feature_set', None) or ''
        feature_cols = [c for c in df.columns if c not in ['name', 'label']]
        if not selected or not isinstance(selected, str):
            return df
        want = set(s.strip().lower() for s in selected.split('_') if s.strip())
        if not want:
            return df
        keep = ['name', 'label']
        for col in df.columns:
            if col in keep:
                continue
            cat = self._feature_column_category(col)
            if cat and cat in want:
                keep.append(col)
        to_drop = [c for c in feature_cols if c not in keep]
        if not to_drop:
            return df
        return df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')

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
            # 다중분류(Train_GRUprocess_multi)와 동일: Train 탭에서 선택한 피처만 학습에 사용
            df = self._filter_columns_by_selected_features(df)

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
        """모델 저장 (다중분류와 동일: training_results run 폴더가 있으면 그 안에 model.pkl / scaler.pkl 또는 model.h5)"""
        if not getattr(self, "save_training_outputs", True):
            print("[INFO] 결과 저장 OFF: save_model2 생략")
            return
        if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
            run_dir = getattr(self, '_run_output_dir', None)
            if run_dir and os.path.isdir(run_dir):
                folder_path = run_dir
                pklname = os.path.join(folder_path, "model.pkl")
                self.scalername = os.path.join(folder_path, "scaler.pkl")
            else:
                folder_path = os.getcwd()
                pklname = os.path.join(folder_path, str(self.csv_path + "_" + self.aimodel + "model.pkl"))
                self.scalername = os.path.join(folder_path, str(self.csv_path + "_" + self.aimodel + "scaler.pkl"))
            joblib.dump(self.model, pklname)
            with open(self.scalername, 'wb') as f:
                joblib.dump(self.scaler, f)
                f.close()
            print(f"[INFO] 모델/스케일러 저장: {folder_path}")

        elif self.index == 1:
            run_dir = getattr(self, '_run_output_dir', None)
            if run_dir and os.path.isdir(run_dir):
                h5_path = os.path.join(run_dir, "model.h5")
                self.model.save(h5_path)
                print(f"[INFO] LSTM 모델 저장: {h5_path}")
            else:
                self.model.save(str(self.extension + '\\' + 'model_bin.h5'))

    def apply_simhash(self, df):
        """Simhash 적용"""
        df.columns = df.columns.astype(str)
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            if column == 'sequence':
                # sequence는 0도 유효값으로 취급 (결측과 구분)
                df[column] = df[column].apply(
                    lambda v: self.calculate_simhash_lib(v, zero_as_missing=False, missing_sentinel=-99999998)
                )
            else:
                df[column] = df[column].apply(self.calculate_simhash_lib)
        return df

    def plot_feature_importance(self, importance_df):
        if not getattr(self, 'show_feature_importance_plot', False):
            return
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
        # 다중분류와 동일: 케이스/training_results/{모델}_{타임스탬프}/ 에 모델·히스토리 저장
        base_dir = os.path.dirname(self.csv_path)
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.aimodel
        if getattr(self, "save_training_outputs", True):
            run_folder = os.path.join(base_dir, "training_results", f"{model_name}_{self._run_timestamp}")
            os.makedirs(run_folder, exist_ok=True)
            self._run_output_dir = run_folder
            print(f"[INFO] 학습 결과 저장 폴더: {run_folder}")
        else:
            self._run_output_dir = None
            print("[INFO] 결과 저장 OFF: training_results 폴더 생성 생략")

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
        # 다중분류와 동일: 학습 산출물을 training_results/{모델}_{타임스탬프}/ 에 저장
        base_dir = os.path.dirname(self.csv_path)
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.aimodel
        if getattr(self, "save_training_outputs", True):
            run_folder = os.path.join(base_dir, "training_results", f"{model_name}_{self._run_timestamp}")
            os.makedirs(run_folder, exist_ok=True)
            self._run_output_dir = run_folder
            print(f"[INFO] 학습 결과 저장 폴더: {run_folder}")
        else:
            self._run_output_dir = None
            print("[INFO] 결과 저장 OFF: training_results 폴더 생성 생략")

        names = df['name']
        X = df.drop(columns=['label', 'name'])

        # 'label' 컬럼을 출력 변수로 설정
        y = df['label'].astype("int")

        # 데이터 분할
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            X, y, names, test_size=0.25, random_state=42
        )

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

        pd.set_option('display.width', 1000)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(results_df)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test_labels, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)


        y_scores = self.model.predict_proba(X_test_scaled)
        y_scores_positive = y_scores[:, 1]
        lb = LabelBinarizer()
        y_test_binarized = lb.fit_transform(y_test)

        # Align y_scores with the expected output
        # Ensure y_scores has the same number of columns as unique classes in y_train
        if y_scores.shape[1] != len(lb.classes_):
            # Add missing classes with zero probabilities
            full_scores = np.zeros((y_scores.shape[0], len(lb.classes_)))
            for idx, cls in enumerate(lb.classes_):
                if cls in self.model.classes_:
                    full_scores[:, idx] = y_scores[:, list(self.model.classes_).index(cls)]
            y_scores = full_scores

        # Compute AUROC
        auroc = roc_auc_score(y_test_binarized, y_scores_positive)

        # AUPR (multi-class)
        from sklearn.metrics import average_precision_score
        aupr = average_precision_score(y_test_binarized, y_scores_positive)

        self.auroc = auroc
        self.aupr = aupr

        print(f"AUROC: {auroc:.6f}")
        print(f"AUPR: {aupr:.6f}")

        # Print evaluation metrics
        if self.index != 5:
            accuracy = accuracy_score(y_test_labels, y_pred)
            unique_labels = np.unique(np.concatenate([np.asarray(y_test_labels), np.asarray(y_pred)]))
            is_binary = len(unique_labels) == 2
            pos_label = None
            if is_binary:
                _mc = getattr(self.model, "classes_", None)
                model_classes = list(_mc) if _mc is not None else []
                if len(model_classes) >= 2:
                    pos_label = model_classes[1]
                else:
                    pos_label = sorted(unique_labels, key=lambda x: str(x))[1]

            if is_binary:
                weightedprecision = precision_score(
                    y_test_labels, y_pred, average='weighted', zero_division=0
                )
                microprecision = precision_score(
                    y_test_labels, y_pred, average='micro', zero_division=0
                )
                macroprecision = precision_score(
                    y_test_labels, y_pred, average='binary', pos_label=pos_label, zero_division=0
                )
                weightedrecall = recall_score(
                    y_test_labels, y_pred, average='weighted', zero_division=0
                )
                microrecall = recall_score(
                    y_test_labels, y_pred, average='micro', zero_division=0
                )
                macrorecall = recall_score(
                    y_test_labels, y_pred, average='binary', pos_label=pos_label, zero_division=0
                )
                weightedf1 = f1_score(
                    y_test_labels, y_pred, average='weighted', zero_division=0
                )
                microf1 = f1_score(
                    y_test_labels, y_pred, average='micro', zero_division=0
                )
                macrof1 = f1_score(
                    y_test_labels, y_pred, average='binary', pos_label=pos_label, zero_division=0
                )
            else:
                weightedprecision = precision_score(
                    y_test_labels, y_pred, average='weighted', zero_division=0
                )
                microprecision = precision_score(
                    y_test_labels, y_pred, average='micro', zero_division=0
                )
                macroprecision = precision_score(
                    y_test_labels, y_pred, average='macro', zero_division=0
                )
                weightedrecall = recall_score(
                    y_test_labels, y_pred, average='weighted', zero_division=0
                )
                microrecall = recall_score(
                    y_test_labels, y_pred, average='micro', zero_division=0
                )
                macrorecall = recall_score(
                    y_test_labels, y_pred, average='macro', zero_division=0
                )
                weightedf1 = f1_score(
                    y_test_labels, y_pred, average='weighted', zero_division=0
                )
                microf1 = f1_score(
                    y_test_labels, y_pred, average='micro', zero_division=0
                )
                macrof1 = f1_score(
                    y_test_labels, y_pred, average='macro', zero_division=0
                )

            print(f"Accuracy: {accuracy:.4f}")

            print("*********Precision*************")
            print(f"wightedPrecision: {weightedprecision:.4f}")
            print(f"microPrecision: {microprecision:.4f}")
            if is_binary:
                print(f"binaryPrecision(pos={pos_label}): {macroprecision:.4f}")
            else:
                print(f"macroPrecision: {macroprecision:.4f}")

            print("*********Recall*************")
            print(f"wightedRecall: {weightedrecall:.4f}")
            print(f"microRecall: {microrecall:.4f}")
            if is_binary:
                print(f"binaryRecall(pos={pos_label}): {macrorecall:.4f}")
            else:
                print(f"macroRecall: {macrorecall:.4f}")

            print("*********f1score*************")
            print(f"weightedF1 Score: {weightedf1:.4f}")
            print(f"microF1 Score: {microf1:.4f}")
            if is_binary:
                print(f"binaryF1 Score(pos={pos_label}): {macrof1:.4f}")
            else:
                print(f"macroF1 Score: {macrof1:.4f}")
            self.accuracy = accuracy
            self.y_pred = y_pred

            print("**************************")
            print("**************************")
            print("**************************")
            print(f"AUROC: {auroc:.6f}")
            print(f"AUPR: {aupr:.6f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"macroPrecision: {macroprecision:.4f}")
            print(f"macroRecall: {macrorecall:.4f}")
            print(f"macroF1 Score: {macrof1:.4f}")


            message = f"Accuracy: {accuracy:.4f}, Precision: {macroprecision:.4f}, Recall: {macrorecall:.4f}, F1 Score: {macrof1:.4f}"
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

                importance_path = str(self.aimodel + "feature_importance.csv")
                save_dir = getattr(self, '_run_output_dir', None) or os.path.dirname(self.csv_path)
                file_path = os.path.join(save_dir, importance_path)
                if getattr(self, "save_training_outputs", True):
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
            importance_path = str(self.aimodel + "feature_importance.csv")
            save_dir = getattr(self, '_run_output_dir', None) or os.path.dirname(self.csv_path)
            file_path = os.path.join(save_dir, importance_path)
            if getattr(self, "save_training_outputs", True):
                importance_df.to_csv(file_path, index=False)

        # 학습에 사용된 피처 목록 (다중분류와 동일: config/feature.json)
        self.feature_list = X.columns.tolist()
        _base = os.path.dirname(self.csv_path)
        if getattr(self, "save_training_outputs", True):
            config_dir = os.path.join(_base, "config")
            os.makedirs(config_dir, exist_ok=True)
            jsonpath = os.path.join(config_dir, "feature.json")
            with open(jsonpath, 'w') as f:
                json.dump(self.feature_list, f)
            print(f"[INFO] feature.json 생성: {jsonpath} (피처 {len(self.feature_list)}개, CSV: {os.path.basename(self.csv_path)})")

        # 학습 결과 저장 (히스토리)
        if self.index != 5 and getattr(self, "save_training_outputs", True):  # 분류 모델인 경우에만
            self.save_training_history(
                results_df, original_labels, conf_matrix, y_test_labels, y_pred, y_pred,
                accuracy, weightedprecision, macroprecision, weightedrecall, macrorecall,
                weightedf1, macrof1, auroc, aupr
            )

        # print("******************은지********************")
        # print(recall_score(y_test_labels, y_pred, average=None))  # 클래스별 Recall
        # print(classification_report(y_test_labels, y_pred))


        return self.model, accuracy




        # 성능 평가
        # y_pred = self.model.predict(X_test_scaled)
        # if self.index == 4:
        #     threshold = 0.5
        #     y_pred = [1 if y >= threshold else 0 for y in y_pred]
        # accuracy = accuracy_score(y_test, y_pred)
        # message = f"정확도 {accuracy}%로 학습되었습니다."
        # self.show_alert(message)
        # print(f"Model Accuracy: {accuracy:.2f}")
        # self.accuracy = accuracy
        #
        # if hasattr(self.model, 'feature_importances_'):
        #     feature_importances = self.model.feature_importances_
        #     importance_df = pd.DataFrame({
        #         'Feature': X.columns,
        #         'Importance': feature_importances
        #     }).sort_values(by='Importance', ascending=False)
        #
        #     print("Feature Importance:")
        #     self.importance_df = importance_df
        #
        #     # 피처 중요도 시각화
        #     self.plot_feature_importance(importance_df)
        #     importance_path = os.path.join(str("bin_" + self.aimodel + "feature_importance.csv"))
        #     file_path = os.path.join(os.path.dirname(self.csv_path), importance_path)
        #     importance_df.to_csv(file_path, index=False)
        #
        # # 추후 변경 필요 --> 파일이름을 피처 반영되게 / self.csv_path랑 동일 경로에 feature.json저장
        # self.feature_list = X.columns.tolist()
        # jsonpath = os.path.join(os.path.dirname(self.csv_path), "feature.json")
        # with open(jsonpath, 'w') as f:
        #     json.dump(self.feature_list, f)

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
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 텍스트 선택/복사 가능
        message_label.setWordWrap(True)
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

    def _load_label_mapping_for_history(self, base_dir):
        """Train_GRUprocess_multi.TrainClass와 동일 경로에서 label_mapping.json 로드."""
        if not getattr(self, "csv_path", None):
            return None
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        case_config = None
        if getattr(self, "case_direc", None) and str(self.case_direc).strip():
            case_config = os.path.join(os.path.abspath(self.case_direc), "config")
        base_config = os.path.join(base_dir, "config")
        possible_paths = []
        if getattr(self, "mapping_json_path", None) and os.path.isfile(self.mapping_json_path):
            possible_paths.append(os.path.abspath(self.mapping_json_path))
        if case_config and os.path.isdir(case_config):
            possible_paths.extend(
                [
                    os.path.join(case_config, f"{csv_name}_label_mapping.json"),
                    os.path.join(case_config, "label_mapping.json"),
                ]
            )
        possible_paths.extend(
            [
                os.path.join(base_config, f"{csv_name}_label_mapping.json"),
                os.path.join(base_config, "label_mapping.json"),
                os.path.join(base_dir, f"{csv_name}_label_mapping.json"),
                os.path.join(base_dir, "label_mapping.json"),
                os.path.join(os.path.dirname(base_dir), f"{csv_name}_label_mapping.json"),
            ]
        )
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return build_label_display_map_from_mapping_json(data)
                except Exception as e:
                    print(f"[WARN] 레이블 매핑 로드 실패 {path}: {e}")
        return None

    @staticmethod
    def _label_to_text(label_val, label_map):
        if not label_map:
            return str(label_val)
        raw = str(label_val).strip() if label_val is not None else ""
        if raw in label_map:
            return label_map[raw]
        try:
            key = str(int(float(label_val)))
            return label_map.get(key, raw or str(label_val))
        except (ValueError, TypeError, OverflowError):
            return label_map.get(raw, str(label_val))

    @staticmethod
    def calculate_simhash_lib(value, zero_as_missing=True, missing_sentinel=-99999999):
        """Detect와 동일: 1.0, "1.0", 1, "1" -> "1"로 정규화하여 simhash 일치"""
        try:
            if value in [None, ""] or (isinstance(value, float) and np.isnan(value)):
                return missing_sentinel
            if zero_as_missing:
                try:
                    if float(value) == 0.0:
                        return missing_sentinel
                except Exception:
                    pass
        except Exception:
            pass
        try:
            v = float(value)
            if not np.isfinite(v):  # inf, -inf, NaN 처리 (int 변환 시 OverflowError 방지)
                return missing_sentinel
            if v == int(v):
                value = str(int(v))
            else:
                value = str(v)
        except (ValueError, TypeError, OverflowError):
            value = str(value)
        try:
            try:
                simval = Simhash(str(value)).value
            except Exception:
                simval = Simhash(str(value)[:200]).value
        except Exception:
            simval = missing_sentinel
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

    def save_training_history(self, results_df, original_labels, conf_matrix, y_test_labels, y_pred, y_pred_display,
                              accuracy, weightedprecision, macroprecision, weightedrecall, macrorecall,
                              weightedf1, macrof1, auroc, aupr):
        """학습 결과를 다중분류와 동일하게 training_results/{모델}_{타임스탬프}/ 및 config/training_history_all.json 에 저장"""
        if not getattr(self, "save_training_outputs", True):
            print("[INFO] 결과 저장 OFF: 학습 히스토리 저장 생략")
            return
        try:
            base_dir = os.path.dirname(self.csv_path)
            timestamp = getattr(self, '_run_timestamp', None) or datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.aimodel
            run_folder = getattr(self, '_run_output_dir', None)
            if not run_folder or not os.path.isdir(run_folder):
                run_folder = os.path.join(base_dir, "training_results", f"{model_name}_{timestamp}")
                os.makedirs(run_folder, exist_ok=True)
                self._run_output_dir = run_folder

            # 1. 예측 결과 전체 저장 (label_mapping.json 기준 표시명 + 원시 클래스 값)
            prediction_results = results_df.copy()
            label_map = self._load_label_mapping_for_history(base_dir)
            ar = np.asarray(original_labels["label"].values)
            pr = np.asarray(y_pred_display)
            prediction_results["실제_레이블_raw"] = ar
            prediction_results["예측_레이블_raw"] = pr
            to_name = lambda v: self._label_to_text(v, label_map)
            prediction_results["실제_레이블"] = [to_name(x) for x in ar]
            prediction_results["예측_레이블"] = [to_name(x) for x in pr]
            prediction_results["성공여부"] = (ar == pr)
            prediction_results["성공여부"] = prediction_results["성공여부"].map({True: "성공", False: "실패"})

            prediction_file = os.path.join(run_folder, f"training_history_{model_name}_{timestamp}_predictions.csv")
            prediction_results.to_csv(prediction_file, index=False, encoding='utf-8-sig')
            print(f"[INFO] 예측 결과 저장: {prediction_file}")

            # 2. 오탐 케이스만 저장 (실제 레이블과 예측 레이블이 다른 경우)
            misclassified = prediction_results[prediction_results['성공여부'] == '실패'].copy()
            if len(misclassified) > 0:
                misclassified_file = os.path.join(run_folder, f"training_history_{model_name}_{timestamp}_misclassified.csv")
                misclassified.to_csv(misclassified_file, index=False, encoding='utf-8-sig')
                print(f"[INFO] 오탐 케이스 저장: {misclassified_file} ({len(misclassified)}건)")

            # 3. Confusion Matrix 저장
            conf_matrix_df = pd.DataFrame(conf_matrix)
            conf_matrix_file = os.path.join(run_folder, f"training_history_{model_name}_{timestamp}_confusion_matrix.csv")
            conf_matrix_df.to_csv(conf_matrix_file, index=False, encoding='utf-8-sig')
            print(f"[INFO] Confusion Matrix 저장: {conf_matrix_file}")

            # 4. Classification Report 저장
            report = classification_report(y_test_labels, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_file = os.path.join(run_folder, f"training_history_{model_name}_{timestamp}_classification_report.csv")
            report_df.to_csv(report_file, index=True, encoding='utf-8-sig')
            print(f"[INFO] Classification Report 저장: {report_file}")

            # 5. 학습 히스토리 JSON 저장 (메트릭 요약)
            selected_features = getattr(self, 'selected_feature_set', None) or ''
            history_data = {
                'timestamp': timestamp,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': model_name,
                'run_folder': run_folder,
                'classmode': self.classmode,
                'csv_path': self.csv_path,
                'selected_features': selected_features,
                'metrics': {
                    'accuracy': float(accuracy),
                    'weighted_precision': float(weightedprecision),
                    'macro_precision': float(macroprecision),
                    'weighted_recall': float(weightedrecall),
                    'macro_recall': float(macrorecall),
                    'weighted_f1': float(weightedf1),
                    'macro_f1': float(macrof1),
                    'auroc': float(auroc),
                    'aupr': float(aupr)
                },
                'test_samples': len(y_test_labels),
                'correct_predictions': int(np.sum(y_test_labels == y_pred)),
                'misclassified': int(np.sum(y_test_labels != y_pred)),
                'misclassification_rate': float(np.sum(y_test_labels != y_pred) / len(y_test_labels))
            }

            history_file = os.path.join(run_folder, f"training_history_{model_name}_{timestamp}_summary.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 학습 히스토리 저장: {history_file}")

            # 6. 전체 히스토리 (다중분류와 동일: config/training_history_all.json)
            config_dir = os.path.join(base_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            all_history_file = os.path.join(config_dir, "training_history_all.json")
            if os.path.exists(all_history_file):
                with open(all_history_file, 'r', encoding='utf-8') as f:
                    all_history = json.load(f)
            else:
                all_history = []

            all_history.append(history_data)
            with open(all_history_file, 'w', encoding='utf-8') as f:
                json.dump(all_history, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 전체 히스토리 업데이트: {all_history_file}")

        except Exception as e:
            print(f"[WARN] 학습 결과 저장 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

        print("Classification Report:")
        print(classification_report(y_test_labels, y_pred))
