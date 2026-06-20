import json
import math
import pickle
import re
import plotly.express as px
import pyautogui
import seaborn as sns
from PyQt5.QtCore import Qt
from keras.optimizers import Adam
from keras.regularizers import l2
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
from sklearn.preprocessing import LabelEncoder
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QMessageBox, QFileDialog, \
    QTableWidgetItem, QPushButton, QVBoxLayout, QLabel, QDialog
from PyQt5 import uic, QtWidgets
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import traceback
import time
from datetime import datetime


def _numeric_class_atoms_from_label_cell(label_cell):
    """숫자 클래스 목록이면 정규화 원자 리스트, 아니면 None(OEI·'O / E / I' 등은 한 덩어리)."""
    s = str(label_cell or "").strip()
    if not s:
        return None
    if " / " in s:
        parts = [p.strip() for p in re.split(r"\s*/\s*", s) if p.strip()]
    else:
        parts = [s]
    norm = []
    for p in parts:
        try:
            v = float(p)
            if not np.isfinite(v):
                return None
            norm.append(str(int(v)))
        except (ValueError, TypeError, OverflowError):
            return None
    return norm


def build_label_display_map_from_mapping_json(data):
    """
    label_mapping.json 전체 dict -> { '0': 'GALAXYS7EDGE', '1': 'G8', ... }.
    우선순위: groups_detail > label_name_display_overrides > label_to_group.
    숫자 복합('0 / 1 / 2')만 원자별 매핑 추가. 문자 복합('O / E / I')은 전체 문자열 키만 사용.
    """
    out = {}
    if not isinstance(data, dict):
        return out

    def add_keys(raw_lbl, display, only_if_missing=False):
        if display is None or not str(display).strip():
            return
        disp = str(display).strip()
        s = str(raw_lbl).strip() if raw_lbl is not None else ""
        if not s:
            return
        keys = [s]
        try:
            nk = str(int(float(s)))
            if nk not in keys:
                keys.append(nk)
        except (ValueError, TypeError, OverflowError):
            pass
        for key in keys:
            if only_if_missing and key in out:
                continue
            if not only_if_missing or key not in out:
                out[key] = disp

    for entry in data.get("groups_detail", []) or data.get("groups", []) or []:
        if not isinstance(entry, dict):
            continue
        lnm = entry.get("label_name")
        if lnm is None or not str(lnm).strip():
            continue
        disp = str(lnm).strip()
        lbl_raw = entry.get("label", "")
        lbl = str(lbl_raw).strip() if lbl_raw is not None else ""
        if not lbl:
            continue
        add_keys(lbl, disp)
        atoms = _numeric_class_atoms_from_label_cell(lbl)
        if atoms:
            for ak in atoms:
                add_keys(ak, disp, only_if_missing=True)

    for ent in data.get("label_name_display_overrides") or []:
        if not isinstance(ent, dict):
            continue
        fl = ent.get("full_label", ent.get("label"))
        disp = ent.get("display", "")
        if fl is not None and str(disp).strip():
            add_keys(fl, disp)

    ltg = data.get("label_to_group") or {}
    if isinstance(ltg, dict):
        for k, v in ltg.items():
            ks = str(k).strip() if k is not None else ""
            vs = str(v).strip() if v is not None else ""
            if not ks or not vs:
                continue
            try:
                nk = str(int(float(ks)))
            except (ValueError, TypeError, OverflowError):
                nk = ks
            if ks not in out and nk not in out:
                add_keys(ks, vs)
            ltg_atoms = _numeric_class_atoms_from_label_cell(ks)
            if ltg_atoms and vs and vs != ks:
                for ak in ltg_atoms:
                    add_keys(ak, vs, only_if_missing=True)

    if not out:
        for entry in data.get("groups_detail", []) or data.get("groups", []) or []:
            if not isinstance(entry, dict):
                continue
            lbl = str(entry.get("label", "")).strip()
            grp = str(entry.get("group", "") or entry.get("pattern", "")).strip()
            if lbl and grp and lbl not in out:
                add_keys(lbl, grp)
    return out


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
        print("학습csv : ", csv_path)
        print("선택한 모델 : ", model)

        try:
            if hasattr(self, "progress_callback") and callable(self.progress_callback):
                self.progress_callback("데이터 전처리 중...")
        except Exception:
            pass

        self.csv_path = csv_path
        self.classmode = classmode
        self.index = trainindex
        self.aimodel = model
        df, _ = self.preprocess_data(self.csv_path, is_train=True)
        try:
            df= df.drop(columns='md5')
        except:
            pass
        df = self._sanitize_labels_for_training(df)
        self.extension = os.path.basename(os.path.dirname(self.csv_path))
        # 훈련 데이터와 테스트 데이터로 분할
        #df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

        # 훈련 데이터 전처리
        #df_train = df_train.drop(columns='label')
        try:
            if hasattr(self, "progress_callback") and callable(self.progress_callback):
                self.progress_callback("Simhash 적용 중...")
        except Exception:
            pass
        df_train_processed = self.apply_simhash(df)

        # 모델 훈련
        try:
            if hasattr(self, "progress_callback") and callable(self.progress_callback):
                self.progress_callback("모델 학습/탐색 중... (시간이 걸릴 수 있습니다)")
        except Exception:
            pass
        self.train_model(df_train_processed)
        #baseline_model, baseline_accuracy =self.train_baseline_model(df_train_processed)

        #print("베이스라인 정확도", baseline_accuracy)
        print(f"----------validation--------------")
        if getattr(self, "save_training_outputs", True):
            self.save_model2()
        else:
            print("[INFO] 결과 저장 OFF: 모델/스케일러 저장 생략")
        # self.original_df_test = df_test
        # df_test = df_test.drop(columns='label')
        # df_test_processed = self.apply_simhash(df_test)
        #
        # predicted_data = self.predict_data(df_test_processed)
        # predicted_datalabel = predicted_data['label']
        # results, success_failure, results_df = self.analyze_prediction(predicted_data, self.original_df_test[['name', 'label']])
        # actual_labels = self.original_df_test['label']
        # actual_labels = actual_labels.astype(int)
        # predicted_labels = predicted_datalabel
        #
        # conf_matrix = self.confusion_matrix2(actual_labels, predicted_labels)
        # print(conf_matrix)
        #
        # pd.set_option('display.width', 1000)
        #
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # #print(success_failure)
        # print(results_df)
        # #
        # # # 예측 성공률 계산
        # #
        # total = len(results_df)
        # success = sum([1 for row in success_failure.values() if "예측 성공" in row])
        # success_rate = (success / total) * 100
        #
        # print(f"예측 성공률: {success_rate:.2f}%")
        # accuracy = accuracy_score(actual_labels, predicted_labels)
        # precision = precision_score(actual_labels, predicted_labels, average = 'weighted')
        # recall = recall_score(actual_labels, predicted_labels, average = 'weighted')
        # f1 = f1_score(actual_labels, predicted_labels, average = 'weighted')
        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1:.4f}")





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

    @staticmethod
    def _feature_column_category(col_name):
        """
        컬럼 이름 → Create 피처 종류 매핑.
        - sequence → structure sequence (seq)
        - SPS_* / PPS_* → SPS/PPS (sps)
        - GOP → GOP frame (gop)
        - 그 외 → structure value (val)
        Create에서 피처 해제/설정에 따라 학습 시 해당 컬럼 삭제/추가됨.
        """
        if col_name in ('name', 'label', 'md5'):
            return None
        if col_name == 'sequence':
            return 'seq'   # structure sequence
        if isinstance(col_name, str) and (col_name.startswith('SPS_') or col_name.startswith('PPS_') or col_name in ('SPS', 'PPS')):
            return 'sps'   # SPS/PPS
        if col_name == 'GOP':
            return 'gop'   # GOP frame
        if col_name == 'GOP compression':
            return 'ratio'
        return 'val'       # structure value (그 외 전체)

    def _sanitize_labels_for_training(self, df):
        """
        학습 전 label 컬럼을 안전하게 정제한다.
        - NaN/inf 제거
        - 정수로 해석 불가한 값 제거
        - 정수형 라벨로 변환
        """
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
        """선택된 피처(selected_feature_set)에 해당하는 컬럼만 남긴다. 비어 있으면 필터 없음."""
        selected = getattr(self, 'selected_feature_set', None) or ''
        feature_cols = [c for c in df.columns if c not in ['name', 'label']]
        print(f"[DEBUG 피처필터] CSV 피처 컬럼 수: {len(feature_cols)}, selected_feature_set='{selected}'")
        if not selected or not isinstance(selected, str):
            print(f"[DEBUG 피처필터] selected_feature_set 비어 있음 → 필터 없이 전체 사용")
            return df
        want = set(s.strip().lower() for s in selected.split('_') if s.strip())
        if not want:
            print(f"[DEBUG 피처필터] 파싱된 want 비어 있음 → 필터 없이 전체 사용")
            return df
        print(f"[DEBUG 피처필터] 사용할 피처 종류( want ): {sorted(want)}")
        # 컬럼별 분류 수집 (디버깅용)
        col_to_cat = {}
        for col in feature_cols:
            col_to_cat[col] = self._feature_column_category(col)
        by_cat = {}
        for col, cat in col_to_cat.items():
            by_cat.setdefault(cat or '?', []).append(col)
        for cat in sorted(by_cat.keys()):
            print(f"[DEBUG 피처필터]   분류 '{cat}': {len(by_cat[cat])}개 컬럼")
        keep = ['name', 'label']
        for col in df.columns:
            if col in keep:
                continue
            cat = self._feature_column_category(col)
            if cat and cat in want:
                keep.append(col)
        to_drop = [c for c in feature_cols if c not in keep]
        kept_feature_count = len(keep) - 2
        if not to_drop:
            print(f"[DEBUG 피처필터] 제거할 컬럼 없음. 최종 학습 피처 수: {kept_feature_count}")
            return df
        df = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')
        # 제거된 컬럼을 분류별로 요약 (None 분류는 '?'로 통일해 sorted 호환)
        dropped_by_cat = {}
        for c in to_drop:
            cat = col_to_cat.get(c) or '?'
            dropped_by_cat.setdefault(cat, []).append(c)
        print(f"[DEBUG 피처필터] 제거된 컬럼(분류별): ", end="")
        for cat in sorted(dropped_by_cat.keys()):
            n = len(dropped_by_cat[cat])
            print(f"'{cat}' {n}개 ", end="")
        print("")
        max_show = 15
        for cat in sorted(dropped_by_cat.keys()):
            names = dropped_by_cat[cat]
            show = names[:max_show]
            suffix = f" ... 외 {len(names)-max_show}개" if len(names) > max_show else ""
            print(f"[DEBUG 피처필터]   제거 '{cat}': {show}{suffix}")
        print(f"[DEBUG 피처필터] 유지된 컬럼(분류별): ", end="")
        kept_by_cat = {}
        for c in (c for c in keep if c not in ('name', 'label')):
            cat = col_to_cat.get(c) or '?'
            kept_by_cat.setdefault(cat, []).append(c)
        for cat in sorted(kept_by_cat.keys()):
            print(f"'{cat}' {len(kept_by_cat[cat])}개 ", end="")
        print("")
        print(f"[DEBUG 피처필터] 결과: {len(feature_cols)}개 → {kept_feature_count}개 사용, {len(to_drop)}개 제외")
        return df

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
            # Create에서 선택한 피처(val/seq/sps/gop)만 사용
            df = self._filter_columns_by_selected_features(df)

        else:
            features = df.columns[1:-1]
            df.columns = ['name'] + list(features) + ['label']
            original_labels = df[['name', 'label']]
            df = df[1:]
        return df, original_labels

    @staticmethod
    def calculate_simhash_lib(value, zero_as_missing=True, missing_sentinel=-99999999):
        """Detect와 동일: 1.0, '1.0', 1, '1' 정규화 후 simhash 계산."""
        try:
            if value in [None, ""] or (isinstance(value, float) and math.isnan(value)):
                return missing_sentinel
            if zero_as_missing:
                try:
                    if float(value) == 0.0:
                        return missing_sentinel
                except Exception:
                    pass
        except Exception:
            pass
        # 숫자 정규화: pandas가 CSV에서 1->1.0으로 읽으므로 1.0과 1을 동일하게
        try:
            v = float(value)
            if not math.isfinite(v):  # inf, -inf, NaN 처리 (int 변환 시 OverflowError 방지)
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
                simval = Simhash(str(value[:200])).value
        except Exception as e:
            print(e)
            simval = missing_sentinel
        return simval

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
        """config 폴더(케이스 루트) 및 학습 CSV 디렉터리에서 label_mapping.json을 찾아 레이블 번호 -> 텍스트 이름 딕셔너리 반환."""
        if not getattr(self, 'csv_path', None):
            return None
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        # 1) 케이스 루트의 config 폴더 우선 (createtraining에서 case_direc 전달 시)
        case_config = None
        if getattr(self, 'case_direc', None) and str(self.case_direc).strip():
            case_config = os.path.join(os.path.abspath(self.case_direc), "config")
        # 2) CSV와 같은 디렉터리 안의 config
        base_config = os.path.join(base_dir, "config")
        possible_paths = []
        # createtraining에서 이미 찾은 매핑 경로(보통 config 내)가 있으면 최우선 사용
        if getattr(self, 'mapping_json_path', None) and os.path.isfile(self.mapping_json_path):
            possible_paths.append(os.path.abspath(self.mapping_json_path))
        if case_config and os.path.isdir(case_config):
            possible_paths.extend([
                os.path.join(case_config, f"{csv_name}_label_mapping.json"),
                os.path.join(case_config, "label_mapping.json"),
            ])
        possible_paths.extend([
            os.path.join(base_config, f"{csv_name}_label_mapping.json"),
            os.path.join(base_config, "label_mapping.json"),
            os.path.join(base_dir, f"{csv_name}_label_mapping.json"),
            os.path.join(base_dir, "label_mapping.json"),
            os.path.join(os.path.dirname(base_dir), f"{csv_name}_label_mapping.json"),
        ])
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return build_label_display_map_from_mapping_json(data)
                except Exception as e:
                    print(f"[WARN] 레이블 매핑 로드 실패 {path}: {e}")
        return None

    def _label_to_text(self, label_val, label_to_group):
        """레이블 값을 매핑 dict로 표시 문자열로 변환. 없으면 원값 문자열."""
        if not label_to_group:
            return str(label_val)
        raw = str(label_val).strip() if label_val is not None else ""
        if raw in label_to_group:
            return label_to_group[raw]
        try:
            key = str(int(float(label_val)))
            return label_to_group.get(key, raw or str(label_val))
        except (ValueError, TypeError, OverflowError):
            return label_to_group.get(raw, str(label_val))

    @staticmethod
    def _filename_to_group_pattern(filename):
        """파일명에서 그룹 패턴 추출 (예: sn_OEI_563.mp4 -> sn_OEI_*). JSON 매핑과 무관하게 실제 파일 기준."""
        if not filename or not isinstance(filename, str):
            return str(filename) if filename else ''
        name = os.path.splitext(filename.strip())[0]
        if not name:
            return filename.strip()
        # 끝의 _숫자(들) 를 _* 로 치환 (예: sn_OEI_563 -> sn_OEI_*)
        pattern = re.sub(r'_\d+$', '_*', name)
        return pattern if pattern else name

    def save_training_history(self, results_df, original_labels, conf_matrix, y_test_labels, y_pred, y_pred_display,
                              accuracy, weightedprecision, macroprecision, weightedrecall, macrorecall,
                              weightedf1, macrof1, auroc, aupr):
        """학습 결과를 케이스 내 training_results/{모델명}_{타임스탬프}/ 폴더에 저장"""
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

            # 1. 예측 결과 전체 저장 (파일명, 실제 레이블, 예측 레이블, 성공/실패)
            # name 기준으로 병합하여 행 순서 불일치 방지 (results_df는 groupby 순서, original_labels는 테스트 순서)
            actual_df = original_labels.rename(columns={'label': '실제_레이블_raw'})[['name', '실제_레이블_raw']]
            pred_df = pd.DataFrame({'name': original_labels['name'].values, '예측_레이블_raw': y_pred_display})
            prediction_results = results_df[['name']].merge(actual_df, on='name', how='left')
            prediction_results = prediction_results.merge(pred_df, on='name', how='left')
            label_map = self._load_label_mapping_for_history(base_dir)
            to_name = lambda v: self._label_to_text(v, label_map)
            prediction_results['실제_레이블'] = prediction_results['실제_레이블_raw'].map(to_name)
            prediction_results['예측_레이블'] = prediction_results['예측_레이블_raw'].map(to_name)
            prediction_results['성공여부'] = (
                prediction_results['실제_레이블_raw'] == prediction_results['예측_레이블_raw']
            ).map({True: '성공', False: '실패'})
            prediction_results['result'] = prediction_results.apply(
                lambda r: f"기존 label : {r['실제_레이블']}, 예측 label : {r['예측_레이블']}", axis=1)
            prediction_results['success_failure'] = prediction_results['성공여부'].map({'성공': '예측 성공', '실패': '!!!예측 실패!!!'})

            prediction_results['실제_레이블_이름'] = prediction_results['name'].apply(
                lambda fn: self._filename_to_group_pattern(fn))
            prediction_results['예측_레이블_이름'] = prediction_results['예측_레이블_raw'].map(to_name)
            prediction_results = prediction_results[
                [
                    'name',
                    '실제_레이블_raw',
                    '예측_레이블_raw',
                    'result',
                    'success_failure',
                    '실제_레이블',
                    '실제_레이블_이름',
                    '예측_레이블',
                    '예측_레이블_이름',
                    '성공여부',
                ]
            ]

            prediction_file = os.path.join(run_folder, f"training_history_{model_name}_{timestamp}_predictions.csv")
            prediction_results.to_csv(prediction_file, index=False, encoding='utf-8-sig')
            print(f"[INFO] 예측 결과 저장: {prediction_file}")
            
            # 2. 오탐 케이스만 저장 (실제 레이블과 예측 레이블이 다른 경우, 텍스트 컬럼 포함)
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
            # metrics는 "핵심 지표"만 앞쪽에 두고, 나머지는 metrics_detail로 분리
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
                    'macro_precision': float(macroprecision),
                    'macro_recall': float(macrorecall),
                    'macro_f1': float(macrof1),
                    'auroc': float(auroc),
                    'aupr': float(aupr)
                },
                'metrics_detail': {
                    'weighted_precision': float(weightedprecision),
                    'weighted_recall': float(weightedrecall),
                    'weighted_f1': float(weightedf1),
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
            
            # 6. 전체 히스토리 파일에 추가 (config에 저장, 불러올 때도 config 우선)
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

    def train_model(self, df):
        try:
            if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
                model, accuracy = self.ensemble(df)
                # ensemble() 내부에서 show_alert를 띄우지만, 혹시라도 여기까지 왔는데 팝업이 없으면 안전하게 한 번 더 표시
                try:
                    self.show_alert(f"학습 완료: Accuracy {accuracy * 100:.2f}%")
                except Exception:
                    pass
            elif self.index == 1:
                self.lstm(df)
        except Exception as e:
            tb = traceback.format_exc()
            print("[ERROR] multi train_model 실패:", e)
            print(tb)
            try:
                self.show_alert(f"학습 중 오류가 발생했습니다.\n\n{e}\n\n상세:\n{tb}")
            except Exception:
                pass
            raise

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
        # 학습 결과 저장용 run 폴더 미리 생성 (모델/히스토리 모두 여기 저장)
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
        # NOTE: 전체 GridSearch는 (조합 수 × CV) 만큼 학습을 반복하므로 매우 오래 걸릴 수 있습니다.
        # 여기서는 RandomizedSearch를 기본으로 사용해 시간을 현실적인 수준으로 줄입니다.
        params_xgb = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'n_estimators': [150, 200, 250, 300],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_lambda': [1.0, 2.0, 5.0],
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

        # class_weight_dict는 "라벨값 -> weight" 매핑이어야 합니다.
        # (기존 코드의 dict(enumerate(...))는 0..N-1 키가 되어 KeyError가 발생합니다)
        classes = np.unique(y_train.astype(int))
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.astype(int))
        class_weight_dict = dict(zip(classes, class_weights))

        # pandas/numpy dtype 차이로 KeyError가 나지 않도록 int로 정규화 + 기본값 1.0
        sample_weights = y_train.map(lambda x: float(class_weight_dict.get(int(x), 1.0)))
        if self.index == 0:  # XGBoost

            start = time.time()
            # tree_method='hist'는 CPU에서 일반적으로 더 빠릅니다.
            self.model = xgb.XGBClassifier(
                random_state=42,
                tree_method='hist',
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            # 기존 GridSearchCV는 조합 수가 많아 시간이 과도하게 걸릴 수 있어 RandomizedSearch로 변경
            grid_search = RandomizedSearchCV(
                self.model,
                params_xgb,
                n_iter=20,
                cv=3,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            # XGBoost는 내부적으로 클래스가 0..K-1 형태일 때 가장 안정적입니다.
            self._label_encoder = LabelEncoder()
            y_train_encoded = self._label_encoder.fit_transform(y_train.astype(int))
            grid_search.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights)
            self.model = grid_search.best_estimator_
            print(f"[INFO] XGBoost 탐색/학습 소요시간: {time.time() - start:.1f}초")

        elif self.index == 2:  # RandomForest
            self.model = RandomForestClassifier(class_weight=class_weight_dict)
            grid_search = RandomizedSearchCV(self.model, params_rf, n_iter=10, cv=3, scoring='accuracy',
                                             random_state=42, n_jobs=-1, verbose=1)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_

        elif self.index == 3:  # LightGBM

            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))

            # LightGBM 모델 초기화
            base_model = LGBMClassifier(objective='multiclass', class_weight=class_weight_dict, verbose=-1,random_state=42)

            # RandomizedSearchCV로 하이퍼파라미터 튜닝
            grid_search = RandomizedSearchCV(
                base_model,
                params_lgbm,
                n_iter=10,
                cv=3,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_

        elif self.index == 4:  # Logistic Regression
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))

            # LogisticRegression에 전달
            self.model = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial',
                                            class_weight=class_weight_dict)
            self.model.fit(X_train_scaled, y_train)

        # Model evaluation
        if self.index == 0 and hasattr(self, "_label_encoder"):
            # XGBoost는 encoded label 공간에서 평가(확률/클래스 정렬 문제 방지)
            y_test_labels = self._label_encoder.transform(y_test.astype(int))
            y_pred = self.model.predict(X_test_scaled).astype(int)
            # 분석용(사람이 보는 결과)은 원래 라벨로 복원
            y_pred_display = self._label_encoder.inverse_transform(y_pred)
        else:
            y_test_labels = y_test.astype(int)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_display = y_pred
        # Combine test data with predicted labels
        df_test = pd.DataFrame(X_test, columns=X.columns)
        df_test['name'] = names_test.values
        df_test['label'] = y_pred_display

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

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Save the plot as an image
        confusion_path = f"confusion_matrix_{self.model}.png"
        plt.savefig("confusion_matrix.png")

        # NOTE: GUI 환경에서 plt.show()는 창을 닫기 전까지 실행이 멈춘 것처럼 보입니다.
        # 학습이 "끝났는데 안 끝난 것처럼" 보이는 원인이 될 수 있어 show()는 호출하지 않습니다.
        plt.close()

        y_scores = self.model.predict_proba(X_test_scaled)
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
        auroc = roc_auc_score(y_test_binarized, y_scores, multi_class='ovo', average='weighted')

        # AUPR (multi-class)
        from sklearn.metrics import average_precision_score
        aupr = average_precision_score(y_test, y_scores, average='weighted')

        self.auroc = auroc
        self.aupr = aupr

        print(f"AUROC: {auroc:.6f}")
        print(f"AUPR: {aupr:.6f}")

        # Print evaluation metrics
        if self.index != 5:
            accuracy = accuracy_score(y_test_labels, y_pred)
            weightedprecision = precision_score(y_test_labels, y_pred, average='weighted')
            microprecision = precision_score(y_test_labels, y_pred, average='micro')
            macroprecision = precision_score(y_test_labels, y_pred, average='macro')
            weightedrecall = recall_score(y_test_labels, y_pred, average='weighted')
            microrecall = recall_score(y_test_labels, y_pred, average='micro')
            macrorecall = recall_score(y_test_labels, y_pred, average='macro')
            weightedf1 = f1_score(y_test_labels, y_pred, average='weighted')
            microf1 = f1_score(y_test_labels, y_pred, average='micro')
            macrof1 = f1_score(y_test_labels, y_pred, average='macro')

            print(f"Accuracy: {accuracy:.4f}")

            print("*********Precision*************")
            print(f"wightedPrecision: {weightedprecision:.4f}")
            print(f"microPrecision: {microprecision:.4f}")
            print(f"macroPrecision: {macroprecision:.4f}")

            print("*********Recall*************")
            print(f"wightedRecall: {weightedrecall:.4f}")
            print(f"microRecall: {microrecall:.4f}")
            print(f"macroRecall: {macrorecall:.4f}")

            print("*********f1score*************")
            print(f"weightedF1 Score: {weightedf1:.4f}")
            print(f"microF1 Score: {microf1:.4f}")
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

            importance_path = str(self.aimodel + "feature_importance.csv")
            save_dir = getattr(self, '_run_output_dir', None) or os.path.dirname(self.csv_path)
            file_path = os.path.join(save_dir, importance_path)
            if getattr(self, "save_training_outputs", True):
                importance_df.to_csv(file_path, index=False)

        # 학습에 사용된 피처 목록을 config/feature.json에 저장
        self.feature_list = X.columns.tolist()
        if getattr(self, "save_training_outputs", True):
            base_dir = os.path.dirname(self.csv_path)
            config_dir = os.path.join(base_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            jsonpath = os.path.join(config_dir, "feature.json")
            with open(jsonpath, 'w') as f:
                json.dump(self.feature_list, f)
            print(f"[INFO] feature.json 생성: {jsonpath} (피처 {len(self.feature_list)}개, CSV: {os.path.basename(self.csv_path)})")
        
        # 학습 결과 저장 (히스토리)
        if self.index != 5 and getattr(self, "save_training_outputs", True):  # 분류 모델인 경우에만
            self.save_training_history(
                results_df, original_labels, conf_matrix, y_test_labels, y_pred, y_pred_display,
                accuracy, weightedprecision, macroprecision, weightedrecall, macrorecall,
                weightedf1, macrof1, auroc, aupr
            )

        # print("******************은지********************")
        # print(recall_score(y_test_labels, y_pred, average=None))  # 클래스별 Recall
        # print(classification_report(y_test_labels, y_pred))


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

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=-1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=-1)

        # Train the model with class weights
        model.fit(X_train_scaled, y_train, epochs=300, batch_size=16, validation_data=(X_test_scaled, y_test),
                  class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr])

        self.model = model

    def save_model2(self):
        """모델 저장 (training_results/ run 폴더가 있으면 그 안에, 없으면 기존 방식)"""
        if not getattr(self, "save_training_outputs", True):
            print("[INFO] 결과 저장 OFF: save_model2 생략")
            return
        if self.index == 0 or self.index == 2 or self.index == 3 or self.index == 4:
            run_dir = getattr(self, '_run_output_dir', None)
            if run_dir and os.path.isdir(run_dir):
                folder_path = run_dir
                pklname = os.path.join(folder_path, "model.pkl")
                self.scalername = os.path.join(folder_path, "scaler.pkl")
                enc_basename = "label_encoder.pkl"
            else:
                folder_path = os.getcwd()
                pklname = os.path.join(folder_path, str(self.csv_path+"_" + self.aimodel + "model.pkl"))
                self.scalername = os.path.join(folder_path, str(self.csv_path+"_" + self.aimodel + "scaler.pkl"))
                enc_basename = str(self.csv_path+"_" + self.aimodel + "label_encoder.pkl")

            joblib.dump(self.model, pklname)
            with open(self.scalername, 'wb') as f:
                joblib.dump(self.scaler, f)
            
            if hasattr(self, "_label_encoder") and self._label_encoder is not None:
                enc_path = os.path.join(folder_path, enc_basename)
                joblib.dump(self._label_encoder, enc_path)
                print(f"[INFO] LabelEncoder 저장: {os.path.basename(enc_path)}")
            print(f"[INFO] 모델/스케일러 저장: {folder_path}")

        elif self.index == 1:
            run_dir = getattr(self, '_run_output_dir', None)
            if run_dir and os.path.isdir(run_dir):
                h5_path = os.path.join(run_dir, "model.h5")
                self.model.save(h5_path)
                print(f"[INFO] LSTM 모델 저장: {h5_path}")
            else:
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
