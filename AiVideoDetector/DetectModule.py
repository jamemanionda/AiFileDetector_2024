import os
import sys
import json
import struct
import numpy as np
import pandas as pd
from simhash import Simhash
import joblib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QCheckBox, QVBoxLayout, QWidget, QPushButton, QLabel
)
from PyQt5.QtCore import Qt


def _copyable_msg(parent, icon, title, text, buttons=QMessageBox.Ok):
    """텍스트 선택/복사 가능한 QMessageBox (createtraining 의존 없이 독립 구현)"""
    msg = QMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    for label in msg.findChildren(QLabel):
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return msg.exec_()


# createtraining.py의 함수들을 import
try:
    # 모듈 레벨 함수 import
    from createtraining import _process_one_file, _compute_file_hash
    from extractframe_single import extractGOP
    from extract_sps import parse_sps
    from pps import analyzesps
    CREATETRAINING_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] createtraining.py 모듈을 import할 수 없습니다: {e}")
    CREATETRAINING_AVAILABLE = False

class PredictionModule(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP4 File Prediction Module")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.index = 0  # Choose your model type (0: RandomForest/XGBoost, 1: LSTM, etc.)
        
        # 기본값 설정
        self.classmode = "Android_Device(val+gop)_"
        self.aimodel = "Xgboost"
        self.csv_path = None
        self.pklpath = None
        self.scalerpath = None
        
        # Initialize feature states (학습 시와 동일한 옵션 사용)
        self.structure_val_state = False
        self.structure_seq_state = False
        self.frame_gop_state = False
        self.frame_ratio_state = False
        self.frame_sps_state = False
        
        # seqdict 로딩 (createtraining.py와 동일)
        self.seqdict = {}
        self.load_seqdict()
        
        self.setup_ui()
        self.load_model_and_scaler()

    def load_seqdict(self):
        """시퀀스 딕셔너리 로딩 (createtraining.py와 동일)"""
        try:
            excel_file = os.path.join('mp4', '_dict.xlsx')
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file)
                self.seqdict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                print(f"[INFO] seqdict 로드 완료: {len(self.seqdict)}개 항목")
            else:
                print(f"[WARN] seqdict 파일을 찾을 수 없습니다: {excel_file}")
        except Exception as e:
            print(f"[WARN] seqdict 로드 실패: {e}")

    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()

        # Add checkboxes for feature selection
        self.structure_val_box = QCheckBox("Structure Value")
        self.structure_seq_box = QCheckBox("Structure Sequence")
        self.frame_gop_box = QCheckBox("GOP")
        self.frame_sps_box = QCheckBox("SPS")

        self.structure_val_box.stateChanged.connect(self.update_structure_val_state)
        self.structure_seq_box.stateChanged.connect(self.update_structure_seq_state)
        self.frame_gop_box.stateChanged.connect(self.update_frame_gop_state)
        self.frame_sps_box.stateChanged.connect(self.update_frame_sps_state)

        layout.addWidget(self.structure_val_box)
        layout.addWidget(self.structure_seq_box)
        layout.addWidget(self.frame_gop_box)
        layout.addWidget(self.frame_sps_box)

        # Add button to select and predict file
        predict_button = QPushButton("Select MP4 File and Predict")
        predict_button.clicked.connect(self.select_file)
        layout.addWidget(predict_button)

        # Set the layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def load_model_and_scaler(self):
        """Load the model and scaler if available."""
        folder_path = os.getcwd()
        
        # 기본값 사용
        pklname = str(self.classmode + self.aimodel + "model.pkl")
        scalername = str(self.classmode + self.aimodel + "scaler.pkl")

        model_path = os.path.join(folder_path, pklname)
        scaler_path = os.path.join(folder_path, scalername)

        try:
            if self.index in [0, 2, 3]:  # Ensemble models
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    self.pklpath = model_path
                    self.scalerpath = scaler_path
                    print(f"[INFO] Model 로드 완료: {pklname}")
                    print(f"[INFO] Scaler 로드 완료: {scalername}")
                    
                    # LabelEncoder 로드 시도
                    enc_filename = pklname.replace("model.pkl", "label_encoder.pkl")
                    enc_path = os.path.join(folder_path, enc_filename)
                    if os.path.exists(enc_path):
                        try:
                            self.label_encoder = joblib.load(enc_path)
                            print(f"[INFO] LabelEncoder 로드 완료: {enc_filename}")
                        except Exception as e:
                            print(f"[WARN] LabelEncoder 로드 실패: {e}")
                            self.label_encoder = None
                    else:
                        self.label_encoder = None
                else:
                    print(f"[WARN] Model 또는 Scaler 파일을 찾을 수 없습니다.")
                    print(f"  - Model 경로: {model_path}")
                    print(f"  - Scaler 경로: {scaler_path}")
        except Exception as e:
            _copyable_msg(self, QMessageBox.Critical, "Error", f"Failed to load model: {str(e)}")

    def update_structure_val_state(self, state):
        self.structure_val_state = state == 2

    def update_structure_seq_state(self, state):
        self.structure_seq_state = state == 2

    def update_frame_gop_state(self, state):
        self.frame_gop_state = state == 2

    def update_frame_sps_state(self, state):
        self.frame_sps_state = state == 2

    def select_file(self):
        """Open a file dialog to select an MP4 file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select MP4 File", filter="MP4 Files (*.mp4)")
        if file_path:
            self.predict(file_path)

    def extract_features_from_file(self, file_path):
        """
        createtraining.py의 _process_one_file과 동일한 방식으로 특징 추출
        """
        if not CREATETRAINING_AVAILABLE:
            _copyable_msg(self, QMessageBox.Critical, "Error", "createtraining.py 모듈을 사용할 수 없습니다. createtraining.py 파일이 같은 디렉토리에 있는지 확인하세요.")
            return None
        
        try:
            # state_flags 준비 (createtraining.py와 동일한 형식)
            state_flags = {
                'structure_val_state': self.structure_val_state,
                'structure_seq_state': self.structure_seq_state,
                'frame_gop_state': self.frame_gop_state,
                'frame_ratio_state': self.frame_ratio_state,
                'frame_sps_state': self.frame_sps_state
            }
            
            # _process_one_file 호출 (createtraining.py와 동일)
            result = _process_one_file(file_path, state_flags, self.seqdict.copy())
            
            # 결과를 리스트 형태로 반환 (파일 1개 기준)
            return [result]
        except Exception as e:
            _copyable_msg(self, QMessageBox.Critical, "Error", f"특징 추출 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def flatten_features_for_prediction(self, data):
        """
        createtraining.py의 flatten_features_for_prediction과 동일한 로직
        """
        flattened_rows = []
        global_fieldnames = set()

        for file_data in data:
            flattened = {}
            key_count_local = {}  # Local key duplication count for each file
            for key, value in file_data:
                if key != 'GOP':
                    if key in key_count_local:
                        key_count_local[key] += 1
                        key_with_count = f"{key}({key_count_local[key]})"
                    else:
                        key_count_local[key] = 1
                        key_with_count = key

                    if isinstance(value, str) and ":" in value:
                        # SPS는 analyzesps가 반환한 "profile_idc:66, constraint_set0_flag:0, ..." 형식
                        if key == 'SPS' and ',' in value:
                            sps_attrs = [attr.strip() for attr in value.split(",")]
                            for sps_attr in sps_attrs:
                                if ":" in sps_attr:
                                    sps_name, sps_val = sps_attr.split(":", 1)
                                    sps_name_clean = sps_name.strip()
                                    sps_val_clean = sps_val.strip()
                                    sps_field_name = f"{key}_{sps_name_clean}"
                                    flattened[sps_field_name] = sps_val_clean
                                    global_fieldnames.add(sps_field_name)
                        else:
                            # 일반적인 박스 속성 파싱
                            attributes = [attr.strip() for attr in value.split(",")]
                            for attr in attributes:
                                if ":" in attr:
                                    attr_name, attr_value = attr.split(":", 1)
                                    attr_name_clean = attr_name.strip()
                                    attr_value_clean = attr_value.strip()
                                    
                                    if attr_name_clean == "Entries" and attr_value_clean.startswith("['") and "Duration:" in attr_value_clean:
                                        entries_str = attr_value_clean.strip("[]'\"")
                                        entry_parts = entries_str.split(",")
                                        for entry_part in entry_parts:
                                            if ":" in entry_part:
                                                entry_key, entry_val = entry_part.split(":", 1)
                                                entry_key_clean = entry_key.strip()
                                                entry_val_clean = entry_val.strip()
                                                if entry_key_clean in ["Media Time", "Rate"]:
                                                    entry_field_name = f"{key_with_count}_{entry_key_clean}"
                                                    flattened[entry_field_name] = entry_val_clean
                                                    global_fieldnames.add(entry_field_name)
                                    else:
                                        attr_field_name = f"{key_with_count}_{attr_name_clean}"
                                        flattened[attr_field_name] = attr_value_clean
                                        global_fieldnames.add(attr_field_name)
                                else:
                                    flattened[key_with_count] = value
                                    global_fieldnames.add(key_with_count)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and ":" in item:
                                item_attrs = [ia.strip() for ia in item.split(",")]
                                for item_attr in item_attrs:
                                    if ":" in item_attr:
                                        item_key, item_val = item_attr.split(":", 1)
                                        item_key_clean = item_key.strip()
                                        item_val_clean = item_val.strip()
                                        item_field_name = f"{key_with_count}_{item_key_clean}"
                                        flattened[item_field_name] = item_val_clean
                                        global_fieldnames.add(item_field_name)
                            else:
                                flattened[key_with_count] = str(item)
                                global_fieldnames.add(key_with_count)
                    else:
                        flattened[key_with_count] = value
                        global_fieldnames.add(key_with_count)
                else:
                    # Handle the 'GOP' key separately
                    # GOP는 이미 "30,30,30" 형식의 쉼표로 구분된 문자열이므로 그대로 사용
                    flattened[key] = value if value else ''
                    global_fieldnames.add(key)

            flattened_rows.append(flattened)

        df = pd.DataFrame(flattened_rows)

        # 모든 필드가 포함되도록 빈 값으로 채우기
        for field in global_fieldnames:
            if field not in df.columns:
                df[field] = None

        return df

    def apply_simhash(self, df):
        """
        createtraining.py의 apply_simhash와 동일한 로직 (간소화 버전)
        모든 문자열 컬럼에 simhash 적용
        """
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                try:
                    df_copy[col] = df_copy[col].apply(
                        lambda x: Simhash(str(x)).value if pd.notna(x) and str(x).strip() else 0
                    )
                except Exception:
                    df_copy[col] = 0
        return df_copy

    def predict(self, file_path):
        """Perform prediction on the MP4 file using createtraining.py와 동일한 방식."""
        try:
            if not self.model or not self.scaler:
                _copyable_msg(self, QMessageBox.Warning, "Error", "Model 또는 Scaler가 로드되지 않았습니다. 먼저 모델을 학습하세요.")
                return

            # 1. 특징 추출 (createtraining.py와 동일)
            print(f"[INFO] 특징 추출 시작: {os.path.basename(file_path)}")
            feature_data = self.extract_features_from_file(file_path)
            if feature_data is None:
                return

            # 2. Flatten (createtraining.py와 동일)
            print(f"[INFO] 특징 Flatten 시작")
            structured_data = self.flatten_features_for_prediction(feature_data)
            print(f"[INFO] Flatten 완료: {structured_data.shape[1]}개 피처")

            # 3. feature.json 로드 (run 폴더 → config 폴더 순, 학습 시 사용된 컬럼 순서)
            base_dir = os.path.dirname(os.path.abspath(self.pklpath)) if self.pklpath else os.getcwd()
            jsonpath = os.path.join(base_dir, "feature.json")
            if not os.path.exists(jsonpath) and "training_results" in os.path.normpath(base_dir):
                case_root = os.path.dirname(os.path.dirname(base_dir))
                config_feature = os.path.join(case_root, "config", "feature.json")
                if os.path.exists(config_feature):
                    jsonpath = config_feature
            model_features = None
            
            try:
                with open(jsonpath, 'r', encoding='utf-8') as f:
                    model_features = json.load(f)
                print(f"[INFO] feature.json 로드: {len(model_features)}개 피처")
            except FileNotFoundError:
                # scaler에서 피처 가져오기 시도
                try:
                    model_features = list(getattr(self.scaler, "feature_names_in_", []))
                    if model_features:
                        print(f"[INFO] feature.json 없음, scaler에서 피처 가져옴: {len(model_features)}개")
                except Exception:
                    model_features = [str(c) for c in structured_data.columns]
                    print(f"[WARN] feature.json 없음, 현재 컬럼 사용: {len(model_features)}개")
            except Exception as e:
                print(f"[WARN] feature.json 읽기 실패: {e}")
                try:
                    model_features = list(getattr(self.scaler, "feature_names_in_", []))
                except Exception:
                    model_features = [str(c) for c in structured_data.columns]

            # 4. 전처리 (createtraining.py와 동일)
            df = structured_data.copy()
            
            # md5 제거
            df = df.drop(columns='md5', errors='ignore')
            # name 제거
            df = df.drop(columns=[col for col in df.columns if col == 'name'], errors='ignore')
            # 컬럼명을 문자열로 변환
            df.columns = df.columns.astype(str)
            
            # simhash 적용
            df = self.apply_simhash(df)
            
            # label 제거 (있을 경우)
            df = df.drop(columns=[col for col in df.columns if col == 'label'], errors='ignore')

            # 5. feature.json과 컬럼 맞추기
            if model_features:
                missing = [c for c in model_features if c not in df.columns]
                extra = [c for c in df.columns if c not in model_features]
                
                if missing:
                    missing_ratio = len(missing) / float(len(model_features))
                    print(f"[WARN] 누락 피처: {len(missing)}개 ({missing_ratio:.1%})")
                    if missing_ratio > 0.3:
                        _copyable_msg(
                            self, QMessageBox.Warning, "Warning",
                            f"입력 피처가 학습 피처와 많이 다릅니다.\n"
                            f"누락 피처: {len(missing)}개 ({missing_ratio:.1%})\n"
                            f"예측 정확도가 낮을 수 있습니다."
                        )
                
                # feature.json 순서에 맞춤 (누락된 컬럼은 0으로 채움)
                df = df.reindex(columns=model_features, fill_value=0)
            else:
                # scaler의 feature_names_in_ 사용
                try:
                    scaler_features = list(getattr(self.scaler, "feature_names_in_", []))
                    if scaler_features:
                        df = df.reindex(columns=scaler_features, fill_value=0)
                except Exception:
                    pass

            # 6. 스케일링 및 예측
            print(f"[INFO] 예측 시작: {df.shape[1]}개 피처")
            X_new_scaled = self.scaler.transform(df)
            y_pred = self.model.predict(X_new_scaled)
            y_pred_probs = self.model.predict_proba(X_new_scaled) if hasattr(self.model, 'predict_proba') else None

            # LabelEncoder가 있으면 원래 라벨로 복원
            if self.label_encoder is not None:
                try:
                    y_pred = self.label_encoder.inverse_transform(y_pred.astype(int))
                except Exception as e:
                    print(f"[WARN] LabelEncoder inverse_transform 실패: {e}")

            # 예측 확률 계산
            pred_label = int(y_pred[0]) if isinstance(y_pred[0], (int, np.integer)) else y_pred[0]
            pred_prob = y_pred_probs[0, np.argmax(y_pred_probs[0])] if y_pred_probs is not None else None

            # 결과 표시
            result_msg = f"예측 완료!\n\n파일: {os.path.basename(file_path)}\n예측 라벨: {pred_label}"
            if pred_prob is not None:
                result_msg += f"\n예측 확률: {pred_prob:.4f} ({pred_prob*100:.2f}%)"
            
            print(f"[INFO] 예측 결과: 라벨={pred_label}, 확률={pred_prob}")
            self.show_message(result_msg)

        except Exception as e:
            import traceback
            error_msg = f"Prediction failed: {str(e)}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            _copyable_msg(self, QMessageBox.Critical, "Error", error_msg)

    def show_message(self, message):
        """Show a message box with the provided message (텍스트 선택/복사 가능)."""
        _copyable_msg(self, QMessageBox.Information, "Prediction Results", message)

    def run(self):
        """Launch the application."""
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    predictor = PredictionModule()
    predictor.run()
    sys.exit(app.exec_())
