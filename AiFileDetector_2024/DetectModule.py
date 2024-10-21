import os
import sys
import struct
import numpy as np
import pandas as pd
from simhash import Simhash
import joblib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QCheckBox, QVBoxLayout, QWidget, QPushButton
)

class PredictionModule(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP4 File Prediction Module")
        self.model = None
        self.scaler = None
        self.index = 0  # Choose your model type (0: RandomForest/XGBoost, 1: LSTM, etc.)

        # Initialize feature states
        self.structure_val_state = False
        self.structure_seq_state = False
        self.frame_gop_state = False
        self.frame_sps_state = False

        self.setup_ui()
        self.load_model_and_scaler()

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
        model_path = os.path.join(folder_path, "Xgboostmodel.pkl")
        scaler_path = os.path.join(folder_path, "Xgboostscaler.pkl")

        try:
            if self.index in [0, 2, 3]:  # Ensemble models
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    print("Model and scaler loaded successfully.")
                else:
                    print("Model or scaler not found. Train the model first.")
            # elif self.index == 1:  # LSTM model
            #     if os.path.exists(model_path) and os.path.exists(scaler_path):
            #         self.model = load_model(model_path)
            #         self.scaler = joblib.load(scaler_path)
            #         print("LSTM model and scaler loaded successfully.")
            #     else:
            #         print("LSTM model or scaler not found. Train the model first.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

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

    def extract_mp4_features(self, file_path):
        """Extract features from the MP4 file based on the selected states."""
        features = []

        def parse_box(f, end_position, depth=0):
            while f.tell() < end_position:
                box_header = f.read(8)
                if len(box_header) < 8:
                    break

                box_size, box_type = struct.unpack(">I4s", box_header)
                box_type = box_type.decode('utf-8', errors='ignore')
                actual_box_size = box_size if box_size != 1 else struct.unpack(">Q", f.read(8))[0]
                box_end_position = f.tell() + (actual_box_size - 8)

                if box_type in ['moov', 'trak', 'mdia', 'minf', 'stbl']:
                    parse_box(f, box_end_position, depth + 1)
                else:
                    f.seek(box_end_position)

                # Collect features based on state
                if self.structure_val_state:
                    features.append(Simhash(box_type).value)
                if self.structure_seq_state:
                    features.append(len(box_type))

        with open(file_path, 'rb') as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            parse_box(f, file_size)

        if self.frame_gop_state:
            features.append(self.extract_gop(file_path))

        if self.frame_sps_state:
            features.append(self.extract_sps(file_path))

        return features

    def extract_gop(self, file_path):
        """Dummy GOP extraction function (replace with actual logic)."""
        return Simhash("GOP").value

    def extract_sps(self, file_path):
        """Dummy SPS extraction function (replace with actual logic)."""
        return Simhash("SPS").value

    def predict(self, file_path):
        """Perform prediction on the MP4 file."""
        try:
            features = self.extract_mp4_features(file_path)

            # Scale the extracted features
            features_df = pd.DataFrame([features])
            features_scaled = self.scaler.transform(features_df)

            # Predict the label using the loaded model
            if self.index == 1:  # LSTM model
                features_scaled = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
                y_pred = np.argmax(self.model.predict(features_scaled), axis=1)
            else:
                y_pred = self.model.predict(features_scaled)

            # Display the prediction result
            self.show_message(f"Prediction completed!\nFile: {os.path.basename(file_path)}\nPredicted Label: {y_pred[0]}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def show_message(self, message):
        """Show a message box with the provided message."""
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Prediction Results")
        msg_box.setText(message)
        msg_box.exec_()

    def run(self):
        """Launch the application."""
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    predictor = PredictionModule()
    predictor.run()
    sys.exit(app.exec_())
