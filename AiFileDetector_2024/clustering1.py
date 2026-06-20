import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyexpat import features
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, \
    precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.python.keras.models import load_model
import pandas as pd
import os
from datetime import datetime
from simhash import Simhash
from sklearn.metrics import accuracy_score
import sys
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QApplication, QWidget, QFileSystemModel, QMainWindow, QMessageBox, QFileDialog, \
    QTableWidgetItem
from PyQt5 import uic, QtWidgets
import joblib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

'''
device_lib.list_local_devices()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
form_class = uic.loadUiType("Training.ui")[0]

with tf.device('/GPU:0'):
'''


class trainClustering(QMainWindow):  # QMainWindow, form_class

    def __init__(self):
        super(trainClustering, self).__init__()
        self.choice = 0
        self.file_paths = []
        # self.dpath = 'E:\\'
        self.model = None


        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())
        self.xlsfileext = '.csv'
        self.filter_files_by_extension(self.xlsfileext)


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
                df = pd.read_csv(file_name, encoding='UTF-8')
                self.display_dataframe(df)
            except Exception as e:
                self.tableWidget.setRowCount(0)
                self.tableWidget.setColumnCount(0)
                self.show_error_message("CSV 파일을 읽는 중 오류가 발생했습니다: " + str(e))



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

    def gotrain(self, csv_path, classmode=None):
        self.csv_path = csv_path
        self._prepare_cluster_output_paths(csv_path)
        df, orignal_labels = self.preprocess_data(csv_path, is_train=True)
        self.extension = os.path.basename(os.path.dirname(csv_path))
        df_processed = self.apply_simhash(df)
        return self.optimize_clustering(df_processed, orignal_labels, classmode)

    def _prepare_cluster_output_paths(self, csv_path):
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        csv_base = os.path.splitext(os.path.basename(csv_path))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(csv_dir, "clustering_results")
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"{csv_base}_clustering_{stamp}")
        self.cluster_output_base = output_base
        self.last_cluster_output_paths = {
            "output_dir": output_dir,
            "result_csv": f"{output_base}_results.csv",
            "metrics_csv": f"{output_base}_metrics.csv",
            "elbow_figure": f"{output_base}_elbow_silhouette.png",
            "pca_figure": f"{output_base}_pca.png",
        }
        return self.last_cluster_output_paths

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

        sample_df = pd.read_csv(filepath, nrows=1, header=None)
        tempvalue = sample_df.iloc[0, 0]

        if tempvalue != 'name':
            df = pd.read_csv(filepath, header=1)
        else:
            df = pd.read_csv(filepath)

        # 'name'과 'label' 열 분리
        try:
            df = df.drop(columns=['md5'])
        except:
            pass
        if 'label' in df.columns:
            original_labels = df[['name', 'label']]
            df = df.drop(columns=['label'])
        else:
            original_labels = None

        if 'name' in df.columns:
            df = df.drop(columns=['name'])



        return df, original_labels

    @staticmethod
    def calculate_simhash_lib(value):
        try:
            try:
                simval = Simhash(str(value)).value
            except:
                simval = Simhash(str(value[:100])).value
        except:
            simval = 0
        return simval

    def apply_simhash(self, df):
        """Simhash 적용"""
        df.columns = df.columns.astype(str)
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            df[column] = df[column].apply(self.calculate_simhash_lib)
        return df

    def _cluster_count_from_labels(self, original_labels, classmode, n_samples):
        """Train 탭의 binary/multi 선택을 클러스터 수 결정에 반영."""
        mode = (classmode or "").lower()
        if mode.startswith("bin"):
            return 2 if n_samples > 2 else None
        if not mode.startswith("mul"):
            return None
        if original_labels is None or 'label' not in original_labels.columns:
            print("[WARN] multi 클러스터링: label 컬럼이 없어 silhouette 기준으로 클러스터 수를 선택합니다.")
            return None

        labels = original_labels['label'].dropna().unique()
        label_count = len(labels)
        if label_count < 2:
            print("[WARN] multi 클러스터링: 유효한 라벨이 2개 미만이라 silhouette 기준으로 클러스터 수를 선택합니다.")
            return None
        if label_count >= n_samples:
            label_count = max(2, n_samples - 1)
        return int(label_count)

    def optimize_clustering(self, df, original_labels, classmode=None):
        """클러스터링 최적화"""
        # 데이터 스케일링
        # Scale features
        features = df.select_dtypes(include=[np.number])
        scaled_features = StandardScaler().fit_transform(features)

        # Dimensionality reduction using PCA
        pca = PCA(n_components=0.999)
        #
        scaled_features_pca = pca.fit_transform(scaled_features)

        # Find optimal number of clusters for KMeans
        ssd = []
        silhouette_avg = []
        n_samples = len(features)
        max_clusters = min(59, n_samples - 1)
        if max_clusters < 2:
            raise ValueError("클러스터링에는 최소 3개 이상의 샘플이 필요합니다.")
        target_n_clusters = self._cluster_count_from_labels(original_labels, classmode, n_samples)
        range_n_clusters = range(2, max_clusters + 1)
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=15)
            kmeans.fit(scaled_features_pca)
            ssd.append(kmeans.inertia_)
            cluster_labels = kmeans.labels_
            silhouette_avg.append(silhouette_score(scaled_features_pca, cluster_labels))

        output_paths = getattr(self, "last_cluster_output_paths", {})

        # Plot elbow and silhouette scores
        fig_elbow = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range_n_clusters, ssd, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')

        plt.subplot(1, 2, 2)
        plt.plot(range_n_clusters, silhouette_avg, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis For Optimal k')
        plt.tight_layout()
        if output_paths.get("elbow_figure"):
            fig_elbow.savefig(output_paths["elbow_figure"], dpi=180, bbox_inches="tight")
            print(f"[INFO] Elbow/Silhouette figure 저장: {output_paths['elbow_figure']}")
        plt.show()

        # Determine optimal number of clusters
        if target_n_clusters is not None:
            optimal_n_clusters = min(max(2, target_n_clusters), max_clusters)
            print(f"[INFO] {classmode or 'auto'} 모드: 라벨 기준 클러스터 수 {optimal_n_clusters}개를 사용합니다.")
        else:
            optimal_n_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
            print(f"[INFO] auto 모드: silhouette 기준 클러스터 수 {optimal_n_clusters}개를 사용합니다.")
        self.kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=15).fit(scaled_features_pca)

        # Other clustering methods (DBSCAN, Agglomerative Clustering)
        dbscan_eps = np.percentile(cdist(scaled_features_pca, scaled_features_pca, 'euclidean'), 90)
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=2).fit(scaled_features_pca)
        self.agglo = AgglomerativeClustering(n_clusters=optimal_n_clusters).fit(scaled_features_pca)

        # Display clusters
        df['cluster'] = self.kmeans.labels_
        self.compare_clusters_with_labels(df, original_labels)

        metrics_rows = [
            self.evaluate_clustering(scaled_features_pca, original_labels, self.kmeans.labels_, "KMeans"),
            self.evaluate_clustering(scaled_features_pca, original_labels, self.dbscan.labels_, "DBSCAN"),
            self.evaluate_clustering(scaled_features_pca, original_labels, self.agglo.labels_, "AgglomerativeClustering"),
        ]

        if original_labels is not None:
            result_df = original_labels.reset_index(drop=True).copy()
        else:
            result_df = pd.DataFrame({"row_index": range(len(df))})
        result_df["kmeans_cluster"] = self.kmeans.labels_
        result_df["dbscan_cluster"] = self.dbscan.labels_
        result_df["agglomerative_cluster"] = self.agglo.labels_
        if output_paths.get("result_csv"):
            result_df.to_csv(output_paths["result_csv"], index=False, encoding="utf-8-sig")
            print(f"[INFO] 클러스터링 결과 CSV 저장: {output_paths['result_csv']}")
        if output_paths.get("metrics_csv"):
            pd.DataFrame(metrics_rows).to_csv(output_paths["metrics_csv"], index=False, encoding="utf-8-sig")
            print(f"[INFO] 클러스터링 평가지표 CSV 저장: {output_paths['metrics_csv']}")

        # PCA visualization
        self.visualize_pca(df, scaled_features, self.kmeans.labels_)
        return output_paths

    def visualize_pca(self, df, scaled_features, cluster_labels):
        """PCA를 사용하여 클러스터 결과 시각화"""
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        df_pca = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
        df_pca['cluster'] = cluster_labels
        df_pca['cluster_label'] = df_pca['cluster'].astype(str)

        unique_clusters = sorted(
            df_pca['cluster_label'].unique(),
            key=lambda value: int(value) if str(value).lstrip("-").isdigit() else str(value)
        )
        palette = dict(zip(unique_clusters, sns.color_palette("husl", n_colors=len(unique_clusters))))
        show_legend = len(unique_clusters) <= 20

        def robust_limits(values):
            values = np.asarray(values.dropna(), dtype=float)
            if len(values) == 0:
                return None
            low, high = np.percentile(values, [1, 99])
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = float(np.min(values)), float(np.max(values))
            if low == high:
                padding = 1.0
            else:
                padding = (high - low) * 0.08
            return low - padding, high + padding

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        x_col = 'principal component 1'
        y_col = 'principal component 2'

        for ax, title in zip(axes, ["PCA of Clusters - Full", "PCA of Clusters - Zoom"]):
            sns.scatterplot(
                x=x_col,
                y=y_col,
                hue='cluster_label',
                data=df_pca,
                palette=palette,
                s=36,
                alpha=0.85,
                linewidth=0.25,
                edgecolor='white',
                legend=show_legend,
                ax=ax
            )
            ax.set_title(title)
            ax.grid(True, alpha=0.25)

        x_lim = robust_limits(df_pca[x_col])
        y_lim = robust_limits(df_pca[y_col])
        if x_lim and y_lim:
            axes[1].set_xlim(*x_lim)
            axes[1].set_ylim(*y_lim)
            inside_zoom = (
                df_pca[x_col].between(x_lim[0], x_lim[1])
                & df_pca[y_col].between(y_lim[0], y_lim[1])
            )
            outlier_count = int((~inside_zoom).sum())
            if outlier_count:
                axes[1].set_title(f"PCA of Clusters - Zoom ({outlier_count} outliers hidden)")

        centroids = df_pca.groupby('cluster_label')[[x_col, y_col]].median()
        for cluster_name, row in centroids.iterrows():
            x_val = float(row[x_col])
            y_val = float(row[y_col])
            if x_lim and y_lim and not (x_lim[0] <= x_val <= x_lim[1] and y_lim[0] <= y_val <= y_lim[1]):
                continue
            axes[1].annotate(
                cluster_name,
                (x_val, y_val),
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', alpha=0.7, linewidth=0.3)
            )

        if show_legend:
            for ax in axes:
                ax.legend(title='cluster', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        else:
            print(f"[INFO] 클러스터가 {len(unique_clusters)}개라 범례 대신 확대 그래프에 클러스터 번호를 표시합니다.")

        plt.tight_layout()
        output_paths = getattr(self, "last_cluster_output_paths", {})
        if output_paths.get("pca_figure"):
            fig.savefig(output_paths["pca_figure"], dpi=180, bbox_inches="tight")
            print(f"[INFO] PCA figure 저장: {output_paths['pca_figure']}")
        plt.show()

    def evaluate_clustering(self, scaled_features, original_labels, cluster_labels, method):
        """클러스터링 평가"""
        """Evaluate clustering without original labels."""
        result = {
            "method": method,
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "error": "",
        }
        try:
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            davies_bouldin_avg = davies_bouldin_score(scaled_features, cluster_labels)
            print(f'{method} - Silhouette Score: {silhouette_avg}')
            print(f'{method} - Davies-Bouldin Index: {davies_bouldin_avg}')
            result["silhouette_score"] = silhouette_avg
            result["davies_bouldin_index"] = davies_bouldin_avg
        except ValueError as e:
            print(f"{method} - Unable to calculate silhouette score: {e}")
            result["error"] = str(e)
        return result

    def compare_clusters_with_labels(self, df, original_labels):
        """Map data points to clusters with names from original labels."""
        # Merge 'name' from original_labels with cluster assignments in df
        df_with_names = df.merge(original_labels[['name']], left_index=True, right_index=True)

        # Display the data point name and assigned cluster
        print("\nData point to cluster mapping:")
        for index, row in df_with_names.iterrows():
            print(f"Data point {row['name']} -> Cluster {row['cluster']}")

        # 평가 지표 계산
        # accuracy = accuracy_score(original_labels['label'], df['predicted_label'])
        # precision = precision_score(original_labels['label'], df['predicted_label'], average='weighted',
        #                             zero_division=1)
        # recall = recall_score(original_labels['label'], df['predicted_label'], average='weighted', zero_division=1)
        # f1 = f1_score(original_labels['label'], df['predicted_label'], average='weighted', zero_division=1)
        #
        # print(f'Accuracy: {accuracy}')
        # print(f'Precision: {precision}')
        # print(f'Recall: {recall}')
        # print(f'F1 Score: {f1}')

        # 'name' 열을 추가하기 위해 original_labels와 병합
        # df_with_names = df.merge(original_labels[['name']], left_index=True, right_index=True)
        # self.display_cluster_mapping(df_with_names)


    def display_cluster_mapping(self, df):
        """클러스터링 결과 출력"""
        print("\nData point to cluster mapping:")
        for index, row in df.iterrows():
            print(f"Data point {row['name']} -> Cluster {row['cluster']}")

    def predict_cluster(self, df):
        """새 데이터 클러스터 예측"""
        df_processed = self.apply_simhash(df)
        features = df_processed[df_processed.columns[1:]]
        scaled_features = MinMaxScaler().fit_transform(features)

        predicted_clusters = self.kmeans.predict(scaled_features)
        df_processed['predicted_cluster'] = predicted_clusters

        cluster_label_map = {}
        for cluster in df_processed['predicted_cluster'].unique():
            cluster_data = df_processed[df_processed['predicted_cluster'] == cluster]
            most_common_label = cluster_data['label'].mode()[0]
            cluster_label_map[cluster] = most_common_label

        df_processed['predicted_label'] = df_processed['predicted_cluster'].map(cluster_label_map)

        return df_processed


if __name__ == "__main__":
    app = QApplication(sys.argv)
    data_preprocessor = trainClustering()

    data_preprocessor.show()
    app.exec_()
