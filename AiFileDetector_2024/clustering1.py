
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, \
    precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.python.keras.models import load_model
import pandas as pd
import os
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

    def gotrain(self, csv_path):
        df, original_labels = self.preprocess_data(csv_path, is_train=True)
        self.extension = os.path.basename(os.path.dirname(csv_path))
        df_processed = self.apply_simhash(df)
        self.optimize_clustering(df_processed, original_labels)

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
            simval = Simhash(str(value)).value
        except :
            simval = 0
        return simval

    def apply_simhash(self, df):
        """Simhash 적용"""
        df.columns = df.columns.astype(str)
        columns_to_process = [col for col in df.columns if col not in ['name', 'label']]
        for column in columns_to_process:
            df[column] = df[column].apply(self.calculate_simhash_lib)
        return df

    def optimize_clustering(self, df, original_labels):
        """클러스터링 최적화"""
        # 데이터 스케일링
        features = df[df.columns[1:]]  # 첫 번째 열은 'name'이므로 제외
        scaled_features = StandardScaler().fit_transform(features)  # StandardScaler로 변경

        # 최적의 클러스터 수 찾기 (엘보우 방법)
        ssd = []
        silhouette_avg = []
        range_n_clusters = range(2, 11)
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(scaled_features)
            ssd.append(kmeans.inertia_)

            # 실루엣 점수 계산
            cluster_labels = kmeans.labels_
            silhouette_avg.append(silhouette_score(scaled_features, cluster_labels))

        # 엘보우 방법과 실루엣 점수 시각화
        plt.figure(figsize=(12, 5))

        # 엘보우 방법
        plt.subplot(1, 2, 1)
        plt.plot(range_n_clusters, ssd, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')

        # 실루엣 분석
        plt.subplot(1, 2, 2)
        plt.plot(range_n_clusters, silhouette_avg, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis For Optimal k')
        plt.show()

        # 최적의 클러스터 수 선택 (여기서는 실루엣 점수를 기준으로 선택)
        optimal_n_clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
        print(f'Optimal number of clusters (by silhouette score): {optimal_n_clusters}')

        # KMeans 클러스터링
        self.kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, init='k-means++', n_init=10).fit(
            scaled_features)

        # DBSCAN 클러스터링 (eps와 min_samples 조정)
        self.dbscan = DBSCAN(eps=1, min_samples=2).fit(scaled_features)

        # AgglomerativeClustering 클러스터링
        self.agglo = AgglomerativeClustering(n_clusters=optimal_n_clusters).fit(scaled_features)

        # 클러스터링 결과 비교 및 평가
        self.evaluate_clustering(scaled_features, original_labels, self.kmeans.labels_, "KMeans")
        self.evaluate_clustering(scaled_features, original_labels, self.dbscan.labels_, "DBSCAN")
        self.evaluate_clustering(scaled_features, original_labels, self.agglo.labels_, "AgglomerativeClustering")

        # KMeans 클러스터링 결과를 사용하여 시각화
        df['cluster'] = self.kmeans.labels_
        self.compare_clusters_with_labels(df, original_labels)

        # PCA 시각화
        self.visualize_pca(df, scaled_features, self.kmeans.labels_)
    def visualize_pca(self, df, scaled_features, cluster_labels):
        """PCA를 사용하여 클러스터 결과 시각화"""
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        df_pca = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
        df_pca['cluster'] = cluster_labels

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='principal component 1', y='principal component 2', hue='cluster', data=df_pca, palette='viridis')
        plt.title('PCA of Clusters')
        plt.show()

    def evaluate_clustering(self, scaled_features, original_labels, cluster_labels, method):
        """클러스터링 평가"""
        # 실루엣 점수
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        # 다비스-볼딘 지수
        davies_bouldin_avg = davies_bouldin_score(scaled_features, cluster_labels)

        print(f'{method} - Silhouette Score: {silhouette_avg}')
        print(f'{method} - Davies-Bouldin Index: {davies_bouldin_avg}')

    def compare_clusters_with_labels(self, df, original_labels):
        """클러스터링 결과와 실제 레이블 비교"""
        cluster_label_map = {}
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            most_common_label = cluster_data['label'].mode()[0]
            cluster_label_map[cluster] = most_common_label

        df['predicted_label'] = df['cluster'].map(cluster_label_map)

        # 평가 지표 계산
        accuracy = accuracy_score(df['label'], df['predicted_label'])
        precision = precision_score(df['label'], df['predicted_label'], average='weighted')
        recall = recall_score(df['label'], df['predicted_label'], average='weighted')
        f1 = f1_score(df['label'], df['predicted_label'], average='weighted')

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        self.display_cluster_mapping(df)

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
