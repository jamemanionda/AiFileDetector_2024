import sys
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication


class combinedclass():
    def __init__(self):
        super(combinedclass, self).__init__()
        self.choice = 0

        while 1:
            csv_path1 = self.showFileDialog()
            csv_path2 = self.showFileDialog()
            result_csv = 'Y:\결과물들\라벨링\\1108 2차 실험\\combined.csv'
            self.update_or_add_rows(csv_path1, csv_path2, result_csv)

    def update_or_add_rows(self, csv1_path, csv2_path, output_path):
        # CSV 파일 불러오기
        df1 = pd.read_csv(csv1_path, low_memory=False)
        df2 = pd.read_csv(csv2_path, low_memory=False)

        # 모든 컬럼을 고려하여 업데이트하기 위해 df1과 df2의 모든 컬럼을 합친 set을 만듭니다.
        all_columns = set(df1.columns).union(set(df2.columns))
        df1 = df1.reindex(columns=all_columns, fill_value=None)  # df1에 없는 컬럼을 추가하고 빈 값으로 채웁니다.

        # 각 행에 대해 hash 값이 있는지 확인하고 처리
        for _, row in df2.iterrows():
            hash_value = row['md5']
            name_value = row['name']

            # df1에 동일한 hash 값이 있는지 확인
            if hash_value in df1['md5'].values:
                # hash 값이 있으면 name 값을 업데이트
                df1.loc[df1['md5'] == hash_value, 'name'] = name_value
            else:
                # hash 값이 없으면 df1에 행을 추가
                new_row = pd.DataFrame([{col: row[col] if col in row else None for col in df1.columns}])
                df1 = pd.concat([df1, new_row], ignore_index=True)  # pd.concat을 사용하여 새로운 행을 추가합니다.

        # name, md5, label 순서로 컬럼을 정렬하고 나머지 컬럼은 그대로 뒤에 추가
        first_columns = ['name', 'md5', 'label']
        remaining_columns = [col for col in df1.columns if col not in first_columns]
        df1 = df1[first_columns + remaining_columns]

        # 결과를 저장
        df1.to_csv(output_path, index=False)
        print("추출완료")

    def showFileDialog(self):
        # 파일 다이얼로그를 띄워서 파일 선택
        csv_path, _ = QFileDialog.getOpenFileName()

        # 선택한 파일 경로를 출력
        if csv_path:
            print(f'선택된 파일 경로: {csv_path}')
        else:
            print('파일이 선택되지 않았습니다.')

        return csv_path


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = combinedclass()
    app.exec_()
