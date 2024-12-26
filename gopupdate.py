import os
import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog


class columnclass():
    def __init__(self):
        super(columnclass, self).__init__()

        while 1:
            csv_path1 = self.showFileDialog()
            directory, filename = os.path.split(csv_path1)
            csv_path2 = "Y:\\4차\\GOP기준\\_train_2412011610_processed_.csv"
            result_csv = f'adjustGOP_{filename}'
            csv_path1 = os.path.join(directory, result_csv)
            self.update_csv_with_gop(csv_path1, csv_path2, result_csv)

    def showFileDialog(self):
        # 파일 다이얼로그를 띄워서 파일 선택
        csv_path, _ = QFileDialog.getOpenFileName()

        # 선택한 파일 경로를 출력
        if csv_path:
            print(f'선택된 파일 경로: {csv_path}')
        else:
            print('파일이 선택되지 않았습니다.')

        return csv_path
    def update_csv_with_gop(slef, first_csv_path, second_csv_path, output_csv_path):
        # 첫 번째 CSV 파일 읽기
        df1 = pd.read_csv(first_csv_path)

        # 두 번째 CSV 파일 읽기
        df2 = pd.read_csv(second_csv_path)

        # MD5 값을 기준으로 병합
        merged_df = df1.merge(df2[['md5', 'GOP']], on='md5', how='left', suffixes=('', '_new'))

        # GOP_new 값이 비어있지 않은 경우에만 GOP 값을 업데이트
        merged_df['GOP'] = merged_df.apply(lambda row: row['GOP_new'] if pd.notna(row['GOP_new']) else row['GOP'], axis=1)

        # 필요 없는 GOP_new 컬럼 제거
        merged_df.drop(columns=['GOP_new'], inplace=True)

        # 결과를 새로운 CSV 파일로 저장
        merged_df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = columnclass()
    app.exec_()
