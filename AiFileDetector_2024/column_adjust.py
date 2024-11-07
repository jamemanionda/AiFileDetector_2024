import sys

import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QApplication


class columnclass():
    def __init__(self):
        super(columnclass, self).__init__()

    def showFileDialog(self):
        # 파일 다이얼로그를 띄워서 파일 선택
        csv_path, _ = QFileDialog.getOpenFileName()

        # 선택한 파일 경로를 출력
        if csv_path:
            print(f'선택된 파일 경로: {csv_path}')
        else:
            print('파일이 선택되지 않았습니다.')

        return csv_path


# 파일 경로 설정
first_csv_path = 'first_file.csv'  # 첫 번째 CSV 파일 경로 (컬럼 조회용)
second_csv_path = 'second_file.csv'  # 두 번째 CSV 파일 경로 (필터링용)
output_csv_path = 'filtered_second_file.csv'  # 결과 저장 파일 경로

# 첫 번째 CSV 파일에서 컬럼 조회
first_df = pd.read_csv(first_csv_path)
first_csv_columns = first_df.columns  # 첫 번째 파일의 컬럼명 리스트

# 두 번째 CSV 파일을 읽고, 첫 번째 파일의 컬럼명과 일치하는 컬럼만 선택
second_df = pd.read_csv(second_csv_path)
filtered_df = second_df[first_csv_columns.intersection(second_df.columns)]  # 일치하는 컬럼만 선택

# 결과를 새로운 CSV 파일로 저장
filtered_df.to_csv(output_csv_path, index=False)

print("필터링이 완료되었습니다. 결과는 filtered_second_file.csv에 저장되었습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = columnclass()
    app.exec_()
