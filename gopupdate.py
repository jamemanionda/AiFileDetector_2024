import os
import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog


class ColumnClass():
    def __init__(self):
        super(ColumnClass, self).__init__()

        while True:
            csv_path1 = self.showFileDialog()
            directory, filename = os.path.split(csv_path1)
            csv_path2 = "Y:\\4차\\GOP기준\\_train_2412011610_processed_.csv"
            result_csv = f'adjustGOPwithName_{filename}'
            result_path = os.path.join(directory, result_csv)
            self.update_csv_with_gop(csv_path1, csv_path2, result_path)

    def showFileDialog(self):
        # 파일 다이얼로그를 띄워서 파일 선택
        csv_path, _ = QFileDialog.getOpenFileName()

        # 선택한 파일 경로를 출력
        if csv_path:
            print(f'선택된 파일 경로: {csv_path}')
        else:
            print('파일이 선택되지 않았습니다.')

        return csv_path

    def load_csv_with_fallback(self, file_path):
        """
        첫 번째 행에서 컬럼 이름을 찾지 못하면 두 번째 행을 컬럼 이름으로 설정.
        """
        try:
            # 먼저 기본적으로 CSV 파일을 읽음
            df = pd.read_csv(file_path)

            # 'md5' 컬럼이 존재하지 않으면 두 번째 행을 헤더로 설정
            if 'md5' not in df.columns:
                df = pd.read_csv(file_path, header=1)
                print(f"'md5'가 첫 번째 행에서 발견되지 않아 두 번째 행을 헤더로 사용했습니다.")

            return df
        except Exception as e:
            print(f"CSV 파일을 읽는 도중 오류 발생: {e}")
            return pd.DataFrame()  # 오류 발생 시 빈 데이터프레임 반환

    def update_csv_with_gop(self, first_csv_path, second_csv_path, output_csv_path):
        # 첫 번째 CSV 파일 읽기
        df1 = self.load_csv_with_fallback(first_csv_path)

        # 두 번째 CSV 파일 읽기
        df2 = self.load_csv_with_fallback(second_csv_path)

        # 'md5' 키가 없는 경우 처리
        if 'name' not in df1.columns or 'name' not in df2.columns:
            print("두 파일 중 하나에서 'name' 컬럼을 찾을 수 없습니다.")
            return

        # MD5 값을 기준으로 병합
        merged_df = df1.merge(df2[['name', 'GOP']], on='name', how='left', suffixes=('', '_new'))

        # GOP_new 값이 비어있지 않은 경우에만 GOP 값을 업데이트
        merged_df['GOP'] = merged_df.apply(lambda row: row['GOP_new'] if pd.notna(row['GOP_new']) else row['GOP'], axis=1)

        # 필요 없는 GOP_new 컬럼 제거
        merged_df.drop(columns=['GOP_new'], inplace=True)

        # 결과를 새로운 CSV 파일로 저장
        merged_df.to_csv(output_csv_path, index=False)
        print(f"결과 CSV가 저장되었습니다: {output_csv_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ColumnClass()
    app.exec_()
