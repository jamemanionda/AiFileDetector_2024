import subprocess
import os
import re

# h264_analyze 실행 파일 경로 설정



def analyzesps(file_path):
    # 분석할 H.264 비트스트림 파일 설정
    H264_ANALYZE_PATH = "./h264Bitstream/h264_analyze.exe"  # 필요 시 경로 수정
    INPUT_FILE = file_path

    # h264_analyze 프로그램 실행 및 출력 처리
    try:
        result = subprocess.run(
            [H264_ANALYZE_PATH, INPUT_FILE],
            capture_output=True,
            text=True
        )
        # 실행 결과 확인
        if result.returncode != 0:
            print("Error:", result.stderr)
            raise RuntimeError("h264_analyze failed to execute properly")

        # 정규식을 사용하여 (이름, 값) 형식의 데이터 추출
        pattern = re.compile(r'sps->([a-zA-Z0-9_.]+): (\d+)')
        parsed_data = pattern.findall(result.stdout)

        # 데이터를 "key:value" 형식의 문자열로 변환
        sps_data = ', '.join([f"{name}:{value}" for name, value in parsed_data])

        return sps_data

    except Exception as e:
        print(f"Error occurred: {e}")






