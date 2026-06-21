import subprocess
import os
import re

# h264_analyze 실행 파일 경로 설정



def analyzesps(file_path):
    # 분석할 H.264 비트스트림 파일 설정
    # 실행 파일/입력 파일은 절대경로로 실행(작업 디렉토리 영향을 제거)
    candidates = [
        "./h264Bitstream/h264_analyze.exe",
        "./h264bitstream/h264_analyze.exe",
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(script_dir, "h264Bitstream", "h264_analyze.exe"),
        os.path.join(script_dir, "h264bitstream", "h264_analyze.exe"),
    ]
    H264_ANALYZE_PATH = None
    for cand in candidates:
        try:
            cand_abs = os.path.abspath(cand)
            if os.path.exists(cand_abs):
                H264_ANALYZE_PATH = cand_abs
                break
        except Exception:
            continue
    if not H264_ANALYZE_PATH:
        # 기존 상대경로 fallback (그래도 실행 시도)
        H264_ANALYZE_PATH = os.path.abspath("./h264Bitstream/h264_analyze.exe")

    INPUT_FILE = os.path.abspath(file_path)

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
        return ""






