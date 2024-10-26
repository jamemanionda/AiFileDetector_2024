import os
import subprocess
import pandas as pd
import math
from moviepy.editor import VideoFileClip


def get_video_info(video_path):
    # ffprobe를 사용해 동영상 정보 추출
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,bit_rate,r_frame_rate,pix_fmt', '-of', 'default=noprint_wrappers=1', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = {}

    # ffprobe 출력 파싱
    for line in result.stdout.splitlines():
        key, value = line.split('=')
        info[key.strip()] = value.strip()

    return info


def get_bit_depth(pix_fmt):
    # 픽셀 포맷에 따른 비트 깊이 추정 (일반적인 경우 사용)
    # 각 포맷은 다양한 색상 정보와 비트 깊이를 갖습니다
    # 일부 일반적인 포맷에 대해 매핑합니다
    bit_depth_map = {
        'yuv420p': 8,
        'yuv422p': 8,
        'yuv444p': 8,
        'yuv420p10le': 10,
        'yuv422p10le': 10,
        'yuv444p10le': 10,
        'yuv420p12le': 12,
        'yuv422p12le': 12,
        'yuv444p12le': 12
        # 필요한 경우 다른 픽셀 포맷도 추가
    }
    return bit_depth_map.get(pix_fmt, 8)  # 매핑되지 않은 포맷은 8비트로 가정


def calculate_gop_compression_ratio(video_path):
    # 동영상 정보 가져오기
    video_info = get_video_info(video_path)

    # 필요한 정보 추출 및 변환
    width = int(video_info['width'])
    height = int(video_info['height'])
    bitrate = int(video_info['bit_rate'])  # 압축된 비트레이트
    frame_rate = eval(video_info['r_frame_rate'])  # 프레임 속도 (예: "30/1" -> 30)
    pix_fmt = video_info['pix_fmt']  # 픽셀 포맷 추출

    # 픽셀 포맷에 따른 비트 깊이 가져오기
    bit_depth = get_bit_depth(pix_fmt)

    # 동영상 클립 열기
    clip = VideoFileClip(video_path)

    # 비디오 지속 시간 (초)
    duration = clip.duration

    # 동영상의 총 프레임 수
    frame_count = frame_rate * duration

    # YUV 형식 및 비트 깊이를 바탕으로 비압축 데이터 크기 계산 (bits)
    # YUV 4:2:0 포맷의 경우 1.5바이트(8비트)로 가정
    if '420' in pix_fmt:
        uncompressed_data_size = frame_count * (width * height * 1.5) * bit_depth
    elif '422' in pix_fmt:
        uncompressed_data_size = frame_count * (width * height * 2) * bit_depth
    elif '444' in pix_fmt:
        uncompressed_data_size = frame_count * (width * height * 3) * bit_depth
    else:
        # 매핑되지 않은 포맷은 4:2:0 기준으로 계산
        uncompressed_data_size = frame_count * (width * height * 1.5) * bit_depth

    # 압축률 계산
    compression_ratio = uncompressed_data_size / (bitrate * duration)

    clip.close()

    # 소수점 첫째 자리에서 반올림
    return round(compression_ratio)


def process_videos_in_folder(video_path):
    if os.path.isfile(video_path):
        try:
            # 압축률 계산
            compression_ratio = calculate_gop_compression_ratio(video_path)
            return compression_ratio
        except Exception as e:
            print(f"Failed to process")

    # 데이터프레임 생성
    #df = pd.DataFrame(results)

    # CSV 파일로 저장
    #df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    #print(f"Results saved to {output_csv_path}")


