import os
import struct

file_path = ""

def read_box_header(f):
    """MP4 박스의 헤더를 읽고 크기와 타입 반환"""
    header = f.read(8)
    if len(header) < 8:
        return None, None, None
    box_size, box_type = struct.unpack(">I4s", header)
    box_type = box_type.decode("utf-8", errors="ignore").strip()

    if box_size == 1:  # 'large size' 처리
        large_size = struct.unpack(">Q", f.read(8))[0]
        return large_size, box_type, 16
    else:
        return box_size, box_type, 8

def save_nal_unit(data):
    """NAL Unit 형식으로 데이터를 파일에 저장 (00 00 01 헤더 포함)"""
    nal_unit = b'\x00\x00\x00\x01' + data  # NAL Unit 헤더 추가
    return nal_unit

def extract_sps_pps(data, file_path):
    """avcC 데이터에서 SPS와 PPS 값을 분리 및 추출"""
    offset = 0
    sps_size = struct.unpack(">H", data[offset:offset + 2])[0]  # SPS 크기 (2바이트)
    offset += 2

    sps_data = data[offset:offset + sps_size]  # SPS 데이터 추출
    print(f"SPS 데이터:\n{sps_data.hex()}")

    offset += sps_size
    offset += 1  # SPS 데이터 뒤의 1바이트 건너뛰기

    pps_size = struct.unpack(">H", data[offset:offset + 2])[0]  # PPS 크기 (2바이트)
    offset += 2

    pps_data = data[offset:offset + pps_size]  # PPS 데이터 추출
    print(f"PPS 데이터:\n{pps_data.hex()}")

    # SPS와 PPS 데이터를 NAL Unit 형식으로 각각 파일에 저장
    sps_data = save_nal_unit(sps_data)
    pps_data = save_nal_unit(pps_data)

    nalu = sps_data+pps_data
    file_name = os.path.basename(file_path)
    file_name += '.264'
    base_path = os.getcwd()
    file_path = os.path.join(base_path, file_name)
    with open(file_path, 'wb') as f:
        f.write(nalu)
    print(f"{nalu}에 NAL Unit 저장 완료!")

def find_avcc_in_avc1(f, avc1_end_pos, file_path):
    """avc1 박스 내에서 avcC 구조를 탐지하고 크기 및 SPS/PPS 추출"""
    while f.tell() < avc1_end_pos+1:
        current_pos = f.tell()
        data = f.read(4)

        if data == b'avcC':
            f.seek(current_pos - 4)
            size_data = f.read(4)
            avcc_size = struct.unpack(">I", size_data)[0]

            print(f"avcC 발견! 위치: {current_pos}, 크기: {avcc_size}")

            f.seek(current_pos + 10)  # avcC 이후 데이터로 이동
            avcc_data = f.read(avcc_size - 10)  # 헤더 제외한 데이터

            extract_sps_pps(avcc_data, file_path)  # SPS/PPS 추출
            return

        f.seek(current_pos + 1)

    print("avcC를 찾지 못했습니다.")


def parse_box(f, end_position, depth=0, max_depth=100):

    if depth > max_depth:
        print("최대 재귀 깊이 도달 에러")
        return

    while f.tell() < end_position:
        box_header = f.read(8)  # 첫 8Bytes Box 헤더


        if len(box_header) < 8:
            break




        try:
            box_size, box_type1 = struct.unpack(">I4s", box_header)  # size 4Bytes, type 4Bytes 추출
            box_type = box_type1.decode("utf-8").strip()
            if not box_type.isprintable() or len(box_type) != 4:
                raise ValueError(f"잘못된 박스 타입 감지: {box_type}")
        except Exception as e:
            print(f"박스 타입 디코딩 오류: {e}, 위치: {f.tell() - 8}")
            f.seek(0, 1)  # 잘못된 헤더일 경우 8바이트만 건너뛰고 계속
            continue


        if box_size == 0:  # 파일의 끝까지 Box가 확장됨을 의미
            break
        elif box_size == 1:  # 실제 크기는 다음 8Bytes에 저장됨
            large_size = f.read(8)
            actual_box_size = struct.unpack(">Q", large_size)[0]
        else:
            actual_box_size = box_size

        box_end_position = f.tell() + (actual_box_size - 8 if box_size == 1 else box_size - 8)

        try:

            if box_type in ('moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd'):
                parse_box(f, box_end_position, depth + 1, max_depth)
        except Exception as e:
            pass

        if box_type == 'avc1':
            print("avc1 박스 발견! 내부에서 avcC 탐색 중...")
            fname = f.name
            find_avcc_in_avc1(f, box_end_position, f.name)
            f.seek(end_position)
            return



            # 다음 Box로 이동
        f.seek(box_end_position)











        if f.tell() > end_position:
            print(f"경계 초과 오류 - 현재 위치: {f.tell()}, 끝: {end_position}")
            break



def parse_sps(file_path):
    with open(file_path, 'rb') as f:
        file_size = f.seek(0, 2)  # 파일 끝으로 이동하여 크기 계산
        f.seek(0)  # 파일 시작 위치로 이동
        parse_box(f, file_size)  # 파싱 시작