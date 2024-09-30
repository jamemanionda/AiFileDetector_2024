import pandas as pd
import subprocess
import json
#1. Input_video

input_video = './O_GalaxyZFlip3_852.mp4'

result = subprocess.run(
    ['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_frames', '-of', 'json', input_video],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

frame_data = json.loads(result.stdout)
frame_types_string = ""

if 'frames' in frame_data:
    frames = frame_data['frames']
    for frame in frames:
        if 'pict_type' in frame:
            frame_types_string += frame['pict_type']
else:
    print("에러")

print(f"프레임 Type: {frame_types_string}")

excel_file = 'sample.csv'

df = pd.read_csv(excel_file)

video_name = input_video.split('/')[-1]

row_index = df[df['Name'] == video_name].index

if not row_index.empty:
    df.at[row_index[0], 'GOP'] = frame_types_string
    print(f"{video_name} GOP 정보 업데이트 완료")
else:
    print(f"{video_name}가 존재하지 않음")

df.to_csv(excel_file, index=False)
