import pandas as pd
import subprocess
import json
import os

video_folder = 'Y:\\version2\\test'
excel_file = 'sample.csv'
# df = pd.read_csv(excel_file)
# Read the Excel file into a DataFrame

def extractGOP(video_file):

    if video_file.endswith(('.mp4','.MP4')):
        input_video = os.path.join(video_folder, video_file)

        # Step 3: Extract frame types from the video
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_frames', '-of', 'json', input_video],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        frame_data = json.loads(result.stdout)
        frame_types_string = ""  # Initialize a string to store frame types

        if 'frames' in frame_data:
            frames = frame_data['frames']
            for frame in frames:
                if 'pict_type' in frame:
                    # Append frame type to the string
                    frame_types_string += frame['pict_type']
        else:
            print(f"No frames found or error in extracting frames for {video_file}.")

        print(f"Frame Types String for {video_file}: {frame_types_string}")  # Output the constructed frame types string

        # Step 4: Parse the frame_types_string to count 'P' frames after each 'I'
        p_counts = []  # List to store the number of 'P' frames between 'I' frames
        count = 0
        for i in range(len(frame_types_string)):
            if frame_types_string[i] == 'I':
                if count > 0:  # Add the previous count if there was an 'I' before
                    p_counts.append(count)
                count = 0  # Reset count after each 'I'
            elif frame_types_string[i] == 'P':
                count += 1

        if count > 0:
            p_counts.append(count)  # Add the last count if any

        # Print out the counts of 'P' frames after each 'I' frame
        for idx, p_count in enumerate(p_counts):
            framestr = f"I frame {idx + 1}: P frames count = {p_count}"



        # Step 5: Update Excel file with GOP information

        #video_name = video_file  # Extract just the filename

        # Find the row index where the video name matches input_video
        #row_index = df[df['Name'] == video_name].index

        #if not row_index.empty:
            # Update the 'GOP' column with the frame_types_string
        #    df.at[row_index[0], 'GOP'] = frame_types_string
        #    print(f"GOP updated for {video_name} in CSV.")
        #    print(f"*************************************")
        #else:
            # Add a new row with the video name and GOP information
        #    new_row = pd.DataFrame({'Name': [video_name], 'GOP': [frame_types_string]})
        #    df = pd.concat([df, new_row], ignore_index=True)
        #    print(f"Video {video_name} not found in CSV. Added new entry with GOP data.")
    frame_types_string = extract_pattern_with_repeats_lcs(frame_types_string)
    print(frame_types_string)
        #df.to_csv(excel_file, index=False)
    return frame_types_string

def extract_pattern_with_repeats_lcs(s):
    # Split by 'I'
    scount = s.count('I')
    segments = s.split('I')
    count = 0
    formatted_segments = {}

    for segment in segments[1:]:
        formatted = "I"
        i = 0
        while i < len(segment):
            count += 1
            if count == scount :
                break

            max_pattern = ""
            max_count = 0
            for j in range(i + 1, len(segment) + 1):
                current_substr = segment[i:j]
                remaining_string = segment[j:]
                lcs = find_lcs(current_substr, remaining_string)
                if lcs and segment[i:].startswith(lcs * (segment[i:].count(lcs))):
                    repeat_count = segment[i:].count(lcs)
                    if len(lcs) * repeat_count <= len(segment[i:]):
                        max_pattern = lcs
                        max_count = repeat_count
                        break

            if max_pattern and max_count > 1:
                formatted += f"{{{max_pattern}}}_{max_count}"
                i += len(max_pattern) * max_count
            else:
                if max_count == 1 or max_pattern == "":
                    formatted += segment[i]
                else:
                    formatted += f"{{{segment[i]}}}(1)"
                i += 1
        if formatted!='I':
            formatted_segments[formatted]=1
        else:
            print("I임")

    result_with_counts = []
    current_pattern = ""
    current_count = 0

    for pattern, value in formatted_segments.items():
        result_with_counts.append(f"{pattern}:{value}")

    result_with_counts = ', '.join(result_with_counts)
    return result_with_counts

def find_lcs(str1, str2):
    m = len(str1)
    n = len(str2)
    lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
    length = 0
    lcs_end = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                if lcs_matrix[i][j] > length:
                    length = lcs_matrix[i][j]
                    lcs_end = i
            else:
                lcs_matrix[i][j] = 0

    return str1[lcs_end - length:lcs_end]



print("CSV file updated successfully.")
