import pandas as pd
import subprocess
import json
import os

video_folder = 'Y:\\version2\\test'
excel_file = 'sample.csv'

# Read the Excel file into a DataFrame
df = pd.read_csv(excel_file)

for video_file in os.listdir(video_folder):
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
            print(f"I frame {idx + 1}: P frames count = {p_count}")

        # Step 5: Update Excel file with GOP information
        video_name = video_file  # Extract just the filename

        # Find the row index where the video name matches input_video
        row_index = df[df['Name'] == video_name].index

        if not row_index.empty:
            # Update the 'GOP' column with the frame_types_string
            df.at[row_index[0], 'GOP'] = frame_types_string
            print(f"GOP updated for {video_name} in CSV.")
            print(f"*************************************")
        else:
            # Add a new row with the video name and GOP information
            new_row = pd.DataFrame({'Name': [video_name], 'GOP': [frame_types_string]})
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Video {video_name} not found in CSV. Added new entry with GOP data.")

# Step 6: Save the updated DataFrame back to the CSV file
df.to_csv(excel_file, index=False)
print("CSV file updated successfully.")
