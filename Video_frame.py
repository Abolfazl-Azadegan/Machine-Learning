import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    success = True

    print("Extracting frames...")

    while success:
        # Read one frame at a time
        success, frame = video.read()

        if success:
            # Save the frame as a JPEG file
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            frame_count += 1

    # Release the video object
    video.release()
    print(f"Finished extracting {frame_count} frames to '{output_folder}'.")


def main():
    # Initialize Tkinter root (without opening a GUI window)
    Tk().withdraw()

    # Open a file dialog for the user to select a video file
    print("Please select a video file...")
    video_path = askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])

    if not video_path:
        print("No video file selected. Exiting program.")
        return

    # Ask user for output folder
    print("Please select an output folder to save frames...")
    output_folder = askdirectory()

    if not output_folder:
        print("No output folder selected. Exiting program.")
        return

    # Call the function to extract frames
    extract_frames(video_path, output_folder)

if __name__ == "__main__":
    main()
