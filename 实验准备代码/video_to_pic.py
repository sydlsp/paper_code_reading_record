import cv2
import os

input_video_path="./video/video_clip/subclip.mp4"


# 创建保存视频帧的文件夹
output_dir="./video/frames"
os.makedirs(output_dir, exist_ok=True)

# 读取视频
cap=cv2.VideoCapture(input_video_path)

frame_count=0
# 设置帧间隔
frame_interval=30

while cap.isOpened():
    # ret: bool, frame: 包含读取到的视频帧的图像数据
    ret,frame=cap.read()

    if not ret:
        break
    if frame_count % frame_interval == 0:
        frame_filename=os.path.join(output_dir,f"{frame_count}.jpg")
        cv2.imwrite(frame_filename,frame)

    frame_count+=1

cap.release()
cv2.destroyAllWindows()


