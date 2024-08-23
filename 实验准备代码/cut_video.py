from moviepy.video.io.VideoFileClip import VideoFileClip

video_path="./video/videoplayback.mp4"

video_file=VideoFileClip(video_path)

start_time=120
end_time=125

subclip = video_file.subclip(start_time, end_time)

subclip.write_videofile("./video/video_clip/subclip.mp4",fps=subclip.fps)
