from utils import read_video, save_video
from trackers import PlayerTracker
def main():
    # reading video
    input_video_path = 'input_videos\input_video.mp4'
    video_frames = read_video(input_video_path)

    # detecting player
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    player_detections = player_tracker.detect_frame(video_frames)

    # draw output
    
    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    save_video(output_video_frames, "output_videos\output_video.avi")







    
if __name__=='__main__':
    main()