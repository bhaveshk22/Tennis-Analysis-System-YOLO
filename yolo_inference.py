from ultralytics import YOLO

model = YOLO('yolovs=8x')

result = model.track('input_videos\input_video.mp4', save=True)
# print('boxes')
# for box in result[0].boxes:
#     print(box)