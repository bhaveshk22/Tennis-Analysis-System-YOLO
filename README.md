
# Tennis Analysis System YOLO

This project is a Tennis Analysis System using the YOLO (You Only Look Once) object detection model. It leverages state-of-the-art computer vision techniques to analyze tennis matches. It uses YOLOv5 and YOLOv8 for player and ball detection and employs the ResNet50 model for keypoints detection.

## Demo

https://github.com/user-attachments/assets/d17a6fd9-da89-469c-b905-2a9312983518



## Features
- Player Detection
- Ball Tracking
- Key Points Detection
- Visualizations for analysis

## Installation

1. Clone the repository
    ```
    git clone https://github.com/bhaveshk22/Tennis-Analysis-System-YOLO.git
    cd Tennis-Analysis-System-YOLO
    ```

2. Install the required dependencies
    ```
    pip install -r requirements.txt
    ```

3. Download the YOLOv5, YOLOv8, and ResNet50 models as per the instructions provided in the    respective directories.

## Usage

1. Ensure you have a video of a tennis match ready for analysis.
2. Run the analysis script with the video file as input
    ```
    python main.py 
    ```
3. The output will be a video with detected objects and a summary of match statistics.


## Models Used
- **YOLOv5:** Used for detecting players and the ball.
- **YOLOv8:** Also used for detecting players and the ball with improved accuracy and performance.
- **ResNet50:** Utilized for keypoints detection to analyze players' movements and postures.


## Results
- Detailed analysis of player movements.
- Accurate ball tracking throughout the match.
- Key points detection providing insights into player positions and actions.


## Lessons Learned
1. **Model Selection:** Choosing the right model is crucial. YOLOv5 and YOLOv8 were selected for their balance between speed and accuracy in object detection, while ResNet50 provided robust performance for keypoints detection.
2. **Data Preparation:** Preprocessing data and ensuring quality inputs significantly impact the model's performance.
3. **Integration Challenges:** Integrating different models for a cohesive analysis system required meticulous planning and testing to ensure compatibility and performance.
4. **Performance Optimization:** Continuous optimization is necessary to handle real-time video processing efficiently.



## Authors

[@Bhavesh](https://github.com/bhaveshk22/) 
