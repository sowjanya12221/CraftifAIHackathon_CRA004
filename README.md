# Parallel Object Detection and Human Pose Estimation Using PipeGen

## 1. Project Overview

This project demonstrates parallel object detection and human pose estimation using PipeGen. It processes live webcam frames through two perception branches and combines their results into a single visual output.

The object detection branch uses YOLOv8n to identify objects, while the human pose estimation branch uses YOLO11n Pose to estimate human body keypoints. The final output displays object bounding boxes, class labels, and human pose skeletons on the same video frame.

## 2. Problem Statement

Real-time video analysis may require object detection and human pose estimation to be performed on the same frame. Processing these tasks separately and combining their results can introduce additional processing overhead.

This project demonstrates a parallel perception pipeline that performs both tasks on video frames and combines their results for visualization.

## 3. Proposed Solution

The proposed system uses PipeGen to execute a perception pipeline with two parallel branches.

- **Object Detection:** YOLOv8n detects objects from 80 COCO classes and produces bounding boxes, class labels, and confidence scores.
- **Human Pose Estimation:** YOLO11n Pose estimates human body keypoints and generates a 17-keypoint skeleton.
- **Result Visualization:** The outputs of both branches are combined and displayed on the original video frame.

The system accepts live webcam input and displays the processed results.

## 4. Technologies Used

- PipeGen
- Python
- YOLOv8n Object Detection
- YOLO11n Pose Estimation
- ONNX models
- TensorRT FP16
- OpenCV
- NumPy
- ONNX Runtime for the Python implementation
- DeepStream configuration and parser files

## 5. Pipeline Architecture

The pipeline consists of eight nodes that process each video frame.

### Input

The camera node captures live video frames from the webcam.

### Branch 1: Object Detection

1. **Preprocessing:** Converts the frame from BGR to RGB, resizes it to 640 × 640, and normalizes the input.
2. **YOLOv8n Inference:** Performs object detection.
3. **Detection Postprocessing:** Filters predictions, calculates bounding boxes, and suppresses overlapping detections.

### Branch 2: Human Pose Estimation

1. **Preprocessing:** Converts the frame to RGB, applies letterbox resizing, and records scaling and padding information.
2. **YOLO11n Pose Inference:** Estimates human bounding boxes and body keypoints.
3. **Pose Postprocessing:** Processes the pose predictions and maps the keypoint coordinates back to the original frame.

### Final Output

The display node combines the original frame with the detection and pose results. Object bounding boxes and labels are drawn first, followed by the human pose skeleton.

### Workflow Diagram

```text
                       Live Webcam
                            |
                       Camera Node
                            |
                 +----------+----------+
                 |                     |
                 v                     v
         Object Detection       Pose Estimation
                 |                     |
          Preprocessing          Preprocessing
                 |                     |
             YOLOv8n              YOLO11n Pose
                 |                     |
        Detection Postprocess   Pose Postprocess
                 |                     |
                 +----------+----------+
                            |
                      Display Node
## 6. Models Used

### YOLOv8n — Object Detection

- **Task:** Object detection
- **Classes:** 80 COCO object classes
- **Input Size:** 640 × 640
- **Model File:** `yolov8n.onnx`
- **Output:** Object bounding boxes, class labels, and confidence scores

### YOLO11n Pose — Human Pose Estimation

- **Task:** Human pose estimation
- **Keypoints:** 17 COCO human body keypoints
- **Input Size:** 640 × 640
- **Model File:** `yolo11n-pose.onnx`
- **Output:** Human pose keypoints and skeleton visualization

---

## 7. Model Optimization

The models were compiled using **TensorRT FP16** for the PipeGen pipeline.

The reported engine build times were:

| Model | Engine Build Time |
|---|---|
| YOLOv8n | Approximately 18.2 seconds |
| YOLO11n Pose | Approximately 23.5 seconds |

**Note:** These values represent model engine build times, not inference latency or frames per second (FPS).

---

## 8. Repository Structure

The repository contains the pipeline description, Python implementation, model files, and DeepStream-related files.

```text
pipegen-parallel-object-pose-detection/
├── PipelineIR.json
├── engine_paths.json
├── detectors.py
├── yolov8_pose.py
├── yolov8/
│   ├── YOLOv8.py
│   └── utils.py
├── models/
│   ├── yolov8n.onnx
│   ├── yolo11n-pose.onnx
│   ├── download.sh
│   └── engine_paths.json
├── deepstream/
│   ├── config/
│   ├── labels/
│   └── parsers/
├── run.sh
├── screenshots/
└── README.md
## 9. Execution

### 9.1 PipeGen Pipeline Execution

The PipeGen perception pipeline was executed using the **Run Pipeline** button in the PipeGen environment.

The pipeline uses a live webcam feed and processes each frame through two parallel perception branches.

**Branch 1: Object Detection**
1. Capture frames from the webcam.
2. Resize the frames to 640 × 640.
3. Perform object detection using YOLOv8n.
4. Process the detections to obtain bounding boxes, class labels, and confidence scores.

**Branch 2: Human Pose Estimation**
1. Prepare the input frames using letterbox preprocessing.
2. Resize the input to 640 × 640 while maintaining the required aspect ratio.
3. Perform human pose estimation using YOLO11n Pose.
4. Process the results to obtain human body keypoints and skeleton information.

**Final Output**

The results from both branches are combined and displayed on the screen. The output includes object bounding boxes, object labels, confidence scores, human pose keypoints, and skeleton visualization.

### 9.2 Running the Python Implementation

The repository also contains a separate Python implementation that uses the ONNX model files for object detection and human pose estimation.

**Prerequisites**
- Python installed and configured.
- Required Python dependencies installed.
- ONNX model files available in the expected locations.
- A working webcam or an input video file.

**Run with the default webcam**

```bash
./run.sh
```

**Run with a video file and save the output**

```bash
./run.sh --input people.mp4 --output out.mp4 --no-show
```

**Display available command-line options**

```bash
./run.sh --help
```

When the display window is open, press `q` or `Esc` to quit.

**Note:** The commands above apply only to the Python implementation. They are separate from the PipeGen pipeline, which was executed using the Run Pipeline button in the PipeGen environment. The Python implementation requires its own dependencies and model configuration.

---
## 10. Screenshots and Pipeline Documentation

This section presents the live output, model optimization reports, pipeline architecture, workflow stages, and PipeGen pipeline configuration.

### 10.1 Live Object Detection and Human Pose Estimation

The live output demonstrates object detection and human pose estimation using a webcam feed. Bounding boxes and object labels are displayed alongside human pose keypoints and skeleton visualization.

![Live Detection Output](screenshots/live-detection.png)

### 10.2 Object Detection Model Report

This screenshot shows the model compilation or inspection report for the YOLOv8n object detection model.

![Object Detection Report](screenshots/object-detection-report.png)

### 10.3 Human Pose Estimation Model Report

This screenshot shows the model compilation or inspection report for the YOLO11n Pose model.

![Pose Estimation Report](screenshots/pose-estimation-report.png)

### 10.4 Pipeline Architecture

The architecture illustrates the two parallel branches: object detection using YOLOv8n and human pose estimation using YOLO11n Pose. Their results are combined to produce the final display output.

![Pipeline Architecture](screenshots/pipeline-architecture.png)

### 10.5 PipeGen Workflow Stages

This screenshot presents the workflow stages used to configure and execute the perception pipeline in PipeGen.

![PipeGen Workflow Stages](screenshots/pipegen-workflow-stages.png)

### 10.6 PipeGen IR and Node Graph

This screenshot shows the PipeGen intermediate representation (IR) and node graph, including the processing stages and connections between pipeline components.

![PipeGen IR and Node Graph](screenshots/pipegen-ir-node-graph.png)
## 11. Current Limitations

- The current PipeGen prototype displays object detection and human pose estimation results on the screen.
- Automated alerts, external dashboards, and message-broker integrations are not implemented.
- Performance depends on the hardware, input video, and execution environment.
- Additional testing is required under different lighting conditions, camera positions, and crowded scenes.
- End-to-end latency and frames per second (FPS) have not been measured or reported.
- Local model paths and environment-specific dependencies may require configuration when running the project on another computer.

---

## 12. Future Improvements

- Measure end-to-end latency and frames per second (FPS) to evaluate pipeline performance.
- Optimize inference and preprocessing for the target hardware.
- Add optional structured output for object detections and human pose keypoints.
- Test the pipeline with different camera inputs, lighting conditions, and video scenarios.
- Improve portability by documenting model setup, dependencies, and environment configuration.

---

## 13. Conclusion

This project demonstrates a parallel perception pipeline for object detection and human pose estimation using PipeGen.

YOLOv8n identifies objects in the input frames, while YOLO11n Pose estimates human body keypoints. The results from both branches are combined into a single visual output.

The project provides a foundation for further development of real-time computer vision applications and demonstrates how parallel perception tasks can be integrated into a single pipeline.

---

## Acknowledgements

This project uses YOLO-family models and open-source computer vision tools. Credit should be given to the respective model authors and software libraries in accordance with their licenses and usage requirements.
                 
