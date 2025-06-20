# Pizza Counting System - EATLAB Project

A smart AI-powered system uses YOLO to detect and track pizzas and pizza boxes in video streams, count pizza sales, and store data in a database. Developed for the EATLAB AI Engineer interview, it processes video files, webcam feeds, or live CCTV streams.

## <span style="color:red">**MUST READ**</span>
For interviewer who will review my project please go to this link first to see my project explanation: 

[Project Explanation](https://www.notion.so/CHU-HOANG-THIEN-LONG-EATLAB-Interview-project-explnation-216d8c535eb680a5aeeaf3c44a054f3e?pvs=4)

## Installation

1. Clone the repo:
   ```bash
    https://github.com/harrydawitch/EATLAB-interview.git
    cd EATLAB-interview
   ```

2. Create a virtual environment (optional)
   ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```
3. Install dependencies:
   ```bash 
   pip install -r requirements.tx
   ```
## Usage 

The `main.py` script runs a YOLOv11 model for object detection and tracking. It supports:

* Video files (.mp4, .avi, etc.)
* Realtime streams (webcam or RTSP URLs)
* Customizable thresholds and model settings
* Saving output videos with detection overlays

####  Running the Script:
Execute the script using Python from the command line. The general syntax is:
```bash
python main.py --model <weights_path> --tracker <tracker_config> --input_source <source> --input_type <type> --conf <confidence> --iou <iou> --output <output_video>
```

#### Examples:
* To run the model on a webcam feed (e.g., default webcam):
    ```bash
    python main.py --input_source 0 --input_type realtime --output captured/webcam_output.avi
    ```

* To run the model on a CCTV stream with custom confidence and IoU:
    ```bash
    python main.py --input_source rtsp://example.com/stream --input_type realtime --conf 0.5 --iou 0.7 --output captured/cctv_output.avi
    ```

* To run the model on a video files (.mp4):
    ```bash
    python main.py --input_source your_video.mp4 --input_type video --output output.avi
    ```

### Command-Line Arguments

| Argument       | Description                                                  | Default Value                              | Options/Examples                      |
|----------------|--------------------------------------------------------------|--------------------------------------------|---------------------------------------|
| `--model`      | Path to YOLO model weights file                              | `weights/best.pt`                          | `weights/custom.pt`                   |
| `--tracker`    | Path to tracker configuration file                           | `tracker.yaml`                             | `configs/custom_tracker.yaml`         |
|`--input_source`| Video file path, webcam index, or CCTV URL                 | `videos.mp4` | `0`, `rtsp://example.com/stream`     |
| `--input_type` | Type of input source                                         | `video`                                    | `video`, `realtime`                   |
| `--conf`       | Confidence threshold for detection                           | `0.45`                                     | `0.5` (float between 0 and 1)         |
| `--iou`        | Intersection over Union threshold for detection              | `0.6`                                      | `0.7` (float between 0 and 1)         |
| `--output`     | Output video file path                                       | `captured/output.avi`                      | `captured/custom_output.avi`          |

## Project Structure
```pqsql
EATLAB-interview/
├──training              # Training folder
|    ├─ train
|    ├─ valid
|    ├─ data.yaml
|    ├─ train.py
|
├──weights               # YOLO weight files
|    ├──best.pt
|
├── main.py              # Entry point for the app
├── model.py             # Model wrapper with YOLO and processor
├── process.py           # Logic for matching and zone
├── requirements.txt     # Python dependencies
├── tracker.yaml         # botsort config for tracking object
```



