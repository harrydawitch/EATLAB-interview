import argparse
from model import Model

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO model on video or real-time webcam/CCTV feed.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="weights/best.pt", 
        help="Path to YOLO model weights (e.g., weights/best.pt)"
    )
    parser.add_argument(
        "--tracker", 
        type=str, 
        default="tracker.yaml", 
        help="Path to tracker configuration file (e.g., tracker.yaml)"
    )
    parser.add_argument(
        "--input_source", 
        type=str,
        default="videos/1465_CH02_20250607170555_172408.mp4", 
        help="Video file path, webcam index (e.g., 0), or CCTV URL (e.g., rtsp://...)"
    )
    parser.add_argument(
        "--input_type", 
        type=str, 
        choices=["video", "realtime"], 
        default="video", 
        help="Input type: 'video' for video file, 'realtime' for webcam/CCTV"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="captured/output.avi", 
        help="Output video file name (e.g., output.avi)"
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Initialize the Model
    model = Model(
        model=args.model,
        tracker=args.tracker,
        input_source=args.input_source,
        input_type=args.input_type,
        video_output_name=args.output
    )

    # Setup and run the model
    try:
        model._setup()
        model.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()