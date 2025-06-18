import cv2
from ultralytics import YOLO
from process import Processor
from db.data import append_df_to_csv
import pandas as pd

class Model:
    def __init__(
        self, 
        model: str, 
        tracker: str, 
        input_source: str, 
        input_type: str = "video",  
        video_output_name: str = "video.avi"
    ):
        self.model = YOLO(model, verbose=True)
        self.processor = Processor()
        self.tracker = tracker
        self.input_source = input_source  # Path for video, device index or URL for webcam/CCTV
        self.input_type = input_type.lower()  # 'video' or 'realtime'
        self.output_name = video_output_name
        self.db_path = "db/data.csv"
        
        self.captured = None
        self.video_writer = None
        
    def __call__(self):
        return self._run()
    
    def _setup(self):
        # Initialize video capture based on input type
        if self.input_type == "video":
            self.captured = cv2.VideoCapture(self.input_source)
        elif self.input_type == "realtime":
            # For webcam (e.g., 0) or CCTV (e.g., RTSP/HTTP URL)
            try:
                self.captured = cv2.VideoCapture(int(self.input_source))
            except:
                raise ValueError("Error: Could not open webcam or CCTV stream.")
        else:
            raise ValueError("Invalid input_type. Use 'video' or 'realtime'.")
        
        assert self.captured.isOpened(), f"Error opening {'video file' if self.input_type == 'video' else 'webcam/CCTV stream'}"
        
        w, h, fps = (int(self.captured.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        # Ensure valid FPS (default to 30 if FPS is 0, common for webcams)
        fps = fps if fps > 0 else 30
        self.video_writer = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def run(self):
        # Process each frame
        while self.captured.isOpened():
            success, frame = self.captured.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            # Run YOLO detection + tracking
            results = self.model.track(source=frame, persist=True, tracker=self.tracker, conf=0.45)

            # Process detection results
            sales = self.processor(results)
            
            if self.processor.count:
                
                sales_df = self.processor.get_new_sales_df()
                if sales_df is not None:
                    append_df_to_csv(sales_df, self.db_path)         
                
                self.processor.count= False

            # Draw sales info on the frame
            result_frame = results[0].plot()
            cv2.putText(
                result_frame,
                text=f"Number of pizza sales: {sales}",
                org=(100, 50),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(255, 0, 0),
                thickness=3
            )

            # Write to video file and show on screen
            self.video_writer.write(result_frame)
            cv2.imshow("Pizza Sale Tracker", result_frame)

            # For real-time, reduce delay to make it feel live
            delay = 1 if self.input_type == "realtime" else 10
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

        # Clean up
        self.captured.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
def main():
  
    realtime_model = Model(
        model="weights/best.pt",
        tracker="tracker.yaml",
        input_source=0,  
        input_type="realtime",
        video_output_name="realtime_output.avi"
    )
    
    realtime_model._setup()
    realtime_model.run()
    
if __name__ == "__main__":
    main()