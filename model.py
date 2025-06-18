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
                 video_path: str, 
                 video_output_name= "video.avi"
                 ):
        
        self.model = YOLO(model, verbose= True)
        self.processor = Processor()
        self.tracker = tracker
        self.video = video_path
        self.output_name = video_output_name
        self.db_path= "db/data.csv"
        
        self.captured = None
        self.video_writer = None
        
    def __call__(self):
        return self._run()
    
    
    def _setup(self):
        self.captured = cv2.VideoCapture(self.video)
        assert self.captured.isOpened(), "Error reading video file"
        
        w, h, fps = (int(self.captured.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        self.video_writer = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


    def run(self):
        # Process each frame
        while self.captured.isOpened():
            success, frame = self.captured.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            # Run YOLO detection + tracking
            results = self.model.track(source= frame, persist= True, tracker=self.tracker, conf= 0.35)

            # Process detection results
            sales = self.processor(results)
            
            sales_df = self.processor.get_new_sales_df()
            if sales_df is not None:
                append_df_to_csv(sales_df, self.db_path)            

            # Draw sales info on the frame
            result_frame = results[0].plot()
            cv2.putText(result_frame,
                        text=f"Number of pizza sales: {sales}",
                        org=(100, 50),
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        fontScale=0.8,
                        color=(0, 255, 0),
                        thickness=2)

            # Write to video file and show on screen
            self.video_writer.write(result_frame)
            cv2.imshow("Pizza Sale Tracker", result_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up
        self.captured.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
def main():
    model = Model("weights/best.pt",
                  tracker= "tracker.yaml",
                  video_path= "videos/1465_CH02_20250607170555_172408.mp4",
                  video_output_name= "video1.avi"
                  )
    
    model._setup()
    model.run()
    
if "__main__" == __name__:
    main()
    