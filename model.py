import cv2
from ultralytics import YOLO
from process import Processor
import pandas as pd
import os
import torch

class Model:
    def __init__(
        self,
        model: str,
        tracker: str,
        conf: int,
        iou: int,
        input_source: str,
        input_type: str = "video",
        video_output_name: str = "captured/video.avi",

        
    ):
        
        self.model = YOLO(model, verbose=True)
        self.processor = Processor()
        self.tracker = tracker
        self.input_source = input_source 
        self.input_type = input_type.lower()
        self.output_name = video_output_name
        self.db_path = "db/data.csv"
        self.conf = conf
        self.iou = iou
        
        self.captured = None
        self.video_writer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self):
        self._setup()
        self._run()

    def append_df_to_csv(self, df: pd.DataFrame, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        mode = "a" if os.path.exists(file_path) else "w"
        header = not os.path.exists(file_path)
        df.to_csv(
            file_path,
            mode=mode,
            header=header,
            index=False,
            encoding="utf-8"
        )

    def _setup(self):
        # Initialize capture
        if self.input_type == "video":
            self.captured = cv2.VideoCapture(self.input_source)
        elif self.input_type == "realtime":
            # Distinguish between webcam and cctv
            
            if str(self.input_source).isdigit():
                self.captured = cv2.VideoCapture(int(self.input_source)) # "0", "1" for webcams

            # rtsp:// or http:// URLs
            else:
                self.captured = cv2.VideoCapture(self.input_source)
            
        else:
            raise ValueError("input_type must be 'video' or 'realtime'")

        if not self.captured.isOpened():
            src = 'video file' if self.input_type=='video' else 'stream'
            raise RuntimeError(f"Error opening {src}")

        w = int(self.captured.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.captured.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.captured.get(cv2.CAP_PROP_FPS)) or 30
        
        os.makedirs(os.path.dirname(self.output_name), exist_ok=True)
        self.video_writer = cv2.VideoWriter(
            self.output_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

    def _run(self):
        while True:
            success, frame = self.captured.read()
            if not success:
                print("Finished processing or empty frame.")
                break

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                iou = self.iou,
                device= self.device
            )
            sales = self.processor(results)

            if self.processor.count:
                sales_df = self.processor.get_new_sales_df()
                if sales_df is not None:
                    self.append_df_to_csv(sales_df, self.db_path)
                self.processor.count = False

            frame_out = results[0].plot()
            cv2.putText(
                frame_out,
                f"Number of pizza sales: {sales}",
                (150, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 200, 100),
                2
            )

            self.video_writer.write(frame_out)
            cv2.imshow("Pizza Sale Tracker", frame_out)

            delay = 1 if self.input_type == "realtime" else 10
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        self.captured.release()
        self.video_writer.release()
        cv2.destroyAllWindows()


def main():
    model = Model(
        model="weights/best.pt",
        tracker="tracker.yaml",
        input_source=0,
        input_type="realtime",
        video_output_name="realtime_output.avi",
        conf = 0.5,
        iou= 0.5
    )
    model()

if __name__ == "__main__":
    main()
