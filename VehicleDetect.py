import torch
import cv2 as cv
import argparse

from EmissionCounter import EmissionCounter
from VehicleCounter import VehicleCounter
from sys import argv
from pathlib import Path


class vehicle_detection:
    def __init__(self, path_to_model, conf=0.55, iou=0.45, device='cuda' if torch.cuda.is_available() else 'cpu', time_to_avg=5, display=False):
        self.path_to_model = Path(path_to_model)
        self.model = self.select_custom_model(self.path_to_model.__str__())
        self.device = device
        self.model.to(self.device)
        self.model.conf = conf
        self.model.iou = iou
        self.time_to_avg = time_to_avg
        self.VehicleCounter = VehicleCounter()
        self.emision_counter = EmissionCounter(30)
        self.display = display
        self.bgr = {
            'Car': (10,10, 255),
            'Jeepney': (86,170,255),
            'Motorcycle': (86,255,255),
            'Tricycle': (0,255,0),
            'Truck': (255,127,0),
            'Utility Vehicle': (255,0,127),
        }

    def select_custom_model(self, path_to_model):
        """
        Function to select a custom YOLOv5 model
        """
        # returns model
        return torch.hub.load("ultralytics/yolov5", "custom", path_to_model)


    def analyze_frame(self, frame):
        result = self.model([frame])
        # return result, labels, coords
        return result, result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]

    def int_to_label(self, n):
        return self.model.names[int(n)]

    def process_frame(self, result, frame):
        """
        Plot Bounding Boxes on the frame using the results
        """
        _, labels, coord = result
        x = frame.shape[1]
        y = frame.shape[0]
        for i in range(len(labels)):
            row = coord[i]
            x1 = int(row[0]*x)
            y1 = int(row[1]*y)
            x2 = int(row[2]*x)
            y2 = int(row[3]*y)
            curr_vehicle = self.int_to_label(labels[i])
            self.VehicleCounter.increment_vehicle(curr_vehicle)

            rect_thickness = 1
            cv.rectangle(frame, (x1,y1), (x2,y2), self.bgr[curr_vehicle], rect_thickness)
            cv.putText(frame, curr_vehicle, (x1,y1-rect_thickness), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.bgr[curr_vehicle], 2)

        self.emision_counter.add_emission_count(self.VehicleCounter.get_vehicle_count())
        self.VehicleCounter.reset_count()

        return frame

    def process_video(self, source):
        """
        Analyzes video or webcam
        :param source: 0 if webcam and the path to the file if its a video
        """
        if source == "0":
            source = 0
        capture = cv.VideoCapture(source)

        # saving the video
        fps = capture.get(5)
        output_dir = Path('output')
        filename = 'test_file1.mp4'
        result_vid = cv.VideoWriter(output_dir.joinpath(filename).__str__(), cv.VideoWriter_fourcc(*'mp4v'), fps, (int(capture.get(3)), int(capture.get(4))))

        self.emision_counter.set_queue_num(fps*self.time_to_avg)
        frame_counter = 0
        while(capture.isOpened()):
            ret, frame = capture.read()

            # !! IMPORTANT !!
            # Do not remove the following check
            # It verifies if there is a frame or not
            if not ret:
                break

            result = self.analyze_frame(frame)
            new_frame = self.process_frame(result, frame)

            # only update the displayed counter every second
            if frame_counter%int(fps) == 0:
                emission_count = self.emision_counter.get_emission_count()
                prompt = f'Total emissions: {round(emission_count,2)} PM2.5/km'

            cv.putText(new_frame, prompt, (int(new_frame.shape[1]*0.01)+2,int(new_frame.shape[0]*0.05)+2), cv.FONT_HERSHEY_SIMPLEX, 1, (10,10,10), 2)
            cv.putText(new_frame, prompt, (int(new_frame.shape[1]*0.01),int(new_frame.shape[0]*0.05)), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


            if self.display:
                cv.imshow('HAZY',new_frame)

            result_vid.write(new_frame)
            result[0].print()
            frame_counter += 1

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        result_vid.release()
        capture.release()
        cv.destroyAllWindows()

def main(file, weights, conf, iou, device, time_to_avg, display=True):
    # YOLOv5 model
    detector = vehicle_detection(
        weights,
        conf=conf,
        iou=iou,
        device=device,
        time_to_avg=time_to_avg,
        display=display
        )
    detector.process_video(file)


if __name__ == "__main__":
    proj_desc = " Hazy A software for approximating PM2.5 emission from traffic footage."
    parser = argparse.ArgumentParser(description='Hazy')
    parser.add_argument('filepath', metavar='filepath', type=str, help='Location of the video file')
    parser.add_argument('--weights', metavar='weights', default='weights/Best.pt', type=str, help='path location of the weights that will be used')
    parser.add_argument('--conf', metavar='conf', default=0.55, type=int, help='Set confidence threshold')
    parser.add_argument('--iou', metavar='iou', default=0.45, type=int, help='Set IOU')
    parser.add_argument('--device', metavar='device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Set Device to use to CUDA or CPU')
    parser.add_argument('--time-to-avg', metavar='time_to_avg', default=5, type=int, help='Set the amount of time the program will average the values')
    parser.add_argument('--no-display', action='store_false', help='option to disable display')
    args = parser.parse_args()
    main(
        args.filepath,
        args.weights,
        args.conf,
        args.iou,
        args.device,
        args.time_to_avg,
        display=args.no_display,
        )
