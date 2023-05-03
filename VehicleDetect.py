import torch
import cv2 as cv

from sys import argv
from pathlib import Path
from statistics import mean
from collections import deque

class emission_counter:
    def __init__(self, queue_num):
        self.emission_queue = deque()
        self.queue_num = queue_num
        # TODO: change into actual values
        self.emission_values = {
            'Car': 0.0221,
            'Jeepney': 0.8466,
            'Motorcycle': 0.0336,
            'Tricycle': 0.0562,
            'Truck': 0.7519,
            'Utility Vehicle': 0.143,
        }

    def add_emission_count(self, vehicle_count):
        total_emission = 0

        if len(self.emission_queue) == self.queue_num:
            self.emission_queue.popleft()

        for key in vehicle_count:
            total_emission += self.get_emission_of(key) * vehicle_count[key]

        self.emission_queue.append(total_emission)
    
    def get_emission_of(self, vehicle):
        return self.emission_values[vehicle]

    def get_emission_count(self):
        # print(self.emission_queue)
        return mean(self.emission_queue)

    def set_queue_num(self, new_num):
        self.queue_num = new_num

class vehicle_counter:
    def __init__(self):
        self.vehicle_count = {
            'Car': 0,
            'Jeepney': 0,
            'Motorcycle': 0,
            'Tricycle': 0,
            'Truck': 0,
            'Utility Vehicle': 0,
        }

    def reset_count(self):
        for key in self.vehicle_count:
            self.vehicle_count[key] = 0

    def get_vehicle_count(self, type='all'):
        if type == 'all':
            return self.vehicle_count
        else:
            return self.vehicle_count[type]

    def increment_vehicle(self, type):
        self.vehicle_count[type] += 1

class vehicle_detection:
    def __init__(self, path_to_model, conf=0.55, iou=0.45, device='cuda' if torch.cuda.is_available() else 'cpu', time_to_avg=5, display=False):
        self.path_to_model = Path(path_to_model)
        self.model = self.select_custom_model(self.path_to_model.__str__())
        self.device = device
        self.model.to(self.device)
        self.model.conf = conf
        self.model.iou = iou
        self.time_to_avg = time_to_avg
        self.vehicle_counter = vehicle_counter()
        self.emision_counter = emission_counter(30)
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
        # return torch.hub.load("ultralytics/yolov5", "custom", path_to_model)


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
            self.vehicle_counter.increment_vehicle(curr_vehicle)

            rect_thickness = 1
            cv.rectangle(frame, (x1,y1), (x2,y2), self.bgr[curr_vehicle], rect_thickness)
            cv.putText(frame, curr_vehicle, (x1,y1-rect_thickness), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.bgr[curr_vehicle], 2)

        self.emision_counter.add_emission_count(self.vehicle_counter.get_vehicle_count())
        self.vehicle_counter.reset_count()

        return frame

    def process_video(self, source):
        """
        Analyzes video or webcam
        :param source: 0 if webcam and the path to the file if its a video
        """
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

def main(file, weights='weights/Best.pt', display=False):

    # YOLOv5 model
    # detector = vehicle_detection("/home/jg/Downloads/git/yolov5/weights/Apr24Best.pt")
    detector = vehicle_detection(weights, display=display)
    # detector.process_video("/home/jg/Documents/College4.2/SP/test1/VID_20230225_172405.mp4")
    detector.process_video(file)


if __name__ == "__main__":
    main(argv[1])
