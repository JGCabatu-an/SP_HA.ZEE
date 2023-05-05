from collections import deque
from statistics import mean

class EmissionCounter:
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
