from collections import deque
from statistics import mean

class EmissionCounter:
    def __init__(self, queue_num):
        self.emission_queue = deque()
        self.queue_num = queue_num
        self.emission_values = {
            'Car': 0.0221,
            'Jeepney': 0.8466,
            'Motorcycle': 0.0336,
            'Tricycle': 0.0562,
            'Truck': 0.7519,
            'Utility Vehicle': 0.143,
        }
    def calculate_emission(self, vehicle_count):
        total_emission = 0

        for key in vehicle_count:
            total_emission += self.get_emission_of(key) * vehicle_count[key]

        return total_emission


    def add_emission_count(self, vehicle_count):
        if len(self.emission_queue) == self.queue_num:
            self.emission_queue.popleft()

        self.emission_queue.append(self.calculate_emission(vehicle_count))
    
    def get_emission_of(self, vehicle):
        return self.emission_values[vehicle]

    def get_emission_count(self):
        return mean(self.emission_queue)
    
    def get_curr_emission_count(self):
        return self.emission_queue[-1]

    def set_queue_num(self, new_num):
        self.queue_num = new_num
