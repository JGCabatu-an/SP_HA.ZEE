from statistics import mean

class EmissionCounter:
    def __init__(self, queue_num):
        # self.frame_count = 0
        self.emission_values = {
            'PM' : {
                'Car': 0.0221,
                'Jeepney': 0.8466,
                'Motorcycle': 0.0336,
                'Tricycle': 0.0562,
                'Truck': 0.7519,
                'Utility Vehicle': 0.143
            },
            'CO2' : {
                'Car': 109.8958,
                'Jeepney': 668.7415,
                'Motorcycle': 60.0983,
                'Tricycle': 66.8747,
                'Truck': 842.0852,
                'Utility Vehicle': 92.4039
            },
            'CH4': {
                'Car': 0.7408 ,
                'Jeepney': 0.2357 ,
                'Motorcycle': 2.3022 ,
                'Tricycle': 4.0906,
                'Truck': 0.3648,
                'Utility Vehicle': 0.3538
            },
            'N2O': {
                'Car': 0.0099 ,
                'Jeepney': 0.0316,
                'Motorcycle': 0.0015 ,
                'Tricycle': 0.0021,
                'Truck': 0.0226,
                'Utility Vehicle': 0.0063
            }
            
        }
    
    # def inc_frame_count(self):
    #     self.frame_count += 1

    def calculate_emission(self, vehicle_count, emission_type):
        total_emission = 0

        for key in vehicle_count:
            total_emission += self.get_emission_of(key, emission_type) * vehicle_count[key]

        return total_emission


    # def add_emission_count(self, vehicle_count):
    #     if len(self.emission_queue) == self.queue_num:
    #         self.emission_queue.popleft()

    #     self.emission_queue.append(self.calculate_emission(vehicle_count))
    
    def get_emission_of(self, vehicle, emission_type):
        return self.emission_values[emission_type][vehicle]

    # def get_emission_count(self):
    #     return mean(self.emission_queue)
    
    # def get_curr_emission_count(self):
    #     return self.emission_queue[-1]

    # def set_queue_num(self, new_num):
    #     self.queue_num = new_num
