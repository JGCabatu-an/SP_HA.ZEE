class VehicleCounter:
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