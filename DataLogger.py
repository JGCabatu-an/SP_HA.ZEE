import datetime
import csv

from os import path, makedirs

from EmissionCounter import EmissionCounter
from VehicleCounter import VehicleCounter

# TODO: Save data to csv file
class DataLogger:
    def __init__(self):
        self.headers = ['datetime', 'PM2.5', 'Cars', 'Jeepney', 'Motorcycle', 'Tricycle', 'Truck', 'Utility Vehicle']

        # Figure out the directory name
        dir_id =''
        while path.exists(path.join(f'{path.dirname(path.realpath(__file__))}/outputs/run{dir_id}/')):
            dir_id = dir_id + 1 if dir_id != '' else 1
        
        self.dir_name = path.join(f'{path.dirname(path.realpath(__file__))}/outputs/run{dir_id}/')
        makedirs(self.dir_name)
        self.filepath = f'{self.dir_name}{datetime.datetime.now()}.csv'
        with open(self.filepath, 'w', encoding='UTF-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)

    def get_dir(self):
        return self.dir_name
    
    def write_row(self, data):
        '''
        Writes one row in the csv file
        :param data: the PM2.5 and object data in an array
        '''
        with open(self.filepath, 'a', encoding='UTF-8', newline='') as file:
            temp = [datetime.datetime.now()] + data
            writer = csv.writer(file)
            writer.writerow(temp)