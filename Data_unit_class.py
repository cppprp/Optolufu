import pickle
""" Single animal definition"""

class Unit:
    mapped_points = None
    mapped_points_norm = None
    index = []
    ref_frame = None
    insp_frame = None
    exp_frame = None
    NS_wave = []
    WE_wave = []
    EW_wave = []
    def __init__(self, name="", date="", fps = ""):
        self.name = name
        self.date = date
        self.FPS = fps

    def read_meta(self, file_path):
        with open(file_path, 'r') as file:
            line = file.readline()
            self.date = line.split(' ')[-2].strip()

            line = file.readline()
            self.name = line.split(' ')[-1].strip()
            file.readline()

            line = file.readline()
            self.FPS = line.split(' ')[-1].strip()


    def save_class(self, pickle_path):
        with open(pickle_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_class(cls, pickle_path):
        with open(pickle_path, 'rb') as file:
            obj = pickle.load(file)
        return obj

