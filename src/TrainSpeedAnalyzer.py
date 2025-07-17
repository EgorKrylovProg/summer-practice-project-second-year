class TrainSpeedAnalyzer:
    def __init__(self, telega_length=8.650, num_axles=4):
        self.telega_length = telega_length
        self.num_axles = num_axles

    def calculate_speeds(self, timestamps, peaks):
        speeds = []
        for i in range(0, len(peaks) - self.num_axles + 1, self.num_axles):
            group = peaks[i:i + self.num_axles]
            if len(group) == self.num_axles:
                delta_t = timestamps[group[-1]] - timestamps[group[0]]
                if delta_t > 0:
                    speed = (self.telega_length / delta_t) * 3.6
                    speeds.append(speed)
        return speeds

    def print_speeds(self, speeds):
        if not speeds:
            print("  [i] Тележки не обнаружены (недостаточно пиков)")
        else:
            for i, speed in enumerate(speeds, 1):
                print(f"  Тележка {i}: {speed:.2f} км/ч")