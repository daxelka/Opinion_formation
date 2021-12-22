class Simulation:
    def __init__(self, parameter_iterator, initial_values_iterator, model_run):
        self.parameter_iterator = parameter_iterator
        self.initial_values_iterator = initial_values_iterator
        self.model_run = model_run

    def run(self):
        results = {}

        for parameter in self.parameter_iterator():
            for initial_value in self.initial_values_iterator():
                result = self.model_run(parameter, initial_value)

                if results.get(parameter) is None:
                    results[parameter] = []

                if result:
                    results[parameter].append(result)

        return results
