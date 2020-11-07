class PreprocessingPipeline(object):
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, inp, metadata):
        output = inp
        for p in self._preprocessors:
            output, metadata = p(output, metadata)
        return output, metadata
