import logging.config
import training.views as tv
import prediction.views as pv

from stdlib import jsonx


# -------------------------------------------------------------------------
# Minimal implementation of Request/Body
# -------------------------------------------------------------------------

class Body:
    def __init__(self, body):
        self._body = body

    def decode(self, encode):
        return jsonx.dumps(self._body)
# end


class Request:

    def __init__(self, jfile):
        self.body = Body(jsonx.load(jfile))
    pass
# end


# -------------------------------------------------------------------------
# training
# prediction
# -------------------------------------------------------------------------

def training_test():
    request = Request("test_data/training-20240501_084354.json")
    tv.training(request)
    pass


def prediction_test():
    # request = Request("test_data/prediction-20240501_085438.json")
    request = Request("test_data/prediction-20240503_144002.json")
    pv.prediction(request)
    pass


# -------------------------------------------------------------------------
# training
# prediction
# -------------------------------------------------------------------------

def main():
    training_test()
    # prediction_test()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
