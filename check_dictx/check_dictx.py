import numpy as np
from stdlib.dictx import dict

def main():

    d = dict(None)
    d = dict({})
    d = dict({'a': 1})
    d = dict()

    data_props = {'num_historical_numeric': 4,
                  'num_historical_categorical': 6,
                  'num_static_numeric': 10,
                  'num_static_categorical': 11,
                  'num_future_numeric': 2,
                  'num_future_categorical': 3,
                  'historical_categorical_cardinalities': (1 + np.random.randint(10, size=6)).tolist(),
                  'static_categorical_cardinalities': (1 + np.random.randint(10, size=11)).tolist(),
                  'future_categorical_cardinalities': (1 + np.random.randint(10, size=3)).tolist(),
                  }
    configuration = {
        'model':
            {
                'dropout': 0.05,
                'state_size': 64,
                'output_quantiles': [0.1, 0.5, 0.9],
                'lstm_layers': 2,
                'attention_heads': 4
            },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    configuration = dict(configuration)

    print(configuration.task_type)
    configuration.task_type = 'classification'
    print(configuration.task_type)
    print(configuration.model.dropout)
    print(configuration.get('model.dropout'))
    print(configuration.get('model.dropoutx'))

    pass


if __name__ == "__main__":
    main()
