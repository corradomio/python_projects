from iPredict_17.train_predict import import_from, json_load


def create_model(name, config):
    klass = import_from(config['class'])
    params = {} | config
    del params['class']
    return klass(**params)


def main():
    config = json_load("./iPredict_17/models_config.json")
    for name in config:
        if name.startswith("#"):
            continue
        m = create_model(name, config[name])
    pass


if __name__ == "__main__":
    main()
