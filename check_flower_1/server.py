import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), server_address="127.0.0.1:8080")

