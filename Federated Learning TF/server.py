import flwr as fl

fl.server.start_server(
    # Spacify IP address and port for communication
    server_address="0.0.0.0:8080",
    # Run 3 rounds of federated training
    config=fl.server.ServerConfig(num_rounds=3)
)