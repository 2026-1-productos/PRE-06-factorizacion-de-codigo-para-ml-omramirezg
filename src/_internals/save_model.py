import os
import pickle


def save_model(model, save_path):
    # Asegurarnos de que la carpeta 'models/' exista
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as file:
        pickle.dump(model, file)
