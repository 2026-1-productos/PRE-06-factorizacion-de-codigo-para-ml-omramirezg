# Busque los mejores parametros de un modelo knn...

import os
import sys

# Esto le dice a Python que busque módulos en la carpeta padre (la raíz del proyecto)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.neighbors import KNeighborsRegressor
from src._internals.prepare_data import prepare_data
from src._internals.calculate_metrics import calculate_metrics
from src._internals.print_metrics import print_metrics
from src._internals.save_model_if_better import save_model_if_better


def main():
    x_train, x_test, y_train, y_test = prepare_data()

    # NOTA: Aquí deberías implementar tu ciclo para buscar la cantidad de vecinos
    estimator = KNeighborsRegressor(n_neighbors=5)
    estimator.fit(x_train, y_train)

    print(f"\n{estimator}:")

    # Metricas de entrenamiento
    y_pred_train = estimator.predict(x_train)
    mse_train, mae_train, r2_train = calculate_metrics(y_train, y_pred_train)
    print_metrics(mse_train, mae_train, r2_train, "Metricas de entrenamiento:")

    # Metricas de testing
    y_pred_test = estimator.predict(x_test)
    mse_test, mae_test, r2_test = calculate_metrics(y_test, y_pred_test)
    print_metrics(mse_test, mae_test, r2_test, "Metricas de testing:")

    # Guardar si es mejor
    save_model_if_better(estimator, x_test, y_test)


if __name__ == "__main__":
    main()
