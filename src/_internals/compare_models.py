from sklearn.metrics import r2_score


def compare_models(new_model, best_model, x_test, y_test):
    """Devuelve el modelo que tenga un mejor R2 en los datos de prueba."""
    if best_model is None:
        return new_model

    new_pred = new_model.predict(x_test)
    best_pred = best_model.predict(x_test)

    new_score = r2_score(y_test, new_pred)
    best_score = r2_score(y_test, best_pred)

    if new_score > best_score:
        return new_model
    return best_model
