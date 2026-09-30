import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from crud_service import FILE_TRABAJO
from nombres_variables import obtener_nombre_amigable

warnings.filterwarnings('ignore')

N_FOLDS = 5
N_ITER_BUSQUEDA = 20
RANDOM_STATE = 42
METRICA_SINTONIA = 'f1'
SCORING_BUSQUEDA = {
    'f1': 'f1',
    'sensibilidad': 'recall',
    'precision': 'precision',
    'roc_auc': 'roc_auc',
}


def obtener_dataframe_procesado():
    """Lee el CSV y realiza el preprocesamiento inicial para ML"""
    df = pd.read_csv(FILE_TRABAJO)

    df = df.drop(columns=['ID'])

    columnas_bool = df.select_dtypes(include=['bool']).columns
    df[columnas_bool] = df[columnas_bool].astype(int)

    df.rename(columns={'genero': 'generoMasculino'}, inplace=True)
    df['generoMasculino'] = df['generoMasculino'].map({'Masculino': 1, 'Femenino': 0}).astype(int)

    return df


def preprocesar_para_ml(df):
    """Aplica encoding y estandarización para ML"""
    columnas_strings = ['servicioTelefonico', 'servicioDeInternet', 'tipoDeContrato', 'metodoDePago']
    categorias_a_eliminar = ['No', 'No', 'Mes a mes', 'Cheque enviado por correo']
    columnas_escalar = ['antiguedadEnMeses', 'cuentasMensuales', 'cobroTotal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop=categorias_a_eliminar, handle_unknown='ignore', sparse_output=False), columnas_strings),
            ('num', StandardScaler(), columnas_escalar)
        ],
        remainder='passthrough'
    )
    preprocessor.set_output(transform="pandas")

    df_procesado = preprocessor.fit_transform(df)

    columnas_cat = [col for col in df_procesado.columns if col.startswith('cat__')]
    df_procesado[columnas_cat] = df_procesado[columnas_cat].astype(int)

    df_procesado.columns = df_procesado.columns.str.replace('num__', '', regex=False)
    df_procesado.columns = df_procesado.columns.str.replace('remainder__', '', regex=False)

    df_procesado.rename(columns={
        'cat__servicioTelefonico_Múltiples líneas': 'MultiplesLineas',
        'cat__servicioTelefonico_Una línea': 'UnicaLinea',
        'cat__servicioDeInternet_DSL': 'TieneDSL',
        'cat__servicioDeInternet_Fibra óptica': 'TieneFibraOptica',
        'cat__tipoDeContrato_Dos años': 'ContratoDosAnos',
        'cat__tipoDeContrato_Un año': 'ContratoUnAno',
        'cat__metodoDePago_Cheque electrónico': 'PagoChequeElectronico',
        'cat__metodoDePago_Tarjeta de crédito (automático)': 'PagoTarjetaCredito',
        'cat__metodoDePago_Transferencia bancaria (automático)': 'PagoTransferenciaBancaria'
    }, inplace=True)

    return df_procesado


def eliminar_columnas_problematicas(df):
    """Elimina columnas con alta multicolinealidad detectada en el EDA"""
    columnas_a_eliminar = ['cobroTotal', 'generoMasculino', 'UnicaLinea', 'TieneDSL', 'TieneFibraOptica']
    return df.drop(columns=columnas_a_eliminar)


def preparar_datos_ml():
    """Pipeline de preparación de datos para ML (sin partición train/test)"""
    df = obtener_dataframe_procesado()
    df_procesado = preprocesar_para_ml(df)
    df_limpio = eliminar_columnas_problematicas(df_procesado)

    y = df_limpio['canceloServicio']
    X = df_limpio.drop(columns=['canceloServicio'])

    return X, y


def dividir_datos(X, y, test_size=0.20, random_state=RANDOM_STATE):
    """Divide los datos en conjuntos de entrenamiento y prueba"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def crear_kfold():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def ranking_univariado(X_train, y_train):
    """Ordena variables por F de ANOVA (SelectKBest / f_classif) sobre el train."""
    f_scores, p_valores = f_classif(X_train, y_train)
    ranking = pd.DataFrame({
        'variable': X_train.columns,
        'f_score': f_scores,
        'p_valor': p_valores,
    }).sort_values('f_score', ascending=False).reset_index(drop=True)
    return ranking


def seleccionar_features(X_train, y_train, X_test):
    """
    Selección de features solo con el conjunto de entrenamiento:
    1) ranking univariado (f_classif) para interpretar
    2) SelectFromModel con un RF preliminar (umbral mediana)
    """
    ranking = ranking_univariado(X_train, y_train)

    estimador = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector = SelectFromModel(estimador, threshold='median')
    selector.fit(X_train, y_train)

    columnas = X_train.columns[selector.get_support()].tolist()
    if len(columnas) < 5:
        columnas = ranking.head(8)['variable'].tolist()

    return X_train[columnas], X_test[columnas], columnas, ranking


def _metricas_clase_positiva(conf_matrix):
    """Métricas de calidad a partir de la matriz de confusión (clase 1 = abandonador)."""
    tn, fp, fn, tp = conf_matrix.ravel()
    total = tn + fp + fn + tp
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (
        2 * precision * sensibilidad / (precision + sensibilidad)
        if (precision + sensibilidad) else 0.0
    )
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': accuracy,
        'precision': precision,
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'f1': f1,
    }


def evaluar_modelo(model, X_test, y_test, nombre_modelo):
    """Evalúa un modelo en el conjunto de prueba y devuelve métricas"""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No abandonó', 'Abandonó'])
    metricas = _metricas_clase_positiva(conf_matrix)

    print(f"\n=== Resultados {nombre_modelo} (test) ===")
    print(f"Exactitud (Accuracy): {metricas['accuracy']:.4f}")
    print(f"Precisión (Precision, abandonadores): {metricas['precision']:.4f}")
    print(f"Sensibilidad (Recall): {metricas['sensibilidad']:.4f}")
    print(f"Especificidad (Specificity): {metricas['especificidad']:.4f}")
    print(f"F1: {metricas['f1']:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(report)

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': y_pred,
        'metricas': metricas,
    }


def evaluar_cv(modelo, X_train, y_train, cv, nombre_modelo):
    """Validación cruzada estratificada sobre el entrenamiento (sin tocar el test)."""
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'sensibilidad': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }
    resultados = cross_validate(
        modelo, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )
    resumen = {}
    print(f"\n=== Validación cruzada {nombre_modelo} ({N_FOLDS}-fold) ===")
    
    # Mapeo de nombres de métricas a español con inglés entre paréntesis
    nombres_metricas = {
        'accuracy': 'Exactitud (Accuracy)',
        'precision': 'Precisión (Precision)',
        'sensibilidad': 'Sensibilidad (Recall)',
        'f1': 'F1',
        'roc_auc': 'ROC-AUC',
    }
    
    for nombre_score in scoring:
        valores = resultados[f'test_{nombre_score}']
        resumen[nombre_score] = {
            'media': float(np.mean(valores)),
            'desvio': float(np.std(valores)),
        }
        nombre_mostrar = nombres_metricas.get(nombre_score, nombre_score)
        print(
            f"{nombre_mostrar}: {resumen[nombre_score]['media']:.4f} "
            f"(± {resumen[nombre_score]['desvio']:.4f})"
        )
    return resumen


def entrenar_modelo_logistico(X_train, y_train):
    """Entrena un modelo de Regresión Logística (baseline sin sintonía)."""
    model = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def extraer_metricas_sintonia(busqueda):
    """F1 (métrica de sintonía) y sensibilidad de la mejor configuración en CV."""
    idx = int(busqueda.best_index_)
    cvres = busqueda.cv_results_
    return {
        'best_params': busqueda.best_params_,
        'best_score_cv': float(busqueda.best_score_),
        'sintonia': {
            'f1': float(cvres['mean_test_f1'][idx]),
            'f1_std': float(cvres['std_test_f1'][idx]),
            'sensibilidad': float(cvres['mean_test_sensibilidad'][idx]),
            'sensibilidad_std': float(cvres['std_test_sensibilidad'][idx]),
            'precision': float(cvres['mean_test_precision'][idx]),
            'roc_auc': float(cvres['mean_test_roc_auc'][idx]),
        },
    }


def _imprimir_sintonia(nombre, busqueda):
    metricas = extraer_metricas_sintonia(busqueda)['sintonia']
    print(f"\n=== Sintonía {nombre} ===")
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(
        f"Métrica de sintonía (F1): {metricas['f1']:.4f} "
        f"(± {metricas['f1_std']:.4f})"
    )
    print(
        f"Sensibilidad (Recall) de esa configuración: {metricas['sensibilidad']:.4f} "
        f"(± {metricas['sensibilidad_std']:.4f})"
    )


def sintonizar_regresion_logistica(X_train, y_train, cv):
    """Baseline: grilla de regularización C (L2 y L1) sobre k-fold."""
    modelo = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=2000,
    )
    grilla = [
        {
            'solver': ['lbfgs'],
            'penalty': ['l2'],
            'C': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        {
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'C': [0.05, 0.1, 0.5, 1.0, 5.0],
        },
    ]
    busqueda = GridSearchCV(
        modelo,
        grilla,
        scoring=SCORING_BUSQUEDA,
        refit=METRICA_SINTONIA,
        cv=cv,
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)
    _imprimir_sintonia('Regresión Logística', busqueda)
    return busqueda


def sintonizar_random_forest(X_train, y_train, cv):
    modelo = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    distribuciones = {
        'n_estimators': [80, 150, 250, 400],
        'max_depth': [4, 6, 8, 12, 18, None],
        'min_samples_leaf': [1, 2, 4, 8],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    busqueda = RandomizedSearchCV(
        modelo,
        distribuciones,
        n_iter=N_ITER_BUSQUEDA,
        scoring=SCORING_BUSQUEDA,
        refit=METRICA_SINTONIA,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)
    _imprimir_sintonia('Random Forest', busqueda)
    return busqueda


def sintonizar_xgboost(X_train, y_train, cv):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    modelo = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
    )
    distribuciones = {
        'n_estimators': [80, 150, 250, 400],
        'max_depth': [2, 3, 4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0],
    }
    busqueda = RandomizedSearchCV(
        modelo,
        distribuciones,
        n_iter=N_ITER_BUSQUEDA,
        scoring=SCORING_BUSQUEDA,
        refit=METRICA_SINTONIA,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)
    _imprimir_sintonia('XGBoost', busqueda)
    return busqueda


def obtener_correlaciones(X, y):
    """Calcula la matriz de correlación sin mostrar gráficos"""
    df_temp = X.copy()
    df_temp['canceloServicio'] = y

    # Renombrar columnas a nombres amigables
    df_temp = df_temp.rename(columns={col: obtener_nombre_amigable(col, es_procesada=True) for col in df_temp.columns})

    corr_matrix = df_temp.corr()

    print('Correlación con la Cancelación (churn):')
    print(corr_matrix['Canceló el servicio'].sort_values(ascending=False))

    return corr_matrix


def _params_legibles(params):
    return {clave: (None if valor is None else valor) for clave, valor in params.items()}


def formatear_reporte_pipeline(resultados):
    """Texto para la pestaña de modelos de la UI."""
    sel = resultados['seleccion']
    
    # Renombrar columnas seleccionadas a nombres amigables
    columnas_amigables = [obtener_nombre_amigable(col, es_procesada=True) for col in sel['columnas']]
    
    lineas = [
        "=== Pipeline de Machine Learning Completado ===",
        "",
        f"Registros totales: {resultados['n_total']}",
        f"Entrenamiento: {resultados['n_train']} | Prueba: {resultados['n_test']}",
        "Métrica de sintonía: F1 (se elige la configuración que maximiza F1 en CV)",
        f"Validación cruzada: {N_FOLDS}-fold estratificado",
        f"Iteraciones de búsqueda (RF y XGBoost): {N_ITER_BUSQUEDA}",
        "",
        "=== Selección de features (solo train) ===",
        f"Variables iniciales: {sel['n_inicial']}",
        "Variables seleccionadas (" + str(len(sel['columnas'])) + "): " + ', '.join(columnas_amigables),
        "",
        "Ranking univariado (F-score, primeras 8):",
    ]
    for _, fila in sel['ranking'].head(8).iterrows():
        nombre_amigable = obtener_nombre_amigable(fila['variable'], es_procesada=True)
        lineas.append(f"  - {nombre_amigable}: F={fila['f_score']:.2f}, p={fila['p_valor']:.4g}")

    for clave, titulo in (
        ('resultados_log', 'Regresión Logística (baseline)'),
        ('resultados_rf', 'Random Forest'),
        ('resultados_xgb', 'XGBoost'),
    ):
        bloque = resultados[clave]
        m = bloque['metricas']
        cv = bloque['cv']
        s = bloque['sintonia']
        lineas.extend([
            "",
            f"=== {titulo} ===",
            f"Hiperparámetros: {_params_legibles(bloque['best_params'])}",
            f"F1 de sintonía (CV de la búsqueda): {s['f1']:.4f} (± {s['f1_std']:.4f})",
            f"Sensibilidad de esa misma configuración: {s['sensibilidad']:.4f} (± {s['sensibilidad_std']:.4f})",
            "",
            f"CV {N_FOLDS}-fold sobre el modelo sintonizado:",
            f"  F1: {cv['f1']['media']:.4f} (± {cv['f1']['desvio']:.4f})",
            f"  Sensibilidad (Recall): {cv['sensibilidad']['media']:.4f} (± {cv['sensibilidad']['desvio']:.4f})",
            f"  Precisión (Precision): {cv['precision']['media']:.4f} (± {cv['precision']['desvio']:.4f})",
            f"  Exactitud (Accuracy): {cv['accuracy']['media']:.4f} (± {cv['accuracy']['desvio']:.4f})",
            f"  ROC-AUC: {cv['roc_auc']['media']:.4f} (± {cv['roc_auc']['desvio']:.4f})",
            "",
            "Evaluación en test (matriz de confusión):",
            f"  VN={m['tn']}  FP={m['fp']}  FN={m['fn']}  VP={m['tp']}",
            f"  Exactitud (Accuracy): {m['accuracy']:.4f}",
            f"  Precisión (Precision): {m['precision']:.4f}",
            f"  Sensibilidad (Recall): {m['sensibilidad']:.4f}",
            f"  Especificidad (Specificity): {m['especificidad']:.4f}",
            f"  F1: {m['f1']:.4f}",
            "",
            bloque['classification_report'],
        ])

    return "\n".join(lineas)


def ejecutar_pipeline_completo():
    """
    Ciclo completo:
    preprocesamiento → split → selección de features → k-fold + sintonía
    (logística, RF, XGBoost) → evaluación final en test.
    """
    print("=== Preparando datos ===")
    X, y = preparar_datos_ml()

    print("=== Mostrando correlaciones ===")
    obtener_correlaciones(X, y)

    print("=== Dividiendo datos ===")
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    print(f"Registros totales: {len(X)}")
    print(f"Registros para entrenar: {len(X_train)}")
    print(f"Registros para probar: {len(X_test)}")

    print("=== Selección de features ===")
    x_train_sel, x_test_sel, columnas, ranking = seleccionar_features(X_train, y_train, X_test)
    print(f"Variables seleccionadas ({len(columnas)}): {columnas}")

    cv = crear_kfold()

    print("\n=== Entrenando baseline: Regresión Logística ===")
    busqueda_log = sintonizar_regresion_logistica(x_train_sel, y_train, cv)
    cv_log = evaluar_cv(busqueda_log.best_estimator_, x_train_sel, y_train, cv, "Regresión Logística")
    resultados_log = evaluar_modelo(busqueda_log.best_estimator_, x_test_sel, y_test, "Regresión Logística")
    resultados_log.update(extraer_metricas_sintonia(busqueda_log))
    resultados_log['cv'] = cv_log

    print("\n=== Entrenando Random Forest (sintonía + k-fold) ===")
    busqueda_rf = sintonizar_random_forest(x_train_sel, y_train, cv)
    cv_rf = evaluar_cv(busqueda_rf.best_estimator_, x_train_sel, y_train, cv, "Random Forest")
    resultados_rf = evaluar_modelo(busqueda_rf.best_estimator_, x_test_sel, y_test, "Random Forest")
    resultados_rf.update(extraer_metricas_sintonia(busqueda_rf))
    resultados_rf['cv'] = cv_rf

    print("\n=== Entrenando XGBoost (sintonía + k-fold) ===")
    busqueda_xgb = sintonizar_xgboost(x_train_sel, y_train, cv)
    cv_xgb = evaluar_cv(busqueda_xgb.best_estimator_, x_train_sel, y_train, cv, "XGBoost")
    resultados_xgb = evaluar_modelo(busqueda_xgb.best_estimator_, x_test_sel, y_test, "XGBoost")
    resultados_xgb.update(extraer_metricas_sintonia(busqueda_xgb))
    resultados_xgb['cv'] = cv_xgb

    return {
        'X_train': x_train_sel,
        'X_test': x_test_sel,
        'y_train': y_train,
        'y_test': y_test,
        'n_total': len(X),
        'n_train': len(x_train_sel),
        'n_test': len(x_test_sel),
        'model_log': busqueda_log.best_estimator_,
        'model_rf': busqueda_rf.best_estimator_,
        'model_xgb': busqueda_xgb.best_estimator_,
        'resultados_log': resultados_log,
        'resultados_rf': resultados_rf,
        'resultados_xgb': resultados_xgb,
        'seleccion': {
            'columnas': columnas,
            'ranking': ranking,
            'n_inicial': X_train.shape[1],
        },
    }
