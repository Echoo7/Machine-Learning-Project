import pandas as pd
import numpy as np
import datetime


def MLR_model_estimator(model, features, feature_names=None, model_feature_names=None, show_steps=True):
    """
    Estimate travel time using a trained linear model and a single travel's characteristics.

    Parameters
    - model: trained sklearn linear model (e.g. LinearRegression)
    - features: list, dict, or pandas.Series representing one travel row similar to `df_model1` row
    - feature_names: if `features` is a list, this is the list of column names in the same order
      Default assumed order when `None` and `features` is list: ['distance','travel_time','datetime_1h','day','month']
    - model_feature_names: list of feature names (in the same order as model.coef_). If omitted, the function
      will try to read `model.feature_names_in_`. If that's not present, you must provide this argument.
    - show_steps: If True, prints progress and debugging info at each step.

    Behavior (architecture):
    1) Skip "useless" features by only extracting the useful ones (distance, datetime/day/hour, month, etc.)
    2) Extract/derive needed fields: distance, day, hour, month
    3) Convert extracted information to model input (handle dummy encoding used with drop_first=True for days/hours)
    4) Use model.coef_ and model.intercept_ to compute predicted travel time

    Returns
    - predicted travel time (float)

    Notes
    - The function is defensive: it prints where it failed if an exception happens.
    - It supports models trained with dummy columns like `day_1..day_6` (Monday is the reference, all zeros).

    Example usage:
    >>> from functions.estimator import estimate_travel_time
    >>> # Suppose model was trained on X with columns like ['distance','day_1',...,'hour_1',...]
    >>> sample_row = ['1200', None, '2020-01-07 08:15:00', 1, 1]  # if feature_names matches
    >>> pred = estimate_travel_time(model, sample_row, feature_names=['distance','travel_time','datetime_1h','day','month'], model_feature_names=X.columns)

    """
    step = "start"
    try:
        if show_steps:
            print("[STEP] Starting estimation")
        # Normalize input to dict-like
        if isinstance(features, pd.Series):
            base = features.to_dict()
        elif isinstance(features, dict):
            base = features.copy()
        elif isinstance(features, (list, tuple)):
            if feature_names is None:
                # sensible default assumed as user described
                feature_names = ['distance', 'travel_time', 'datetime_1h', 'day', 'month']
                if show_steps:
                    print("[INFO] feature_names not provided; using default:", feature_names)
            if len(feature_names) != len(features):
                raise ValueError(f"feature_names length ({len(feature_names)}) does not match features length ({len(features)})")
            base = dict(zip(feature_names, features))
        else:
            raise TypeError("features must be a list, tuple, dict, or pandas.Series")

        step = "extracted_base"
        if show_steps:
            print("[STEP] Extracted base features:", {k: base.get(k) for k in ['distance','day','datetime_1h','month']})

        # Extract useful values
        # Distance
        step = "distance"
        if 'distance' in base and base['distance'] is not None:
            distance = float(base['distance'])
        else:
            raise KeyError("'distance' not found in input features")
        if show_steps:
            print(f"[STEP] distance = {distance}")

        # Day: prefer explicit 'day' if provided, else derive from datetime_1h
        step = "day"
        day = None
        if 'day' in base and base.get('day') is not None:
            day = int(base['day'])
            if show_steps:
                print(f"[STEP] Using provided day = {day}")
        elif 'datetime_1h' in base and base.get('datetime_1h') is not None:
            dtval = base['datetime_1h']
            if isinstance(dtval, str):
                try:
                    dt = pd.to_datetime(dtval)
                except Exception:
                    dt = None
            elif isinstance(dtval, (pd.Timestamp, datetime.datetime)):
                dt = pd.to_datetime(dtval)
            else:
                dt = pd.to_datetime(dtval)
            if pd.isna(dt):
                raise ValueError("Cannot parse 'datetime_1h' to derive day/hour")
            day = int(dt.dayofweek)
            if show_steps:
                print(f"[STEP] Derived day from datetime_1h = {day}")
        else:
            raise KeyError("Cannot determine 'day' (no 'day' nor 'datetime_1h' provided)")

        # Hour: may be needed for model
        step = "hour"
        hour = None
        if 'hour' in base and base.get('hour') is not None:
            hour = int(base['hour'])
            if show_steps:
                print(f"[STEP] Using provided hour = {hour}")
        elif 'datetime_1h' in base and base.get('datetime_1h') is not None:
            dtval = base['datetime_1h']
            if isinstance(dtval, str):
                dt = pd.to_datetime(dtval)
            elif isinstance(dtval, (pd.Timestamp, datetime.datetime)):
                dt = pd.to_datetime(dtval)
            else:
                dt = pd.to_datetime(dtval)
            hour = int(dt.hour)
            if show_steps:
                print(f"[STEP] Derived hour from datetime_1h = {hour}")
        else:
            # hour not strictly required; leave as None
            if show_steps:
                print("[INFO] No hour provided or derivable; hour features (if any) will be set to 0")

        # Month
        step = "month"
        month = None
        if 'month' in base and base.get('month') is not None:
            month = int(base['month'])
            if show_steps:
                print(f"[STEP] Using provided month = {month}")
        elif 'datetime_1h' in base and base.get('datetime_1h') is not None:
            dtval = base['datetime_1h']
            if isinstance(dtval, str):
                dt = pd.to_datetime(dtval)
            elif isinstance(dtval, (pd.Timestamp, datetime.datetime)):
                dt = pd.to_datetime(dtval)
            else:
                dt = pd.to_datetime(dtval)
            month = int(dt.month)
            if show_steps:
                print(f"[STEP] Derived month from datetime_1h = {month}")
        else:
            if show_steps:
                print("[INFO] No month provided; month feature (if any) will be set to 0")

        # Determine model feature names (order must match model.coef_)
        step = "model_features"
        if model_feature_names is None:
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
                if show_steps:
                    print("[STEP] Obtained model_feature_names from model.feature_names_in_")
            else:
                raise ValueError("model_feature_names not provided and model.feature_names_in_ not available. Provide model_feature_names list matching the order of model.coef_.")
        else:
            model_feature_names = list(model_feature_names)
            if show_steps:
                print("[STEP] Using provided model_feature_names")

        if show_steps:
            print(f"[INFO] model will expect {len(model_feature_names)} features: {model_feature_names}")

        # Build input vector x in same order as model_feature_names
        step = "build_input_vector"
        x_vals = []
        for fname in model_feature_names:
            if fname == 'intercept':
                # some bookkeeping variants include intercept in columns; but model.coef_ will not contain it.
                # We'll ignore intercept entries in feature names if present here (set 0)
                x_vals.append(0.0)
                continue

            if fname == 'distance':
                x_vals.append(float(distance))
                continue

            if fname == 'month':
                x_vals.append(float(month) if month is not None else 0.0)
                continue

            # days encoded as day_1 ... day_6 (drop_first=True): Monday (0) is when all day_* == 0
            if fname.startswith('day_'):
                try:
                    day_index = int(fname.split('_')[1])
                except Exception:
                    # fallback: if not numeric, try to find in base
                    val = float(base.get(fname, 0.0)) if base.get(fname) is not None else 0.0
                    x_vals.append(val)
                    continue
                val = 1.0 if day == day_index else 0.0
                x_vals.append(val)
                continue

            # hours encoded as hour_1 ... hour_23 (drop_first=True): hour 0 is reference
            if fname.startswith('hour_'):
                try:
                    hour_index = int(fname.split('_')[1])
                except Exception:
                    val = float(base.get(fname, 0.0)) if base.get(fname) is not None else 0.0
                    x_vals.append(val)
                    continue
                val = 1.0 if (hour is not None and hour == hour_index) else 0.0
                x_vals.append(val)
                continue

            # generic fallback: if the model feature is present in base, use it
            if fname in base and base.get(fname) is not None:
                try:
                    x_vals.append(float(base.get(fname)))
                except Exception:
                    x_vals.append(0.0)
                continue

            # else set default 0.0
            x_vals.append(0.0)

        x_arr = np.array(x_vals, dtype=float)
        if show_steps:
            print("[STEP] Built input vector (first 10 values shown):", x_arr[:10])

        # Get coefficients
        step = "coeffs"
        if not hasattr(model, 'coef_'):
            raise ValueError('Provided model does not expose coef_')
        coefs = np.array(model.coef_, dtype=float)
        intercept = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
        if show_steps:
            print(f"[STEP] Model intercept = {intercept}")
            print(f"[STEP] Model has {len(coefs)} coefficients; input vector length = {len(x_arr)}")

        if len(coefs) != len(x_arr):
            # allow some flexibility: if model_feature_names contains 'intercept' but coefs doesn't, try to remove intercept positions
            # Find indices where model_feature_names != 'intercept'
            non_intercept_indices = [i for i, n in enumerate(model_feature_names) if n != 'intercept']
            if len(non_intercept_indices) == len(coefs):
                x_arr_reduced = x_arr[non_intercept_indices]
                x_arr = x_arr_reduced
                if show_steps:
                    print("[INFO] Adjusted input vector to match coef_ length by removing 'intercept' positions")
            else:
                raise ValueError(f"Length mismatch between model.coef_ ({len(coefs)}) and built input vector ({len(x_arr)}). Check model_feature_names.")

        step = "predict"
        pred = intercept + float(np.dot(coefs, x_arr))
        if show_steps:
            print(f"[RESULT] Predicted travel time = {pred}")

        return pred

    except Exception as e:
        # helpful debug message including the last step
        print(f"[ERROR] Estimation failed at step '{step}': {e}")
        # optionally re-raise if you want to crash
        raise
