from django.shortcuts import render
import numpy as np
from .views import model
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
from django.conf import settings
import joblib
import json


# Load persisted normality model and feature ranges if available; fall back to CSV init
def _load_normality_artifacts():
    try:
        iso_path = Path(settings.BASE_DIR) / 'performance' / 'normality.pkl'
        feats_path = Path(settings.BASE_DIR) / 'performance' / 'normality_feature_names.json'
        if iso_path.exists() and feats_path.exists():
            iso = joblib.load(iso_path)
            with open(feats_path, 'r') as f:
                feat_names = json.load(f)
            # For soft constraints, attempt to load CSV and compute mins/maxs on these features
            mins = {}
            maxs = {}
            try:
                csv_path = Path(settings.BASE_DIR).parent / 'StudentPerformance.csv'
                df = pd.read_csv(csv_path)
                # Map Yes/No
                if 'Extracurricular Activities' in df and df['Extracurricular Activities'].dtype == object:
                    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0}).fillna(0)
                # Rebuild engineered features if present
                if 'study_to_sleep_ratio' in feat_names:
                    df['study_to_sleep_ratio'] = df['Hours Studied'] / (df['Sleep Hours'] + 1)
                if 'practice_rate' in feat_names:
                    df['practice_rate'] = df['Sample Question Papers Practiced'] / (df['Hours Studied'] + 1)
                sub = df[feat_names]
                mins = sub.min().to_dict()
                maxs = sub.max().to_dict()
            except Exception:
                mins, maxs = {}, {}
            return iso, feat_names, mins, maxs
    except Exception:
        pass
    # Fallback: try building from CSV like before
    try:
        csv_path = Path(settings.BASE_DIR).parent / 'StudentPerformance.csv'
        df = pd.read_csv(csv_path)
        if df['Extracurricular Activities'].dtype == object:
            df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0}).fillna(0)
        df['study_to_sleep_ratio'] = df['Hours Studied'] / (df['Sleep Hours'] + 1)
        df['practice_rate'] = df['Sample Question Papers Practiced'] / (df['Hours Studied'] + 1)
        feat_names = [
            'Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced','study_to_sleep_ratio','practice_rate'
        ]
        features = df[feat_names]
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(features)
        mins = features.min().to_dict()
        maxs = features.max().to_dict()
        return iso, feat_names, mins, maxs
    except Exception:
        return None, [], {}, {}


_ISO_MODEL, _ISO_FEATURES, _MIN_FEATS, _MAX_FEATS = _load_normality_artifacts()


def predict_form(request):
    context = {}
    if request.method == 'POST':
        try:
            hours_studied = float(request.POST.get('hours_studied', 0))
            previous_scores = float(request.POST.get('previous_scores', 0))
            extracurricular = 1 if request.POST.get('extracurricular') in ['on', 'true', '1', 'yes'] else 0
            sleep_hours = float(request.POST.get('sleep_hours', 0))
            sample_papers = float(request.POST.get('sample_papers', 0))

            # Build DataFrame with training column names + engineered features
            row = {
                'Hours Studied': hours_studied,
                'Previous Scores': previous_scores,
                'Extracurricular Activities': float(extracurricular),
                'Sleep Hours': sleep_hours,
                'Sample Question Papers Practiced': sample_papers,
            }
            row['study_to_sleep_ratio'] = hours_studied / (sleep_hours + 1)
            row['practice_rate'] = sample_papers / (hours_studied + 1)

            errors = []
            # Physical constraints
            if sleep_hours <= 0 or sleep_hours >= 24:
                errors.append('Sleep hours must be between 0 and 24 (exclusive).')
            if hours_studied < 0 or hours_studied > 18:
                errors.append('Study hours should be between 0 and 18 per day.')
            if hours_studied + sleep_hours > 24:
                errors.append('Study hours + sleep hours cannot exceed 24 hours.')

            # Soft training-range constraints if available
            if _MIN_FEATS and _MAX_FEATS:
                for k in ['Hours Studied','Previous Scores','Sleep Hours','Sample Question Papers Practiced','study_to_sleep_ratio','practice_rate']:
                    if k in _MIN_FEATS and k in _MAX_FEATS:
                        v = float(row[k])
                        if not (_MIN_FEATS[k] <= v <= _MAX_FEATS[k]):
                            errors.append(f"{k} should be between {round(_MIN_FEATS[k],2)} and {round(_MAX_FEATS[k],2)} based on training data.")

            # ML abnormality check
            if _ISO_MODEL is not None and _ISO_FEATURES:
                X_row = pd.DataFrame([row], columns=_ISO_FEATURES)
                if int(_ISO_MODEL.predict(X_row)[0]) == -1:
                    errors.append('Inputs appear abnormal compared to training data. Please adjust values.')

            if errors:
                context['error'] = '\n'.join(errors)
            else:
                # Use DataFrame with original training feature names for prediction
                X_pred = pd.DataFrame([row], columns=['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced'])
                prediction = model.predict(X_pred)[0]
                context['predicted_performance_index'] = round(float(prediction), 2)
            context['form_values'] = {
                'hours_studied': hours_studied,
                'previous_scores': previous_scores,
                'extracurricular': bool(extracurricular),
                'sleep_hours': sleep_hours,
                'sample_papers': sample_papers,
            }
        except Exception:
            context['error'] = 'Unable to compute prediction. Please check your inputs.'
    return render(request, 'performance/predict_form.html', context)
