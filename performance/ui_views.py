from django.shortcuts import render
import numpy as np
from .views import model


def predict_form(request):
    context = {}
    if request.method == 'POST':
        try:
            hours_studied = float(request.POST.get('hours_studied', 0))
            previous_scores = float(request.POST.get('previous_scores', 0))
            extracurricular = 1 if request.POST.get('extracurricular') in ['on', 'true', '1', 'yes'] else 0
            sleep_hours = float(request.POST.get('sleep_hours', 0))
            sample_papers = float(request.POST.get('sample_papers', 0))

            features = np.array([[hours_studied,
                                   previous_scores,
                                   extracurricular,
                                   sleep_hours,
                                   sample_papers]])

            prediction = model.predict(features)[0]
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
