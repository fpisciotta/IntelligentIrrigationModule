from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from sklearn.externals import joblib
from django.views.decorators.csrf import csrf_exempt
import os.path,json,base64,sys, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))

@csrf_exempt 
def predict(request):
    if request.method == 'POST':

        clf = joblib.load(os.path.join(BASE, "model.pkl"))
        body_unicode = request.body.decode('utf-8')
        print(body_unicode)
        body = json.loads(body_unicode)
        print(body['parameters'])
        content = body['parameters'].split(' ')
        content = [float(x) for x in content]
        content = np.array(content).astype(np.float)
        print(content)
        try:
            result = clf.predict(content)
            resp = { 'prediction': result[0]}
        except:
            return HttpResponse(sys.exc_info()[0], content_type="application/json")
        return HttpResponse(json.dumps(resp), content_type="application/json")


def index(request):
    print("TEST")
    return HttpResponse("<html><body><h1>Use /api/predict for the intelligent irrigation module</h1></body></html>")
