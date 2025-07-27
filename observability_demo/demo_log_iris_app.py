from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import joblib
import numpy as np
import pandas as pd
import logging
import time
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Set up structured logging
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(
                    json.dumps(
                               {
                            "severity": "%(levelname)s",
                            "message": "%(message)s",
                            "timestamp": "%(asctime)s",
                        }))
handler.setFormatter(formatter)
logger.addHandler(handler)


# FastAPI app
app = FastAPI(title="Iris Classifier API")

# Load model
model = joblib.load("feast_iris_model.joblib")

# Load label encoder
label_encoder = joblib.load("feast_iris_label_encoder.joblib") 

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    
# Simulated flags, normally these would be set by various parts of the code
# e.g. if model load is taking time due to weights being large, 
#  then is_ready would be False until the model is loaded.
app_state = {"is_ready": False, "is_alive": True}


# ENDPOINTS

# Monitoring and healthcheck endpoints
@app.on_event("startup")
async def startup_event():
    import time
    time.sleep(2)  # simulate work, normally this would be model loading
    app_state["is_ready"] = True
    
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )


# Main App endpoints
@app.get("/")
def root():
    return {"message": "Welcome to the Iris API. Whatever you post is classified!"}

@app.post("/predict/")
async def predict_iris_class(data: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        
        try:
            input_df = pd.DataFrame([data.dict()])
            prediction = model.predict(input_df)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            result = {"The iris species that you bringeth is": predicted_class}
            
            latency = round((time.time() - start_time) * 1000, 2)
            logger.info(json.dumps({
                                    "event": "prediction",
                                    "trace_id": trace_id,
                                    "input": data.dict(),
                                    "result": result,
                                    "latency_ms": latency,
                                    "status": "success"
                                }))
            return result
        
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")