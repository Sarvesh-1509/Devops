from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os

# Create FastAPI app
app = FastAPI(
    title="Boston Housing Pricing API",
    description="Predict Boston housing prices using a trained ML model",
    version="1.0"
)

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# The 13 feature columns required by the model
FEATURE_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the bare-bones HTML form."""
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"features": FEATURE_NAMES, "prediction": None}
    )


@app.post("/", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    CRIM: float = Form(0.0),
    ZN: float = Form(0.0),
    INDUS: float = Form(0.0),
    CHAS: float = Form(0.0),
    NOX: float = Form(0.0),
    RM: float = Form(0.0),
    AGE: float = Form(0.0),
    DIS: float = Form(0.0),
    RAD: float = Form(0.0),
    TAX: float = Form(0.0),
    PTRATIO: float = Form(0.0),
    B: float = Form(0.0),
    LSTAT: float = Form(0.0)
):
    """Handle form submission and return HTML with prediction."""
    if model is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html", 
            context={
                "features": FEATURE_NAMES, 
                "prediction": "Error: Model not loaded."
            }
        )

    # Prepare input data as a DataFrame to match the format used during training
    input_data = pd.DataFrame([[
        CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
    ]], columns=FEATURE_NAMES)
    
    # Make prediction
    pred_value = model.predict(input_data)[0]
    
    # Return the template with the prediction
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={
            "features": FEATURE_NAMES, 
            "prediction": f"${pred_value * 1000:,.2f}" # Target is usually in $1000s
        }
    )
