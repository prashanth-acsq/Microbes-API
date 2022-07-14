from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from static.utils import feature_distribution, train_model, get_logs, get_specific_logs, infer

VERSION: str = "0.0.1"

STATIC_PATH: str = "static"

origins = [
    "http://localhost:4005",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    solidity: float
    eccentricity: float
    equiv_diameter: float
    extrema: float
    filled_area: float
    extent: float
    orientation: float
    euler_number: float
    major_axis_length: float
    minor_axis_length: float
    perimeter: float
    convex_area: float
    area: float
    raddi: float
    


@app.get("/")
async def root():
    return JSONResponse({
        "statusText" : "Root Endpoint for Microbes API",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/version")
async def version():
    return JSONResponse({
        "statusCode" : 200,
        "statusText" : "Microbes API Version Fetch Successful",
        "version" : VERSION,
    })


@app.get("/distribution/{feature_name}")
async def get_feature_distribution(feature_name: str):
    imageData = feature_distribution(feature_name)
    return JSONResponse({
        "statusText" : "Distribution Fetch Successfulr",
        "statusCode" : 200,
        "imageData" : imageData,
        "message" : f"Distribution of Feature '{feature_name}'"
    })


@app.get("/train")
async def train():
    acc_model_fold_name = train_model()
    return JSONResponse({
        "statusText" : "Training Complete",
        "statusCode" : 200,
        "best_acc_model": f"{acc_model_fold_name.split('_')[0]}, {acc_model_fold_name.split('_')[1]}",
    })


@app.get("/train/logs")
async def train_logs_specific_model():
    logs = get_logs()
    if logs is not None:
        return JSONResponse({
            "statusText" : "Log Fetch Complete",
            "statusCode" : 200,
            "logs" : logs,
        })
    else:
        return JSONResponse({
            "statusText" : "No LogFile Found",
            "statusCode" : 404,
        })


@app.get("/train/logs/{model_name}/{fold}")
async def train_logs_specific_model(model_name: str, fold: str):
    logs = get_specific_logs(model_name, int(fold))
    if logs is not None:
        return JSONResponse({
            "statusText" : "Log Fetch Complete",
            "statusCode" : 200,
            "logs" : logs,
        })
    else:
        return JSONResponse({
            "statusText" : "No Log Found",
            "statusCode" : 404,
        })
    

@app.get("/infer")
async def get_infer():
    return JSONResponse({
        "statusText" : "Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.post("/infer")
async def post_infer(data: Data):
    y_pred, y_pred_proba = infer([   
        data.solidity,
        data.eccentricity,
        data.equiv_diameter,
        data.extrema,
        data.filled_area,
        data.extent,
        data.orientation,
        data.euler_number,
        data.major_axis_length,
        data.minor_axis_length,
        data.perimeter,
        data.convex_area,
        data.area,
        data.raddi,
    ])

    if y_pred is not None and y_pred_proba is not None:
        return JSONResponse({
            "statusText": "Inference Complete", 
            "statusCode": 200, 
            "prediction": str(y_pred), 
            "probability": str(y_pred_proba[0, 1]),
        })
    else:
        return JSONResponse({
            "statusText" : "Error in performing inference",
            "statusCode" : 404,
        })