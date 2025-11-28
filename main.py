import sys
import os
# Add the project root directory to the Python path to resolve import errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import base64
import json
import shutil
import tempfile
from typing import Any, Dict, List

# Configure Matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from PIL import Image
from ultralytics import YOLO

# Internal imports for database and schemas
from app import crud, models, schemas
from app.database import engine, get_db

class AnalysisResponse(schemas.BaseModel):
    answer: str
    annotated_image_url: str
    patient: schemas.Patient

# --- Configuration & Initialization ---

# Define directories
GENERATED_DIR = "generated"
STATIC_DIR = "static"
os.makedirs(GENERATED_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="MedhaVision Medical Imaging Analysis")

# --- Static File Serving ---
app.mount(f"/{GENERATED_DIR}", StaticFiles(directory=GENERATED_DIR), name="generated")
app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name="static")

# --- Model Loading ---
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(script_dir, '..', '..', 'weights', 'xray.pt')
    fracture_model = YOLO("./weights/xray.pt")
    tumor_model = YOLO("./weights/xray.pt")
    # Corrected path for the tumor model
    # tumor_model = YOLO("https://github.com/roboflow-ai/brain-tumor-detection-yolov8/releases/download/v1/braintumor_best.pt")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- LangChain Agent Setup ---
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
tools = [] 

@tool
def run_xray_fracture(image_path: str) -> Dict[str, Any]:
    """Detects BONE FRACTURES on X-RAY images."""
    print("Running Fracture Model")
    results = fracture_model.predict(source=image_path, conf=0.25, save=False, imgsz=640)
    r = results[0]
    return {
        "workflow": "xray_fracture", "image_path": image_path,
        "boxes_xywhn": r.boxes.xywhn.tolist() if r.boxes else [],
        "scores": r.boxes.conf.tolist() if r.boxes else [],
    }

@tool
def run_brain_ct_tumor(image_path: str) -> Dict[str, Any]:
    """Detects TUMORS / LESIONS on BRAIN CT SCAN images."""
    print("Running Tumor Model")
    results = tumor_model.predict(source=image_path, conf=0.25, save=False, imgsz=640)
    r = results[0]
    return {
        "workflow": "ct_tumor", "image_path": image_path,
        "boxes_xywhn": r.boxes.xywhn.tolist() if r.boxes else [],
        "scores": r.boxes.conf.tolist() if r.boxes else [],
    }

@tool
def contextual_q_and_a(findings: str, user_prompt: str) -> str:
    """
    Answers user questions about medical imaging analysis results.

    Args:
        findings: A JSON string detailing the analysis findings (workflow, boxes, scores).
        user_prompt: The user's question regarding the findings.

    Returns:
        A natural language response based on the provided findings and question.
        This function is designed to be descriptive and avoid making a medical diagnosis.
    """
    system_prompt = (
        "You are a helpful medical imaging assistant. Your role is to interpret the "
        "results of a detection model and answer the user's question based on those "
        "results. DO NOT provide a medical diagnosis.\n\n"
        "Here are the findings from the analysis:\n{findings}\n\n"
        "Based on these findings, provide a clear and concise answer to the user's question."
    )
    
    formatted_prompt = system_prompt.format(findings=findings)
    
    # Using the existing llm instance for a direct call
    response = llm.invoke([
        ("system", formatted_prompt),
        ("user", user_prompt)
    ])
    return response.content

tools.extend([run_xray_fracture, run_brain_ct_tumor, contextual_q_and_a])
scan_agent = create_agent(model=llm, tools=tools)

# --- Core Analysis Logic (Synchronous) ---
# Note: these are blocking calls and should be run in FastAPI's thread pool,
# which is done automatically for `def` endpoints.

def analyze_scan(image_path: str):
    """Invokes the LangChain agent to analyze a single image."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    user_content = [
        {"type": "text", "text": (
            "You are a triage assistant for medical imaging. Look at the image and "
            "decide if it is an X-ray for fracture detection or a brain CT for tumor detection. "
            "Call EXACTLY ONE tool: 'run_xray_fracture' for X-RAYs or 'run_brain_ct_tumor' for BRAIN CTs. "
            f"Use this exact path: {image_path}"
        )},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    state = {"messages": [("user", user_content)]}
    # .invoke() is a blocking call
    return scan_agent.invoke(state)

def extract_tool_result(result):
    """Extracts the JSON tool result from the agent's messages."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage):
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return None
    return None

def save_analysis_result(tool_result: Dict[str, Any]) -> str | None:
    """Overlays bounding boxes and saves the annotated image."""
    if not tool_result or not tool_result.get("image_path"): return None
    image_path = tool_result["image_path"]
    boxes = tool_result.get("boxes_xywhn", [])
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return None

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    img_width, img_height = img.size
    for box in boxes:
        x_center, y_center, w, h = box
        box_width, box_height = w * img_width, h * img_height
        x_top_left = (x_center * img_width) - (box_width / 2)
        y_top_left = (y_center * img_height) - (box_height / 2)
        rect = patches.Rectangle((x_top_left, y_top_left), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.title(f"Analysis: {tool_result.get('workflow', 'Unknown')}")
    
    output_filename = f"{os.path.basename(image_path)}_{tempfile._get_candidate_names().__next__()}.jpg"
    output_path = os.path.join(GENERATED_DIR, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return f"/{GENERATED_DIR}/{output_filename}"

# --- API Endpoints ---

@app.post("/api/analyze/", response_model=schemas.Patient)
def analyze_images_endpoint(
    patient_name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts patient name and images, analyzes them, and saves results to the database.
    This is a synchronous endpoint and will be run by FastAPI in a thread pool.
    """
    patient = crud.get_patient_by_name(db, name=patient_name)
    if not patient:
        patient = crud.create_patient(db, patient=schemas.PatientCreate(name=patient_name))
    
    patient_id = patient.id

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # These are blocking, CPU-bound/IO-bound operations
            agent_result = analyze_scan(temp_path)
            tool_result = extract_tool_result(agent_result)
            annotated_image_url = save_analysis_result(tool_result) if tool_result else None
            
            image_data = schemas.ImageCreate(
                original_filename=file.filename,
                annotated_image_url=annotated_image_url,
                analysis_workflow=tool_result.get("workflow", "error") if tool_result else "error",
                scores=tool_result.get("scores", []) if tool_result else [],
                boxes_xywhn=tool_result.get("boxes_xywhn", []) if tool_result else [],
            )
            crud.create_patient_image(db, image=image_data, patient_id=patient_id)

    # Fetch the complete patient object with all relationships loaded
    updated_patient = crud.get_patient(db, patient_id=patient_id)
    return updated_patient

@app.get("/api/patients/", response_model=List[schemas.Patient])
def list_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Returns a list of all patients."""
    patients = crud.get_patients(db, skip=skip, limit=limit)
    return patients

@app.get("/api/patients/{patient_id}", response_model=schemas.Patient)
def get_patient_details(patient_id: int, db: Session = Depends(get_db)):
    """Returns details and all images for a single patient."""
    patient = crud.get_patient(db, patient_id=patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# --- Frontend Serving ---

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """
    Serves the frontend SPA.
    This catch-all route returns the index.html for any path not matching an API route or a static file,
    enabling client-side routing.
    """
    static_file_path = os.path.join(STATIC_DIR, full_path)
    if full_path and os.path.isfile(static_file_path):
        return FileResponse(static_file_path)
    
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
