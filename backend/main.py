from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.routes import router as data_router

app = FastAPI(
    title="Hospital Data Analytics API",
    description="Backend API for serving hospital patient analytics data.",
    version="1.0.0"
)

# Configure CORS so Streamlit can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes definition
app.include_router(data_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
