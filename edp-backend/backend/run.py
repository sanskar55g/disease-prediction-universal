 PY
import uvicorn
 
if __name__ == "__main__":
    # Only watch source code. Watching the whole folder also picked up writes to
    # medical_history.db, which restarted the server on every consultation.
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        reload_includes=["*.py"],
    )
 