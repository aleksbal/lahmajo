from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests

app = FastAPI(debug=True)

class Query(BaseModel):
    prompt: str
    model: str = "llama3.2"


@app.post("/generate")
async def generate_text(query: Query):
    try:
        print(query.model)
        print(query.prompt)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={'model': query.model, 'stream': False, 'prompt': query.prompt}
        )
        response.raise_for_status()
        
        return {"generated_text": response.json()["response"]}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/tags")
async def generate_text(query: Query):
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")