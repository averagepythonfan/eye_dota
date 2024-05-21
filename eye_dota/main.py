from fastapi import FastAPI
import uvicorn
from routers import main


app = FastAPI()

@app.get("/healthcheck")
async def healthcheck():
    return {"message": "server is up"}


app.include_router(main)


if __name__ == "__main__":
    uvicorn.run(
        'main:app',
        host="0.0.0.0",
        port=9090,
        log_level="info",
        reload=True
    )