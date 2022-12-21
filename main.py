import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from application.init_routers import init_routers

app = FastAPI(title='keras_model_predict')

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_routers(app)

if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)




