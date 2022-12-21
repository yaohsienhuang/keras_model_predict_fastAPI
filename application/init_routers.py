from .routers import get_gpu_memory_status, get_model_info, predict, predict_move_image

def init_routers(app):
    app.include_router(get_gpu_memory_status.router)
    app.include_router(get_model_info.router)
    app.include_router(predict.router)
    app.include_router(predict_move_image.router)

