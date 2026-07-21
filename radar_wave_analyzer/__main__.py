"""支持 python -m radar_wave_analyzer 启动。"""
from .app import app

if __name__ == '__main__':
    app.run_server(threaded=True)
