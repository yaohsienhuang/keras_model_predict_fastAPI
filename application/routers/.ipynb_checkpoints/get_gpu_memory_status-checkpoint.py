from fastapi import APIRouter
import subprocess as sp
from starlette.requests import Request
from ..utility.setLogging import logger

router = APIRouter()

@router.get('/gpu_memory_status', tags=['Common'])
async def get_gpu_memory_status(request: Request):
    '''
    主要功能：確認 GPU free memory free 與 used 狀態
    '''
    client_host_ip = request.client.host
    logger.info(f'[get_gpu_memory_status] request_ip={client_host_ip} ')
    
    memory_status=dict()
    command_fmt = "nvidia-smi --query-gpu=memory.%s --format=csv"
    try:
        for status in ['free','used']:
            command=command_fmt%status
            memory_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            for i, x in enumerate(memory_info):
                if f'device:{i}' in memory_status.keys():
                    memory_status[f'device:{i}'].append(f'{status}:{str(x.split()[0])}')
                else:
                    memory_status[f'device:{i}']= [f'{status}:{str(x.split()[0])}']

        return {
            'status': 'success' if len(memory_status) else 'error',
            'message': memory_status
        }
    
    except Exception as e:
        logger.warning(f'[get_gpu_memory_status] {str(e)} ')
        