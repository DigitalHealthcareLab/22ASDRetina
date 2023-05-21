from pathlib import Path
import shutil
import os
from datetime import datetime

class print_logger :
    def __init__(self, raw_log_path, log_name : str) : 
        log_path = Path(raw_log_path)
        log_path.mkdir(exist_ok=True, parents=True) 
        
        #exist_ok를 지정하면 폴더가 존재하지 않으면 생성하고, 존재하는 경우에는 아무것도 하지 않습니다.
        #parents: True인 경우 상위 path가 없는 경우 새로 생성함, Flase인 경우 상위 path가 없으면 FileNotFountError를 발생함

        self.log_path = log_path.joinpath(log_name).with_suffix('.txt')            
        self.check_exists()
        

    def __call__(self, log, end = '\n') : 
        now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        print(f'[{now}] {log}', end = end)
        with open(self.log_path, 'a') as f :
            f.write(f'[{now}] {log}{end}')

    def check_exists(self) : 
        try :  
            if self.log_path.exists() : 
                shutil.rmtree(self.log_path)
        except : 
                os.remove(self.log_path)    