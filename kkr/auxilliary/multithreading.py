from queue import Queue
from threading import Thread
from tqdm import tqdm
import multiprocessing as mp
import traceback 

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""

    def __init__(self, tasks, device, log:callable=None):
        Thread.__init__(self)
        self.device = device
        self.tasks = tasks
        self.daemon = True
        self.log = log
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, device=self.device, **kargs)
            except Exception as e:
                print(e)
                print(traceback.print_exc() )
            finally:
                self.tasks.task_done()
                if self.log:
                    self.log(1)


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""

    def __init__(self, devices, pbar):
        
        self.tasks = Queue(len(devices))
        for device in devices:
            Worker(self.tasks, device, pbar)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


                
def tmap(function, devices, log=False):
    def f(data):            
        pool = ThreadPool(devices, log)
        result = {}
        for i, args in enumerate(data):
            pool.add_task(function, *args, result=result, run=i)
        pool.wait_completion()
   
        return result
    return f
