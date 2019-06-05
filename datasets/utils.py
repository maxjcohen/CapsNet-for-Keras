import threading

def reshapeBatch(generator):
    '''This function reshape the output of a generator (x, y) as ([x, y], [y, x])
    for capsule training.
    '''
    for batch in generator:
        x, y = batch
        yield ([x, y], [y, x])


class ReshapeBatch():
    def __init__(self, iterator):
        self.it = iterator
        
        self.n = self.it.n
        self.batch_size = self.it.batch_size
        
        self.lock = threading.Lock()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            x, y = next(self.it)
            return ([x, y], [y, x])