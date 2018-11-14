def reshapeBatch(generator):
    '''This function reshape the output of a generator (x, y) as ([x, y], [y, x])
    for capsule training.
    '''
    for batch in generator:
        x, y = batch
        yield ([x, y], [y, x])
