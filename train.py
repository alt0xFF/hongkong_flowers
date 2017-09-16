# custom modules
from options import Options

def train():
    # setting up, options contains all our params
    options = Options()
    
    # set training mode
    options.mode = 0
    
    # choose model from library
    if options.library == 'pytorch':
        from core.pytorch.model import Model
        model = Model(options)
    elif options.library == 'keras':
        from core.keras.model import Model
        model = Model(options)
    
    # fit model
    model.fit(options)
        
if __name__=='__main__':
    train()