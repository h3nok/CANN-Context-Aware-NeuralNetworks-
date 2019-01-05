
def JE(object):
    def __init__(self):
        self._patches = []
        self._ordering = 0

    def sort(self,patches,ordering):
        pass

def je(patches, order=0):
    print ("Entering joint entropy function...")
    if not isinstance(patches,dict):
        raise ValueError("Patches must contain patch_data and je rank!")
    if len(patches) < 4:
        raise ValueError("You must supply patch container with at least 4 patches")
        
