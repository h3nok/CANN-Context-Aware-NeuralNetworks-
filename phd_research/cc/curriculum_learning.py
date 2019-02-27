import map_measure


class Curriculum(object):
    def __init__(self):
        self.syllabus = Syllabus()
        self.batch = None
    def Propose(self, batch):
        self.batch = batch

    def _Assess(self):
        pass
class Syllabus(object):
    pass