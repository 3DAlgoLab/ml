class Obj:
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        print("reset called")


o = Obj()
