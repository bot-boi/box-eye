from vectormath import Vector2


class Point(Vector2):
    def __new__(cls, x=None, y=None):
        # i dont really understand this
        cls = super().__new__(cls, x=x, y=y)
        return cls.astype('uint32')

    def __str__(self):
        return f'{self.x},{self.y}'
