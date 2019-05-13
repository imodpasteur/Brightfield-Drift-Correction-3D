from unittest import TestCase
from bfdc.template import generate_circle_template, generate_moving_circle_2D

import matplotlib.pyplot as plt


class TestTemplate(TestCase):
    
    def test_circle(self):
        px = 20
        size = 500
        expected_shape = (size/px, size/px) 
        circle = generate_circle_template(px=px, size=size)
        for a, b in zip(expected_shape, circle.shape):
            self.assertEqual(a, b)
        if True:
            plt.imshow(circle)
            plt.show()

    def test_moving_circle(self):
        movie = generate_moving_circle_2D()
        self.assertEqual(len(movie), 10)