import unittest
from node import *


class NodeTests(unittest.TestCase):
    def test_product(self):
        # x * ( x^2 + 1 )
        x = InputNode("x")
        two = ConstantNode(2)
        x2 = ExponentiateNode(x, two)
        one = ConstantNode(1)
        sum_ = AddNode(x2, one)
        product = MultiplyNode(x, sum_)
        product.forward(feed_dict={'x':3})
        self.assertEqual(30, product.value)
        self.assertEqual(3, x.value)
        self.assertEqual(2, two.value)
        self.assertEqual(1, one.value)
        self.assertEqual(10, sum_.value)
        self.assertEqual(9, x2.value)
        product.backward()
        self.assertEqual(28, x.grad)
        self.assertEqual(1, product.grad)
        self.assertEqual(3, sum_.grad)
        self.assertEqual(3, one.grad)
        self.assertEqual(3, x2.grad)
        product.clear_grad_recursive()
        self.assertIsNone(x.grad)
        self.assertIsNone(product.grad)
        self.assertEqual(3, x.value)
        self.assertEqual(30, product.value)
        product.backward()
        self.assertEqual(28, x.grad)
        self.assertEqual(1, product.grad)

        # now with operators
        x = InputNode("x")
        one = ConstantNode(1)
        two = ConstantNode(2)
        product = x * (x ** two + one)

        product.forward(feed_dict={"x":3})
        self.assertEqual(30, product.value)
        self.assertEqual(3, x.value)
        product.backward()
        self.assertEqual(28, x.grad)
        self.assertEqual(1, product.grad)


if __name__ == '__main__':
    unittest.main()