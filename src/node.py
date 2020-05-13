import math

class Node:
    '''Node interface'''

    def __init__(self):
        self.children = []
        self.parents = []
        self.value = None
        self.grad = None
    
    def add_child(self, *children):
        '''add 1 or more children to this node
        updates children's parents as well
        '''
        for child in children:
            self.children.append(child)
            child.parents.append(self)
        
    def get_children(self):
        '''return the children of this node
        (the nodes directly used to evaluate this node in a forward pass)
        '''
        return self.children

    def get_parents(self):
        '''return the parents of this node
        (the nodes that directly use this node in a forward pass)
        '''
        return self.parents

    def forward(self, feed_dict):
        '''evaluate the whole computation graph up to this node given
        a feed_dict, which maps input names to their values.
        If a needed input name is not provided, an error will be raised
        EFFECT: modifies self.value and values of children
        '''
        fwd = lambda node : node.forward(feed_dict)
        child_values = list(map(fwd, self.get_children()))
        self.value = self.evaluate(child_values)
        return self.value


    def backward(self):
        '''back-propagate this node's gradient into its children
        node must have a value (this will happen if you do a forward pass through this node)
        should make this so you can't call the public .backward() on the same node twice like pytorch
        '''
        if self.value is None:
            raise RuntimeError("Must run forward pass before back propagation (attempted to back-prop node with no value)")

        if self.grad is None:
            # this node is the first one in the backprop
            self.grad = 1
        # now grad is not None
        children = self.get_children()
        for child, dself_dchild in zip(children, self.derivatives([child.value for child in children])):
            if child.grad is None:
                child.grad = 0
            child.grad += self.grad * dself_dchild
        for child in children:
            child.backward()
        return self.grad

    def clear_grad(self):
        '''erase stored gradient
        '''
        self.grad = None
    
    def clear_grad_recursive(self):
        self.clear_grad()
        for child in self.get_children():
            child.clear_grad_recursive()
    
    def __add__(self, other):
        return AddNode(self, other)
    
    def __sub(self, other):
        return SubtractNode(self, other)
    
    def __mul__(self, other):
        return MultiplyNode(self, other)
    
    def __truediv__(self, other):
        return DivideNode(self, other)
    
    def __pow__(self, other):
        return ExponentiateNode(self, other)

    ################## abstract methods that must be implemented ##################

    def evaluate(self, child_values):
        '''evaluate this node given the values of the child nodes
        '''
        raise NotImplementedError

    def derivatives(self, child_values):
        '''partial self / partial child for each child
        returns a list of numbers in the same order as the child_values they were calculated from
        '''
        raise NotImplementedError


class InputNode(Node):
    '''Input placeholder node
    Has no children, requires feed_dict to evaluate'''

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
    
    def derivatives(self, child_values):
        # this should never be called
        return [0]
    
    def forward(self, feed_dict):
        if self.name not in feed_dict:
            raise KeyError("Value for input name " + str(self.name) + " missing from feed dict.")
        else:
            self.value = feed_dict[self.name]
            return self.value


class ConstantNode(Node):
    '''Constant value node
    Has no children'''

    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value
    
    def evaluate(self, child_values):
        return self.value
    
    def derivatives(self, child_values):
        # this should never be called
        return [0]


def SumNode(Node):
    '''Addition of two numbers. Special case to be abstracted later
    the code actually works for any amount of numbers, but the constructor forces two numbers
    '''

    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_child(*children)
    
    def evaluate(self, child_values):
        return sum(child_values)
    
    def derivatives(self, child_values):
        return [1 for _ in child_values]


class BinopNode(Node):
    '''Abstract binary operation node which takes in two child nodes, an evaluation function, and
    a differentiation function
    '''

    def __init__(self, left, right, evaluate, derivatives, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_child(left, right)
        self.evaluate_ = evaluate # might be weird setting a method to a function value
        self.derivatives_ = derivatives
    
    def evaluate(self, child_values):
        x, y = child_values
        return self.evaluate_(x, y)
    
    def derivatives(self, child_values):
        x, y = child_values
        return self.derivatives_(x, y)



class AddNode(BinopNode):
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(left, right, lambda x, y: x + y, lambda x, y: [1, 1])


class SubtractNode(BinopNode):
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(left, right, lambda x, y: x - y, lambda x, y: [1, -1])


class MultiplyNode(BinopNode):
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(left, right, lambda x, y: x * y, lambda x, y: [y, x])


class DivideNode(BinopNode):
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(left, right, lambda x, y: x / y, lambda x, y: [1 / y, -x / (y**2)]) # quosh


class ExponentiateNode(BinopNode):
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(left, right, lambda x, y: x ** y, lambda x, y: [y * x ** (y-1), x**y * math.log(x)])
