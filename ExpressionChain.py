from MathExpression import MathExpression
from itertools import chain


class ExpressionChain:
    """ An iterable which stores MathExpressions and allows the use of previous expressions as variables when iterated
    """
    __author__ = "Thomas Schweich"

    def __init__(self, variables=None, operators=None, modules=None,
                 fallbackFunc=None):
        """Takes an initial argument for 'variables', which is then appended to as expressions are evaluated

        The remaining options stay constant, and are simply used to customize the MathExpression
        """
        self.formulae = []
        self.iterator = None
        self.variables = {} if not variables else variables
        self.operators = operators
        self.modules = modules
        self.fallbackFunc = fallbackFunc
        self.finalized = False

    def addVariable(self, name, value):
        self.variables.update({name:value})

    def addExp(self, expression, name=""):
        """Adds a string to be evaluated using MathExpression and a name by which it can be referred to in the future

        Names should be added without angle brackets, and, per MathExpression's usage, used with angle brackets.
        If the expression will not be referred to in the future (for instance, the last expression), one can omit the
        name argument; it will be stored in the resulting MathExpressions' dictionaries under the empty string
        """
        self.formulae.append((expression, name))

    def __iter__(self):
        """Creates an iterator from the given expressions in order, returns self ready for use in a loop"""
        self.iterator = chain(*self.formulae)
        return self

    def next(self):
        """Generates the next result from .formulae, allowing use of previously defined variables"""
        formula = next(self.iterator)
        print "Formula: %s" % formula
        name = next(self.iterator)
        print "Name: %s" % name
        exp = MathExpression(formula, variables=self.variables, operators=self.operators, modules=self.modules,
                             fallbackFunc=self.fallbackFunc)
        exp.evaluate()
        self.variables.update({name: exp.expression})
        return exp.expression, name

    def __len__(self):
        return len(self.formulae)

    @staticmethod
    def testCreate():
        chain = ExpressionChain()
        chain.addExp("10 + 5", "Bob")
        chain.addExp("<Bob> + 5", "Joe")
        chain.addExp("<Joe> + 5")
        for i in chain:
            print "Result of chain procedure: %s" % i
