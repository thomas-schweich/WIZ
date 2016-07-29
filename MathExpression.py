import operator
import collections
import numpy as np
import math
import re
import functools
from threading import Thread


def forceReversible(func):
    """Tries reversing the arguments of the two argument function func if the original raises a TypeError"""
    @functools.wraps(func)
    def wrapper(arg0, arg1):
        print "Arg 0: %s\nArg 1: %s" % (str(arg0), str(arg1))
        try:
            return func(arg0, arg1)
        except TypeError:
            return func(arg1, arg0)
    return wrapper


class MathExpression:
    """A parser for mathematical expressions. Does not depend on any other modules in dataManipulation.

    Allows the specification of a list of operators which are used to break down a string, and which are assigned to a
    function which takes exactly two arguments through an OrderedDict. Interpretation is done on *every* part of the
    expression, thus making it impossible to execute arbitrary code. In order to maintain "safety," the input must
    be a single string. Parsing is completed in groups of 3 in the format token + operator + token. This is done
    iteratively. Expressions are recursively broken into sub-expressions through
    the use of matching parenthesis, and evaluated from the inside out.
    Function calls are evaluated in the format func(arg0, arg1...). Kwargs are not supported. A
    backup function in the format  backup(func, *args) can be specified to handle arguments which are passed to a
    function but are of improper type; for instance, using only the first index of an array as arguments
    for certain functions, etc.
    """

    __author__ = "Thomas Schweich"

    @staticmethod
    def returnDict(key, value):
        return {key: value}

    operators = collections.OrderedDict(
        (("(", None), (")", None), (",", None), ("^", forceReversible(operator.pow)), ("/", forceReversible(operator.div)),
         ("*", forceReversible(operator.mul)), ("+", forceReversible(operator.add)), ("-", forceReversible(operator.sub))))
    modules = (np, math)

    def __init__(self, expression, variables=None, operators=operators, modules=modules, fallbackFunc=None):
        self.variables = variables
        self.operators = operators
        self.modules = modules
        self.fallbackFunc = fallbackFunc
        self.expression = self.genFromString(expression)
        self.loops = 0
        self.result = None

    def genFromString(self, string):
        """Separates string by .operators using regex"""
        operators = self.operators.keys()
        operators.sort(key=lambda x: -len(x))
        exp = re.findall(r'<.*?>|' + "|".join(["%s" % re.escape(op) for op in operators]) + '|[\.\w]+', string)
        print exp
        return exp

    def getEvaluationThread(self):
        """Returns a Thread who's target is evaluate() which can be started and joined at your leisure"""
        return Thread(target=self.evaluate())

    def evaluate(self):
        """Calls evaluateExpression() on .expression"""
        self.expression = self.evaluateExpression(self.expression)

    def evaluateExpression(self, exp):
        """Recursively evaluates expressions starting with innermost parenthesis, working outwards

        Iteratively solves sub-expressions (grouped by parenthesis) in the order of .operators
        Note: Order of operations is strict. Equal precedence not yet allowed.
        """
        # TODO Equal operator precedence
        if len(exp) >= 3:
            print "-New Starting expression: %s" % str(exp)
            rightInner = exp.index(")") if ")" in exp else len(exp)
            print "Right inner parenthesis index: %d" % rightInner
            leftSide = exp[:rightInner]
            leftInner = len(leftSide) - leftSide[::-1].index("(") if "(" in leftSide else 0
            print "Left inner parenthesis index: %d" % leftInner
            subExp = leftSide[leftInner:]
            print "Sub Expression: " + str(subExp)
            callerIndex = leftInner - 2
            if callerIndex > -1 and exp[callerIndex] not in self.operators:
                print "Calling function...."
                # Call function if in format something(arg0, arg1...) if "something" is not an operator
                args = []
                while "," in subExp:
                    args.append(self.evaluateExpression(list(subExp[:subExp.index(",")])))
                    del subExp[:subExp.index(",") + 1]
                args.append(self.evaluateExpression(subExp))
                print "Arguments: " + str(args)
                funcToCall = self._interpret(exp[callerIndex])
                try:
                    result = funcToCall(*args)
                except:
                    try:
                        result = self.fallbackFunc(funcToCall, *args)
                    except Exception as e:
                        raise MathExpression.ParseFailure(str(funcToCall), e)
                print "Result: " + str(result)
                del exp[callerIndex:rightInner + 1]
                exp.insert(callerIndex, result)
                print "Expression after replacement" + str(exp)
                print "....call complete"
            else:
                print "Evaluating expression...."
                # Otherwise, evaluate the expression within the parenthesis, replacing the range with the result
                newExp = subExp[:]
                for op in self.operators:
                    for index, part in enumerate(subExp):
                        self.loops += 1
                        if part == op:  # Changing to allow for equal level operators
                            newLocation = newExp.index(part)
                            prevIndex = newLocation - 1
                            nextIndex = newLocation + 1
                            try:
                                prev = self._interpret(newExp[prevIndex])
                                nxt = self._interpret(newExp[nextIndex])
                            except IndexError as i:
                                raise MathExpression.ParseFailure(part, i)
                            print "Combining %s with %s using '%s' operator" % (str(prev), str(nxt), str(part))
                            if not (isinstance(prev, np.ndarray) and not isinstance(nxt, np.ndarray)):
                                solution = self.operators[part](prev, nxt)
                            else:
                                raise MathExpression.SyntaxError(prev)
                            print "Solution: " + str(solution)
                            del newExp[prevIndex:nextIndex + 1]
                            newExp.insert(prevIndex, solution)
                            print "After replacement with solution: " + str(newExp)
                try:
                    hasParens = exp[leftInner - 1] == "(" and exp[rightInner] == ")"
                except IndexError:
                    raise MathExpression.SyntaxError(exp)
                if len(newExp) == 1:
                    if hasParens:
                        print "Replacing parenthesis and expression"
                        del exp[leftInner - 1:rightInner + 1]
                    else:
                        print "Replacing expression only (parenthesis not found)"
                        del exp[leftInner:rightInner]
                    exp.insert(leftInner-1, newExp[0])
                else:
                    raise MathExpression.SyntaxError(newExp)
                print "New Expression: %s" % str(exp)
                print "....evaluate complete"
            print "Length of expression: %d" % len(exp)
            return self.evaluateExpression(exp)
        else:
            if len(exp) == 1:
                print "Loops: %d" % self.loops
                self.loops = 0
                return self._interpret(exp[0])
            else:
                raise MathExpression.SyntaxError(exp)

    def _interpret(self, string):
        if isinstance(string, str):
            if string[0] == "<" and string[-1] == ">":
                varString = string[1:-1]
                try:
                    print "Trying interpret %s as variable" % varString
                    return self.variables[varString]
                except KeyError as k:
                    raise MathExpression.ParseFailure(string, k)
            else:
                try:
                    print "Trying interpret %s as float" % string
                    return float(string)
                except ValueError:
                    pass
                for module in self.modules:
                    try:
                        print "Trying interpret %s as %s" % (string, str(module))
                        return getattr(module, string)
                    except AttributeError:
                        pass
                raise MathExpression.SyntaxError(string)
                # return string
        else:
            return string

    '''
    def operate(self, operator_, *args):
        # TODO Dict kwargs
        kwargs = [self._interpret(arg) for arg in args if isinstance(arg, dict)]
        args = [self._interpret(arg) for arg in args if not isinstance(arg, dict)]
        return operator_(*args, **kwargs)
    '''

    class ParseFailure(Exception):
        """Represents the expression group (i.e. token + operator + token) and the given exception"""
        def __init__(self, badPart, exception):
            self.badPart = badPart
            self.exception = exception

        def __repr__(self):
            custom = ""
            if self.exception is AttributeError:
                custom += "'%s' not found in namespace. \n" % str(self.badPart)
            if self.badPart in MathExpression.operators:
                custom += "It is likely that the operator was missing an argument. "
            return "%s threw error: %s. %s" % (str(self.badPart), str(self.exception), custom)

        def __str__(self):
            return str(self.__repr__())

    class SyntaxError(Exception):
        """Represents only the expression group (i.e. token + operator + token)"""
        def __init__(self, badPart):
            self.badPart = badPart

        def __repr__(self):
            custom = ""
            if "(" in self.badPart or ")" in self.badPart:
                custom += "It is likely that you are missing a parenthesis."
            return "Syntax error on sub expression: %s. \n%s" % (str(self.badPart), custom)

        def __str__(self):
            return str(self.__repr__())


def limit(max_=None):
    """Return decorator that limits allowed returned values."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            try:
                mag = abs(ret)
            except TypeError:
                pass  # not applicable
            else:
                if mag > max_:
                    raise ValueError(ret)
            return ret
        return wrapper
    return decorator