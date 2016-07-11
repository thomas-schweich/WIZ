import operator
import collections
import numpy as np
import math
import re


class MathExpression:
    __author__ = "Thomas Schweich"

    @staticmethod
    def returnDict(key, value):
        return {key: value}

    operators = collections.OrderedDict(
        (("(", None), (")", None), (",", None), ("^", operator.pow), ("/", operator.div),
         ("*", operator.mul), ("+", operator.add), ("-", operator.sub)))
    modules = (np, math)

    def __init__(self, expression, variables=None, operators=operators, modules=modules):
        self.variables = variables
        self.operators = operators
        self.modules = modules
        self.expression = self.genFromString(expression)
        self.loops = 0

    def genFromString(self, string):
        operators = self.operators.keys()
        operators.sort(key=lambda x: -len(x))
        exp = re.findall(r'<.*?>|' + "|".join(["%s" % re.escape(op) for op in operators]) + '|\w+', string)
        print exp
        return exp

    def evaluate(self):
        ev = self.evaluateExpression(self.expression)
        print ev
        return ev

    def evaluateExpression(self, exp):
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
                    args.append(self.evaluateExpression(subExp[:subExp.index(",")]))
                    del subExp[:subExp.index(",") + 1]
                args.append(self.evaluateExpression(subExp))
                print "Arguments: " + str(args)
                result = self._interpret(exp[callerIndex])(*args)
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
                        if part == op:
                            newLocation = newExp.index(part)
                            # ^ Check - Should be ok though since previous operators are removed
                            prevIndex = newLocation - 1
                            nextIndex = newLocation + 1
                            prev = self._interpret(newExp[prevIndex])
                            nxt = self._interpret(newExp[nextIndex])
                            print "Combining %s with %s using '%s' operator" % (str(prev), str(nxt), str(part))
                            solution = self.operators[part](prev, nxt)
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
                    exp.insert(leftInner-1, newExp[0]) # -1s
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
                raise MathExpression.ParseFailure(string, AttributeError)
                # return string
            # TODO Fallback function (pass function as first arg...)
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
        def __init__(self, badPart, exception):
            self.badPart = badPart
            self.exception = exception

        def __repr__(self):
            custom = ""
            if self.exception is AttributeError:
                custom += "'%s' not found in namespace. " % str(self.badPart)
            if self.badPart in MathExpression.operators:
                custom += "It is likely that you missed a parenthesis. "
            return "%s threw error: %s. %s" % (str(self.badPart), str(self.exception), custom)

        def __str__(self):
            return self.__repr__()

    class SyntaxError(Exception):
        def __init__(self, badPart):
            self.badPart = badPart

        def __repr__(self):
            return "Syntax error on token %s" % str(self.badPart)

        def __str__(self):
            return self.__repr__()