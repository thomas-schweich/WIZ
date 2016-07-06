import operator
import collections
import numpy as np


class MathExpression:
    operators = collections.OrderedDict(
        (("(", None), (")", None), (",", None), ("^", operator.pow), ("/", operator.div), ("*", operator.mul),
         ("+", operator.add), ("-", operator.sub)))

    @staticmethod
    def genFromString(string):
        string = "(" + string + ")"
        string = "".join(string.split())
        expressionList = []
        lastOpIndex = 0
        for index, char in enumerate(string):
            if char in MathExpression.operators:
                if index != lastOpIndex: expressionList.append(string[lastOpIndex:index])
                expressionList.append(char)
                lastOpIndex = index + 1
        print expressionList
        return MathExpression(expressionList)

    def __init__(self, expression, variables=None):
        self.expression = expression
        self.variables = variables

    def evaluate(self):
        ev = self.evaluateExpression(self.expression)
        print ev
        return ev

    def evaluateExpression(self, exp):
        if len(exp) >= 3:
            print "Starting expression: " + str(exp)
            rightInner = exp.index(")") if ")" in exp else len(exp) - 1
            print "Right inner: %d" % rightInner
            leftSide = exp[:rightInner]
            leftInner = len(leftSide) - 1 - leftSide[::-1].index("(") if "(" in leftSide else 0
            print "Left inner: %d" % leftInner
            subExp = leftSide[leftInner + 1:]
            print "Sub Expression: " + str(subExp)
            callerIndex = leftInner - 1
            if exp[callerIndex] not in MathExpression.operators:
                print "Trying call"
                # Call function if in format something(arg0, arg1...) if "something" is not an operator
                args = []
                while "," in subExp:
                    args.append(self.evaluateExpression(subExp[:subExp.index(",")]))
                    del subExp[:subExp.index(",") + 1]
                args.append(self.evaluateExpression(subExp))
                print "Arguments: " + str(args)
                result = self._replace(exp[callerIndex])(*args)
                print "Result: " + str(result)
                print "Expression before deletion" + str(exp)
                del exp[callerIndex:rightInner]
                print "Expression after deletion" + str(exp)
                exp.insert(callerIndex, result)
                print "Expression after insertion" + str(exp)
            else:
                print "Trying evaluate"
                # Otherwise, evaluate the expression within the parenthesis, replacing the range with the result
                for op in MathExpression.operators:
                    for index, part in enumerate(subExp):
                        if part == op:
                            prevIndex = index - 1
                            nextIndex = index + 1
                            prev = self._replace(subExp[prevIndex])
                            nxt = self._replace(subExp[nextIndex])
                            print "Combining " + str(prev) + " with " + str(nxt)
                            solution = MathExpression.operators[part](prev, nxt)
                            print "Solution: " + str(solution)
                            del subExp[prevIndex:nextIndex]
                            subExp[prevIndex] = solution
                            print "New Sub Expression: " + str(subExp)
                del exp[leftInner:rightInner + 1]
                exp.insert(leftInner, subExp[0])
            return self.evaluateExpression(exp)
        else:
            return self._replace(exp[0])

    def _replace(self, string):
        if isinstance(string, str):
            try:
                return float(string)
            except ValueError:
                pass
            if string[0] == "<" and string[-1] == ">":
                varString = string[1:len(string) - 1]
                try:
                    return self.variables[varString]
                except KeyError as k:
                    raise MathExpression.ParseFailure(string, k)
            else:
                try:
                    return getattr(np, string)
                except AttributeError as a:
                    raise MathExpression.ParseFailure(string, a)
        else:
            return string

    class ParseFailure(Exception):
        def __init__(self, badPart, exception):
            self.badPart = badPart
            self.exception = exception

        def __repr__(self):
            return str(self.badPart) + " threw error: " + str(self.exception)

        def __str__(self):
            return self.__repr__()
