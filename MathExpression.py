import operator
import collections
import numpy as np
import math
import exceptions


class MathExpression:
    operators = collections.OrderedDict(
        (("(", None), (")", None), (",", None), ("^", operator.pow), ("/", operator.div), ("*", operator.mul),
         ("+", operator.add), ("-", operator.sub)))
    modules = (np, math)

    def __init__(self, expression, variables=None, operators=operators, modules=modules):
        self.variables = variables
        self.operators = operators
        self.modules = modules
        self.expression = self.genFromString(expression)

    def genFromString(self, string):
        string = "(" + string + ")"
        string = "".join(string.split())
        expressionList = []
        lastOpIndex = 0
        for index, char in enumerate(string):
            if char in self.operators:
                if index != lastOpIndex: expressionList.append(string[lastOpIndex:index])
                expressionList.append(char)
                lastOpIndex = index + 1
        print expressionList
        return expressionList

    def evaluate(self):
        ev = self.evaluateExpression(self.expression)
        print ev
        return ev

    def evaluateExpression(self, exp):
        if len(exp) >= 3:
            print "Starting expression: " + str(exp)
            rightInner = exp.index(")") if ")" in exp else len(exp)
            print "Right inner: %d" % rightInner
            leftSide = exp[:rightInner]
            leftInner = len(leftSide) - leftSide[::-1].index("(") if "(" in leftSide else 0
            print "Left inner: %d" % leftInner
            subExp = leftSide[leftInner:]
            print "Sub Expression: " + str(subExp)
            callerIndex = leftInner - 2
            if callerIndex > -1 and exp[callerIndex] not in self.operators:
                print "Calling function..................................."
                # Call function if in format something(arg0, arg1...) if "something" is not an operator
                args = []
                while "," in subExp:
                    args.append(self.evaluateExpression(subExp[:subExp.index(",")]))
                    del subExp[:subExp.index(",") + 1]
                args.append(self.evaluateExpression(subExp))
                print "Arguments: " + str(args)
                result = self._interpret(exp[callerIndex])(*args)
                print "Result: " + str(result)
                print "Expression before deletion" + str(exp)
                del exp[callerIndex:rightInner + 1]
                print "Expression after deletion" + str(exp)
                exp.insert(callerIndex, result)
                print "Expression after insertion" + str(exp)
                print "....call complete"
            else:
                print "Evaluating expression.................................."
                # Otherwise, evaluate the expression within the parenthesis, replacing the range with the result
                newExp = subExp[:]
                for op in self.operators:
                    for index, part in enumerate(subExp):
                        if part == op:
                            newLocation = newExp.index(part)
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
                        print "Parens found"
                        del exp[leftInner - 1:rightInner + 1]
                    else:
                        print "Left inner: %s Right inner: %s" % (leftInner, rightInner)
                        del exp[leftInner:rightInner]
                    print "After deletion: %s" % str(exp)
                    exp.insert(leftInner-1, newExp[0]) # -1s
                    print "After insertion: %s" % str(exp)
                else:
                    raise MathExpression.SyntaxError(newExp)
                print "New Expression: %s" % str(exp)
                print "...evaluate complete"
            print "Length of expression: %d" % len(exp)
            return self.evaluateExpression(exp)
        else:
            if len(exp) == 1:
                return self._interpret(exp[0])
            else:
                raise MathExpression.SyntaxError(exp)

    def _interpret(self, string):
        if isinstance(string, str):
            if string[0] == "<" and string[-1] == ">":
                varString = string[1:len(string) - 1]
                try:
                    print "Trying interpret %s as variable" % string
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
                #if string in self.operators:
                raise MathExpression.ParseFailure(string, AttributeError)
        else:
            return string

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
