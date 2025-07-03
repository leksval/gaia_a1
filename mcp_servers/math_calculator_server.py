# mcp_servers/math_calculator_server.py
"""
Mathematics and Calculations MCP Server

This server provides specialized tools for mathematical operations, calculations,
statistical analysis, and numerical problem solving.
"""

import json
import logging
from typing import Dict, Any, List, Union, Optional
import math
import statistics
import re
import numpy as np
from sympy import symbols, sympify, solve, simplify, expand, factor, parse_expr
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import io
import base64
import os

# Import assertions for error handling
from tools.assertions import (
    require, ensure, assert_not_none, assert_type, assert_non_empty,
    ConfigurationError, NetworkError, ProcessingError, assert_in_range
)

from mcp_servers.mcp_base import MCPServer, MCPToolDefinition, MCPResourceDefinition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("math_calculator_server")


class MathCalculatorServer(MCPServer):
    """MCP server for advanced mathematical operations and calculations."""
    
    def __init__(self):
        super().__init__(
            name="math_calculator",
            description="Performs mathematical operations, symbolic calculations, statistical analysis, and data visualization"
        )
        # Create a directory for storing temporary plots
        os.makedirs("temp_plots", exist_ok=True)
    
    def initialize(self):
        """Initialize the server with mathematical tools."""
        
        # Register basic calculator tool
        self.register_tool(
            MCPToolDefinition(
                name="calculate",
                description="Calculate the result of a mathematical expression",
                input_schema={
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                output_schema={
                    "result": {"type": "number", "description": "Calculated result"},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "Calculation steps"}
                },
                function=self.calculate
            )
        )
        
        # Register equation solver
        self.register_tool(
            MCPToolDefinition(
                name="solve_equation",
                description="Solve mathematical equations for unknown variables",
                input_schema={
                    "equation": {"type": "string", "description": "Equation to solve (e.g., 'x^2 + 2*x - 3 = 0')"},
                    "variable": {"type": "string", "description": "Variable to solve for (e.g., 'x')"}
                },
                output_schema={
                    "solutions": {"type": "array", "items": {"type": "string"}, "description": "Solutions to the equation"},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "Solution steps"}
                },
                function=self.solve_equation
            )
        )
        
        # Register statistical analysis tool
        self.register_tool(
            MCPToolDefinition(
                name="statistical_analysis",
                description="Perform statistical analysis on a dataset",
                input_schema={
                    "data": {"type": "array", "items": {"type": "number"}, "description": "Numerical dataset to analyze"},
                    "analyses": {
                        "type": "array", 
                        "items": {"type": "string", "enum": ["mean", "median", "mode", "std_dev", "variance", "range", "quartiles", "all"]},
                        "description": "Statistical measures to calculate"
                    }
                },
                output_schema={
                    "results": {"type": "object", "description": "Statistical analysis results"}
                },
                function=self.statistical_analysis
            )
        )
        
        # Register probability calculator
        self.register_tool(
            MCPToolDefinition(
                name="probability_calculator",
                description="Calculate probabilities for common distributions and scenarios",
                input_schema={
                    "distribution_type": {
                        "type": "string", 
                        "enum": ["binomial", "normal", "poisson", "custom"],
                        "description": "Probability distribution to use"
                    },
                    "parameters": {"type": "object", "description": "Parameters for the selected distribution"},
                    "query_type": {"type": "string", "enum": ["pdf", "cdf", "interval"], "description": "Type of probability calculation"}
                },
                output_schema={
                    "probability": {"type": "number", "description": "Calculated probability"},
                    "explanation": {"type": "string", "description": "Explanation of the calculation"}
                },
                function=self.probability_calculator
            )
        )
        
        # Register plot generator
        self.register_tool(
            MCPToolDefinition(
                name="generate_plot",
                description="Create plots and visualizations of mathematical functions or data",
                input_schema={
                    "plot_type": {
                        "type": "string", 
                        "enum": ["function", "scatter", "histogram", "bar", "box"],
                        "description": "Type of plot to generate"
                    },
                    "data": {
                        "type": "object", 
                        "description": "Data or function to plot (format depends on plot_type)"
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional plotting parameters (title, axis labels, etc.)"
                    }
                },
                output_schema={
                    "plot_data": {"type": "string", "description": "Base64-encoded plot image"},
                    "description": {"type": "string", "description": "Description of the plot"}
                },
                function=self.generate_plot
            )
        )
        
        # Register formula lookup resource
        self.register_resource(
            MCPResourceDefinition(
                uri_pattern="math://formula/",
                description="Access common mathematical formulas and identities",
                function=self.get_formula
            )
        )
    
    def calculate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the result of a mathematical expression with steps."""
        assert_not_none(args, "calculation arguments")
        
        expression = args.get("expression", "")
        assert_non_empty(expression.strip(), "mathematical expression")
        
        logger.info(f"Calculating expression: {expression}")
        
        # Sanitize input
        sanitized_expr = re.sub(r'[^0-9+\-*/^().\s]', '', expression)
        sanitized_expr = sanitized_expr.replace('^', '**')
        
        steps = []
        steps.append(f"Original expression: {expression}")
        
        # Evaluate using sympy for safer mathematical expression evaluation
        expr = parse_expr(sanitized_expr)
        result = float(expr.evalf())
        
        # Show intermediate steps for more complex expressions
        if '+' in sanitized_expr or '-' in sanitized_expr or '*' in sanitized_expr or '/' in sanitized_expr:
            steps.append(f"Evaluating: {sanitized_expr}")
            steps.append(f"Result: {result}")
        
        logger.info(f"Calculation result: {result}")
        return {
            "result": result,
            "steps": steps
        }
    
    def solve_equation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Solve an equation for a specified variable."""
        assert_not_none(args, "equation solving arguments")
        
        equation_str = args.get("equation", "")
        variable_str = args.get("variable", "x")
        
        assert_non_empty(equation_str.strip(), "equation")
        assert_non_empty(variable_str.strip(), "variable")
        
        logger.info(f"Solving equation {equation_str} for {variable_str}")
        
        steps = []
        steps.append(f"Original equation: {equation_str}")
        
        # Parse the equation string
        if '=' in equation_str:
            left_side, right_side = equation_str.split('=')
            equation_expr = f"({left_side})-({right_side})"
            steps.append(f"Rearranged to: {equation_expr} = 0")
        else:
            equation_expr = equation_str
            steps.append(f"Interpreted as: {equation_expr} = 0")
        
        # Use sympy to solve
        variable = symbols(variable_str)
        expr = sympify(equation_expr)
        solutions = solve(expr, variable)
        
        # Format solutions
        str_solutions = [str(sol) for sol in solutions]
        steps.append(f"Found {len(solutions)} solution(s):")
        for i, sol in enumerate(str_solutions):
            steps.append(f"Solution {i+1}: {variable_str} = {sol}")
        
        logger.info(f"Found {len(solutions)} solutions to equation")
        return {
            "solutions": str_solutions,
            "steps": steps
        }
    
    def statistical_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on a dataset."""
        assert_not_none(args, "statistical analysis arguments")
        
        data = args.get("data", [])
        analyses = args.get("analyses", ["mean"])
        
        assert_type(data, list, "data")
        assert_type(analyses, list, "analyses")
        assert_non_empty(data, "dataset")
        
        logger.info(f"Performing statistical analysis on dataset of {len(data)} points")
        
        if "all" in analyses:
            analyses = ["mean", "median", "mode", "std_dev", "variance", "range", "quartiles"]
        
        results = {}
        
        for analysis in analyses:
            if analysis == "mean":
                results["mean"] = statistics.mean(data)
            elif analysis == "median":
                results["median"] = statistics.median(data)
            elif analysis == "mode":
                # Handle case where no unique mode exists
                mode_result = statistics.multimode(data)
                if len(mode_result) == 1:
                    results["mode"] = mode_result[0]
                else:
                    results["mode"] = f"Multiple modes: {mode_result}"
            elif analysis == "std_dev":
                results["std_dev"] = statistics.stdev(data) if len(data) > 1 else 0
            elif analysis == "variance":
                results["variance"] = statistics.variance(data) if len(data) > 1 else 0
            elif analysis == "range":
                results["range"] = max(data) - min(data)
            elif analysis == "quartiles":
                sorted_data = sorted(data)
                results["quartiles"] = {
                    "Q1": np.percentile(sorted_data, 25),
                    "Q2": np.percentile(sorted_data, 50),  # median
                    "Q3": np.percentile(sorted_data, 75)
                }
        
        # Add summary
        results["summary"] = f"Dataset with {len(data)} points. "
        if "mean" in results and "std_dev" in results:
            results["summary"] += f"Mean: {results['mean']:.2f}, StdDev: {results['std_dev']:.2f}"
        
        logger.info(f"Statistical analysis complete: {list(results.keys())}")
        return {"results": results}
    
    def probability_calculator(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate probabilities for different distributions."""
        assert_not_none(args, "probability calculation arguments")
        
        dist_type = args.get("distribution_type", "")
        parameters = args.get("parameters", {})
        query_type = args.get("query_type", "")
        
        assert_non_empty(dist_type, "distribution_type")
        assert_non_empty(query_type, "query_type")
        assert_type(parameters, dict, "parameters")
        
        logger.info(f"Calculating {query_type} probability for {dist_type} distribution")
        
        explanation = f"Calculating {query_type} for {dist_type} distribution with parameters {parameters}"
        probability = 0.0
        
        if dist_type == "binomial":
            n = parameters.get("n", 10)  # number of trials
            p = parameters.get("p", 0.5)  # success probability
            
            assert_in_range(n, 1, 10000, "number of trials")
            assert_in_range(p, 0.0, 1.0, "success probability")
            
            if query_type == "pdf":
                k = parameters.get("k", 5)  # number of successes
                assert_in_range(k, 0, n, "number of successes")
                from scipy.stats import binom
                probability = binom.pmf(k, n, p)
                explanation = f"P(X = {k} | n = {n}, p = {p}) = {probability:.6f}"
            elif query_type == "cdf":
                k = parameters.get("k", 5)  # number of successes
                assert_in_range(k, 0, n, "number of successes")
                from scipy.stats import binom
                probability = binom.cdf(k, n, p)
                explanation = f"P(X ≤ {k} | n = {n}, p = {p}) = {probability:.6f}"
            elif query_type == "interval":
                lower = parameters.get("lower", 3)
                upper = parameters.get("upper", 7)
                assert_in_range(lower, 0, n, "lower bound")
                assert_in_range(upper, lower, n, "upper bound")
                from scipy.stats import binom
                probability = binom.cdf(upper, n, p) - binom.cdf(lower - 1, n, p)
                explanation = f"P({lower} ≤ X ≤ {upper} | n = {n}, p = {p}) = {probability:.6f}"
        
        elif dist_type == "normal":
            mu = parameters.get("mean", 0)
            sigma = parameters.get("std_dev", 1)
            
            require(sigma > 0, "Standard deviation must be positive", context={"sigma": sigma})
            
            if query_type == "pdf":
                x = parameters.get("x", 0)
                from scipy.stats import norm
                probability = norm.pdf(x, mu, sigma)
                explanation = f"PDF at x = {x} for Normal(μ = {mu}, σ = {sigma}) = {probability:.6f}"
            elif query_type == "cdf":
                x = parameters.get("x", 0)
                from scipy.stats import norm
                probability = norm.cdf(x, mu, sigma)
                explanation = f"P(X ≤ {x} | μ = {mu}, σ = {sigma}) = {probability:.6f}"
            elif query_type == "interval":
                lower = parameters.get("lower", -1)
                upper = parameters.get("upper", 1)
                require(upper >= lower, "Upper bound must be >= lower bound", 
                       context={"lower": lower, "upper": upper})
                from scipy.stats import norm
                probability = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
                explanation = f"P({lower} ≤ X ≤ {upper} | μ = {mu}, σ = {sigma}) = {probability:.6f}"
        
        logger.info(f"Probability calculation complete: {probability}")
        return {
            "probability": float(probability),
            "explanation": explanation
        }
    
    def generate_plot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plots of mathematical functions or data."""
        assert_not_none(args, "plot generation arguments")
        
        plot_type = args.get("plot_type", "function")
        data = args.get("data", {})
        params = args.get("params", {})
        
        assert_non_empty(plot_type, "plot_type")
        assert_type(data, dict, "data")
        assert_type(params, dict, "params")
        
        logger.info(f"Generating {plot_type} plot")
        
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Set title and labels
        title = params.get("title", f"{plot_type.title()} Plot")
        x_label = params.get("x_label", "x")
        y_label = params.get("y_label", "y")
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(params.get("grid", True))
        
        description = f"{plot_type.title()} plot"
        
        if plot_type == "function":
            # Plot a mathematical function
            func_str = data.get("function", "x**2")
            x_min = data.get("x_min", -10)
            x_max = data.get("x_max", 10)
            num_points = data.get("num_points", 1000)
            
            assert_non_empty(func_str, "function string")
            require(x_max > x_min, "x_max must be greater than x_min", 
                   context={"x_min": x_min, "x_max": x_max})
            assert_in_range(num_points, 10, 10000, "num_points")
            
            x = np.linspace(x_min, x_max, num_points)
            
            # Replace ^ with ** for power
            func_str = func_str.replace("^", "**")
            
            # Parse the expression with sympy
            expr = parse_expr(func_str)
            # Convert to a numpy-compatible lambda function
            f = lambdify('x', expr, modules=['numpy'])
            # Apply the function to x values
            y = f(x)
            plt.plot(x, y)
            description = f"Plot of function {func_str} from x = {x_min} to {x_max}"
        
        elif plot_type == "scatter":
            # Scatter plot of points
            x_data = data.get("x", [])
            y_data = data.get("y", [])
            
            assert_type(x_data, list, "x_data")
            assert_type(y_data, list, "y_data")
            require(len(x_data) == len(y_data), "x and y data must have same length",
                   context={"x_len": len(x_data), "y_len": len(y_data)})
            
            plt.scatter(x_data, y_data)
            description = f"Scatter plot with {len(x_data)} data points"
        
        elif plot_type == "histogram":
            # Histogram of data
            values = data.get("values", [])
            bins = data.get("bins", 10)
            
            assert_type(values, list, "values")
            assert_non_empty(values, "values")
            assert_in_range(bins, 1, 100, "bins")
            
            plt.hist(values, bins=bins)
            description = f"Histogram with {len(values)} data points using {bins} bins"
        
        elif plot_type == "bar":
            # Bar chart
            categories = data.get("categories", [])
            values = data.get("values", [])
            
            assert_type(categories, list, "categories")
            assert_type(values, list, "values")
            require(len(categories) == len(values), "categories and values must have same length",
                   context={"cat_len": len(categories), "val_len": len(values)})
            
            plt.bar(categories, values)
            plt.xticks(rotation=45)
            description = f"Bar chart with {len(categories)} categories"
        
        elif plot_type == "box":
            # Box plot
            all_data = data.get("data", [])
            labels = data.get("labels", [])
            
            assert_type(all_data, list, "data")
            assert_non_empty(all_data, "data")
            
            plt.boxplot(all_data, labels=labels)
            description = f"Box plot with {len(all_data)} datasets"
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close the figure to free memory
        plt.close()
        
        logger.info(f"Plot generation complete: {description}")
        return {
            "plot_data": plot_data,
            "description": description
        }
    
    def get_formula(self, uri: str) -> Dict[str, Any]:
        """Retrieve mathematical formulas by URI."""
        assert_not_none(uri, "formula URI")
        
        # Extract formula name from URI
        formula_name = uri.replace("math://formula/", "").lower()
        
        formulas = {
            "quadratic": {
                "name": "Quadratic Formula",
                "formula": "x = (-b ± √(b² - 4ac)) / 2a",
                "description": "Solves the quadratic equation ax² + bx + c = 0",
                "variables": ["a", "b", "c"]
            },
            "pythagorean": {
                "name": "Pythagorean Theorem",
                "formula": "a² + b² = c²",
                "description": "Relates the sides of a right triangle",
                "variables": ["a", "b", "c"]
            },
            "area_circle": {
                "name": "Area of a Circle",
                "formula": "A = πr²",
                "description": "Calculates the area of a circle with radius r",
                "variables": ["r"]
            }
        }
        
        # If no specific formula requested, return list of available formulas
        if not formula_name or formula_name == "list":
            return {
                "available_formulas": list(formulas.keys()),
                "formula_count": len(formulas)
            }
        
        # Return the requested formula if available
        if formula_name in formulas:
            return formulas[formula_name]
        else:
            return {
                "error": f"Formula '{formula_name}' not found",
                "available_formulas": list(formulas.keys())
            }


def main():
    """Main entry point for the math calculator server."""
    server = MathCalculatorServer()
    server.start()


if __name__ == "__main__":
    main()