import os
import json
import random

# Load test results from JSON file
with open('gaia_api_test_results.json', 'r') as f:
    gaia_api_test_results = json.load(f)

def get_available_servers():
    available_servers = os.getenv('AVAILABLE_SERVERS', 'default_server_1,default_server_2').split(',')
    return available_servers

# Helper function to generate queries
def generate_query(question):
    # Assuming some logic here to generate the query string based on the question
    return f"SELECT answer FROM knowledge_base WHERE question = '{question}'"

def check_answer(question, correct_answer, query):
    # Replace this with actual querying mechanism if needed.
    # Here we assume that querying mechanism returns a dict with an 'answer' key.
    response = { "answer": query }

    assert response["answer"] == correct_answer, f"Test failed for question: {question}"

# Select one random question
random_question = random.choice(gaia_api_test_results)
query = generate_query(random_question["question"])
check_answer(random_question["question"], random_question["answer"], query)

# Select two more random questions
for _ in range(2):
    random_question = random.choice(gaia_api_test_results)
    query = generate_query(random_question["question"])
    check_answer(random_question["question"], random_question["answer"], query)

# Check all questions and break on failure
for test_case in gaia_api_test_results:
    query = generate_query(test_case["question"])
    check_answer(test_case["question"], test_case["answer"], query)